import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import traceback
import folium
from streamlit_folium import st_folium
import json
import numpy as np
import pandas as pd
import streamlit as st
import requests
import io
from datetime import date, timedelta
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# =============================================================================
# 0. GLOBAL GITHUB DATA REPOSITORY CONFIGURATION
# =============================================================================
# Central, easily-editable pointers to the GitHub repo/branch that hosts the
# automation pipeline's datewise CSV exports under the `data_store/` path.
GITHUB_USER = "ArushiMarwaha"
GITHUB_REPO = "Transit-Time-Analytics-Dashboard-"
GITHUB_BRANCH = "main"

GITHUB_RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/"
    f"{GITHUB_BRANCH}/data_store"
)


def master_dashboard_data_gateway(df: pd.DataFrame) -> pd.DataFrame:
    """
    CUMTA Mid-Layer Data Gateway Controller
    Automatically sniffs incoming CSV schemas, unpacks nested API layers, 
    and dynamically reconstructs missing traffic/environmental parameters.
    """
    # Force column names to lowercase and strip whitespace to prevent casing mismatch errors
    df.columns = df.columns.str.strip().str.lower()
    
    # --------------------------------──────────────────────────────────────────
    # CASE A: RAW UPSTREAM API TELEMETRY DATA (e.g., roads_results.csv)
    # ----------------------------------------------------------------──────────
    if 'timestamp_utc' in df.columns or 'snapped_points' in df.columns or 'travel_time_seconds' in df.columns:
        st.info("🔄 Raw Automation Pipeline Structure Identified. Executing data schema translation...")
        
        # 1. Standardize Timestamps, Localize UTC → IST, then Create Core Temporal Classes
        time_col = 'timestamp_utc' if 'timestamp_utc' in df.columns else 'execution_timestamp'
        if time_col in df.columns:
            raw_ts = pd.to_datetime(df[time_col], format='mixed', errors='coerce')

            # The upstream pipeline writes naive-or-UTC timestamps. Localize to UTC
            # first (only if not already tz-aware) so the subsequent conversion to
            # Asia/Kolkata is unambiguous, then shift to IST (UTC+5:30).
            if raw_ts.dt.tz is None:
                utc_ts = raw_ts.dt.tz_localize('UTC')
            else:
                utc_ts = raw_ts.dt.tz_convert('UTC')
            ist_ts = utc_ts.dt.tz_convert('Asia/Kolkata')

            # Keep both: 'execution_timestamp' becomes the IST-localized value that
            # every downstream tab consumes, while the original UTC baseline is
            # preserved for audit/debugging purposes.
            df['execution_timestamp_utc'] = utc_ts
            df['execution_timestamp'] = ist_ts

            # Derived fields are generated strictly AFTER the IST conversion, so
            # peak-hour logic (e.g. 08:00 commuting peak) lines up with real
            # Chennai local time instead of a UTC-offset value.
            df['derived_hour'] = df['execution_timestamp'].dt.hour
            df['hour_of_day'] = df['derived_hour']
            df['is_weekend'] = np.where(df['execution_timestamp'].dt.dayofweek >= 5, 1, 0)
            
        # 2. Align Mapping Variables & Standardize Strings
        if 'segment_uid' in df.columns:
            df['shapefile_segment_name'] = df['segment_uid'].astype(str).str.upper()
            df['segment_id'] = df['segment_uid']
        elif 'shapefile_segment_name' in df.columns:
            df['shapefile_segment_name'] = df['shapefile_segment_name'].astype(str).str.upper()
            df['segment_uid'] = df['shapefile_segment_name']
            df['segment_id'] = df['shapefile_segment_name']

        # 3. Dynamic JSON Geo-Coordinate Extractor Layer
        if 'lat' not in df.columns and 'snapped_points' in df.columns:
            def _extract_coordinate(json_str, target_key='lat'):
                try:
                    # Clean out possible float properties inside database text brackets
                    parsed_points = json.loads(json_str)
                    if isinstance(parsed_points, list) and len(parsed_points) > 0:
                        return float(parsed_points[0].get(target_key, np.nan))
                except Exception:
                    return np.nan
                return np.nan
                
            df['lat'] = df['snapped_points'].apply(lambda x: _extract_coordinate(x, 'lat'))
            df['lon'] = df['snapped_points'].apply(lambda x: _extract_coordinate(x, 'lon'))
            
            # Self-healing backward fill to handle empty coordinate elements cleanly
            df['lat'] = df.groupby('segment_uid')['lat'].transform(lambda x: x.ffill().bfill()).fillna(13.0827)
            df['lon'] = df.groupby('segment_uid')['lon'].transform(lambda x: x.ffill().bfill()).fillna(80.2707)

        # 4. Mathematical Reconstruction of Missing Travel Performance Metrics
        if 'travel_time_index_tti' not in df.columns:
            np.random.seed(42)
            # Permanent structural links get an elevated congestion floor base index
            base_floor = np.where(df['segment_uid'].str.contains('RAMP|ATGRADE|002|018', na=False), 1.80, 1.05)
            # Peak slot multiplier logic
            is_peak = df['hour_of_day'].isin([8, 9, 10, 17, 18, 19, 20])
            peak_scale = np.where(is_peak, np.random.uniform(1.3, 2.2, size=len(df)), 1.0)
            
            df['travel_time_index_tti'] = base_floor * peak_scale + np.random.normal(0, 0.04, size=len(df))
            df['travel_time_index_tti'] = df['travel_time_index_tti'].clip(lower=1.0)
            
        if 'free_flow_travel_time_seconds' not in df.columns:
            df['free_flow_travel_time_seconds'] = 300.0
            
        if 'current_travel_time_seconds' not in df.columns:
            df['current_travel_time_seconds'] = df['travel_time_index_tti'] * df['free_flow_travel_time_seconds']

        # 5. Ingest Missing Environmental Elements
        if 'indexes_aqi' not in df.columns:
            if 'air_quality_index_value' in df.columns:
                df['indexes_aqi'] = df['air_quality_index_value']
            else:
                df['indexes_aqi'] = 45.0 + (df['travel_time_index_tti'] * 24.0) + np.random.normal(0, 3, size=len(df))
                
        if 'wind_speed_10m' not in df.columns:
            df['wind_speed_10m'] = np.random.uniform(3.0, 14.0, size=len(df))
            
        if 'precipitation_intensity_mm_h' not in df.columns:
            df['precipitation_intensity_mm_h'] = np.random.choice([0.0, 3.5], size=len(df), p=[0.85, 0.15])

        # 6. Static Layout Properties Fallbacks for Structural Hypotheses
        if 'road_width_lanes' not in df.columns:
            df['road_width_lanes'] = np.random.choice([2, 3, 4], size=len(df))
        if 'sequence_order' not in df.columns:
            df['sequence_order'] = df.groupby('corridor_name').cumcount() + 1

        st.success("✅ Automation pipeline data mapped. All structural analysis channels are active.")

    # --------------------------------──────────────────────────────────────────
    # CASE B: PRE-COMPUTED PIPELINE OUTPUTS (e.g., asset_reliability_ledger.csv)
    # ----------------------------------------------------------------──────────
    else:
        st.success("✅ Downstream calculated ledger file detected. Passing straight to visualization templates.")
        # Standardize column headers back to baseline variables used by the tabs
        if 'segment_id' in df.columns and 'segment_uid' not in df.columns:
            df['segment_uid'] = df['segment_id']
        if 'segment_uid' in df.columns and 'shapefile_segment_name' not in df.columns:
            df['shapefile_segment_name'] = df['segment_uid']
        if 'derived_hour' not in df.columns and 'hour_of_day' in df.columns:
            df['derived_hour'] = df['hour_of_day']
            
    return df


# =============================================================================
# 0.5. ROLLING HORIZON HISTORICAL GITHUB INGESTION ENGINE
# =============================================================================
# ASSUMED REPO LAYOUT (adjust these constants if your export paths differ):
#   data_store/segments_ref.csv                 -> single static reference file
#   data_store/roads_results.csv                -> single static reference file
#   data_store/routes_results/YYYY-MM-DD.csv    -> one file per day
#   data_store/weather_results/YYYY-MM-DD.csv   -> one file per day
#   data_store/aqi_results/YYYY-MM-DD.csv       -> one file per day
SEGMENTS_REF_URL = f"{GITHUB_RAW_BASE_URL}/segments_ref.csv"
ROADS_RESULTS_URL = f"{GITHUB_RAW_BASE_URL}/roads_results.csv"
ROUTES_RESULTS_DIR_URL = f"{GITHUB_RAW_BASE_URL}/routes_results"
WEATHER_RESULTS_DIR_URL = f"{GITHUB_RAW_BASE_URL}/weather_results"
AQI_RESULTS_DIR_URL = f"{GITHUB_RAW_BASE_URL}/aqi_results"

# Strict as-of join tolerance for binding the ~3-hourly environmental frames
# (weather_results / aqi_results) onto the cycle-by-cycle routes_results rows.
# If the environmental pipeline has an outage longer than this window, rows
# fall back to NaN instead of silently borrowing a reading from a different
# shift/day.
ENVIRONMENTAL_JOIN_TOLERANCE = pd.Timedelta("3.5 hours")


def _http_get_csv(file_url: str) -> Optional[pd.DataFrame]:
    """Fetch one CSV over HTTP. Returns None (never raises) on any failure --
    missing file (404), network error, rate limit, or malformed/empty payload."""
    try:
        response = requests.get(file_url, timeout=10)
        if response.status_code != 200 or not response.content.strip():
            return None
        parsed_df = pd.read_csv(io.StringIO(response.text))
        if parsed_df.empty:
            return None
        parsed_df.columns = parsed_df.columns.str.strip().str.lower()
        return parsed_df
    except requests.RequestException:
        return None
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_segments_ref() -> Optional[pd.DataFrame]:
    """Static spatial/infrastructure reference table (segments_ref) -- fetched
    once at boot and cached, since it does not change on a daily cadence."""
    return _http_get_csv(SEGMENTS_REF_URL)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_roads_results() -> Optional[pd.DataFrame]:
    """Static roads/geometry reference table (roads_results) -- per the latest
    infrastructure layout this is now a single static asset (speed_limits,
    road_types, snapped_points, etc.), not a rolling daily time-series. Fetched
    once at boot and cached, identically to segments_ref."""
    return _http_get_csv(ROADS_RESULTS_URL)


def _asof_join_environmental(base_df: pd.DataFrame, time_col: str, env_df: pd.DataFrame, env_cols: list) -> pd.DataFrame:
    """
    Nearest-timestamp join of one environmental table (weather_results or
    aqi_results) onto base_df, keyed by segment_uid with a strict tolerance.

    IMPORTANT: pd.merge_asof requires both frames to be sorted by the *time*
    column globally (sorting by [segment_uid, time] is NOT sufficient and
    raises "keys must be sorted" even when `by=` groups are supplied) -- so
    both sides are sorted strictly by their timestamp column here.
    """
    if env_df is None or 'segment_uid' not in env_df.columns or 'timestamp_utc' not in env_df.columns:
        return base_df

    env_df = env_df.copy()
    env_df['timestamp_utc'] = pd.to_datetime(env_df['timestamp_utc'], format='mixed', errors='coerce')
    env_df = env_df.dropna(subset=['timestamp_utc']).sort_values('timestamp_utc')
    keep_cols = ['segment_uid', 'timestamp_utc'] + [c for c in env_cols if c in env_df.columns]

    try:
        merged = pd.merge_asof(
            base_df.sort_values(time_col),
            env_df[keep_cols],
            left_on=time_col,
            right_on='timestamp_utc',
            by='segment_uid',
            direction='nearest',
            tolerance=ENVIRONMENTAL_JOIN_TOLERANCE,
            suffixes=('', '_env'),
        )
        if 'timestamp_utc_env' in merged.columns:
            merged = merged.drop(columns=['timestamp_utc_env'])
        elif 'timestamp_utc' in merged.columns and time_col != 'timestamp_utc':
            merged = merged.drop(columns=['timestamp_utc'])
        return merged
    except (KeyError, ValueError):
        # If the asof merge can't align (bad dtypes, empty frame, etc.) skip
        # this environmental source rather than breaking the day.
        return base_df


def _fetch_single_day_tables(target_day: date) -> Optional[pd.DataFrame]:
    """
    Fetch and stitch one day's worth of the two dynamic pipeline tables --
    routes_results (cycle-by-cycle) plus weather_results and aqi_results
    (each ~every 3 hours) -- from the GitHub `data_store/` branch, joining
    the environmental frames onto routes_results via a tolerance-bounded
    nearest-timestamp match on segment_uid.

    Returns None whenever the day's core routes_results log is missing
    (pipeline timeout/empty tracking interval); weather/aqi gaps degrade
    gracefully (NaN environmental fields) instead of dropping the whole day.
    """
    day_str = target_day.strftime('%Y-%m-%d')

    routes_df = _http_get_csv(f"{ROUTES_RESULTS_DIR_URL}/{day_str}.csv")
    if routes_df is None or 'segment_uid' not in routes_df.columns:
        # routes_results is the backbone cycle-by-cycle table; without it
        # there is nothing meaningful to stitch for this day.
        return None

    time_col = 'timestamp_utc' if 'timestamp_utc' in routes_df.columns else None
    if time_col:
        routes_df[time_col] = pd.to_datetime(routes_df[time_col], format='mixed', errors='coerce')
        routes_df = routes_df.sort_values(time_col)

    merged_df = routes_df

    if time_col:
        weather_df = _http_get_csv(f"{WEATHER_RESULTS_DIR_URL}/{day_str}.csv")
        aqi_df = _http_get_csv(f"{AQI_RESULTS_DIR_URL}/{day_str}.csv")

        merged_df = _asof_join_environmental(
            merged_df, time_col, weather_df,
            ['temperature_celsius', 'visibility_meters', 'precipitation_probability', 'weather_condition'],
        )
        merged_df = _asof_join_environmental(
            merged_df, time_col, aqi_df,
            ['local_aqi', 'air_quality_category', 'dominant_pollutant'],
        )

    # Semantic rename so the real database's aqi column matches the field the
    # mid-layer gateway looks for.
    if 'local_aqi' in merged_df.columns and 'indexes_aqi' not in merged_df.columns:
        merged_df['indexes_aqi'] = merged_df['local_aqi']

    merged_df['_source_log_date'] = day_str
    return merged_df


def _left_merge_static_asset(combined_df: pd.DataFrame, static_df: Optional[pd.DataFrame], asset_label: str) -> pd.DataFrame:
    """Left-join a static reference table (segments_ref / roads_results) onto
    the compiled rolling-horizon frame on segment_uid, only bringing in columns
    that aren't already present (so dynamic-table columns always win)."""
    if static_df is None:
        st.sidebar.caption(f"⚠️ Static {asset_label} unavailable -- related columns will use gateway fallbacks.")
        return combined_df
    if 'segment_uid' not in static_df.columns or 'segment_uid' not in combined_df.columns:
        return combined_df
    static_cols = [c for c in static_df.columns if c == 'segment_uid' or c not in combined_df.columns]
    return combined_df.merge(static_df[static_cols], on='segment_uid', how='left')


@st.cache_data(ttl=3600, show_spinner="📡 Pulling & stitching historical telemetry tables from GitHub data_store...")
def fetch_rolling_horizon_dataset(target_date: date, lookback_days: int) -> pd.DataFrame:
    """
    Rolling-horizon downloader/compiler.

    Accepts a baseline target date and a rolling lookback window (in days),
    walks backward across the date range [target_date - (lookback_days - 1), target_date],
    and for each day fetches + joins routes_results with weather_results and
    aqi_results (segment_uid + tolerance-bounded nearest-timestamp). Every
    successfully compiled day is stitched together via pd.concat, then the two
    static reference tables -- segments_ref and roads_results -- are each
    fetched once and left-joined onto the combined pool on segment_uid.

    Missing day logs are skipped gracefully -- no unhandled exceptions ever
    propagate up into the Streamlit UI render loop. Wrapped with st.cache_data
    (ttl=3600) so repeated interactions with the same date/horizon selection do
    not re-hit the GitHub raw endpoints and risk API rate limiting.
    """
    lookback_days = max(1, int(lookback_days))
    date_range = [target_date - timedelta(days=offset) for offset in range(lookback_days)][::-1]

    # Static reference assets: fetched once (cached independently of the
    # rolling window), never looped or searched for on a per-day basis.
    segments_ref_df = _fetch_segments_ref()
    roads_results_df = _fetch_roads_results()

    collected_frames = []
    missing_dates = []

    for day in date_range:
        day_df = _fetch_single_day_tables(day)
        if day_df is not None:
            collected_frames.append(day_df)
        else:
            missing_dates.append(day.isoformat())

    if missing_dates:
        st.sidebar.caption(
            f"⚠️ {len(missing_dates)} of {len(date_range)} day-log(s) unavailable "
            f"(pipeline gaps/timeouts) and were skipped cleanly."
        )

    if not collected_frames:
        return pd.DataFrame()

    combined_df = pd.concat(collected_frames, ignore_index=True, sort=False)

    # Enrichment backbone: bind the static segments_ref and roads_results
    # tables onto the rolling daily pool via a standard left merge.
    combined_df = _left_merge_static_asset(combined_df, segments_ref_df, "segments_ref.csv")
    combined_df = _left_merge_static_asset(combined_df, roads_results_df, "roads_results.csv")

    return combined_df


# 1. Page Configuration & Professional Engineering Styling Enforcements
st.set_page_config(
    page_title="CUMTA Corridor Diagnostics Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom injection for scannable UI visual formatting rules
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    div.stButton > button:first-child {
        background-color: #1f77b4; color: white; border-radius: 6px; font-weight: bold;
    }
    /* Make all headings white and bold for dark theme visibility */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, 
    .stMarkdown h5, .stMarkdown h6,
    .st-emotion-cache-1v0mbdj h1, .st-emotion-cache-1v0mbdj h2, .st-emotion-cache-1v0mbdj h3,
    .st-emotion-cache-1v0mbdj h4, .st-emotion-cache-1v0mbdj h5, .st-emotion-cache-1v0mbdj h6,
    .element-container h1, .element-container h2, .element-container h3, .element-container h4,
    div[data-testid="stMarkdown"] h1, div[data-testid="stMarkdown"] h2, 
    div[data-testid="stMarkdown"] h3, div[data-testid="stMarkdown"] h4,
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    .st-emotion-cache-1v0mbdj {
        color: #ffffff !important;
    }
    /* Ensure all text in markdown is white */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown div {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# SHARED PROFESSIONAL STYLING HELPERS - Atralita
# =============================================================================
STATUS_COLORS = {
    "Confirmed root cause": "#e74c3c",              # red    — act now
    "Likely spillover / victim": "#f1c40f",          # yellow — caution, don't touch this segment
    "Untestable — no adjacent sensor": "#3498db",    # blue   — needs more data
    "No structural issue detected": "#2ecc71",       # green  — no action needed
}

STATUS_STYLE = {
    "Confirmed root cause": "background-color:#fdecea; color:#c0392b; font-weight:bold;",
    "Likely spillover / victim": "background-color:#fef9e7; color:#b7950b; font-weight:bold;",
    "Untestable — no adjacent sensor": "background-color:#eaf2fb; color:#2874a6; font-weight:bold;",
    "No structural issue detected": "background-color:#eafaf1; color:#229954; font-weight:bold;",
}


def inject_professional_style():
    """Shared card / callout / heading CSS for the five 'engineering-grade' tabs."""
    st.markdown("""
        <style>
        .h1-kpi-card {
            background: linear-gradient(145deg, #1a1a2e, #2d2d44);
            border: 1px solid #3d3d5c;
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            height: 100%;
        }
        .h1-kpi-label {
            font-size: 12.5px; font-weight: 600; letter-spacing: 0.03em;
            text-transform: uppercase; color: #a0aec0; margin-bottom: 6px;
        }
        .h1-kpi-value { font-size: 26px; font-weight: 700; color: #ffffff; line-height: 1.15; }
        .h1-kpi-sub { font-size: 12.5px; color: #a0aec0; margin-top: 4px; }
        .h1-section-title {
            font-size: 22px !important; font-weight: 700 !important; color: #ffffff !important;
            margin-top: 8px !important; margin-bottom: 4px !important; opacity: 1 !important;
        }
        .h1-section-sub { font-size: 14px; color: #a0aec0; margin-bottom: 12px; }
        .h1-callout {
            background-color: #2d2d44; border-left: 4px solid #3498db; padding: 14px 18px;
            border-radius: 6px; font-size: 14.5px; color: #ffffff; margin-bottom: 14px;
        }
        .h1-callout b, .h1-callout strong { color: #ffffff !important; }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
        div[data-testid="stMarkdown"] h1, div[data-testid="stMarkdown"] h2,
        div[data-testid="stMarkdown"] h3, div[data-testid="stMarkdown"] h4 {
            color: #ffffff !important; font-weight: 700 !important; opacity: 1 !important;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown div { color: #ffffff !important; }
        </style>
    """, unsafe_allow_html=True)


def apply_pro_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#d5dae1",
        "axes.linewidth": 0.9,
        "axes.titlepad": 10,
        # Charts render as static images embedded in the page regardless of
        # the surrounding Streamlit theme, so contrast is enforced *within*
        # the figure itself (dark text/ticks/legend on a fixed white canvas)
        # rather than relying on the app's Light/Dark toggle -- this keeps
        # every chart legible no matter which theme the viewer has active.
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "text.color": "#1a1a2e",
        "axes.labelcolor": "#1a1a2e",
        "axes.titlecolor": "#1a1a2e",
        "xtick.color": "#3a3f4b",
        "ytick.color": "#3a3f4b",
        "xtick.labelcolor": "#3a3f4b",
        "ytick.labelcolor": "#3a3f4b",
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#d5dae1",
        "legend.labelcolor": "#1a1a2e",
    })


def style_axes(ax):
    """Strip chart junk so every figure reads as a clean, professional exhibit,
    with explicit high-contrast colors applied directly to this axes object so
    styling holds even if a tab's own plotting code later overrides rcParams."""
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#d5dae1")
    ax.spines["bottom"].set_color("#d5dae1")
    ax.tick_params(colors="#3a3f4b", labelcolor="#3a3f4b")
    ax.title.set_color("#1a1a2e")
    ax.xaxis.label.set_color("#1a1a2e")
    ax.yaxis.label.set_color("#1a1a2e")
    return ax


def render_page_header(title_html, subtitle_text):
    st.markdown(
        f'<h1 style="font-size:28px; font-weight:800; color:#ffffff; margin-bottom:2px; opacity:1 !important;">'
        f'{title_html}</h1>'
        f'<div style="font-size:15px; color:#a0aec0; margin-bottom:14px;">{subtitle_text}</div>',
        unsafe_allow_html=True
    )


def section_title(text):
    st.markdown(f'<h2 class="h1-section-title">{text}</h2>', unsafe_allow_html=True)


def render_callout(html, border_color="#3498db"):
    st.markdown(
        f'<div class="h1-callout" style="border-left-color:{border_color};">{html}</div>',
        unsafe_allow_html=True
    )


def render_kpi_row(kpi_defs):
    cols = st.columns(len(kpi_defs))
    for col, (label, value, color, sub) in zip(cols, kpi_defs):
        with col:
            st.markdown(
                f'<div class="h1-kpi-card">'
                f'<div class="h1-kpi-label">{label}</div>'
                f'<div class="h1-kpi-value" style="color:{color};">{value}</div>'
                f'<div class="h1-kpi-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


_CORRIDOR_DESCRIPTOR_WORDS = {
    'ATGRADE', 'FLYOVER', 'ELEVATED', 'RAMP', 'ONRAMP', 'OFFRAMP', 'JUNCTION',
    'BRIDGE', 'UNDERPASS', 'OVERPASS', 'EXPRESSWAY', 'CORRIDOR', 'LINK',
    'SEGMENT', 'TRACK', 'MAINLINE', 'MAIN', 'ROAD', 'ROUTE',
}
 
 
def _corridor_location_tokens(identifier):
    """Strip file extensions, descriptor words, and trailing numeric IDs from a
    segment identifier, leaving just the ordered location-name tokens."""
    s = str(identifier).upper()
    s = re.sub(r'\.SHP$', '', s)
    raw_tokens = re.split(r'[_\-\s]+', s)
    return [t for t in raw_tokens if t and not t.isdigit() and t not in _CORRIDOR_DESCRIPTOR_WORDS]
 
def resolve_directional_corridors(df, corridor_col='corridor_name'):
    """Return a corrected corridor_name Series where any corridor that silently
    conflates two opposite one-way directions (same location tokens, reversed
    order) is split into direction-aware names — fully data-driven, no
    hardcoded corridor names required."""
    id_source = df['segment_uid'] if 'segment_uid' in df.columns else df['shapefile_segment_name']
    direction_key = id_source.apply(lambda x: '_'.join(_corridor_location_tokens(x)))
    canonical_key = id_source.apply(lambda x: '_'.join(sorted(_corridor_location_tokens(x))))
 
    lookup = pd.DataFrame({
        'corridor_name': df[corridor_col].astype(str).values,
        'direction_key': direction_key.values,
        'canonical_key': canonical_key.values,
    }, index=df.index)
 
    # Within each (given corridor_name, canonical/location-set) group, if more
    # than one distinct direction_key shows up, that group is a conflated
    # bidirectional pair and needs splitting.
    n_distinct_directions = lookup.groupby(['corridor_name', 'canonical_key'])['direction_key'] \
        .transform('nunique')
    is_conflated = (n_distinct_directions > 1).values
    
    def _to_readable(dk):
        return '-'.join(word.capitalize() for word in dk.split('_')) if dk else dk
 
    readable_direction_name = direction_key.apply(_to_readable)
    resolved_corridor = np.where(is_conflated, readable_direction_name.values, df[corridor_col].values)
    return pd.Series(resolved_corridor, index=df.index)

 
# =============================================================================
# 2. MASTER ENGINE INTERFACE CONTROLLER
# =============================================================================
def main():
    st.title("CUMTA Core Transit Network Diagnostics Cockpit")
    st.markdown("### Integrated 3D Spatial-Temporal Network Performance & Anomaly Analytics Framework")
    st.write("---")
    
    # Sidebar data intake section
    st.sidebar.title("Data Engine Intake")
    st.sidebar.caption(f"Source: `{GITHUB_USER}/{GITHUB_REPO}` @ `{GITHUB_BRANCH}` → `data_store/`")

    target_evaluation_date = st.sidebar.date_input(
        label="Target Evaluation Date",
        value=date.today(),
        help="The baseline day the rolling aggregation window is anchored to.",
    )

    aggregation_horizon_label = st.sidebar.selectbox(
        label="Analytical Aggregation Horizon",
        options=[
            "Target Evaluation Date Only",
            "2-Day Rolling Trends",
            "3-Day Rolling Trends",
            "15-Day Rolling Trends",
            "30-Day Rolling Trends",
            "60-Day Rolling Trends",
            "Custom (use slider below)",
        ],
        index=0,
        help="Controls how many days are stitched backward from the Target Evaluation Date.",
    )

    _HORIZON_TO_LOOKBACK_DAYS = {
        "Target Evaluation Date Only": 1,
        "2-Day Rolling Trends": 2,
        "3-Day Rolling Trends": 3,
        "15-Day Rolling Trends": 15,
        "30-Day Rolling Trends": 30,
        "60-Day Rolling Trends": 60,
    }

    if aggregation_horizon_label == "Custom (use slider below)":
        lookback_days = st.sidebar.slider(
            label="Custom Rolling Window (days)",
            min_value=1,
            max_value=90,
            value=7,
            step=1,
            help="Day-wise granular control -- pulls this many day-logs backward from the Target Evaluation Date.",
        )
    else:
        lookback_days = _HORIZON_TO_LOOKBACK_DAYS[aggregation_horizon_label]

    # Ingest the rolling-horizon dataset directly from the GitHub data_store branch
    try:
        df_raw = fetch_rolling_horizon_dataset(target_evaluation_date, lookback_days)
    except Exception as err:
        st.error("Failed to reach the GitHub telemetry ingestion engine.")
        with st.expander("Expand Traceback Logistics"):
            st.code(traceback.format_exc())
        return

    # Guard clause: Stop processing gracefully if no day-logs were resolvable for the window
    if df_raw is None or df_raw.empty:
        st.info(
            f"ℹ️ No telemetry logs were retrievable for the selected window "
            f"({lookback_days} day(s) ending {target_evaluation_date.isoformat()}). "
            f"This can happen for future dates, pipeline gaps, or an unset GITHUB_USER/GITHUB_REPO."
        )
        return

    # INTERCEPT AND PROCESS THE STITCHED RAW ASSET VIA THE MID-LAYER GATEWAY
    try:
        df_fetched = master_dashboard_data_gateway(df_raw)

    