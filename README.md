# CUMTA Core Transit Network Diagnostics Cockpit

**Technical Reference Manual & README**

## Executive overview

The **CUMTA Core Transit Network Diagnostics Cockpit** is a publication-grade spatial-temporal analytics framework designed for the Chennai Unified Metropolitan Transport Authority (CUMTA). It integrates high-frequency macroscopic traffic telemetry, environmental measurements, and static GIS infrastructure mappings to diagnose bottlenecks, identify structural choke points, quantify commuter uncertainty, and support data-driven capital expenditure (CapEx) planning.

At a system level, the cockpit combines a rolling-horizon ingestion pipeline, schema-normalizing gateway logic, spatial-environmental joins, advanced diagnostic modules, and an embedded natural-language assistant. The result is a unified operational analytics environment for corridor surveillance, root-cause analysis, reliability measurement, and policy prioritization.

***

## System architecture

### Ingestion architecture

The repository is configured around a centralized raw-data pattern using the following repository constants:

```python
GITHUB_USER = "ArushiMarwaha"
GITHUB_REPO = "Transit-Time-Analytics-Dashboard-"
GITHUB_BRANCH = "main"
```

These constants define a dynamic `GITHUB_RAW_BASE_URL` that points directly to the `data_store/` directory in the GitHub repository.

### Rolling-horizon retrieval engine

The `fetch_rolling_horizon_dataset()` engine implements a rolling lookback strategy over the interval:

$$
[target\_date - lookback\_days + 1,\; target\_date]
$$

Key behaviors:

- Uses robust HTTP CSV retrieval through `_http_get_csv()`.
- Falls back automatically to local workspace mirrors in `data_store/` when remote retrieval fails.
- Fetches static reference assets such as `segments_ref.csv` and `roads_results.csv` once at application boot.
- Caches static assets using `st.cache_data` for performance.
- Prevents many-to-many merge explosions by collapsing `roads_results.csv` to one unique row per segment through `_collapse_to_one_row_per_segment()`.

### Environmental alignment layer

The `_asof_join_environmental()` module binds approximately 3-hourly environmental observations from weather and AQI feeds onto cycle-level telemetry rows keyed by `segment_uid`.

Core alignment rule:

- `ENVIRONMENTAL_JOIN_TOLERANCE = pd.Timedelta("3.5 hours")`

This tolerance prevents silent borrowing of environmental data across incompatible temporal windows or operational shifts.

### Mid-layer gateway

The `master_dashboard_data_gateway` acts as a schema-sniffing and normalization layer. Its major responsibilities include:

- Standardizing inconsistent column names.
- Converting UTC timestamps into Indian Standard Time (`Asia/Kolkata`).
- Extracting latitude/longitude coordinates from nested JSON-like `snapped_points` payloads.
- Applying self-healing geographic backfill when coordinates are partially missing.
- Reconstructing traffic-performance indicators when upstream feeds are incomplete.

### AI assistant layer

The floating assistant rendered by `render_ai_assistant_chat` provides conversational diagnostics and inline micro-analytics via `_render_micro_chart`.

It follows a resilient three-tier execution hierarchy:

1. **Tier 1:** Google Gemini (`gemini-1.5-flash`) through the `google-genai` SDK.
2. **Tier 2:** Anthropic Claude (`claude-sonnet-4-6`) through a REST fallback path.
3. **Tier 3:** A rule-based transport-domain parser backed by `_DOMAIN_KB`.

***

## Data flow

The end-to-end system workflow can be summarized as:

1. Load static and rolling-horizon raw datasets.
2. Normalize schemas and timestamps.
3. Enrich segment telemetry with geometry and environmental context.
4. Compute derived mobility, reliability, and policy features.
5. Render diagnostic tabs, maps, tables, and model outputs.
6. Expose analyst-facing natural-language interpretation through the embedded assistant.

### Primary data domains

| Domain | Typical contents | Operational role |
|---|---|---|
| Traffic telemetry | Travel times, free-flow travel time, segment observations, sequence order | Core congestion diagnostics |
| Static GIS reference | Segment geometry, lane counts, signal proximity, bus-stop proximity | Structural bottleneck analysis |
| Weather data | Rainfall intensity, visibility, atmospheric context | Weather sensitivity estimation |
| AQI / environmental data | AQI and related environmental proxies | Traffic volume and idling proxy diagnostics |

***

## Diagnostic tabs

### Tab 0: Dataset overview and audit table

**Business question:** What is the current schema, completeness, and spatial footprint of the monitored network?

**Key outputs:**

- Network-wide observation counts.
- Active corridor tallies.
- Variable type profiles.
- Missing-value auditing.
- Spatial congestion map.

**Core formulas:**

$$
\overline{TTI}_s = \frac{1}{N_s}\sum_{i=1}^{N_s} TTI_{s,i}
$$

$$
\text{Total Delay Hours}_s = \frac{1}{3600}\sum_{i=1}^{N_s} \max(0,\; t_{\text{current},i} - t_{\text{free-flow},i})
$$

**Visuals:**

- Macro spatial congestion map.
- Corridor-wise leaderboard sorted by mean TTI.

**Policy action:**

- Immediate review of heavy bottleneck corridors.
- Signal-timing optimization queue for moderate-delay corridors.

### Tab 1: Hypothesis 1 — Systemic bottleneck localization

**Business question:** Which segments are true root-cause bottlenecks, and which are merely spillover victims?

**Transformation logic:**

- Segment-level sequencing using `sequence_order` within each `corridor_name`.
- Segment-specific congestion thresholds based on `P90(TTI)`.

**Root-cause rule:**

A segment is treated as a confirmed root cause when all of the following hold:

- `TTI_t > P90(TTI_s)`.
- The immediate upstream neighbor is uncongested.
- Breakdown persists into interval `t + 1`.
- Repetition count is at least 2.

**Multi-Criteria Bottleneck Index (MCBI):**

$$
\mathrm{MCBI}_s = 0.25 \tilde{P}_{90} + 0.20 \tilde{F}_{\text{cong}} + 0.25 (1 - \tilde{H}_{\text{onset}}) + 0.30 \tilde{E}_{\text{rc}}
$$

**Visuals:**

- MCBI leaderboard.
- Segment congestion heatmap.
- Top segment weekday-versus-weekend profiles.

**Policy action:**

Concentrate engineering interventions on root-cause segments and avoid premature CapEx on downstream victim nodes.

### Tab 2: Hypothesis 2 — Temporal peak profiling

**Business question:** When does capacity fail, how long does clearance take, and how do patterns shift on weekends?

**Methods:**

- Aggregate TTI by `time_of_day` and `is_weekend`.
- Estimate dynamic corridor-specific failure thresholds from the 90th percentile.
- Use Wilcoxon signed-rank tests for peak versus off-peak elevation.

**Core logic:**

- Peak failure minute is identified by `argmax_t(average TTI_t)`.
- Clearance duration is measured until failure rate falls to 25 percent or less.

**Visuals:**

- Weekday versus weekend failure-rate bars.
- Diurnal velocity degradation line plots.
- Hour-by-day-type congestion heatmaps.

**Policy action:**

Target signal retiming at empirically verified peak-onset windows.

### Tab 3: Hypothesis 3 — Geometric constraints and structural choke points

**Business question:** Are permanent infrastructure constraints causing localized recurring congestion?

**Feature engineering:**

- `delta_lanes = L_s - L_{s+1}`
- `signal_density_proxy = 1000 / D_sig`
- `friction_bus = 1 / (D_bus × L_s)`

**Analytical logic:**

- Persistent and temporal structural classes defined in a 2D off-peak versus peak TTI dispersion space.
- Mann-Whitney U test compares lane-drop segments with uniform segments.

**Visuals:**

- 2D structural dispersion scatter.
- Partial dependence plots for bus friction and signal density.

**Policy action:**

- Quadrant I persistent segments: lane widening, bus bay recessing, structural redesign.
- Quadrant II temporal segments: adaptive signal retiming.

### Tab 4: Hypothesis 4 — Weather-driven variance

**Business question:** How strongly do rainfall and visibility degrade network performance?

**Methods:**

- Attach rainfall intensity and visibility to telemetry.
- Estimate micro-segment rain sensitivity by OLS regression of TTI on rainfall.
- Use multivariate OLS with hour harmonics and segment fixed effects.

**Visuals:**

- Rainfall versus TTI scatter with fitted line.
- Visibility versus TTI elasticity curves.

**Policy action:**

Prioritize drainage upgrades and surface-friction treatments on high weather-sensitivity corridors.

### Tab 5: Hypothesis 5 — Tidal flow asymmetry

**Business question:** Is directional imbalance strong enough to justify reversible-lane or asymmetric-phasing strategies?

**Tidal Split Coefficient:**

$$
\Lambda_{s,h} = \frac{\text{Median}(TTI_{s,d,h})}{\text{Median}(TTI_{s,\bar{d},h})}
$$

**Inference suite:**

- Shapiro-Wilk test on directional differences.
- Wilcoxon signed-rank test.
- Kolmogorov-Smirnov week-over-week stability check.

**Decision rule:**

- AM inversion pressure: `Λ >= 1.8`
- PM inversion pressure: `Λ <= 0.55`

**Visuals:**

- Lambda hourly divergence profiles.
- Direction A versus Direction B heatmaps.

**Policy action:**

Use reversible lanes where geometry allows, or asymmetric signal phasing where medians are fixed.

### Tab 6: Hypothesis 6 — Commuter uncertainty and predictability

**Business question:** Which segments impose the highest planning burden through unreliable travel times?

**Preprocessing:**

- Remove one-off spikes using the IQR upper fence `P75 + 1.5 × IQR`.

**Metrics:**

$$
\mathrm{BTI}_s = \frac{P_{95}(TT_s) - \mu(TT_s)}{\mu(TT_s)} \times 100\%
$$

$$
\mathrm{PTI}_s = \frac{P_{95}(TT_s)}{FF_s}
$$

**Variance model:**

$$
\ln(\sigma^2) \sim \ln(\overline{TTI}) + D_{\text{signal}}
$$

**Visuals:**

- BTI ranking bars.
- Log-mean versus log-variance scatter.
- Signal-proximity PDPs and Levene test tables.

**Policy action:**

Stage rapid incident-response teams close to segments with very high BTI and unstable weekly variance.

### Tab 7: Hypothesis 7 — Flyover exits and downstream gradients

**Business question:** Do flyovers resolve congestion, or displace it to immediate downstream exits?

**Pairing logic:**

- Match each flyover segment to its next downstream segment using `sequence_order`.
- Align synchronized telemetry by timestamp.

**Key metric:**

- **Displacement Rate:** share of intervals where the flyover is free-flowing while the downstream exit is congested.

**Modeling:**

- Cross-validated Random Forest classifier.

**Visuals:**

- Flyover versus exit hourly TTI profiles.
- Feature-importance bars.

**Policy action:**

Treat flyovers and exits as integrated systems and intervene at the downstream discharge bottleneck.

### Tab 8: Hypothesis 8 — Spatial length dilution bias

**Business question:** Does aggregation across long road stretches hide severe local congestion?

**Method:**

- Group consecutive micro-segments into configurable macro-segments using `GROUP_SIZE`.
- Compare weighted macro-segment TTI against the worst constituent micro-segment.

**Outputs:**

- Dilution gap.
- Underreporting percentage.
- Macro-versus-micro explainability drivers using Random Forest regression.

**Visuals:**

- Micro-versus-macro hourly profiles.
- Feature-importance charts.

**Policy action:**

Maintain micro-segment monitoring in high-variability corridors to avoid masking short severe queues.

### Tab 9: Hypothesis 9 — Unsupervised taxonomy clustering

**Business question:** Which segments map cleanly to one policy archetype, and which require blended intervention packages?

**Feature space:**

A standardized six-dimensional vector:

- AM TTI
- PM TTI
- Off-peak TTI
- BTI
- CV
- Lambda max

**Models:**

- 4-component Gaussian Mixture Model via Expectation-Maximization.
- PCA projection into two interpretable axes.
- Bootstrap stability using Adjusted Rand Index.

**Hybrid flagging rule:**

- Primary membership probability `< 0.85`
- Secondary membership probability `>= 0.30`

**Visuals:**

- PCA policy-archetype scatter.
- Temporal drift slope plot.
- Soft-membership heatmap.
- Parallel-coordinates view.

**Policy action:**

Assign blended CapEx packages to hybrid segments rather than a single policy template.

### Tab 10: Hypothesis 10 — Traffic volume via AQI proxy

**Business question:** Can AQI distinguish high-volume idling gridlock from low-volume physical blockages?

**Methods:**

- Align AQI telemetry with traffic telemetry.
- Use cross-correlation lags to identify temporal structure.
- Model AQI using multivariate OLS with TTI, wind, precipitation, and hour controls.
- Apply SHAP-style attribution logic to separate traffic-generated emissions from external pollution drivers.

**Visuals:**

- TTI versus AQI polynomial fit scatter.
- SHAP attribution bars.
- Observed-versus-forecast validation panels.

**Policy action:**

- High TTI + high AQI: capacity-management interventions such as bus-priority treatment.
- High TTI + flat AQI: likely physical blockage requiring incident-response operations.

***

## Data dictionary

| Variable name | Ingestion source | Data type | Formula / origin logic | Diagnostic function |
|---|---|---|---|---|
| `shapefile_segment_name` | Upstream telemetry | String | Standardized uppercase UID (`segment_uid`) | Primary spatial segment identifier |
| `travel_time_index_tti` | Gateway / computed | Float | `current_travel_time / free_flow_travel_time` | Core congestion index |
| `execution_timestamp` | Gateway / converted | Datetime | UTC to `Asia/Kolkata` conversion | Temporal anchor |
| `derived_hour` | Gateway / computed | Integer | `execution_timestamp.dt.hour` | Diurnal binning |
| `is_weekend` | Gateway / computed | Integer | `1 if dayofweek >= 5 else 0` | Day-type segmentation |
| `delta_lanes` | Spatial reference | Float | `L_s - L_{s+1}` | Geometric bottleneck flag |
| `signal_density_proxy` | Spatial reference | Float | `1000 / max(D_signal, 1)` | Queue back-pressure proxy |
| `friction_bus` | Spatial reference | Float | `1 / (D_bus × L_s)` | Transit-friction proxy |
| `rainfall_intensity_mm_hr` | Weather API | Float | Ingested precipitation intensity | Weather sensitivity driver |
| `visibility_meters` | Weather API | Float | Ingested atmospheric visibility | Sightline elasticity driver |
| `indexes_aqi` | Environment API | Float | AQI or estimated `45 + (TTI × 24)` | Idling and emissions proxy |
| `direction_track` | Gateway / heuristic | String | Heuristic mapping to Direction A or B | Tidal pairing key |
| `lambda_ratio` | Gateway / computed | Float | `Median(TTI_A) / Median(TTI_B)` | Directional asymmetry metric |
| `bti_val` | Computed module | Float | `(P95(TT) - mean(TT)) / mean(TT) × 100%` | Commuter buffer index |
| `pti_val` | Computed module | Float | `P95(TT) / FreeFlow` | Planning time multiplier |
| `cv_val` | Computed module | Float | `sd(TTI) / mean(TTI)` | Relative variability |
| `mcbi_score` | Computed module | Float | Weighted severity-frequency-onset-root-cause score | Engineering triage priority |

***

## Statistical and machine-learning cross-check suite

The cockpit includes a formal cross-validation layer that tests whether rule-based diagnostics are supported by predictive or unsupervised models.

### H1 breakdown risk prediction

A custom NumPy logistic-regression implementation estimates next-interval queue-propagation probability using:

- Current TTI.
- Congestion-state flags.
- Upstream condition.
- Cyclical hour harmonics.
- Historical failure-rate features.

Sigmoid link:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### H2 diurnal failure-probability model

A harmonic logistic-regression framework models multi-peak daily congestion shapes using first- and second-order cyclical sine/cosine terms with corridor interactions.

### H6 heteroscedastic uncertainty expansion

Closed-form OLS estimates whether travel-time variance expands non-linearly with congestion:

$$
\ln(\sigma^2) \sim \ln(\overline{TTI}) + D_{\text{signal}}
$$

A positive fitted congestion elasticity supports structural unpredictability rather than random noise.

### H7 sequential displacement classifier

A cross-validated RandomForestClassifier predicts downstream exit congestion using paired flyover-exit telemetry, testing whether elevated infrastructure displaces rather than eliminates bottlenecks.

### H9 unsupervised soft taxonomy

The clustering subsystem standardizes a six-dimensional feature vector, fits a four-component Gaussian Mixture Model, projects onto PCA axes, and assesses clustering stability with bootstrap Adjusted Rand Index, targeting stability around `ARI >= 0.82`.

***

## Interpretation guidance

### Reading the diagnostic system

The cockpit is designed to distinguish among five broad policy narratives:

- **Root-cause operational bottlenecks** that require engineering attention.
- **Spillover victims** that improve once upstream failures are addressed.
- **Structural choke points** caused by geometry and fixed infrastructure.
- **Environmental degradation regimes** that amplify congestion under rain or poor visibility.
- **Uncertainty-heavy commuter corridors** that require reliability-focused interventions.

### Policy logic by diagnostic class

| Diagnostic class | Typical signal | Preferred intervention |
|---|---|---|
| Root-cause bottleneck | High MCBI, repeated causal onset | Civil engineering or junction redesign |
| Temporal overload | Peak-only failure, low off-peak burden | Adaptive signal retiming |
| Structural choke point | High peak and high off-peak TTI | Lane widening, bus bay redesign, geometric fixes |
| Weather-sensitive corridor | High rainfall or visibility beta | Drainage, surfacing, monsoon hardening |
| High uncertainty corridor | High BTI / PTI / unstable variance | Incident-response staging and traveler information |
| Directionally asymmetric corridor | Extreme Lambda inversion | Reversible lanes or directional phasing |

***

## Repository organization

A recommended repository skeleton for maintainability is shown below.

```text
Transit-Time-Analytics-Dashboard-/
├── app.py
├── dashboard/
│   ├── tabs/
│   ├── models/
│   ├── visuals/
│   └── ai/
├── data_store/
│   ├── routes_results_*.csv
│   ├── weather_results_*.csv
│   ├── aqi_results_*.csv
│   ├── roads_results.csv
│   └── segments_ref.csv
├── utils/
│   ├── ingestion.py
│   ├── gateway.py
│   ├── joins.py
│   └── metrics.py
└── README.md
```

***

## Operational assumptions and safeguards

- Segment identifiers must remain stable across traffic, GIS, weather, and AQI feeds.
- Environmental joins should never exceed the configured tolerance window.
- Static reference assets should be de-duplicated to one row per segment before merging.
- Timezone conversion to IST must occur before any diurnal or weekend segmentation.
- Micro-segment resolution should be preserved whenever aggregation could hide queue tails.
- Cross-check models are intended to validate, not replace, domain rule logic.

***

## Recommended extensions

To strengthen future versions of the cockpit, the following enhancements are recommended:

- Bayesian hierarchical reliability modeling for corridor-level shrinkage.
- Bootstrap confidence intervals for MCBI and BTI rankings.
- Spatial autocorrelation diagnostics for adjacent segment spillovers.
- Causal intervention tracking for before-after signal and CapEx evaluation.
- Automated report export pipelines for policy briefing notes.
- Drift monitoring for schema evolution and upstream API changes.

***

## Quick start

### Prerequisites

- Python environment with Streamlit, pandas, NumPy, scikit-learn, and plotting libraries.
- Access to the configured GitHub raw-data repository or synchronized local mirror.
- Valid API credentials where external AI or environmental services are used.

### Boot sequence

1. Load static references and initialize cache.
2. Resolve rolling-horizon traffic and environmental datasets.
3. Pass raw inputs through `master_dashboard_data_gateway`.
4. Materialize feature layers and hypothesis tables.
5. Render dashboard tabs and activate assistant services.

### Minimum validation checks

- Static tables de-duplicate correctly.
- Timestamps localize to IST without null inflation.
- Segment geocoding backfill succeeds on malformed `snapped_points` rows.
- Environmental joins stay within 3.5-hour tolerance.
- Hypothesis tabs gracefully degrade when one supporting feed is missing.

***

