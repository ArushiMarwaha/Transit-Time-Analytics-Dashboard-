CUMTA Core Transit Network Diagnostics Cockpit: Technical Reference Manual & README
1. Executive Overview & System Architecture
System Summary
The CUMTA Core Transit Network Diagnostics Cockpit is an advanced, publication-grade spatial-temporal analytics framework engineered for the Chennai Unified Metropolitan Transport Authority (CUMTA). The system ingests high-frequency macroscopic traffic telemetry, environmental data, and static GIS infrastructure mappings to diagnose network bottlenecks, evaluate geometric and environmental constraints, quantify commuter uncertainty, and generate data-driven capital expenditure (CapEx) policies.
Ingestion Architecture
GitHub Repository Pointers: Centralized raw data repository configuration leveraging GITHUB_USER = "ArushiMarwaha", GITHUB_REPO = "Transit-Time-Analytics-Dashboard-", and GITHUB_BRANCH = "main", establishing a dynamic base URL (GITHUB_RAW_BASE_URL) pointing directly to the data_store/ directory.
The fetch_rolling_horizon_dataset() Engine: Implements a rolling lookback architecture that walks backward across a date range [target_date - lookback_days + 1, target_date]. It performs robust HTTP CSV retrieval (_http_get_csv) with an automatic fallback trail to local workspace mirrors (data_store/) to ensure uninterrupted execution during network latency or pipeline delays. Static reference assets (segments_ref.csv and roads_results.csv) are fetched once at boot and cached via Streamlit (st.cache_data). To prevent many-to-many merge explosions from repeated snapshot logs, roads_results.csv is dynamically collapsed to a unique one-row-per-segment reference table using _collapse_to_one_row_per_segment().
Environmental Alignment (_asof_join_environmental): Binds ~3-hourly environmental frames (weather_results and aqi_results) onto cycle-by-cycle telemetry rows (routes_results) keyed by segment_uid. It enforces a strict tolerance window of ENVIRONMENTAL_JOIN_TOLERANCE = pd.Timedelta("3.5 hours") to prevent silent data borrowing across shifts.
Mid-Layer Gateway (master_dashboard_data_gateway): Acts as an automated schema sniffer that normalizes column headers, localizes UTC timestamps to Indian Standard Time (Asia/Kolkata IST), extracts geographic coordinates from nested JSON stringified snapped_points with self-healing coordinate backfill, and synthetically reconstructs missing traffic performance indices.
AI Assistant (CUMTA Transit Intelligence Agent)
Rendered via a floating viewport layer (render_ai_assistant_chat), the assistant provides real-time natural language query resolution and inline micro-analytics execution (_render_micro_chart). It utilizes a resilient 3-tier execution fallback architecture:
Tier 1: Google Gemini (gemini-1.5-flash) via the google-genai SDK.
Tier 2: Anthropic Claude (claude-sonnet-4-6) secondary REST fallback.
Tier 3: Robust rule-based domain parser backed by an extensive knowledge base (_DOMAIN_KB) covering core metrics, hypotheses, and proxy transport topics.
2. Comprehensive Tab-by-Tab Technical Specifications
Dataset Overview & Audit Table (Tab 0)
Core Policy & Business Question: Provides a real-time macroscopic review of data schemas, missing value null densities, and spatial footprint distribution across all monitored corridors.
Data Pipeline & Feature Transformations: Consumes raw or pre-computed tabular exports. Computes network-wide observation counts, active corridor tallies, and data type profiles.
Mathematical Formulas & Statistical Logic: Aggregates mean Travel Time Index ($\overline{TTI}_s$), peak-window averages, 95th percentiles ($P_{95}$), and cumulative Total Delay Hours:
$$\overline{TTI}_s = \frac{1}{N_s}\sum_{i=1}^{N_s} TTI_{s,i} \quad\vert\quad \text{Total Delay Hours}_s = \frac{1}{3600}\sum_{i=1}^{N_s} \max(0,\; t_{\text{current},i} - t_{\text{free-flow},i})$$
Visualizations & Chart Breakdown:
Macro Spatial Congestion Map: Interactive Folium map rendering color-coded circle markers for heavy congestion ($TTI \ge 1.25$), moderate delay ($1.05 \le TTI < 1.25$), and free-flow ($TTI < 1.05$).
Corridor-Wise Leaderboard: Tabular ranking sorted by mean TTI with gradient styling.
Actionable CUMTA Policy Intervention: Identifies heavy bottleneck corridors for immediate first-wave capital review and schedules moderate segments for signal-timing optimization.
Hypothesis 1: Systemic Bottleneck Localization (Tab 1)
Core Policy & Business Question: Which specific segments are true root-cause bottlenecks that generate cascading spillover queues, versus victim segments that only slow down due to downstream back-pressure?
Data Pipeline & Feature Transformations: Isolates segment-level sequences using sequence_order within each corridor_name. Computes segment-specific 90th-percentile TTI thresholds ($P_{90}$) to classify congestion states.
Mathematical Formulas & Statistical Logic:
Root-Cause Condition: A segment is a confirmed root cause if $TTI_t > P_{90}(TTI_s)$, its immediate upstream neighbor is clear, the breakdown persists into interval $t+1$, and repetition count $\ge 2$.
Multi-Criteria Bottleneck Index (MCBI): Composite priority score combining tail severity, frequency, early onset, and causal event counts:
$$\mathrm{MCBI}_s = 0.25 \tilde{P}_{90} + 0.20 \tilde{F}_{\text{cong}} + 0.25 (1 - \tilde{H}_{\text{onset}}) + 0.30 \tilde{E}_{\text{rc}}$$
Visualizations & Chart Breakdown:
MCBI Leaderboard: Stacked bar chart breaking down score components.
Segment Congestion Heatmap: Hour-by-hour x segment grid displaying failure density.
Top Segment Profiles: Weekday vs. weekend diurnal TTI comparison against threshold lines.
Actionable CUMTA Policy Intervention: Deploy engineering crews exclusively to confirmed root-cause segments. Avoid civil CapEx on spillover/victim nodes, which self-resolve once upstream bottlenecks are cleared.
Hypothesis 2: Temporal Peak Profiling (Tab 2)
Core Policy & Business Question: At what precise minute does a road's capacity fail, how long does clearance take, and how does this diurnal cycle shift on weekends?
Data Pipeline & Feature Transformations: Aggregates TTI by time_of_day and is_weekend. Calculates dynamic corridor-specific failure thresholds from the 90th percentile.
Mathematical Formulas & Statistical Logic: Identifies peak failure minute via $\arg\max_t(\overline{TTI}_t)$ and measures clearance duration as elapsed minutes until failure rate drops $\le 25\%$. Wilcoxon signed-rank tests verify peak-vs-off-peak elevation.
Visualizations & Chart Breakdown:
Weekday vs. Weekend Failure Rate Bar Chart: Comparing operating failure percentages across corridors.
Diurnal Velocity Degradation Line Plots: Side-by-side weekday and weekend profiles.
Hourly Congestion Ratio Heatmaps: Hour vs. day-type failure proportion grid.
Actionable CUMTA Policy Intervention: Retime traffic signals to target exact peak-hour onset windows confirmed by statistical tests.
Hypothesis 3: Geometric Constraints & Structural Choke Points (Tab 3)
Core Policy & Business Question: Are permanent infrastructure features (lane drops, bus stops, signal density) driving localized congestion, and how do we separate structural failures from transient demand spikes?
Data Pipeline & Feature Transformations: Merges static geometry (road_width_lanes, nearest_signal_dist_meters, nearest_bus_stop_dist_meters) with telemetry. Engineers downstream lane drop delta ($\Delta\text{Lanes} = L_s - L_{s+1}$), signal density ($1000 / D_{\text{sig}}$), and bus friction ($1 / (D_{\text{bus}} \times L_s)$).
Mathematical Formulas & Statistical Logic:
2D Structural Dispersion Matrix: Classifies segments into Quadrant I (Persistent: off-peak TTI $\ge 1.5$ and peak TTI $\ge 2.2$), Quadrant II (Temporal: off-peak $< 1.5$ and peak $\ge 2.2$), and Quadrant III (Nominal).
Mann-Whitney U Test: Tests whether lane-drop segments experience significantly higher TTI than uniform segments without assuming normality.
Visualizations & Chart Breakdown:
2D Dispersion Scatter Plot: Off-peak vs. peak TTI colored by quadrant.
Partial Dependence Plots (PDP): Binned median trendlines for bus friction and signal density.
Actionable_CUMTA Policy Intervention: Allocate capital civil works (lane widening, bus bay recessing) to Quadrant I segments; route Quadrant II segments to adaptive signal retiming.
Hypothesis 4: Weather-Driven Variance (Tab 4)
Core Policy & Business Question: How much does precipitation and reduced visibility degrade network capacity, and can we isolate weather sensitivity from rush-hour demand?
Data Pipeline & Feature Transformations: Binds rainfall intensity ($\text{mm/hr}$) and atmospheric visibility ($\text{meters}$) to segment telemetry, categorizing weather states into dry baseline, light rain, moderate rain, and heavy monsoon anomaly.
Mathematical Formulas & Statistical Logic: Computes micro-segment rain sensitivity slope via OLS regression of TTI on rainfall intensity. Multivariate OLS controls for hour-of-day cyclical harmonics ($\sin, \cos$) and segment fixed effects.
Visualizations & Chart Breakdown:
Rainfall vs. TTI Scatter + Fit Line: Displays link sensitivity slope.
Visibility vs. TTI Elasticity Curves: Evaluates atmospheric sightline degradation.
Actionable CUMTA Policy Intervention: Prioritize stormwater drainage upgrades and surface friction treatments (micro-surfacing) on high-beta-rain segments before monsoon season.
Hypothesis 5: Tidal Flow Asymmetry (Tab 5)
Core Policy & Business Question: Do morning and evening congestion patterns mirror each other, or does severe directional imbalance justify dynamic reversible lane management?
Data Pipeline & Feature Transformations: Resolves opposing direction tracks (Direction A vs. Direction B) using name-token heuristics. Computes the hourly Tidal Split Coefficient ($\Lambda_{s,h}$).
Mathematical Formulas & Statistical Logic:
$$\Lambda_{s,h} = \frac{\text{Median}(TTI_{s,d,h})}{\text{Median}(TTI_{s,\bar{d},h})}$$
Inversion Loop: $\Lambda \ge 1.8$ (AM peak) and $\Lambda \le 0.55$ (PM peak). Validated via Shapiro-Wilk normality testing on differences, Wilcoxon Signed-Rank tests, and Kolmogorov-Smirnov (KS) week-over-week distribution stability tests.
Visualizations & Chart Breakdown:
Lambda Hourly Profile: Divergence curves against 1.0, 1.8, and 0.55 threshold lines.
Direction A vs. B Heatmaps: Side-by-side saturation grids.
Actionable CUMTA Policy Intervention: Implement dynamic reversible lanes with automated bollards where no median barrier exists, or asymmetric green-time phasing where a fixed median barrier is present.
Hypothesis 6: Commuter Uncertainty & Travel Time Predictability (Tab 6)
Core Policy & Business Question: Which segments impose the greatest planning burden on commuters through unpredictable travel times?
Data Pipeline & Feature Transformations: Applies an IQR outlier cleanser ($P_{75} + 1.5 \times IQR$) to filter out one-off crash spikes, isolating recurring peak-hour reliability.
Mathematical Formulas & Statistical Logic:
$$\mathrm{BTI}_s = \frac{P_{95}(TT_s) - \mu(TT_s)}{\mu(TT_s)} \times 100\% \quad\vert\quad \mathrm{PTI}_s = \frac{P_{95}(TT_s)}{FF_s}$$
Heteroscedastic OLS: $\ln(\sigma^2) \sim \ln(\overline{\text{TTI}}) + D_{\text{signal}}$. Levene's test across weekly blocks ($W_1, W_2, W_3$) evaluates variance stability.
Visualizations & Chart Breakdown:
BTI Ranking Bar Chart: Top 15 least reliable segments.
Heteroscedastic Variance Scatter Plot: Log-mean vs. log-variance elasticity fit.
PDP Signal Proximity Curves & Levene Test Tables.
Actionable CUMTA Policy Intervention: Deploy rapid incident response staging teams within 500 m of segments with BTI $\ge 80\%$ and transient Levene results.
Hypothesis 7: The Flyover Exit & Gradients (Tab 7)
Core Policy & Business Question: Does an elevated flyover mainline eliminate congestion, or does it relocate the bottleneck to the immediate downstream exit junction?
Data Pipeline & Feature Transformations: Pairs each flyover segment with its immediate downstream neighbor using sequence_order within the corridor, merging synchronized timestamped telemetry.
Mathematical Formulas & Statistical Logic: Computes the Displacement Rate—the proportion of intervals where the flyover is flowing freely ($TTI \le P_{90}$) while its immediate downstream exit is congested ($TTI > P_{90}$). Evaluated via cross-validated Random Forest classification.
Visualizations & Chart Breakdown:
Flyover vs. Exit Hourly TTI Curves: Comparative diurnal traces.
Feature Importance Bar Charts: Quantifies predictive dependency of exit failure on flyover status.
Actionable CUMTA Policy Intervention: Treat flyover-exit pairs as integrated systems; deploy ramp-metering or exit-lane widening at the downstream bottleneck rather than modifying the flyover mainline.
Hypothesis 8: Spatial Length Dilution Bias (Tab 8)
Core Policy & Business Question: Does analyzing long road stretches artificially hide severe localized traffic jams by averaging slow speeds with fast speeds?
Data Pipeline & Feature Transformations: Dynamically groups consecutive micro-segments into macro-segments of configurable size (GROUP_SIZE) based on sequence_order.
Mathematical Formulas & Statistical Logic: Computes travel-time-weighted macro-segment TTI and compares it against the worst constituent micro-segment's peak TTI to quantify the dilution gap and percentage underreporting.
Visualizations & Chart Breakdown:
Micro-vs-Macro Hourly Profiles: Constituent micro-segment lines plotted alongside combined macro-segment traces.
Random Forest Regressor Importance Charts: Identifies drivers of aggregation loss.
Actionable CUMTA Policy Intervention: Transition network monitoring strictly to micro-segment resolution in high-CV corridors to prevent macro-averaging from masking critical queue tails.
Hypothesis 9: Unsupervised Taxonomy Clustering (Tab 9)
Core Policy & Business Question: Which segments belong cleanly to a single policy archetype, and which ones straddle multiple archetypes, requiring blended multi-track interventions?
Data Pipeline & Feature Transformations: Standardizes a 6D feature vector (AM TTI, PM TTI, Off-Peak TTI, BTI, CV, Lambda max) via z-scoring.
Mathematical Formulas & Statistical Logic: Fits a 4-component Gaussian Mixture Model (GMM) using Expectation-Maximization to output soft membership probabilities ($W_s$) across archetypes. Flags high-risk hybrids when primary probability $< 0.85$ and secondary probability $\ge 0.30$. PCA compresses features into 2 axes ($PC_1$ severity, $PC_2$ unpredictability).
Visualizations & Chart Breakdown:
Multi-Axis PCA Policy Archetype Map: Interactive Plotly scatter with star markers for hybrids.
Temporal Drift Slope Plot: Off-peak $\to$ AM $\to$ PM trajectory lines.
Soft-Membership Heatmap & Parallel Coordinates Grid.
Actionable CUMTA Policy Intervention: Assign blended CapEx packages (e.g., structural widening + adaptive signals) to star-marked hybrid segments rather than single-template fixes.
Hypothesis 10: Traffic Volume via AQI Proxy (Tab 10)
Core Policy & Business Question: Can localized Air Quality Index (AQI) telemetry serve as a proxy to distinguish high-volume vehicle idling gridlock from low-volume physical blockages?
Data Pipeline & Feature Transformations: Aligns Google Environment API AQI readings with traffic telemetry. Applies Cross-Correlation Function (CCF) lags to establish temporal relationships.
Mathematical Formulas & Statistical Logic: Multivariate OLS regression models AQI as a function of TTI, wind speed, precipitation, and hour-of-day controls. SHAP-style attribution separates traffic-induced emissions from external non-traffic pollution sources.
Visualizations & Chart Breakdown:
TTI vs. AQI Polynomial Fit Scatter Plot: Highlights the idling inflection threshold ($TTI = 1.8$).
SHAP Attribution Bar Charts & Observed vs. Forecast Validation Panels.
Actionable CUMTA Policy Intervention: Deploy transit capacity management (dedicated bus lanes) for high TTI + high AQI gridlock; dispatch incident response teams for high TTI + flat AQI physical blockages.
3. Detailed Data Dictionary
Variable Name
Ingestion Source
Data Type
Formula / Origin Logic
Diagnostic Function
shapefile_segment_name
Upstream Telemetry
String
Standardized uppercase UID (segment_uid)
Primary spatial segment identifier
travel_time_index_tti
Gateway / Computed
Float
$\text{current\_travel\_time} / \text{free\_flow\_travel\_time}$
Core relative congestion index
execution_timestamp
Gateway / Converted
Datetime
UTC timestamp converted to Asia/Kolkata IST
Temporal anchoring for diurnal analysis
derived_hour
Gateway / Computed
Integer
Extracted from IST timestamp (execution_timestamp.dt.hour)
Diurnal binning key
is_weekend
Gateway / Computed
Integer
1 if dayofweek $\ge 5$ else 0
Day-type segmentation filter
delta_lanes
Spatial Reference
Float
$L_s - L_{s+1}$ (Downstream lane drop delta)
Geometric bottleneck identification
signal_density_proxy
Spatial Reference
Float
$1000 / \max(D_{\text{signal}}, 1)$
Intersection queue back-pressure proxy
friction_bus
Spatial Reference
Float
$1 / (D_{\text{bus}} \times L_s)$
Intermodal transit friction index
rainfall_intensity_mm_hr
Weather API
Float
Ingested precipitation intensity (mm/h)
Weather-driven capacity degradation driver
visibility_meters
Weather API
Float
Ingested atmospheric visibility scale
Sightline and speed-following elasticity driver
indexes_aqi
Environment API
Float
Localized AQI or estimated via $45 + (TTI \times 24)$
Vehicle idling and emissions proxy
direction_track
Gateway / Heuristic
String
Mapped to Direction A or Direction B
Tidal flow asymmetry pairing key
lambda_ratio
Gateway / Computed
Float
$\text{Median}(TTI_A) / \text{Median}(TTI_B)$
Directional tidal split coefficient
bti_val
Computed Module
Float
$(P_{95}(TT) - \mu(TT)) / \mu(TT) \times 100\%$
Commuter planning buffer time index
pti_val
Computed Module
Float
$P_{95}(TT) / \text{FreeFlow}$
Absolute planning time multiplier
cv_val
Computed Module
Float
$\sigma(TTI) / \mu(TTI)$
Standardized dispersion and erraticism metric
mcbi_score
Computed Module
Float
Weighted composite of tail severity, frequency, onset, and RC count
Multi-criteria engineering triage priority score

4. Machine Learning & Statistical Cross-Check Suite
The dashboard embeds five rigorous econometric and machine learning modules to cross-check empirical rule-based findings:
H1 Breakdown Risk Prediction (NumPy Logistic Regression): Implements a custom gradient descent optimizer with sigmoid activation ($\sigma(z) = 1 / (1 + e^{-z})$) and standardized features (current TTI, congestion flags, upstream state, cyclical hour harmonics, and historical rates) to predict next-interval queue propagation probability.
H2 Diurnal Failure-Probability Model (Harmonic Logistic Regression): Utilizes logistic regression with 1st and 2nd harmonic sine/cosine cyclical encodings (periods 24h and 12h) plus corridor interaction terms to capture multi-peak daily congestion shapes.
H6 Heteroscedastic Uncertainty Expansion (Log-Variance OLS): Fits $\ln(\sigma^2) \sim \ln(\overline{\text{TTI}}) + D_{\text{signal}}$ via closed-form least squares to test whether journey-time variance expands non-linearly with congestion ($\hat\beta_1 > 0$), confirming structural unpredictability.
H7 Sequential Displacement Classifier (Cross-Validated Random Forest): Trains a RandomForestClassifier on paired flyover-exit telemetry to predict downstream exit congestion from flyover status, proving whether flyovers relocate traffic jams.
H9 Unsupervised Soft Taxonomy (Gaussian Mixture Model & PCA): Standardizes a 6D feature vector, fits a 4-component GMM for soft posterior probabilities ($W_{s,k}$), projects data into 2D via PCA, and evaluates stability via bootstrap Adjusted Rand Index (ARI $\ge 0.82$).
