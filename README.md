# CS2 Clutch Survival Analysis

A data-driven approach to modeling post-plant survival probability in Counter-Strike 2 professional matches using empirical positioning data and hazard rate modeling on the map Mirage.

## Overview

This project analyzes post-plant survival in Counter-Strike 2 professional matches on Mirage using Cox proportional hazards regression with Ridge regularization (L2). We process 569 professional demo files from major tournaments (2023-2025) to model how positioning, equipment, utility state, and team composition affect player survival probability during both terrorist post-plant defense and counter-terrorist retake scenarios.

Our models achieve concordance indices of 0.748-0.806, demonstrating strong predictive power for survival outcomes based on empirically-derived spatial features and game state variables. This provides a rigorous, data-driven framework for understanding tactical positioning in competitive CS2.

The core question: given a player's position, equipment, and game state at any moment during post-plant play, what is their instantaneous hazard rate (risk of elimination), and how do these factors change dynamically over time?

## Motivation

Traditional clutch analysis in Counter-Strike relies on win rate percentages (e.g., "1v2 clutches succeed 29% of the time"). These metrics fail to account for:

- Spatial positioning and map geometry
- Utility state (smokes blocking sightlines, mollies denying areas)
- Time pressure from bomb timers
- Weapon loadouts and economy
- Dynamic threat from enemy positions

This project addresses these limitations by modeling survival as a time-dependent process influenced by measurable game state variables.

## Methodology

### Data Collection

We parse 581 professional CS2 demo files from major tournaments (2023-2025) using the awpy Python library. For each bomb plant that results in a retake scenario, we extract:

- Player positions at 64Hz tick rate
- Damage events (weapon type, amount, source)
- Utility usage (grenades thrown, smokes/mollies active)
- Round outcomes and bomb timer states

### Zone Classification

The Mirage map is divided into 47 distinct zones using polygon geometry. Each zone represents a tactically meaningful area (e.g., A_Default, CT_Stairs, Jungle). We classify these zones based on:

- Bombsite proximity (on-site vs. approach zones)
- Visibility relationships between zones
- Connectivity and travel time between positions

Zone definitions are stored as GeoJSON polygons in `data/mirage_zones.json`, enabling precise spatial analysis.

### Episode Tracking

Rather than analyzing entire rounds, we segment player behavior into episodes: continuous periods from retake start until death or round end. Each episode captures:

- Duration (survival time in ticks)
- Censoring (death event vs. survived to round end)
- Initial zone position
- Equipment state (health, armor, weapon type)
- Utility state (active smokes, mollies, flashes)
- Team composition (teammates alive, numerical advantage)
- Spatial features (visibility degree, connectivity degree)
- Crossfire potential

This produces 65,688 episodes (21,360 CT, 44,328 T) for survival analysis across both A-site and B-site scenarios.

### Active Utility Detection

A critical component is tracking utility that affects threat during retakes. We distinguish between:

1. Utility thrown during episodes: Smokes, flashes, HE grenades, and mollies used by players during their positional episodes
2. Active utility at retake start: Pre-existing smokes and mollies still burning from the plant phase

For active utility, we track exact expiration times. A smoke that expires 3 seconds into a retake provides protection initially but then exposes the player to full threat. This time-dependent modeling is essential for accurate hazard rate calculations.

### Movement Speed Calculation

Instead of generic speed categories (walking, running), we use weapon-specific movement speeds derived from CS2 mechanics:

- Knife: 250 units/second
- Pistols: 240 units/second
- Rifles: 215 units/second
- AWP: 200 units/second
- With modifiers for crouching (0.34x) and shift-walking (0.52x)

This precision improves travel time estimates for threat cone modeling.

### Connectivity and Reachability

We model zone connectivity two ways:

1. Manual connectivity graph: Based on map knowledge and common routes
2. Empirical connectivity from movement: Measured from actual player movement patterns in the 581 demos

The empirical approach (`build_connectivity_from_movement.py`) provides realistic travel times that account for how professional players actually navigate the map, rather than theoretical shortest paths.

### Threat Cone Modeling

For any player position, we calculate potential enemy positions based on:

- Last known location
- Time elapsed since detection
- Weapon-based movement speed
- Zones blocked by active utility (smokes prevent vision, mollies deny areas)

Threat probability is weighted using empirical position distributions from the episode data, reflecting actual professional player positioning patterns rather than theoretical proximity assumptions.

## Current Pipeline

### 1. Zone and Connectivity Setup
```
data/mirage_zones.json                     # 291 polygon zones
data/mirage_zone_connectivity.json         # Manual connectivity graph
data/mirage_visibility.json                # Inter-zone visibility
```

### 2. Empirical Connectivity (Optional Enhancement)
```bash
python src/build_connectivity_from_movement.py
# Generates: data/mirage_zone_connectivity_from_movement.json
```

### 3. Episode Data Collection
```bash
python src/analyze_player_positioning.py
# Processes 569 demos
# Output: data/ct_episodes.parquet, data/t_episodes.parquet
# Total: 65,688 episodes (21,360 CT, 44,328 T)
```

### 4. Cox Survival Analysis with Ridge Regularization
```bash
python src/cox_survival_analysis_ridge.py
# Fits 12 Cox models (4 sites × 3 model types)
# Grid search over λ ∈ [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
# 5-fold grouped cross-validation
# Output: results/cox_models/ridge_gridsearch/
# Runtime: ~45-60 minutes
```

### 5. Testing and Validation
```bash
python tests/test_episode_tracking.py
# Validates episode tracking on sample demo
python tests/test_zone_classifier.py
# Validates zone polygon classification
```

## Key Features

### Time-Dependent Utility Modeling

The system tracks utility with precise timing:

```
Active utility at retake: 1 smoke, 0 mollies, 0 recent flashes
  Smoke by Jimpphat: expires in 3.0s
```

This enables modeling threat that changes as utility expires. A position may be safe for 3 seconds (smoke cover) but then become exposed.

### Weapon-Specific Movement

Threat calculations account for enemy weapon loadouts:
- AWPer rotations take longer (200 vs 215 units/sec)
- Knife-running creates faster flanks
- Crouch-walking affects timing windows

### Empirical Grounding

Rather than assumed values, the project measures:
- Actual zone occupancy rates from professional play
- Real travel times between zones
- Utility usage patterns in winning vs. losing rounds

## Repository Structure

```
cs2-clutch-survival-analysis/
├── data/
│   ├── mirage_zones.json                           # Zone polygon definitions
│   ├── mirage_zone_connectivity.json               # Manual connectivity
│   ├── mirage_zone_connectivity_from_movement.json # Empirical connectivity
│   ├── mirage_visibility.json                      # Visibility graph
│   └── mirage_plant_spots.json                     # Bomb plant positions
├── src/
│   ├── analyze_player_positioning.py               # Main pipeline: episodes + utility
│   ├── cox_survival_analysis_ridge.py              # Ridge Cox models with grid search
│   ├── detect_retake.py                            # Retake detection with fallback logic
│   ├── build_connectivity_from_movement.py         # Empirical connectivity builder
│   ├── build_visibility_from_damage.py             # Empirical visibility from damage events
│   ├── zone_classifier.py                          # Polygon-based zone classification
│   ├── zone_connectivity.py                        # Reachability and threat cones
│   ├── movement_speed.py                           # Weapon-specific speeds
│   ├── plant_spot_classifier.py                    # Bomb plant position detection
│   ├── utility_tracker.py                          # Grenade tracking
│   ├── audio_tracker.py                            # Sound event detection
│   └── strategic_intelligence.py                   # CT movement analysis
├── tests/
│   ├── test_episode_tracking.py                    # Episode validation on sample demo
│   ├── test_zone_classifier.py                     # Zone polygon tests
│   ├── test_demo_parse.py                          # Demo parsing tests
│   └── test_postplant_analysis.py                  # Legacy tests
├── paper/
│   └── abstract.md                                 # Research abstract
├── requirements.txt                                # Python dependencies
└── README.md                                      # This file
```

## Dependencies

Core requirements:
- demoparser2 >= 0.26.0 (CS2 demo parsing)
- pandas >= 2.0.0 (data manipulation)
- numpy >= 1.24.0 (numerical operations)
- shapely >= 2.0.0 (polygon geometry)
- scikit-learn >= 1.3.0 (future modeling)
- lifelines >= 0.27.0 (survival analysis)

Install:
```bash
pip install -r requirements.txt
```

## Survival Model Results

### Cox Proportional Hazards with Ridge Regularization

We implemented Cox proportional hazards regression with L2 (Ridge) regularization to model player survival during post-plant scenarios. Models were fit separately for each team-site combination (CT A-site, CT B-site, T A-site, T B-site) with three model specifications:

1. Zone models: Zone dummy variables only (spatial effects)
2. Equipment models: Equipment + utility + team composition (no zones)
3. Full models: Zones + equipment + utility + interactions

### Methodology

Feature standardization: All continuous features (health, armor, visibility degree, connectivity degree, crossfire metrics) standardized to mean=0, std=1 before modeling. Critical for Ridge regression as L2 penalty is scale-dependent.

Regularization: Grid search over λ ∈ [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0] using 5-fold grouped cross-validation (grouped by retake to prevent data leakage).

Interaction terms: Zone × smoke, zone × crossfire, zone × numerical advantage (created AFTER standardization to preserve interpretability).

### Results (Concordance Index)

Using grouped cross-validation and cluster-robust standard errors to account for correlated observations, our models achieve concordance indices ranging from 0.748 to 0.806:

| Model | C-Index | CV Score | Optimal λ | Covariates |
|-------|---------|----------|-----------|------------|
| CT A Full | 0.748 | 0.718 | 0.01 | 104 |
| CT B Full | 0.773 | 0.734 | 0.05 | 87 |
| T A Full | 0.789 | 0.775 | 0.01 | 120 |
| T B Full | 0.806 | 0.786 | 0.05 | 96 |

Interpretation: C-index > 0.75 indicates strong predictive power. The T B-site full model achieves 0.806, demonstrating that positioning, equipment, and utility state accurately predict survival outcomes in post-plant defense scenarios.

### Hazard Ratios (Key Findings)

Hazard ratios (HR) quantify how each factor affects elimination risk:
- HR < 1: Factor reduces risk (protective)
- HR > 1: Factor increases risk (hazardous)

Example findings (from full models):
- Numerical advantage: HR ≈ 0.70-0.85 per additional teammate
- Crossfire (teammate within 5m): HR ≈ 0.75-0.85 (protective)
- Visibility degree: HR ≈ 1.05-1.15 per additional exposed zone
- Active smoke: HR ≈ 0.80-0.90 (protective)
- Position-specific: Certain zones show HR 0.50-0.60 (very safe) vs. HR 1.40-1.80 (very dangerous)

### Model Outputs

All results saved to `results/cox_models/ridge_gridsearch/`:
- 12 fitted model objects (.pkl)
- 12 hazard ratio tables with 95% CIs (.csv)
- 12 forest plots visualizing HRs (.png)
- 12 full model summaries (.html)
- 4 survival curves by position (.png)
- 4 top predictors tables (.csv)
- Model statistics including optimal λ and CV scores (.json)

## Future Enhancements

1. Multi-map support: Extend zone classification and survival modeling to Dust2, Inferno, Nuke, etc.
2. Time-varying covariates: Implement utility expiration as time-dependent covariates in Cox models
3. Stratified analysis: Team-specific and player-specific hazard ratios
4. Real-time prediction: Integration with live demo parsing for broadcast probability overlays
5. Causal inference: Propensity score matching or instrumental variables to estimate causal effects of positioning decisions

## Research Applications

This framework enables several research directions:

- Optimal positioning strategy derivation from empirical survival rates
- Utility usage effectiveness quantification (e.g., smoke timing impact)
- Team-specific tactical pattern analysis
- Real-time win probability estimation for broadcasts

## Notes

The project intentionally avoids several common pitfalls in esports analytics:

- No arbitrary thresholds or magic numbers (all parameters measured or documented)
- No assumed player behavior (movement, positioning, utility measured from data)
- No simplified win percentage models (survival analysis captures time-dependent risk)
- No manual feature engineering of "clutch skill" (outcomes emerge from positioning + utility + timing)

The goal is reproducible, empirically grounded analysis that advances understanding of professional CS2 tactics.

## License

Research project by Pete VanBenthuysen, 2025
