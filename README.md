# CS2 Clutch Survival Analysis

A data-driven approach to modeling post-plant survival probability in Counter-Strike 2 professional matches using empirical positioning data and hazard rate modeling.

## Overview

This project analyzes retake scenarios in CS2 professional play by tracking player positioning, combat outcomes, and utility usage during post-plant situations. Rather than relying on simplified probability estimates or arbitrary assumptions, we build empirical models from 582 professional demo files to understand how position, utility, and time affect survival in clutch scenarios.

The core question: given a player's position at any moment during a retake, what factors determine their likelihood of survival, and how do these factors change over time?

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

We parse 582 professional CS2 demo files from major tournaments (2023-2025) using the awpy Python library. For each bomb plant that results in a retake scenario, we extract:

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

Rather than analyzing entire rounds, we segment player behavior into episodes: continuous periods spent in a single zone. Each episode captures:

- Duration (in ticks)
- Damage taken (gun vs. utility)
- Damage dealt (gun vs. utility)
- Outcome (died, got kill, or survived)
- Utility thrown during the episode
- Round winner (for weighting analysis)

This granular approach reveals how specific positions perform under different conditions.

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
2. Empirical connectivity from movement: Measured from actual player movement patterns in the 582 demos

The empirical approach (`build_connectivity_from_movement.py`) provides realistic travel times that account for how professional players actually navigate the map, rather than theoretical shortest paths.

### Threat Cone Modeling

For any player position, we calculate potential enemy positions based on:

- Last known location
- Time elapsed since detection
- Weapon-based movement speed
- Zones blocked by active utility (smokes prevent vision, mollies deny areas)

Currently, threat probability uses inverse-time weighting (closer zones more likely). Future work will replace this with empirical position distributions from the episode data.

## Current Pipeline

### 1. Zone and Connectivity Setup
```
data/mirage_zones.json                     # 47 polygon zones
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
# Processes 582 demos
# Output: Episode-level data with positioning, damage, utility, outcomes
```

### 4. Testing and Validation
```bash
python tests/test_episode_tracking.py
# Validates episode tracking on sample demo
# Shows active utility detection and expiration times
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
│   ├── analyze_player_positioning.py    # Main pipeline: episodes + utility
│   ├── build_connectivity_from_movement.py # Empirical connectivity builder
│   ├── zone_classifier.py               # Polygon-based zone classification
│   ├── zone_connectivity.py             # Reachability and threat cones
│   ├── movement_speed.py                # Weapon-specific speeds
│   ├── plant_spot_classifier.py         # Bomb plant position detection
│   ├── utility_tracker.py               # Grenade tracking (unused in main pipeline)
│   ├── audio_tracker.py                 # Sound event detection (unused)
│   ├── strategic_intelligence.py        # CT movement analysis (unused)
│   └── postplant_processor.py           # Legacy processing (unused)
├── tests/
│   ├── test_episode_tracking.py         # Episode validation on sample demo
│   ├── test_zone_classifier.py          # Zone polygon tests
│   ├── test_plant_spots.py              # Plant position tests
│   └── test_postplant_analysis.py       # Legacy tests
├── paper/
│   └── abstract.md                      # Research abstract
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## Dependencies

Core requirements:
- awpy >= 1.2.0 (CS2 demo parsing)
- pandas >= 2.0.0 (data manipulation)
- numpy >= 1.24.0 (numerical operations)
- shapely >= 2.0.0 (polygon geometry)
- scikit-learn >= 1.3.0 (future modeling)
- lifelines >= 0.27.0 (survival analysis)

Install:
```bash
pip install -r requirements.txt
```

## Survival Model Development

### Hazard Rate Modeling

The next phase involves building Cox proportional hazards models to estimate per-second elimination risk. Hazard ratios (HR) quantify how each factor affects survival probability:

- HR < 1: Factor reduces elimination risk (safer position)
- HR > 1: Factor increases elimination risk (riskier position)

The model will incorporate:

- Position covariates: Current zone, visibility to plant site, distance to nearest cover
- Threat covariates: Empirical zone occupancy distributions from episode data (replacing inverse-time weighting)
- Economy covariates: Weapon loadouts (rifles, AWP, pistols), armor status, defuse kit availability
- Utility covariates: Active smokes blocking critical lanes (jungle smoked, bomb smoked), mollies denying zones
- Temporal covariates: Bomb timer remaining, time since last enemy detection, alive player counts
- Combat covariates: Time to first contact, damage taken/dealt ratios

This approach enables position-specific survival curves that update dynamically as utility expires, enemies are eliminated, and bomb timer depletes.

### Implementation Plan

1. Build empirical position distributions from collected episode data to replace placeholder inverse-time threat weighting
2. Fit Cox proportional hazards models using episode features with time-dependent covariates
3. Validate model calibration and predictive accuracy on held-out tournaments
4. Generate survival curves for key positions accounting for utility states and economy

## Future Enhancements

1. Multi-map support (currently Mirage only)
2. Integration with real-time demo parsing for live probability overlays during matches
3. Defender (CT) perspective analysis during retakes
4. Team-specific tactical pattern recognition and strategy clustering

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
