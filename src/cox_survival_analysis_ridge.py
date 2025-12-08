"""
Cox Proportional Hazards Survival Analysis for CS2 Post-Plant Positioning

This script fits Cox regression models to analyze:
1. Zone-specific death hazards
2. Equipment effects (health, armor, weapons)
3. Utility state impact (smokes, mollies, flashes)
4. Crossfire effects (teammate proximity)
5. Numerical advantage effects
6. Interaction terms (zone × utility, zone × crossfire, etc.)

Run after analyze_player_positioning.py completes and generates parquet files.

THIS VERSION: Ridge regularization (L2), grid search for λ, 5-fold CV, standardized features
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# CONFIGURATION: Grid search for optimal regularization
LAMBDA_GRID = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
CV_FOLDS = 10

print("="*80)
print("COX PROPORTIONAL HAZARDS SURVIVAL ANALYSIS (RIDGE REGULARIZATION)")
print("="*80)

# ============================================================================
# STANDARDIZATION UTILITY FUNCTION
# ============================================================================

def standardize_continuous_features(df, continuous_cols, fit_params=None):
    """
    Standardize continuous features to mean=0, std=1.
    
    This is CRITICAL for Ridge regression because the L2 penalty is scale-dependent.
    Without standardization, features with larger natural ranges get less penalty
    simply due to their units of measurement, not their predictive importance.
    
    Parameters:
    -----------
    df : DataFrame
        Data containing features to standardize
    continuous_cols : list
        Column names to standardize
    fit_params : dict, optional
        Pre-computed means and stds (for test set)
        
    Returns:
    --------
    df_standardized : DataFrame
        Copy of df with standardized columns
    params : dict
        Mean and std for each column (for future use on test sets)
    """
    df_std = df.copy()
    
    if fit_params is None:
        # Fit on training data
        fit_params = {}
        for col in continuous_cols:
            if col in df_std.columns:
                mean_val = df_std[col].mean()
                std_val = df_std[col].std()
                fit_params[col] = {'mean': mean_val, 'std': std_val}
                
                # Standardize: (x - mean) / std
                if std_val > 0:
                    df_std[col] = (df_std[col] - mean_val) / std_val
                else:
                    df_std[col] = 0  # Constant column, center at 0
    else:
        # Transform using pre-computed params (for test data)
        for col in continuous_cols:
            if col in df_std.columns and col in fit_params:
                mean_val = fit_params[col]['mean']
                std_val = fit_params[col]['std']
                if std_val > 0:
                    df_std[col] = (df_std[col] - mean_val) / std_val
                else:
                    df_std[col] = 0
    
    return df_std, fit_params

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("\n[1/7] Loading episode data from parquet files...")
ct_df = pd.read_parquet('data/ct_episodes.parquet')
t_df = pd.read_parquet('data/t_episodes.parquet')

print(f"Loaded {len(ct_df):,} CT episodes and {len(t_df):,} T episodes")
print(f"Total episodes: {len(ct_df) + len(t_df):,}")

# ============================================================================
# 2. DATA PREPARATION - FLATTEN NESTED STRUCTURES
# ============================================================================

print("\n[2/7] Preparing data for Cox regression...")

def prepare_survival_data(df, team='CT'):
    """
    Prepare episode data for Cox regression.
    
    Key transformations:
    - Flatten nested dicts (armor, weapons, utility)
    - Create survival duration and event indicator
    - Add teammate proximity features (for crossfire analysis)
    - Create interaction terms
    """
    
    df = df.copy()
    
    # ========================================
    # SURVIVAL OUTCOME VARIABLES
    # ========================================
    
    # Create unique retake ID for clustering (episodes in same retake are correlated)
    df['retake_id'] = df['demo_file'] + '_R' + df['round_num'].astype(str) + '_T' + df['plant_tick'].astype(str)
    
    # Duration (convert ticks to seconds: 64 ticks = 1 second)
    df['duration_seconds'] = df['duration_ticks'] / 64.0
    
    # Event indicator (1 = death, 0 = censored)
    # Actual column is 'died' (boolean), not 'event_type'
    df['death_event'] = df['died'].astype(int)
    
    # Filter out invalid durations
    df = df[df['duration_seconds'] > 0].copy()
    
    print(f"  {team} episodes: {len(df):,}")
    print(f"    Deaths: {df['death_event'].sum():,} ({df['death_event'].mean()*100:.1f}%)")
    print(f"    Censored: {(~df['death_event'].astype(bool)).sum():,}")
    print(f"    Mean duration: {df['duration_seconds'].mean():.1f}s")
    
    # ========================================
    # FLATTEN ARMOR (JSON string → columns)
    # ========================================
    
    # player_armor_current is JSON string like: '{"armor_value": 100, "has_helmet": true, "armor_type": "Kevlar"}'
    if 'player_armor_current' in df.columns:
        import json
        def parse_armor(armor_str):
            try:
                if isinstance(armor_str, str):
                    armor_dict = json.loads(armor_str)
                    return armor_dict.get('armor_value', 0), armor_dict.get('has_helmet', False)
                else:
                    return 0, False
            except:
                return 0, False
        
        armor_parsed = df['player_armor_current'].apply(parse_armor)
        df['armor_value'] = armor_parsed.apply(lambda x: x[0])
        df['has_helmet'] = armor_parsed.apply(lambda x: x[1])
    else:
        df['armor_value'] = 0
        df['has_helmet'] = False
    
    # ========================================
    # FLATTEN WEAPONS (string → weapon type)
    # ========================================
    
    # player_weapon_current is already a string like 'AK-47', 'USP-S', etc.
    if 'player_weapon_current' in df.columns:
        df['primary_weapon'] = df['player_weapon_current']
        
        # Simplified weapon grouping: Rifle, Pistol, SMG, Shotgun
        def categorize_weapon(weapon):
            if pd.isna(weapon) or weapon == 'None' or weapon == '':
                return 'None'
            
            # Filter out knives and grenades (non-combat equipment)
            non_weapons = ['Knife', 'Bayonet', 'Flashbang', 'Smoke Grenade', 'High Explosive Grenade', 
                          'Incendiary Grenade', 'Molotov', 'Decoy Grenade', 'Zeus x27']
            if any(nw in weapon for nw in non_weapons):
                return 'None'
            
            # Group ALL rifles together
            rifles = ['AK-47', 'M4A4', 'M4A1-S', 'FAMAS', 'Galil AR', 'SG 553', 'AUG']
            if any(r in weapon for r in rifles):
                return 'Rifle'
            
            # Group ALL pistols together
            pistols = ['Desert Eagle', 'Glock-18', 'USP-S', 'Five-SeveN', 'Tec-9', 
                      'R8 Revolver', 'Dual Berettas', 'CZ75-Auto', 'P250', 'P2000']
            if any(p in weapon for p in pistols):
                return 'Pistol'
            
            # Group ALL SMGs together
            smgs = ['MP9', 'MAC-10', 'MP7', 'MP5-SD', 'UMP-45', 'P90', 'PP-Bizon']
            if any(s in weapon for s in smgs):
                return 'SMG'
            
            # Shotguns
            shotguns = ['Nova', 'XM1014', 'MAG-7', 'Sawed-Off']
            if any(sh in weapon for sh in shotguns):
                return 'Shotgun'
            
            # AWP (keep separate - highly distinctive)
            if weapon == 'AWP':
                return 'AWP'
            
            # Heavy weapons
            heavy = ['Negev', 'M249']
            if any(h in weapon for h in heavy):
                return 'Heavy'
            
            return 'None'
        
        df['weapon_type'] = df['primary_weapon'].apply(categorize_weapon)
    else:
        df['weapon_type'] = 'None'
    
    # ========================================
    # UTILITY PARSING (from JSON string)
    # ========================================
    
    # utility_thrown is JSON string like: '{"smokes": [], "flashes": [1234], "he": [], "mollies": []}'
    if 'utility_thrown' in df.columns:
        import json
        def parse_utility(util_str):
            try:
                if isinstance(util_str, str):
                    util_dict = json.loads(util_str)
                    return (
                        len(util_dict.get('smokes', [])),
                        len(util_dict.get('mollies', [])),
                        len(util_dict.get('flashes', [])),
                        len(util_dict.get('he', []))
                    )
                else:
                    return 0, 0, 0, 0
            except:
                return 0, 0, 0, 0
        
        util_parsed = df['utility_thrown'].apply(parse_utility)
        df['smokes_available'] = util_parsed.apply(lambda x: x[0])
        df['mollies_available'] = util_parsed.apply(lambda x: x[1])
        df['flashes_available'] = util_parsed.apply(lambda x: x[2])
        df['he_grenades_available'] = util_parsed.apply(lambda x: x[3])
    else:
        df['smokes_available'] = 0
        df['mollies_available'] = 0
        df['flashes_available'] = 0
        df['he_grenades_available'] = 0
    
    # Total utility count
    df['total_utility'] = df['smokes_available'] + df['mollies_available'] + df['flashes_available'] + df['he_grenades_available']
    
    # Active utility exposure (from parquet columns: in_smoke, in_molly, flashed)
    if 'in_smoke' in df.columns:
        df['active_in_smoke'] = df['in_smoke']
    else:
        df['active_in_smoke'] = False
        
    if 'in_molly' in df.columns:
        df['active_in_molly'] = df['in_molly']
    else:
        df['active_in_molly'] = False
        
    if 'flashed' in df.columns:
        df['flashed'] = df['flashed']
    else:
        df['flashed'] = False
    
    # ========================================
    # SMOKE/MOLLY TIME REMAINING
    # ========================================
    
    # Parse smoke_time_remaining and molly_time_remaining dicts
    # Format: "{'Zone_Name': 12.5, 'Another_Zone': 8.3}"
    def parse_time_remaining(time_str):
        """Parse time remaining dict and return max time or 0"""
        if pd.isna(time_str) or time_str == '{}':
            return 0.0
        try:
            import ast
            time_dict = ast.literal_eval(time_str)
            if time_dict:
                return max(time_dict.values())  # Longest remaining utility
            return 0.0
        except:
            return 0.0
    
    if 'smoke_time_remaining' in df.columns:
        df['max_smoke_time_left'] = df['smoke_time_remaining'].apply(parse_time_remaining)
    else:
        df['max_smoke_time_left'] = 0.0
    
    if 'molly_time_remaining' in df.columns:
        df['max_molly_time_left'] = df['molly_time_remaining'].apply(parse_time_remaining)
    else:
        df['max_molly_time_left'] = 0.0
    
    # Binary flags: any smoke/molly still active?
    df['has_active_smoke'] = df['max_smoke_time_left'] > 0
    df['has_active_molly'] = df['max_molly_time_left'] > 0
    
    # Categorize time remaining (early vs late utility)
    df['smoke_is_fresh'] = df['max_smoke_time_left'] >= 10.0  # 10+ seconds left
    df['molly_is_fresh'] = df['max_molly_time_left'] >= 3.0   # 3+ seconds left
    
    # ========================================
    # NUMERICAL ADVANTAGE
    # ========================================
    
    # Use numerical_advantage_current (real-time advantage throughout episode)
    # This accounts for kills/deaths during the retake, not just initial plant state
    if 'numerical_advantage_current' in df.columns:
        # Already calculated in episode data (positive = your team has advantage)
        if team == 'CT':
            df['numerical_advantage'] = df['numerical_advantage_current']
        else:
            df['numerical_advantage'] = -df['numerical_advantage_current']  # Flip sign for T perspective
    else:
        # Fallback to plant-time advantage
        if team == 'CT':
            df['numerical_advantage'] = df['ct_count_at_plant'] - df['t_count_at_plant']
        else:
            df['numerical_advantage'] = df['t_count_at_plant'] - df['ct_count_at_plant']
    
    # ========================================
    # CROSSFIRE FEATURES (TEAMMATE COMPOSITION)
    # ========================================
    
    # Load visibility graph for crossfire calculation (empirical data from gun damage)
    visibility_path = Path('data/mirage_visibility_from_damage.json')
    visibility_graph = {}
    if visibility_path.exists():
        with open(visibility_path) as f:
            vis_data = json.load(f)
            visibility_graph = vis_data.get('zones', {})
    
    # Load movement-based zone connectivity for indirect threat calculation
    connectivity_path = Path('data/mirage_zone_connectivity_from_movement.json')
    movement_connectivity = {}
    if connectivity_path.exists():
        with open(connectivity_path) as f:
            movement_connectivity = json.load(f)
    
    # Get top zones across entire dataset to create teammate features
    top_zones_global = df['zone'].value_counts().head(25).index.tolist()
    
    # REMOVED: teammate_at_{zone} features - too sparse, causing convergence failures
    # Equipment model outperforms full model with these features (0.733 vs 0.718)
    
    # Initialize teammate aggregate features (mirror all player attributes)
    df['num_teammates_total'] = 0
    df['has_crossfire_support'] = False
    
    # Teammate quality features (aggregated across teammates)
    df['teammate_avg_health_norm'] = 0.0
    df['teammate_avg_armor_norm'] = 0.0
    df['teammate_helmet_rate'] = 0.0
    df['teammate_avg_utility'] = 0.0
    
    # CT-specific teammate features
    if team == 'CT':
        df['teammate_kit_rate'] = 0.0  # % of teammates with defuse kit
    
    # Teammate weapon composition (counts by category)
    df['num_teammates_rifle'] = 0
    df['num_teammates_pistol'] = 0
    df['num_teammates_smg'] = 0
    df['num_teammates_shotgun'] = 0
    df['num_teammates_awp'] = 0
    
    # Teammate tactical state (counts)
    df['num_teammates_in_smoke'] = 0
    df['num_teammates_in_molly'] = 0
    df['num_teammates_flashed'] = 0
    
    # Teammate damage features
    df['teammate_avg_damage_taken'] = 0.0
    df['teammate_avg_damage_dealt'] = 0.0
    
    # Group by retake to populate teammate composition
    retake_groups = df.groupby(['demo_file', 'round_num', 'plant_tick'])
    
    print(f"  Computing teammate composition features for {len(retake_groups)} retakes...")
    
    for retake_id, group in retake_groups:
        # CRITICAL: Get BASELINE positions only (first episode per player)
        # This prevents information leakage from future movements
        baseline_positions = group.groupby('player_name').first()
        zones_at_baseline = list(baseline_positions['zone'].values)
        zone_counts_baseline = pd.Series(zones_at_baseline).value_counts()
        
        # For each episode in this retake
        for idx in group.index:
            player_zone = df.loc[idx, 'zone']
            player_name = df.loc[idx, 'player_name']
            
            # Get teammates (all players except current one)
            teammates = baseline_positions[baseline_positions.index != player_name]
            teammates_count = len(teammates)
            df.loc[idx, 'num_teammates_total'] = teammates_count
            
            if teammates_count > 0:
                # ========================================
                # TEAMMATE QUALITY AGGREGATES
                # ========================================
                
                # Health and armor (normalized 0-1)
                df.loc[idx, 'teammate_avg_health_norm'] = (teammates['player_hp_current'] / 100.0).mean()
                df.loc[idx, 'teammate_avg_armor_norm'] = (teammates['armor_value'] / 100.0).mean()
                
                # Helmet rate (% of teammates with helmets)
                df.loc[idx, 'teammate_helmet_rate'] = teammates['has_helmet'].mean()
                
                # Utility count
                df.loc[idx, 'teammate_avg_utility'] = teammates['total_utility'].mean()
                
                # Damage features
                teammate_damage_taken = teammates['damage_taken_gun'] + teammates['damage_taken_util']
                teammate_damage_dealt = teammates['damage_dealt_gun'] + teammates['damage_dealt_util']
                df.loc[idx, 'teammate_avg_damage_taken'] = teammate_damage_taken.mean()
                df.loc[idx, 'teammate_avg_damage_dealt'] = teammate_damage_dealt.mean()
                
                # CT-specific: Defuse kit rate among teammates
                if team == 'CT' and 'has_kit' in teammates.columns:
                    df.loc[idx, 'teammate_kit_rate'] = teammates['has_kit'].mean()
                
                # ========================================
                # TEAMMATE WEAPON COMPOSITION
                # ========================================
                
                # weapon_type is ALREADY categorized - just count by category
                teammate_weapons = teammates['weapon_type'].value_counts()
                
                # Count by simplified categories
                df.loc[idx, 'num_teammates_rifle'] = teammate_weapons.get('Rifle', 0)
                df.loc[idx, 'num_teammates_pistol'] = teammate_weapons.get('Pistol', 0)
                df.loc[idx, 'num_teammates_smg'] = teammate_weapons.get('SMG', 0)
                df.loc[idx, 'num_teammates_shotgun'] = teammate_weapons.get('Shotgun', 0)
                df.loc[idx, 'num_teammates_awp'] = teammate_weapons.get('AWP', 0)
                
                # ========================================
                # TEAMMATE TACTICAL STATE
                # ========================================
                
                # Utility effects (if columns exist)
                if 'in_smoke' in teammates.columns:
                    df.loc[idx, 'num_teammates_in_smoke'] = teammates['in_smoke'].sum()
                
                if 'in_molly' in teammates.columns:
                    df.loc[idx, 'num_teammates_in_molly'] = teammates['in_molly'].sum()
                
                if 'flashed' in teammates.columns:
                    df.loc[idx, 'num_teammates_flashed'] = teammates['flashed'].sum()
            
            # ========================================
            # TEAMMATE POSITIONAL FEATURES - REMOVED
            # ========================================
            
            # REMOVED: teammate_at_{zone} counts - sparse binary features (<1% occurrence)
            # caused convergence failures (NaN in Newton-Raphson) and degraded C-index
            # (Equipment: 0.733 > Full with positions: 0.718)
            # Keeping only aggregate teammate quality features instead
            
            # ADVANCED CROSSFIRE CALCULATION
            # Teammate has crossfire if they can see threat zones (direct + 1-hop swing-out)
            # CRITICAL: Use BASELINE teammate positions only, not future movements
            has_crossfire = False
            if teammates_count > 0 and visibility_graph and movement_connectivity:
                # Build threat zone set for player
                threat_zones = set()
                
                # 1. Direct visibility threats (zones that can see player right now)
                direct_threats = visibility_graph.get(player_zone, [])
                threat_zones.update(direct_threats)
                
                # 2. 1-hop indirect threats (zones enemies can swing from to peek player)
                # For each zone with direct visibility, add zones connected to it
                for visible_zone in direct_threats:
                    connected_zones = movement_connectivity.get(visible_zone, {}).keys()
                    threat_zones.update(connected_zones)
                
                # 3. Check if any teammate AT BASELINE can see these threat zones
                # Include both direct visibility AND 1-hop swing-out potential from baseline
                for teammate_zone in zones_at_baseline:
                    if teammate_zone != player_zone:
                        # Direct visibility from baseline position
                        teammate_visible = set(visibility_graph.get(teammate_zone, []))
                        
                        # Add 1-hop swing-out visibility (zones teammate can swing to and shoot from)
                        teammate_swing_zones = movement_connectivity.get(teammate_zone, {}).keys()
                        for swing_zone in teammate_swing_zones:
                            teammate_visible.update(visibility_graph.get(swing_zone, []))
                        
                        # Crossfire = teammate can cover at least one threat zone (from baseline or 1-hop swing)
                        if threat_zones & teammate_visible:  # Set intersection
                            has_crossfire = True
                            break
            
            # Fallback: if no data available, any teammate = potential crossfire
            if not (visibility_graph and movement_connectivity) and teammates_count > 0:
                has_crossfire = True
            
            df.loc[idx, 'has_crossfire_support'] = has_crossfire
    
    # ========================================
    # NORMALIZE CONTINUOUS VARIABLES
    # ========================================
    
    # Use player_hp_current (real-time health) instead of at_plant
    df['health_norm'] = df['player_hp_current'] / 100.0
    
    # Scale armor to 0-1
    df['armor_norm'] = df['armor_value'] / 100.0
    
    # Add damage variables (useful predictors)
    df['damage_taken_total'] = df['damage_taken_gun'] + df['damage_taken_util']
    df['damage_dealt_total'] = df['damage_dealt_gun'] + df['damage_dealt_util']
    
    # CT-specific: Defuse kit (reduces defuse time from 10s to 5s)
    if team == 'CT':
        df['has_kit'] = df['had_kit_at_plant'].fillna(False).astype(int)
    else:
        df['has_kit'] = 0  # T side never has kits
    
    return df


# Prepare both datasets
ct_prepared = prepare_survival_data(ct_df, team='CT')
t_prepared = prepare_survival_data(t_df, team='T')

# ============================================================================
# 3. MODEL 1: SIMPLE ZONE-ONLY MODEL (Baseline)
# ============================================================================

print("\n[3/7] Fitting Model 1: Zone-Only (Baseline)...")

def fit_zone_model(df, team='CT', site='A'):
    """Fit Cox model with zone as primary covariate (stratified baseline hazard)."""
    
    # Filter to specific site
    df_site = df[df['site'] == site].copy()
    
    # Get top zones (with sufficient sample size)
    zone_counts = df_site['zone'].value_counts()
    top_zones = zone_counts[zone_counts >= 50].index.tolist()
    
    df_site = df_site[df_site['zone'].isin(top_zones)].copy()
    
    print(f"\n  {team} {site}-site: {len(df_site):,} episodes across {len(top_zones)} zones")
    
    # Create dummy variables for zones (exclude reference category)
    zone_dummies = pd.get_dummies(df_site['zone'], prefix='zone', drop_first=True)
    
    # Combine with survival outcomes
    model_df = pd.concat([
        df_site[['duration_seconds', 'death_event', 'retake_id']],
        zone_dummies
    ], axis=1)
    
    # Standardize continuous features (zones are binary, no need to standardize)
    # For zone-only model, no continuous features to standardize
    
    # Ridge penalty grid search with 5-fold CV
    print(f"  Grid search for optimal Ridge penalty (λ) with {CV_FOLDS}-fold CV...")
    group_kfold = GroupKFold(n_splits=CV_FOLDS)
    
    best_lambda = None
    best_cv_score = -np.inf
    lambda_results = {}
    
    for penalizer in LAMBDA_GRID:
        cv_scores = []
        for train_idx, test_idx in group_kfold.split(model_df, groups=model_df['retake_id']):
            train_data = model_df.iloc[train_idx]
            test_data = model_df.iloc[test_idx]
            
            fold_cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
            try:
                fold_cph.fit(train_data, duration_col='duration_seconds', event_col='death_event', cluster_col='retake_id')
                c_index = fold_cph.score(test_data, scoring_method='concordance_index')
                cv_scores.append(c_index)
            except:
                pass
        
        mean_cv = np.mean(cv_scores) if cv_scores else 0.0
        lambda_results[penalizer] = mean_cv
        
        if mean_cv > best_cv_score:
            best_cv_score = mean_cv
            best_lambda = penalizer
    
    print(f"  Best λ: {best_lambda} (CV C-index: {best_cv_score:.3f})")
    print(f"  All λ results: {lambda_results}")
    
    # Fit final model with best penalty
    cph = CoxPHFitter(penalizer=best_lambda, l1_ratio=0.0)
    cph.fit(model_df, duration_col='duration_seconds', event_col='death_event', cluster_col='retake_id')
    
    print(f"  Concordance Index: {cph.concordance_index_:.3f} (CV: {best_cv_score:.3f})")
    
    return cph, df_site, top_zones, best_lambda, best_cv_score


# Fit models for each team/site combination
ct_a_zone_model, ct_a_data, ct_a_zones, ct_a_zone_lambda, ct_a_zone_cv = fit_zone_model(ct_prepared, 'CT', 'A')
ct_b_zone_model, ct_b_data, ct_b_zones, ct_b_zone_lambda, ct_b_zone_cv = fit_zone_model(ct_prepared, 'CT', 'B')
t_a_zone_model, t_a_data, t_a_zones, t_a_zone_lambda, t_a_zone_cv = fit_zone_model(t_prepared, 'T', 'A')
t_b_zone_model, t_b_data, t_b_zones, t_b_zone_lambda, t_b_zone_cv = fit_zone_model(t_prepared, 'T', 'B')

# ============================================================================
# 4. MODEL 2: ZONE + EQUIPMENT MODEL
# ============================================================================

print("\n[4/7] Fitting Model 2: Zone + Equipment...")

def fit_equipment_model(df, team='CT', site='A'):
    """Fit Cox model with zone + health + armor + weapons."""
    
    df_site = df[df['site'] == site].copy()
    zone_counts = df_site['zone'].value_counts()
    top_zones = zone_counts[zone_counts >= 50].index.tolist()
    df_site = df_site[df_site['zone'].isin(top_zones)].copy()
    
    # Zone dummies
    zone_dummies = pd.get_dummies(df_site['zone'], prefix='zone', drop_first=True)
    
    # Weapon type dummies
    weapon_dummies = pd.get_dummies(df_site['weapon_type'], prefix='weapon', drop_first=True)
    
    # Build feature list based on team
    feature_cols = ['duration_seconds', 'death_event', 'retake_id',
                    'health_norm', 'armor_norm', 'has_helmet']
    
    # Add CT-specific features
    if team == 'CT':
        feature_cols.append('has_kit')
    
    # Combine features
    model_df = pd.concat([
        df_site[feature_cols],
        zone_dummies,
        weapon_dummies
    ], axis=1)
    
    # Standardize continuous features BEFORE Ridge regression
    # Critical for fair penalization across features with different scales
    continuous_to_standardize = ['health_norm', 'armor_norm']
    print(f"  Standardizing continuous features: {continuous_to_standardize}")
    
    # Ridge penalty grid search with 5-fold CV
    print(f"  Grid search for optimal Ridge penalty (λ) with {CV_FOLDS}-fold CV...")
    group_kfold = GroupKFold(n_splits=CV_FOLDS)
    
    best_lambda = None
    best_cv_score = -np.inf
    lambda_results = {}
    
    for penalizer in LAMBDA_GRID:
        cv_scores = []
        for train_idx, test_idx in group_kfold.split(model_df, groups=model_df['retake_id']):
            train_data = model_df.iloc[train_idx].copy()
            test_data = model_df.iloc[test_idx].copy()
            
            # Standardize on train, apply same transform to test
            train_data, std_params = standardize_continuous_features(train_data, continuous_to_standardize)
            test_data, _ = standardize_continuous_features(test_data, continuous_to_standardize, fit_params=std_params)
            
            fold_cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
            try:
                fold_cph.fit(train_data, duration_col='duration_seconds', event_col='death_event', cluster_col='retake_id')
                c_index = fold_cph.score(test_data, scoring_method='concordance_index')
                cv_scores.append(c_index)
            except:
                pass
        
        mean_cv = np.mean(cv_scores) if cv_scores else 0.0
        lambda_results[penalizer] = mean_cv
        
        if mean_cv > best_cv_score:
            best_cv_score = mean_cv
            best_lambda = penalizer
    
    print(f"  Best λ: {best_lambda} (CV C-index: {best_cv_score:.3f})")
    print(f"  All λ results: {lambda_results}")
    
    # Fit final model on full dataset (standardized) with best penalty
    model_df_std, _ = standardize_continuous_features(model_df, continuous_to_standardize)
    cph = CoxPHFitter(penalizer=best_lambda, l1_ratio=0.0)
    cph.fit(model_df_std, duration_col='duration_seconds', event_col='death_event', cluster_col='retake_id')
    
    print(f"  {team} {site}-site Equipment Model - C-index: {cph.concordance_index_:.3f} (CV: {best_cv_score:.3f})")
    
    return cph, df_site, best_lambda, best_cv_score


ct_a_equip_model, ct_a_equip_data, ct_a_equip_lambda, ct_a_equip_cv = fit_equipment_model(ct_prepared, 'CT', 'A')
ct_b_equip_model, ct_b_equip_data, ct_b_equip_lambda, ct_b_equip_cv = fit_equipment_model(ct_prepared, 'CT', 'B')
t_a_equip_model, t_a_equip_data, t_a_equip_lambda, t_a_equip_cv = fit_equipment_model(t_prepared, 'T', 'A')
t_b_equip_model, t_b_equip_data, t_b_equip_lambda, t_b_equip_cv = fit_equipment_model(t_prepared, 'T', 'B')

# ============================================================================
# 5. MODEL 3: FULL MODEL (Zone + Equipment + Utility + Crossfire + Advantage)
# ============================================================================

print("\n[5/7] Fitting Model 3: Full Model with Interactions...")

def fit_full_model(df, team='CT', site='A'):
    """
    Fit comprehensive Cox model including:
    - Zone effects
    - Equipment (health, armor, weapons)
    - Utility state (available utility, active exposure)
    - Teammate composition (who is where)
    - Numerical advantage
    - Key interactions (zone × teammate positions, zone × utility, zone × advantage)
    """
    
    df_site = df[df['site'] == site].copy()
    zone_counts = df_site['zone'].value_counts()
    top_zones = zone_counts[zone_counts >= 50].index.tolist()
    df_site = df_site[df_site['zone'].isin(top_zones)].copy()
    
    # Zone and weapon dummies
    zone_dummies = pd.get_dummies(df_site['zone'], prefix='zone', drop_first=True)
    weapon_dummies = pd.get_dummies(df_site['weapon_type'], prefix='weapon', drop_first=True)
    
    # REMOVED: teammate_at_[zone] features entirely from data pipeline
    # These sparse features caused convergence failures and degraded performance
    
    # Build feature list dynamically based on team
    feature_cols = [
        'duration_seconds', 'death_event', 'retake_id',
        # Player equipment
        'health_norm', 'armor_norm', 'has_helmet',
        # Player utility
        'total_utility', 
        'active_in_smoke', 'active_in_molly', 'flashed',
        # Teammate aggregate quality (not positions)
        'teammate_avg_health_norm', 'teammate_avg_armor_norm', 
        'teammate_helmet_rate', 'teammate_avg_utility',
        'num_teammates_rifle', 'num_teammates_pistol', 'num_teammates_smg',
        'num_teammates_awp',
        'num_teammates_in_smoke', 'num_teammates_in_molly', 'num_teammates_flashed',
        'teammate_avg_damage_taken', 'teammate_avg_damage_dealt',
        # Crossfire support (tactical coordination)
        'has_crossfire_support', 'num_teammates_total',
        # Numerical advantage
        'numerical_advantage'
    ]
    
    # Add CT-specific features
    if team == 'CT':
        feature_cols.extend(['has_kit', 'teammate_kit_rate'])
    
    # Main effects
    features = pd.concat([
        df_site[feature_cols],
        zone_dummies,
        weapon_dummies
    ], axis=1)
    
    # FIRST: Identify which interaction terms to create (based on sparsity filters)
    # We'll create the actual interactions AFTER standardization
    
    interaction_specs = []  # Store (zone_col, interaction_type) tuples
    
    # 1. Zone × Crossfire Support: Do certain zones benefit more from teammate support?
    for zone_col in zone_dummies.columns[:15]:
        # Check sparsity using raw data
        interaction_values = features[zone_col] * features['has_crossfire_support']
        if interaction_values.sum() > len(df_site) * 0.03:
            interaction_specs.append((zone_col, 'has_support'))
    
    # 2. Active Molly × Zone: Does molly exposure affect zones differently?
    for zone_col in zone_dummies.columns[:10]:
        zone_mask = features[zone_col] == 1
        if zone_mask.sum() > 0:
            molly_rate_in_zone = features.loc[zone_mask, 'active_in_molly'].mean()
            if molly_rate_in_zone > 0.05:
                interaction_specs.append((zone_col, 'molly'))
    
    # 3. Active Smoke × Zone: Does smoke exposure affect zones differently?
    for zone_col in zone_dummies.columns[:10]:
        zone_mask = features[zone_col] == 1
        if zone_mask.sum() > 0:
            smoke_rate_in_zone = features.loc[zone_mask, 'active_in_smoke'].mean()
            if smoke_rate_in_zone > 0.05:
                interaction_specs.append((zone_col, 'smoke'))
    
    # 4. Numerical Advantage × Zone: Do certain zones benefit more from man-advantage?
    for zone_col in zone_dummies.columns[:10]:
        interaction_specs.append((zone_col, 'advantage'))
    
    print(f"  Will create {len(interaction_specs)} interaction terms AFTER standardization")
    
    # Remove columns with zero or near-zero variance (final safety check)
    # Exclude metadata columns from variance calculation
    feature_cols_only = [col for col in features.columns if col not in ['duration_seconds', 'death_event', 'retake_id']]
    variance = features[feature_cols_only].var()
    low_var_cols = variance[variance < 1e-6].index.tolist()
    if low_var_cols:
        features = features.drop(columns=low_var_cols)
        print(f"  Dropped {len(low_var_cols)} zero-variance columns")
    
    # STANDARDIZE ALL CONTINUOUS FEATURES for fair Ridge penalization
    # This is CRITICAL because Ridge penalty is scale-dependent
    continuous_features_to_standardize = [
        'health_norm', 'armor_norm', 
        'total_utility',
        'teammate_avg_health_norm', 'teammate_avg_armor_norm',
        'teammate_helmet_rate', 'teammate_avg_utility',
        'num_teammates_rifle', 'num_teammates_pistol', 'num_teammates_smg', 'num_teammates_awp',
        'num_teammates_in_smoke', 'num_teammates_in_molly', 'num_teammates_flashed',
        'teammate_avg_damage_taken', 'teammate_avg_damage_dealt',
        'num_teammates_total', 'numerical_advantage'
    ]
    
    # Add CT-specific continuous features
    if team == 'CT':
        continuous_features_to_standardize.append('teammate_kit_rate')
    
    # Only standardize features that actually exist in the dataframe
    continuous_to_std = [col for col in continuous_features_to_standardize if col in features.columns]
    print(f"  Standardizing {len(continuous_to_std)} continuous features for fair Ridge penalty")
    
    # Helper function to add interactions to standardized data
    def add_interaction_terms(df, interaction_specs):
        """Create interaction terms from standardized features."""
        for zone_col, interaction_type in interaction_specs:
            if zone_col not in df.columns:
                continue
                
            if interaction_type == 'has_support':
                interaction_name = f'{zone_col}_x_has_support'
                df[interaction_name] = df[zone_col] * df['has_crossfire_support']
            elif interaction_type == 'molly':
                interaction_name = f'{zone_col}_x_molly'
                df[interaction_name] = df[zone_col] * df['active_in_molly']
            elif interaction_type == 'smoke':
                interaction_name = f'{zone_col}_x_smoke'
                df[interaction_name] = df[zone_col] * df['active_in_smoke']
            elif interaction_type == 'advantage':
                interaction_name = f'{zone_col}_x_advantage'
                df[interaction_name] = df[zone_col] * df['numerical_advantage']
        return df
    
    # Ridge penalty grid search with 5-fold CV
    print(f"  Grid search for optimal Ridge penalty (λ) with {CV_FOLDS}-fold CV...")
    group_kfold = GroupKFold(n_splits=CV_FOLDS)
    
    best_lambda = None
    best_cv_score = -np.inf
    lambda_results = {}
    
    for penalizer in LAMBDA_GRID:
        cv_scores = []
        for train_idx, test_idx in group_kfold.split(features, groups=features['retake_id']):
            train_data = features.iloc[train_idx].copy()
            test_data = features.iloc[test_idx].copy()
            
            # Standardize continuous features (fit on train, apply to test)
            train_data, std_params = standardize_continuous_features(train_data, continuous_to_std)
            test_data, _ = standardize_continuous_features(test_data, continuous_to_std, fit_params=std_params)
            
            # Create interactions from standardized features
            train_data = add_interaction_terms(train_data, interaction_specs)
            test_data = add_interaction_terms(test_data, interaction_specs)
            
            fold_cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
            try:
                fold_cph.fit(train_data, duration_col='duration_seconds', event_col='death_event', cluster_col='retake_id')
                c_index = fold_cph.score(test_data, scoring_method='concordance_index')
                cv_scores.append(c_index)
            except:
                pass
        
        mean_cv = np.mean(cv_scores) if cv_scores else 0.0
        lambda_results[penalizer] = mean_cv
        
        if mean_cv > best_cv_score:
            best_cv_score = mean_cv
            best_lambda = penalizer
    
    print(f"  Best λ: {best_lambda} (CV C-index: {best_cv_score:.3f})")
    print(f"  All λ results: {lambda_results}")
    
    # Fit final model on full dataset (standardized, with interactions) using best penalty
    features_std, _ = standardize_continuous_features(features, continuous_to_std)
    features_std = add_interaction_terms(features_std, interaction_specs)
    cph = CoxPHFitter(penalizer=best_lambda, l1_ratio=0.0)
    cph.fit(features_std, duration_col='duration_seconds', event_col='death_event', cluster_col='retake_id')
    
    # Calculate main effects count properly
    base_features = 18 if team == 'T' else 20  # T has 18, CT has 20 (includes has_kit, teammate_kit_rate)
    num_main_effects = len(zone_dummies.columns) + len(weapon_dummies.columns) + base_features
    
    print(f"  {team} {site}-site Full Model - C-index: {cph.concordance_index_:.3f} (CV: {best_cv_score:.3f})")
    print(f"  Number of covariates: {len(cph.params_)} (Ridge keeps all)")
    print(f"  Main effects: {num_main_effects}")
    print(f"  Interaction terms: {len(interaction_specs)}")
    
    return cph, df_site, features_std, best_lambda, best_cv_score


ct_a_full_model, ct_a_full_data, ct_a_full_features, ct_a_full_lambda, ct_a_full_cv = fit_full_model(ct_prepared, 'CT', 'A')
ct_b_full_model, ct_b_full_data, ct_b_full_features, ct_b_full_lambda, ct_b_full_cv = fit_full_model(ct_prepared, 'CT', 'B')
t_a_full_model, t_a_full_data, t_a_full_features, t_a_full_lambda, t_a_full_cv = fit_full_model(t_prepared, 'T', 'A')
t_b_full_model, t_b_full_data, t_b_full_features, t_b_full_lambda, t_b_full_cv = fit_full_model(t_prepared, 'T', 'B')

# ============================================================================
# 6. GENERATE RESULTS & VISUALIZATIONS
# ============================================================================

print("\n[6/7] Generating results and visualizations...")

# Create results directory for grid search run
results_dir = Path('results/cox_models/ridge_gridsearch')
results_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*80}")
print(f"OUTPUT DIRECTORY: {results_dir.absolute()}")
print(f"{'='*80}")
print("All outputs will be saved to this directory.")
print("Original ridge/ folder remains untouched.\n")

# ========================================
# 6.1 SAVE MODELS AS PICKLE FILES
# ========================================

import pickle

models_to_save = {
    'ct_a_zone_model.pkl': ct_a_zone_model,
    'ct_b_zone_model.pkl': ct_b_zone_model,
    't_a_zone_model.pkl': t_a_zone_model,
    't_b_zone_model.pkl': t_b_zone_model,
    'ct_a_equipment_model.pkl': ct_a_equip_model,
    'ct_b_equipment_model.pkl': ct_b_equip_model,
    't_a_equipment_model.pkl': t_a_equip_model,
    't_b_equipment_model.pkl': t_b_equip_model,
    'ct_a_full_model.pkl': ct_a_full_model,
    'ct_b_full_model.pkl': ct_b_full_model,
    't_a_full_model.pkl': t_a_full_model,
    't_b_full_model.pkl': t_b_full_model
}

for filename, model in models_to_save.items():
    with open(results_dir / filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved model: {filename}")

# ========================================
# 6.2 HAZARD RATIO TABLES
# ========================================

def save_hazard_ratio_table(model, filename, top_n=None):
    """Save hazard ratio table with confidence intervals."""
    
    summary = model.summary
    summary = summary.sort_values('exp(coef)', ascending=False)
    
    # Format for display
    if top_n is not None:
        table = summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].head(top_n)
    else:
        table = summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    table.columns = ['Hazard Ratio', '95% CI Lower', '95% CI Upper', 'p-value']
    
    # Save
    table.to_csv(results_dir / filename)
    print(f"  Saved: {filename} ({len(table)} coefficients)")
    
    return table

# Save tables (ALL covariates to CSV for all 12 models)
# Zone models
ct_a_zone_hr = save_hazard_ratio_table(ct_a_zone_model, 'ct_a_zone_hazard_ratios.csv')
ct_b_zone_hr = save_hazard_ratio_table(ct_b_zone_model, 'ct_b_zone_hazard_ratios.csv')
t_a_zone_hr = save_hazard_ratio_table(t_a_zone_model, 't_a_zone_hazard_ratios.csv')
t_b_zone_hr = save_hazard_ratio_table(t_b_zone_model, 't_b_zone_hazard_ratios.csv')

# Equipment models
ct_a_equip_hr = save_hazard_ratio_table(ct_a_equip_model, 'ct_a_equip_hazard_ratios.csv')
ct_b_equip_hr = save_hazard_ratio_table(ct_b_equip_model, 'ct_b_equip_hazard_ratios.csv')
t_a_equip_hr = save_hazard_ratio_table(t_a_equip_model, 't_a_equip_hazard_ratios.csv')
t_b_equip_hr = save_hazard_ratio_table(t_b_equip_model, 't_b_equip_hazard_ratios.csv')

# Full models
ct_a_full_hr = save_hazard_ratio_table(ct_a_full_model, 'ct_a_full_hazard_ratios.csv')
ct_b_full_hr = save_hazard_ratio_table(ct_b_full_model, 'ct_b_full_hazard_ratios.csv')
t_a_full_hr = save_hazard_ratio_table(t_a_full_model, 't_a_full_hazard_ratios.csv')
t_b_full_hr = save_hazard_ratio_table(t_b_full_model, 't_b_full_hazard_ratios.csv')

# ========================================
# 6.3 FOREST PLOTS (Hazard Ratio Visualization)
# ========================================

def plot_forest(model, title, filename, top_n=20, filter_prefix=None):
    """Generate forest plot of hazard ratios."""
    
    summary = model.summary.copy()
    
    # Filter to specific covariates if requested
    if filter_prefix:
        summary = summary[summary.index.str.startswith(filter_prefix)]
    
    summary = summary.sort_values('exp(coef)').tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    
    y_pos = range(len(summary))
    
    # Plot HR point estimates
    ax.scatter(summary['exp(coef)'], y_pos, s=100, zorder=3, color='darkblue')
    
    # Plot 95% CI error bars
    ax.hlines(y_pos, summary['exp(coef) lower 95%'], summary['exp(coef) upper 95%'], 
              colors='gray', alpha=0.5, linewidth=2)
    
    # Reference line at HR=1
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='No effect (HR=1)')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary.index)
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")

# Generate forest plots for all models
# Zone models
plot_forest(ct_a_zone_model, 'CT A-Site: Zone Hazard Ratios', 'ct_a_zone_forest.png', 
            filter_prefix='zone_')
plot_forest(ct_b_zone_model, 'CT B-Site: Zone Hazard Ratios', 'ct_b_zone_forest.png', 
            filter_prefix='zone_')
plot_forest(t_a_zone_model, 'T A-Site: Zone Hazard Ratios', 't_a_zone_forest.png', 
            filter_prefix='zone_')
plot_forest(t_b_zone_model, 'T B-Site: Zone Hazard Ratios', 't_b_zone_forest.png', 
            filter_prefix='zone_')

# Equipment models
plot_forest(ct_a_equip_model, 'CT A-Site: Equipment Model Top Effects', 'ct_a_equip_forest.png', 
            top_n=20)
plot_forest(ct_b_equip_model, 'CT B-Site: Equipment Model Top Effects', 'ct_b_equip_forest.png', 
            top_n=20)
plot_forest(t_a_equip_model, 'T A-Site: Equipment Model Top Effects', 't_a_equip_forest.png', 
            top_n=20)
plot_forest(t_b_equip_model, 'T B-Site: Equipment Model Top Effects', 't_b_equip_forest.png', 
            top_n=20)

# Full models
plot_forest(ct_a_full_model, 'CT A-Site: Full Model Top Effects', 'ct_a_full_forest.png', 
            top_n=25)
plot_forest(ct_b_full_model, 'CT B-Site: Full Model Top Effects', 'ct_b_full_forest.png', 
            top_n=25)
plot_forest(t_a_full_model, 'T A-Site: Full Model Top Effects', 't_a_full_forest.png', 
            top_n=25)
plot_forest(t_b_full_model, 'T B-Site: Full Model Top Effects', 't_b_full_forest.png', 
            top_n=25)

# ========================================
# 6.4 SURVIVAL CURVES (Kaplan-Meier by Zone)
# ========================================

def plot_survival_curves(df, zones_to_plot, title, filename):
    """Plot Kaplan-Meier survival curves for specific zones."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    kmf = KaplanMeierFitter()
    
    for zone in zones_to_plot:
        zone_data = df[df['zone'] == zone]
        if len(zone_data) < 10:
            continue
        
        kmf.fit(
            durations=zone_data['duration_seconds'],
            event_observed=zone_data['death_event'],
            label=f"{zone} (n={len(zone_data)})"
        )
        kmf.plot_survival_function(ax=ax)
    
    ax.set_xlabel('Time Since Retake Start (seconds)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")

# Plot survival curves for top zones
ct_a_top_zones = ct_a_data['zone'].value_counts().head(8).index.tolist()
plot_survival_curves(ct_a_data, ct_a_top_zones, 
                     'CT A-Site: Survival by Position', 'ct_a_survival_curves.png')

ct_b_top_zones = ct_b_data['zone'].value_counts().head(8).index.tolist()
plot_survival_curves(ct_b_data, ct_b_top_zones, 
                     'CT B-Site: Survival by Position', 'ct_b_survival_curves.png')

t_a_top_zones = t_a_data['zone'].value_counts().head(8).index.tolist()
plot_survival_curves(t_a_data, t_a_top_zones,
                     'T A-Site Post-Plant: Survival by Position', 't_a_survival_curves.png')

t_b_top_zones = t_b_data['zone'].value_counts().head(8).index.tolist()
plot_survival_curves(t_b_data, t_b_top_zones,
                     'T B-Site Post-Plant: Survival by Position', 't_b_survival_curves.png')

# ========================================
# 6.5 KAPLAN-MEIER SURVIVAL CURVES BY ZONE (TOP 5 vs BOTTOM 5)
# ========================================

def plot_km_by_zone(df, team, site, filename, top_n=5):
    """Plot Kaplan-Meier survival curves for top 5 safest vs 5 deadliest zones."""
    
    # Calculate zone death rates to identify safest/deadliest
    zone_death_rates = df.groupby('zone')['death_event'].agg(['sum', 'count'])
    zone_death_rates['death_rate'] = zone_death_rates['sum'] / zone_death_rates['count']
    zone_death_rates = zone_death_rates[zone_death_rates['count'] >= 50]  # Min sample size
    zone_death_rates = zone_death_rates.sort_values('death_rate')
    
    safest_zones = zone_death_rates.head(top_n).index.tolist()
    deadliest_zones = zone_death_rates.tail(top_n).index.tolist()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    kmf = KaplanMeierFitter()
    
    # Plot safest zones
    for zone in safest_zones:
        zone_data = df[df['zone'] == zone]
        if len(zone_data) > 0:
            kmf.fit(zone_data['duration_seconds'], zone_data['death_event'], label=zone)
            kmf.plot_survival_function(ax=ax1, ci_show=False)
    
    ax1.set_title(f'{team} {site}-Site: Safest {top_n} Zones', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Survival Probability', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    
    # Plot deadliest zones
    for zone in deadliest_zones:
        zone_data = df[df['zone'] == zone]
        if len(zone_data) > 0:
            kmf.fit(zone_data['duration_seconds'], zone_data['death_event'], label=zone)
            kmf.plot_survival_function(ax=ax2, ci_show=False)
    
    ax2.set_title(f'{team} {site}-Site: Deadliest {top_n} Zones', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Survival Probability', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")
    
    return safest_zones, deadliest_zones

# Generate KM curves for all 4 sites
print("\nGenerating Kaplan-Meier survival curves by zone...")
ct_a_safest, ct_a_deadliest = plot_km_by_zone(ct_a_full_data, 'CT', 'A', 'ct_a_km_by_zone.png')
ct_b_safest, ct_b_deadliest = plot_km_by_zone(ct_b_full_data, 'CT', 'B', 'ct_b_km_by_zone.png')
t_a_safest, t_a_deadliest = plot_km_by_zone(t_a_full_data, 'T', 'A', 't_a_km_by_zone.png')
t_b_safest, t_b_deadliest = plot_km_by_zone(t_b_full_data, 'T', 'B', 't_b_km_by_zone.png')

# ========================================
# 6.6 TOP 10 PREDICTORS TABLE (FULL MODELS ONLY)
# ========================================

print("\nGenerating Top 10 Predictors Tables...")

def create_top_predictors_table(model, team, site, filename, top_n=10):
    """Create table of most significant predictors from full model."""
    
    summary = model.summary.copy()
    
    # Sort by absolute coefficient size (influence on hazard)
    summary['abs_coef'] = summary['coef'].abs()
    summary = summary.sort_values('abs_coef', ascending=False)
    
    # Get top N predictors
    top_predictors = summary.head(top_n)[['coef', 'exp(coef)', 'exp(coef) lower 95%', 
                                           'exp(coef) upper 95%', 'p']].copy()
    
    top_predictors.columns = ['Coefficient', 'Hazard Ratio', 'HR 95% CI Lower', 
                               'HR 95% CI Upper', 'p-value']
    
    # Add interpretation column
    def interpret_hr(hr):
        if hr > 1.5:
            return 'Strong risk increase'
        elif hr > 1.1:
            return 'Moderate risk increase'
        elif hr > 0.9:
            return 'Minimal effect'
        elif hr > 0.67:
            return 'Moderate protective'
        else:
            return 'Strong protective'
    
    top_predictors['Interpretation'] = top_predictors['Hazard Ratio'].apply(interpret_hr)
    
    # Save
    top_predictors.to_csv(results_dir / filename)
    print(f"  Saved: {filename} - {team} {site}-site top {top_n} predictors")
    
    return top_predictors

# Generate for all 4 full models
ct_a_top = create_top_predictors_table(ct_a_full_model, 'CT', 'A', 'ct_a_top10_predictors.csv')
ct_b_top = create_top_predictors_table(ct_b_full_model, 'CT', 'B', 'ct_b_top10_predictors.csv')
t_a_top = create_top_predictors_table(t_a_full_model, 'T', 'A', 't_a_top10_predictors.csv')
t_b_top = create_top_predictors_table(t_b_full_model, 'T', 'B', 't_b_top10_predictors.csv')

# ========================================
# 6.7 SAVE MODEL SUMMARIES AS HTML
# ========================================

# Zone models
ct_a_zone_model.summary.to_html(results_dir / 'ct_a_zone_model_summary.html')
ct_b_zone_model.summary.to_html(results_dir / 'ct_b_zone_model_summary.html')
t_a_zone_model.summary.to_html(results_dir / 't_a_zone_model_summary.html')
t_b_zone_model.summary.to_html(results_dir / 't_b_zone_model_summary.html')

# Equipment models
ct_a_equip_model.summary.to_html(results_dir / 'ct_a_equip_model_summary.html')
ct_b_equip_model.summary.to_html(results_dir / 'ct_b_equip_model_summary.html')
t_a_equip_model.summary.to_html(results_dir / 't_a_equip_model_summary.html')
t_b_equip_model.summary.to_html(results_dir / 't_b_equip_model_summary.html')

# Full models
ct_a_full_model.summary.to_html(results_dir / 'ct_a_full_model_summary.html')
ct_b_full_model.summary.to_html(results_dir / 'ct_b_full_model_summary.html')
t_a_full_model.summary.to_html(results_dir / 't_a_full_model_summary.html')
t_b_full_model.summary.to_html(results_dir / 't_b_full_model_summary.html')

# ============================================================================
# 7. SAVE SUMMARY STATISTICS
# ============================================================================

print("\n[7/7] Saving summary statistics...")

stats = {
    'ct_a_zone_model': {
        'concordance_index': float(ct_a_zone_model.concordance_index_),
        'cv_concordance_mean': float(ct_a_zone_cv),
        'penalty': float(ct_a_zone_lambda),
        'num_episodes': len(ct_a_data),
        'num_deaths': int(ct_a_data['death_event'].sum()),
        'num_covariates': len(ct_a_zone_model.params_)
    },
    'ct_b_zone_model': {
        'concordance_index': float(ct_b_zone_model.concordance_index_),
        'cv_concordance_mean': float(ct_b_zone_cv),
        'penalty': float(ct_b_zone_lambda),
        'num_episodes': len(ct_b_data),
        'num_deaths': int(ct_b_data['death_event'].sum()),
        'num_covariates': len(ct_b_zone_model.params_)
    },
    't_a_zone_model': {
        'concordance_index': float(t_a_zone_model.concordance_index_),
        'cv_concordance_mean': float(t_a_zone_cv),
        'penalty': float(t_a_zone_lambda),
        'num_episodes': len(t_a_data),
        'num_deaths': int(t_a_data['death_event'].sum()),
        'num_covariates': len(t_a_zone_model.params_)
    },
    't_b_zone_model': {
        'concordance_index': float(t_b_zone_model.concordance_index_),
        'cv_concordance_mean': float(t_b_zone_cv),
        'penalty': float(t_b_zone_lambda),
        'num_episodes': len(t_b_data),
        'num_deaths': int(t_b_data['death_event'].sum()),
        'num_covariates': len(t_b_zone_model.params_)
    },
    'ct_a_equip_model': {
        'concordance_index': float(ct_a_equip_model.concordance_index_),
        'cv_concordance_mean': float(ct_a_equip_cv),
        'penalty': float(ct_a_equip_lambda),
        'num_episodes': len(ct_a_equip_data),
        'num_deaths': int(ct_a_equip_data['death_event'].sum()),
        'num_covariates': len(ct_a_equip_model.params_)
    },
    'ct_b_equip_model': {
        'concordance_index': float(ct_b_equip_model.concordance_index_),
        'cv_concordance_mean': float(ct_b_equip_cv),
        'penalty': float(ct_b_equip_lambda),
        'num_episodes': len(ct_b_equip_data),
        'num_deaths': int(ct_b_equip_data['death_event'].sum()),
        'num_covariates': len(ct_b_equip_model.params_)
    },
    't_a_equip_model': {
        'concordance_index': float(t_a_equip_model.concordance_index_),
        'cv_concordance_mean': float(t_a_equip_cv),
        'penalty': float(t_a_equip_lambda),
        'num_episodes': len(t_a_equip_data),
        'num_deaths': int(t_a_equip_data['death_event'].sum()),
        'num_covariates': len(t_a_equip_model.params_)
    },
    't_b_equip_model': {
        'concordance_index': float(t_b_equip_model.concordance_index_),
        'cv_concordance_mean': float(t_b_equip_cv),
        'penalty': float(t_b_equip_lambda),
        'num_episodes': len(t_b_equip_data),
        'num_deaths': int(t_b_equip_data['death_event'].sum()),
        'num_covariates': len(t_b_equip_model.params_)
    },
    'ct_a_full_model': {
        'concordance_index': float(ct_a_full_model.concordance_index_),
        'cv_concordance_mean': float(ct_a_full_cv),
        'penalty': float(ct_a_full_lambda),
        'num_episodes': len(ct_a_full_data),
        'num_deaths': int(ct_a_full_data['death_event'].sum()),
        'num_covariates': len(ct_a_full_model.params_)
    },
    'ct_b_full_model': {
        'concordance_index': float(ct_b_full_model.concordance_index_),
        'cv_concordance_mean': float(ct_b_full_cv),
        'penalty': float(ct_b_full_lambda),
        'num_episodes': len(ct_b_full_data),
        'num_deaths': int(ct_b_full_data['death_event'].sum()),
        'num_covariates': len(ct_b_full_model.params_)
    },
    't_a_full_model': {
        'concordance_index': float(t_a_full_model.concordance_index_),
        'cv_concordance_mean': float(t_a_full_cv),
        'penalty': float(t_a_full_lambda),
        'num_episodes': len(t_a_full_data),
        'num_deaths': int(t_a_full_data['death_event'].sum()),
        'num_covariates': len(t_a_full_model.params_)
    },
    't_b_full_model': {
        'concordance_index': float(t_b_full_model.concordance_index_),
        'cv_concordance_mean': float(t_b_full_cv),
        'penalty': float(t_b_full_lambda),
        'num_episodes': len(t_b_full_data),
        'num_deaths': int(t_b_full_data['death_event'].sum()),
        'num_covariates': len(t_b_full_model.params_)
    }
}

with open(results_dir / 'model_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\nSaved statistics to: {results_dir / 'model_statistics.json'}")

print("\n" + "="*80)
print("COX SURVIVAL ANALYSIS COMPLETE (RIDGE REGULARIZATION)")
print("="*80)
print(f"\nResults saved to: {results_dir.absolute()}")
print("\nFINAL C-INDICES:")
print(f"  CT A Full: {ct_a_full_model.concordance_index_:.3f} (CV: {ct_a_full_cv:.3f})")
print(f"  CT B Full: {ct_b_full_model.concordance_index_:.3f} (CV: {ct_b_full_cv:.3f})")
print(f"  T A Full: {t_a_full_model.concordance_index_:.3f} (CV: {t_a_full_cv:.3f})")
print(f"  T B Full: {t_b_full_model.concordance_index_:.3f} (CV: {t_b_full_cv:.3f})")
print("="*80)
