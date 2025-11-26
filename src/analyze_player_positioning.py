"""
Analyze player positioning episodes during post-plant scenarios (both CT and T).
Tracks individual player movements through zones with outcomes (damage, kills, deaths).

This provides empirical data for hazard rate modeling instead of simple probability weights.

IMPORTANT: Detects retakes using first CT vs T engagement in/near bomb sites.
Episode-based tracking: each player's time in each zone is a separate episode with outcomes.

Output feeds into:
- Threat cone calculations (zone connectivity + empirical positioning)
- Hazard rate modeling (survival probability given position/time/game state)
- Sound/rotation integration (audio tracker provides information events)
"""

from pathlib import Path
from demoparser2 import DemoParser
from src.zone_classifier import MirageZoneClassifier
from src.zone_connectivity import ZoneConnectivity
from src.plant_spot_classifier import PlantSpotClassifier
from src.detect_retake import detect_retake
from collections import defaultdict
import json
import pandas as pd

print("="*80)
print("ANALYZING POST-PLANT POSITIONING EPISODES FROM ALL DEMOS")
print("="*80)

classifier = MirageZoneClassifier()
connectivity = ZoneConnectivity()
plant_classifier = PlantSpotClassifier()

# Utility weapons for damage categorization
UTILITY_WEAPONS = {'hegrenade', 'molotov', 'incgrenade', 'inferno'}

def categorize_damage(damage_df):
    """Separate gun damage from utility damage for economy/hazard modeling."""
    if len(damage_df) == 0:
        return 0, 0
    
    gun_dmg = damage_df[~damage_df['weapon'].isin(UTILITY_WEAPONS)]['dmg_health'].sum()
    util_dmg = damage_df[damage_df['weapon'].isin(UTILITY_WEAPONS)]['dmg_health'].sum()
    return gun_dmg, util_dmg

def get_player_team(ticks, player_name, tick):
    """Get player's team (CT or T) at a given tick.
    
    Args:
        ticks: DataFrame with player positions including team_num
        player_name: Player name to look up
        tick: Tick to check team at
    
    Returns:
        'CT' if team_num == 3, 'T' if team_num == 2, None if not found
    """
    player_tick = ticks[
        (ticks['name'] == player_name) & 
        (ticks['tick'] == tick)
    ]
    if len(player_tick) > 0:
        team_num = player_tick.iloc[0]['team_num']
        return 'CT' if team_num == 3 else 'T' if team_num == 2 else None
    return None

def count_utility_thrown(grenades_df, smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate, player_name, tick_start, tick_end, classifier, ticks, damage_df, plant_tick=None):
    """Count utility grenades thrown by player in tick window, with zone classification, duration, and team.
    
    Args:
        player_name: Player name to filter by, or None to get all players' utility
        plant_tick: Plant tick for filtering post-plant utility (optional, for smoke filtering)
    
    Returns dict with utility details including expiration times and team:
    {
        'smokes': [{'zone': 'Jungle', 'detonate_tick': 12345, 'expire_tick': 12700, 'team': 'CT', 'thrower_zone': 'CT_Stairs'}, ...],
        'flashes': [{'zone': 'CT_Stairs', 'detonate_tick': 12346, 'effect_duration': 192, 'team': 'T', 'thrower_zone': 'Jungle'}, ...],
        'he': [{'zone': 'A_Default', 'thrower_zone': 'CT_Stairs', 'tick': 12350, 'team': 'CT'}, ...],
        'mollies': [{'zone': 'Stairs', 'thrower_zone': 'Connector', 'start_tick': 12348, 'expire_tick': 12500, 'team': 'T'}, ...]
    }
    """
    result = {
        'smokes': [],
        'flashes': [],
        'he': [],
        'mollies': []
    }
    
    # Handle case where grenades_df might be an empty list (demo parsing issue)
    import pandas as pd
    if not isinstance(grenades_df, pd.DataFrame):
        grenades_df = pd.DataFrame()  # Convert to empty DataFrame
    
    # Find smoke grenades - use DETONATIONS as source of truth since throws might be outside window
    # For smokes active during [tick_start, tick_end]: detonate < tick_end AND expire > tick_start
    smoke_dets_in_window = smoke_detonate[
        (smoke_detonate['tick'] >= tick_start - 20*64) &  # Look back 20s for detonations
        (smoke_detonate['tick'] < tick_end)
    ]
    
    for _, det in smoke_dets_in_window.iterrows():
        detonate_tick = det['tick']
        throw_player = det['user_name']
        entity_id = det['entityid']
        
        # Skip if tracking specific player and this isn't them
        if player_name is not None and throw_player != player_name:
            continue
        
        # Find thrower position - look for throw event within 5s before detonation
        # (only if grenades_df has data)
        thrower_zone = None
        if len(grenades_df) > 0:
            throw_event = grenades_df[
                (grenades_df['user_name'] == throw_player) &
                (grenades_df['weapon'] == 'smokegrenade') &
                (grenades_df['tick'] >= detonate_tick - 5*64) &
                (grenades_df['tick'] <= detonate_tick)
            ]
            
            if len(throw_event) > 0:
                throw_tick = throw_event.iloc[0]['tick']
                thrower_pos = ticks[
                    (ticks['name'] == throw_player) &
                    (ticks['tick'] == throw_tick)
                ]
                if len(thrower_pos) > 0:
                    pos = thrower_pos.iloc[0]
                    thrower_zone = classifier.point_in_zone_A(pos['X'], pos['Y'], pos['Z'])
        
        # Calculate smoke expiration using standard 18-second duration
        # Note: expire events from demo appear unreliable (show smokes lasting longer than possible)
        # Smokes have a fixed 18-second duration in CS2
        expire_tick = detonate_tick + 18 * 64
        
        # Only include smokes that were active during the tracking window
        # Active means: detonated before window ended AND expired after window started
        if detonate_tick < tick_end and expire_tick > tick_start:
            zone = classifier.point_in_zone_A(det['x'], det['y'], det['z'])
            team = get_player_team(ticks, throw_player, detonate_tick)
            if zone and team:
                result['smokes'].append({
                    'zone': zone,
                    'detonate_tick': detonate_tick,
                    'expire_tick': expire_tick,
                    'team': team,
                    'thrower_zone': thrower_zone if thrower_zone else 'unknown'
                })
    
    # Molotov/incgrenade throws (skip if no grenade data available)
    if len(grenades_df) > 0:
        if player_name is None:
            molly_throws = grenades_df[
                (grenades_df['weapon'].isin(['molotov', 'incgrenade'])) &
                (grenades_df['tick'] >= tick_start) &
                (grenades_df['tick'] < tick_end)
            ]
        else:
            molly_throws = grenades_df[
                (grenades_df['user_name'] == player_name) &
                (grenades_df['weapon'].isin(['molotov', 'incgrenade'])) &
                (grenades_df['tick'] >= tick_start) &
                (grenades_df['tick'] < tick_end)
            ]
    else:
        molly_throws = pd.DataFrame()
    
    # If we have grenade_thrown data, use throw->inferno matching
    for _, throw in molly_throws.iterrows():
        throw_tick = throw['tick']
        throw_player = throw['user_name']
        
        # Find thrower position at throw time
        thrower_pos = ticks[
            (ticks['name'] == throw_player) &
            (ticks['tick'] == throw_tick)
        ]
        thrower_zone = None
        if len(thrower_pos) > 0:
            pos = thrower_pos.iloc[0]
            thrower_zone = classifier.point_in_zone_A(pos['X'], pos['Y'], pos['Z'])
        
        # Find corresponding inferno start (within 5 seconds)
        matching_inferno = inferno_start[
            (inferno_start['user_name'] == throw_player) &
            (inferno_start['tick'] >= throw_tick) &
            (inferno_start['tick'] <= throw_tick + 5*64)
        ]
        if len(matching_inferno) > 0:
            inf = matching_inferno.iloc[0]
            start_tick = inf['tick']
            entity_id = inf['entityid']
            
            # Find when this specific molly expires (match by entityid)
            expire_event = inferno_expire[
                (inferno_expire['entityid'] == entity_id) &
                (inferno_expire['tick'] > start_tick)
            ].sort_values('tick')
            
            if len(expire_event) > 0:
                expire_tick = expire_event.iloc[0]['tick']
            else:
                expire_tick = start_tick + 7 * 64
            
            zone = classifier.point_in_zone_A(inf['x'], inf['y'], inf['z'])
            team = get_player_team(ticks, throw_player, start_tick)
            if zone and team:
                result['mollies'].append({
                    'zone': zone,
                    'start_tick': start_tick,
                    'expire_tick': expire_tick,
                    'team': team,
                    'thrower_zone': thrower_zone if thrower_zone else 'unknown'
                })
    
    # Fallback: If no grenade_thrown data, use inferno_start events directly
    if len(grenades_df) == 0:
        inferno_starts_in_window = inferno_start[
            (inferno_start['tick'] >= tick_start) &
            (inferno_start['tick'] < tick_end)
        ]
        
        for _, inf in inferno_starts_in_window.iterrows():
            start_tick = inf['tick']
            throw_player = inf['user_name']
            entity_id = inf['entityid']
            
            # Skip if tracking specific player
            if player_name is not None and throw_player != player_name:
                continue
            
            # Find when this specific molly expires
            expire_event = inferno_expire[
                (inferno_expire['entityid'] == entity_id) &
                (inferno_expire['tick'] > start_tick)
            ].sort_values('tick')
            
            if len(expire_event) > 0:
                expire_tick = expire_event.iloc[0]['tick']
            else:
                expire_tick = start_tick + 7 * 64
            
            zone = classifier.point_in_zone_A(inf['x'], inf['y'], inf['z'])
            team = get_player_team(ticks, throw_player, start_tick)
            if zone and team:
                result['mollies'].append({
                    'zone': zone,
                    'start_tick': start_tick,
                    'expire_tick': expire_tick,
                    'team': team,
                    'thrower_zone': 'unknown'  # No throw data available
                })
    
    # Flashbang throws (skip if no grenade data available)
    if len(grenades_df) > 0:
        if player_name is None:
            flash_throws = grenades_df[
                (grenades_df['weapon'] == 'flashbang') &
                (grenades_df['tick'] >= tick_start) &
                (grenades_df['tick'] < tick_end)
            ]
        else:
            flash_throws = grenades_df[
                (grenades_df['user_name'] == player_name) &
                (grenades_df['weapon'] == 'flashbang') &
                (grenades_df['tick'] >= tick_start) &
                (grenades_df['tick'] < tick_end)
            ]
    else:
        flash_throws = pd.DataFrame()
    
    # If we have grenade_thrown data, use throw->detonate matching
    for _, throw in flash_throws.iterrows():
        throw_tick = throw['tick']
        throw_player = throw['user_name']
        
        # Find thrower position at throw time
        thrower_pos = ticks[
            (ticks['name'] == throw_player) &
            (ticks['tick'] == throw_tick)
        ]
        thrower_zone = None
        if len(thrower_pos) > 0:
            pos = thrower_pos.iloc[0]
            thrower_zone = classifier.point_in_zone_A(pos['X'], pos['Y'], pos['Z'])
        
        # Find corresponding detonation (within 5 seconds)
        matching_det = flash_detonate[
            (flash_detonate['user_name'] == throw_player) &
            (flash_detonate['tick'] >= throw_tick) &
            (flash_detonate['tick'] <= throw_tick + 5*64)
        ]
        if len(matching_det) > 0:
            det = matching_det.iloc[0]
            detonate_tick = det['tick']
            FLASH_EFFECT_DURATION = 3 * 64
            flash_height = det['z']
            
            # Try ground-level zone first, then try actual height
            zone = classifier.point_in_zone_A(det['x'], det['y'], -100)
            if not zone:
                zone = classifier.point_in_zone_A(det['x'], det['y'], det['z'])
            
            # Classify flash type by height
            if flash_height > 100:
                flash_type = 'high_pop'
            elif flash_height > 0:
                flash_type = 'mid'
            else:
                flash_type = 'ground'
            
            team = get_player_team(ticks, throw_player, detonate_tick)
            if team:  # Only require team, zone can be unknown
                result['flashes'].append({
                    'zone': zone if zone else 'unknown',
                    'detonate_tick': detonate_tick,
                    'effect_duration': FLASH_EFFECT_DURATION,
                    'height': flash_height,
                    'flash_type': flash_type,
                    'team': team,
                    'thrower_zone': thrower_zone if thrower_zone else 'unknown'
                })
    
    # Fallback: If no grenade_thrown data, use flash_detonate events directly (like smokes)
    if len(grenades_df) == 0:
        flash_dets_in_window = flash_detonate[
            (flash_detonate['tick'] >= tick_start) &
            (flash_detonate['tick'] < tick_end)
        ]
        
        for _, det in flash_dets_in_window.iterrows():
            detonate_tick = det['tick']
            throw_player = det['user_name']
            
            # Skip if tracking specific player
            if player_name is not None and throw_player != player_name:
                continue
            
            FLASH_EFFECT_DURATION = 3 * 64
            flash_height = det['z']
            
            # Try ground-level zone first, then try actual height
            zone = classifier.point_in_zone_A(det['x'], det['y'], -100)
            if not zone:
                zone = classifier.point_in_zone_A(det['x'], det['y'], det['z'])
            
            # Classify flash type by height
            if flash_height > 100:
                flash_type = 'high_pop'
            elif flash_height > 0:
                flash_type = 'mid'
            else:
                flash_type = 'ground'
            
            team = get_player_team(ticks, throw_player, detonate_tick)
            if team:
                result['flashes'].append({
                    'zone': zone if zone else 'unknown',
                    'detonate_tick': detonate_tick,
                    'effect_duration': FLASH_EFFECT_DURATION,
                    'height': flash_height,
                    'flash_type': flash_type,
                    'team': team,
                    'thrower_zone': 'unknown'  # No throw data available
                })
    
    # HE grenades - infer landing zone from victim positions
    # HE grenade throws (skip if no grenade data available)
    if len(grenades_df) > 0:
        if player_name is None:
            he_throws = grenades_df[
                (grenades_df['weapon'] == 'hegrenade') &
                (grenades_df['tick'] >= tick_start) &
                (grenades_df['tick'] < tick_end)
            ]
        else:
            he_throws = grenades_df[
                (grenades_df['user_name'] == player_name) &
                (grenades_df['weapon'] == 'hegrenade') &
                (grenades_df['tick'] >= tick_start) &
                (grenades_df['tick'] < tick_end)
            ]
    else:
        he_throws = pd.DataFrame()
    
    for _, he_throw in he_throws.iterrows():
        throw_tick = he_throw['tick']
        throw_player = he_throw['user_name']
        team = get_player_team(ticks, throw_player, throw_tick)
        
        # Find thrower position at throw time
        thrower_pos = ticks[
            (ticks['name'] == throw_player) &
            (ticks['tick'] == throw_tick)
        ]
        thrower_zone = None
        if len(thrower_pos) > 0:
            pos = thrower_pos.iloc[0]
            thrower_zone = classifier.point_in_zone_A(pos['X'], pos['Y'], pos['Z'])
        
        # Find HE damage events from this player within 5 seconds after throw
        he_damage = damage_df[
            (damage_df['attacker_name'] == throw_player) &
            (damage_df['weapon'] == 'hegrenade') &
            (damage_df['tick'] >= throw_tick) &
            (damage_df['tick'] <= throw_tick + 5*64)
        ]
        
        if len(he_damage) > 0:
            # Use first victim's position as approximate landing zone
            first_damage = he_damage.iloc[0]
            damage_tick = first_damage['tick']
            
            # Find victim position at damage tick
            victim_pos = ticks[
                (ticks['name'] == first_damage['user_name']) &
                (ticks['tick'] == damage_tick)
            ]
            
            if len(victim_pos) > 0:
                pos = victim_pos.iloc[0]
                landing_zone = classifier.point_in_zone_A(pos['X'], pos['Y'], pos['Z'])
                if landing_zone and team:
                    result['he'].append({
                        'zone': landing_zone,
                        'thrower_zone': thrower_zone if thrower_zone else 'unknown',
                        'tick': damage_tick,
                        'team': team
                    })
            else:
                if team:
                    result['he'].append({
                        'zone': 'unknown',
                        'thrower_zone': thrower_zone if thrower_zone else 'unknown',
                        'tick': throw_tick,
                        'team': team
                    })
        else:
            # HE didn't hit anyone
            if team:
                result['he'].append({
                    'zone': 'unknown',
                    'thrower_zone': thrower_zone if thrower_zone else 'unknown',
                    'tick': throw_tick,
                    'team': team
                })
    
    return result

def get_active_utility_at_tick(smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate, tick, classifier, ticks, window_before=5*64):
    """Get utility that's active (present on map) at given tick, with zone classification and team labels.
    
    Includes:
    - Active smokes (between detonate and expire)
    - Active molotovs (between start and expire)
    - Recent flashbangs (within window_before)
    
    Returns dict:
    {
        'smokes': [{'user_name': 'player1', 'detonate_tick': t1, 'expire_tick': t2, 'zone': 'Jungle', 'team': 'CT'}, ...],
        'mollies': [{'user_name': 'player2', 'start_tick': t3, 'expire_tick': t4, 'zone': 'Stairs', 'team': 'T'}, ...],
        'recent_flashes': [{'user_name': 'player3', 'detonate_tick': t5, 'effect_until_tick': t6, 'zone': 'CT_Stairs', 'team': 'CT'}, ...]
    }
    """
    result = {
        'smokes': [],
        'mollies': [],
        'recent_flashes': []
    }
    
    # Active smokes: detonated before tick, not yet expired
    for _, det in smoke_detonate[smoke_detonate['tick'] <= tick].iterrows():
        # Find expire event for this smoke (match by entityid only to avoid conflicts)
        expire_event = smoke_expire[
            (smoke_expire['entityid'] == det['entityid']) &
            (smoke_expire['tick'] > det['tick'])
        ].sort_values('tick')
        
        # Use expire event if available, otherwise assume standard 18s duration
        if len(expire_event) > 0:
            expire_tick = expire_event.iloc[0]['tick']
        else:
            # Standard smoke duration is ~18 seconds (18 * 64 ticks)
            expire_tick = det['tick'] + 18 * 64
        
        if expire_tick > tick:
            # Smoke still active
            zone = classifier.point_in_zone_A(det['x'], det['y'], det['z'])
            team = get_player_team(ticks, det['user_name'], det['tick'])
            if zone and team:
                result['smokes'].append({
                    'user_name': det['user_name'],
                    'detonate_tick': det['tick'],
                    'expire_tick': expire_tick,
                    'zone': zone,
                    'team': team
                })
    
    # Active molotovs: started before tick, not yet expired
    for _, inf in inferno_start[inferno_start['tick'] <= tick].iterrows():
        # Find expire event (match by entityid)
        expire_event = inferno_expire[
            (inferno_expire['entityid'] == inf['entityid']) &
            (inferno_expire['tick'] > inf['tick'])
        ].sort_values('tick')
        
        # Use expire event if available, otherwise assume standard 7s duration
        if len(expire_event) > 0:
            expire_tick = expire_event.iloc[0]['tick']
        else:
            # Standard molly duration is ~7 seconds (7 * 64 ticks)
            expire_tick = inf['tick'] + 7 * 64
        
        if expire_tick > tick:
            # Molly still active
            zone = classifier.point_in_zone_A(inf['x'], inf['y'], inf['z'])
            team = get_player_team(ticks, inf['user_name'], inf['tick'])
            if zone and team:
                result['mollies'].append({
                    'user_name': inf['user_name'],
                    'start_tick': inf['tick'],
                    'expire_tick': expire_tick,
                    'zone': zone,
                    'team': team
                })
    
    # Recent flashbangs: detonated within window_before
    recent_flashes = flash_detonate[
        (flash_detonate['tick'] <= tick) &
        (flash_detonate['tick'] >= tick - window_before)
    ]
    
    for _, det in recent_flashes.iterrows():
        # Flashes have ~3s effect duration
        FLASH_EFFECT_DURATION = 3 * 64
        effect_until_tick = det['tick'] + FLASH_EFFECT_DURATION
        if effect_until_tick > tick:
            # Flash effect still active
            zone = classifier.point_in_zone_A(det['x'], det['y'], -100)  # Approximate ground Z
            team = get_player_team(ticks, det['user_name'], det['tick'])
            if zone and team:
                result['recent_flashes'].append({
                    'user_name': det['user_name'],
                    'detonate_tick': det['tick'],
                    'effect_until_tick': effect_until_tick,
                    'zone': zone,
                    'team': team
                })
    
    return result

# Track player episodes: list of {player, zone, duration, damage_taken, damage_dealt, died, got_kill}
ct_episodes = []  # CT episodes during retakes
t_episodes = []   # T episodes during post-plant
a_site_retakes = 0
b_site_retakes = 0

# Get all demo files
demo_files = list(Path("research_demos/extracted").rglob("*.dem"))
total_retakes = 0
demos_processed = 0

print(f"\nFound {len(demo_files)} demo files to process")
print("This will take ~45-50 minutes...\n")

for demo_path in demo_files:
    try:
        if demos_processed % 50 == 0:
            print(f"Progress: {demos_processed}/{len(demo_files)} demos, {total_retakes} retakes found...")
        
        parser = DemoParser(str(demo_path))
        
        # Get all bomb plants
        bomb_plants = parser.parse_event('bomb_planted')
        bomb_defused = parser.parse_event('bomb_defused')
        
        # Get player positions
        ticks = parser.parse_ticks(['X', 'Y', 'Z', 'name', 'tick', 'team_num', 'is_alive', 'health', 'has_defuser'])
        
        # Get damage and death events
        damage = parser.parse_event('player_hurt')
        deaths = parser.parse_event('player_death')
        
        # Get round starts and ends for round numbers and winners
        round_starts = parser.parse_event('round_start')
        round_ends = parser.parse_event('round_end')
        
        # Get grenade events for utility tracking
        grenades_thrown = parser.parse_event('grenade_thrown')
        smoke_detonate = parser.parse_event('smokegrenade_detonate')
        smoke_expire = parser.parse_event('smokegrenade_expired')
        inferno_start = parser.parse_event('inferno_startburn')
        inferno_expire = parser.parse_event('inferno_expire')
        flash_detonate = parser.parse_event('flashbang_detonate')
        
        for idx, plant in bomb_plants.iterrows():
            plant_tick = plant['tick']
            
            # Find round number and winner
            round_num = len(round_starts[round_starts['tick'] <= plant_tick])
            
            # Find the round start for this plant
            this_round_start = round_starts[round_starts['tick'] <= plant_tick].sort_values('tick')
            if len(this_round_start) == 0:
                continue
            round_start_tick = this_round_start.iloc[-1]['tick']
            
            # Get round winner - find the NEXT round_end after plant (tick-based, not round number)
            round_end = round_ends[round_ends['tick'] > plant_tick].sort_values('tick')
            winner_team = round_end.iloc[0]['winner'] if len(round_end) > 0 else None
            
            # Get actual round end tick for filtering deaths to this round only
            round_end_tick_actual = round_end.iloc[0]['tick'] if len(round_end) > 0 else plant_tick + 40*64
            
            # Verify it's A-site by checking planter's position (bomb is planted where they stand)
            planter_name = plant['user_name']
            planter_pos = ticks[
                (ticks['tick'] == plant_tick) &
                (ticks['name'] == planter_name)
            ]
            
            if len(planter_pos) == 0:
                continue
                
            pos = planter_pos.iloc[0]
            bomb_zone = classifier.point_in_zone_A(pos['X'], pos['Y'], pos['Z'])
            
            if not bomb_zone:
                continue  # Could not classify plant location
            
            # Identify site based on zone name
            is_b_site = bomb_zone.startswith('B_') or bomb_zone.startswith('Cat_')
            
            # Get specific plant spot classification
            site_letter = 'b' if is_b_site else 'a'
            plant_spot = plant_classifier.classify_plant(pos['X'], pos['Y'], pos['Z'], site_letter)
            
            # Track defuse kits at plant time
            ct_players_at_plant = ticks[
                (ticks['tick'] == plant_tick) & 
                (ticks['team_num'] == 3) &
                (ticks['is_alive'] == True)
            ]
            ct_kits_at_plant = {}
            for _, ct in ct_players_at_plant.iterrows():
                has_kit = ct.get('has_defuser', False)
                if has_kit:
                    ct_kits_at_plant[ct['name']] = True
            kits_available_at_plant = len(ct_kits_at_plant)
            
            # Use detect_retake function
            retake_result = detect_retake(plant_tick, bomb_zone, ticks, damage, classifier, round_end_tick_actual)
            
            if not retake_result['retake_detected']:
                continue  # No retake detected (save round)
            
            retake_tick = retake_result['retake_tick']
            
            # Capture initial conditions at PLANT_TICK (not retake_tick)
            # This is the "starting state" when bomb was planted
            plant_tick_state = ticks[ticks['tick'] == plant_tick]
            
            # Count alive players at plant_tick
            ct_alive_at_plant = plant_tick_state[
                (plant_tick_state['team_num'] == 3) &
                (plant_tick_state['is_alive'] == True)
            ]
            t_alive_at_plant = plant_tick_state[
                (plant_tick_state['team_num'] == 2) &
                (plant_tick_state['is_alive'] == True)
            ]
            
            ct_count_at_plant = len(ct_alive_at_plant['name'].unique())
            t_count_at_plant = len(t_alive_at_plant['name'].unique())
            numerical_advantage = ct_count_at_plant - t_count_at_plant  # Positive = CT advantage, Negative = T advantage
            
            # Capture health at plant_tick - create dict mapping player name to HP
            ct_health_dict = {}
            for _, player_row in ct_alive_at_plant.iterrows():
                ct_health_dict[player_row['name']] = player_row['health']
            
            t_health_dict = {}
            for _, player_row in t_alive_at_plant.iterrows():
                t_health_dict[player_row['name']] = player_row['health']
            
            # Get active utility at retake start (smokes/mollies/flashes already active from pre-plant)
            active_utility = get_active_utility_at_tick(
                smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate,
                retake_tick, classifier, ticks, window_before=20*64  # Look 20 seconds back (smokes last ~18s)
            )
            
            total_retakes += 1
            if is_b_site:
                b_site_retakes += 1
            else:
                a_site_retakes += 1
            
            # Calculate actual round end tick (when round actually ended)
            # Filter deaths to this round only (between plant and round_end_tick_actual)
            round_deaths = deaths[
                (deaths['tick'] >= plant_tick) &
                (deaths['tick'] <= round_end_tick_actual)
            ].copy()
            
            # For CT retakes, find when last T died (or bomb was defused)
            last_t_death_tick = retake_tick  # Default to retake start if no T deaths
            for _, death in round_deaths.iterrows():
                # Look up victim team from ticks (deaths event doesn't have team field)
                victim_team_data = ticks[
                    (ticks['name'] == death['user_name']) &
                    (ticks['tick'] == death['tick'] - 1)  # Check tick before death
                ]
                if len(victim_team_data) > 0:
                    victim_team = victim_team_data.iloc[0]['team_num']
                    if victim_team == 2:  # T died
                        if death['tick'] > last_t_death_tick:
                            last_t_death_tick = death['tick']
            
            # Check if bomb was defused in this round
            bomb_defused_tick = None
            round_defuses = bomb_defused[
                (bomb_defused['tick'] >= retake_tick) &
                (bomb_defused['tick'] <= round_end_tick_actual)
            ]
            if len(round_defuses) > 0:
                bomb_defused_tick = round_defuses.iloc[0]['tick']
            
            # Round ends when: last T dies OR bomb defused OR 15s after retake (whichever comes first)
            round_end_tick = retake_tick + 15*64  # Default cap at 15s
            if bomb_defused_tick is not None:
                round_end_tick = min(round_end_tick, bomb_defused_tick)
            if last_t_death_tick > retake_tick:  # Only use if we found actual T deaths after retake
                round_end_tick = min(round_end_tick, last_t_death_tick)
            
            # Track CT episodes during retake (retake start to actual round end)
            analysis_start = retake_tick
            analysis_end = round_end_tick
            
            # Only track CTs who were ALIVE at retake start
            ct_players = ticks[
                (ticks['tick'] == analysis_start) &
                (ticks['team_num'] == 3) &
                (ticks['is_alive'] == True)
            ]['name'].unique()
            
            # Count players alive at retake (not plant)
            ct_count_at_retake = len(ct_players)
            t_players_at_retake = ticks[
                (ticks['tick'] == analysis_start) &
                (ticks['team_num'] == 2) &
                (ticks['is_alive'] == True)
            ]['name'].unique()
            t_count_at_retake = len(t_players_at_retake)
            
            for player in ct_players:
                # Check if player died during this retake window
                player_death = deaths[
                    (deaths['tick'] >= analysis_start) &
                    (deaths['tick'] <= analysis_end) &
                    (deaths['user_name'] == player)
                ]
                
                # If player died, only track up to death tick; otherwise full window
                if len(player_death) > 0:
                    player_end_tick = player_death.iloc[0]['tick']
                else:
                    player_end_tick = analysis_end
                
                # Track ALL utility thrown during full retake window
                # Use a buffer before retake start to catch setup utility
                ct_util_start = max(plant_tick, analysis_start - 5*64)  # 5s buffer before retake
                all_utility_thrown = count_utility_thrown(
                    grenades_thrown, smoke_detonate, smoke_expire,
                    inferno_start, inferno_expire, flash_detonate,
                    player, ct_util_start, player_end_tick,
                    classifier, ticks, damage
                )
                
                player_ticks = ticks[
                    (ticks['tick'] >= analysis_start) &
                    (ticks['tick'] <= player_end_tick) &
                    (ticks['name'] == player) &
                    (ticks['is_alive'] == True)
                ].sort_values('tick')
                
                if len(player_ticks) == 0:
                    continue
                
                # Check if this CT player picks up a kit during retake window
                kit_pickup_tick = None
                player_had_kit_at_plant = player in ct_kits_at_plant
                for idx in range(1, len(player_ticks)):
                    prev_row = player_ticks.iloc[idx-1]
                    curr_row = player_ticks.iloc[idx]
                    prev_kit = prev_row.get('has_defuser', False) if 'has_defuser' in prev_row.index else False
                    curr_kit = curr_row.get('has_defuser', False) if 'has_defuser' in curr_row.index else False
                    if not prev_kit and curr_kit:
                        kit_pickup_tick = curr_row['tick']
                        break
                
                # Track zone transitions
                current_zone = None
                zone_entry_tick = None
                
                for _, pos in player_ticks.iterrows():
                    zone = classifier.point_in_zone_A(pos['X'], pos['Y'], pos['Z'])
                    
                    if zone != current_zone:
                        # Save previous episode if exists
                        if current_zone and zone_entry_tick:
                            duration_ticks = pos['tick'] - zone_entry_tick
                            
                            # Get damage taken/dealt in this zone during this episode
                            episode_damage = damage[
                                (damage['tick'] >= zone_entry_tick) &
                                (damage['tick'] < pos['tick']) &
                                (damage['user_name'] == player)
                            ]
                            damage_taken_gun, damage_taken_util = categorize_damage(episode_damage)
                            
                            episode_damage_dealt = damage[
                                (damage['tick'] >= zone_entry_tick) &
                                (damage['tick'] < pos['tick']) &
                                (damage['attacker_name'] == player)
                            ]
                            damage_dealt_gun, damage_dealt_util = categorize_damage(episode_damage_dealt)
                            
                            # Check if player died in this zone
                            episode_deaths = deaths[
                                (deaths['tick'] >= zone_entry_tick) &
                                (deaths['tick'] < pos['tick']) &
                                (deaths['user_name'] == player)
                            ]
                            died = len(episode_deaths) > 0
                            
                            # Check if player got kill from this zone
                            episode_kills = deaths[
                                (deaths['tick'] >= zone_entry_tick) &
                                (deaths['tick'] < pos['tick']) &
                                (deaths['attacker_name'] == player)
                            ]  
                            got_kill = len(episode_kills) > 0
                            
                            # Calculate time remaining on bomb at episode start
                            # C4 timer is 40 seconds (40 * 64 ticks)
                            time_since_plant = (zone_entry_tick - plant_tick) / 64.0
                            time_remaining_on_bomb = 40.0 - time_since_plant
                            
                            # Note: utility_thrown is tracked for full window, saved once at end
                            
                            ct_episodes.append({
                                'zone': current_zone,
                                'duration_ticks': duration_ticks,
                                'damage_taken_gun': damage_taken_gun,
                                'damage_taken_util': damage_taken_util,
                                'damage_dealt_gun': damage_dealt_gun,
                                'damage_dealt_util': damage_dealt_util,
                                'died': died,
                                'got_kill': got_kill,
                                'site': 'B' if is_b_site else 'A',
                                'plant_spot': plant_spot,
                                'round_winner': winner_team,
                                'player_name': player,
                                'utility_thrown': {},  # Per-episode tracking removed, using full-window below
                                'time_remaining_on_bomb': time_remaining_on_bomb,  # Time-varying covariate
                                'ct_count_at_plant': ct_count_at_plant,
                                't_count_at_plant': t_count_at_plant,
                                'numerical_advantage': numerical_advantage,
                                'ct_count_at_retake': ct_count_at_retake,
                                't_count_at_retake': t_count_at_retake,
                                'player_hp_at_plant': ct_health_dict.get(player, 0),  # This player's HP
                                'ct_health_at_plant': ct_health_dict,  # All CT health
                                't_health_at_plant': t_health_dict,  # All T health
                                'had_kit_at_plant': player_had_kit_at_plant,
                                'kit_pickup_tick': kit_pickup_tick
                            })
                        
                        # Start new episode
                        current_zone = zone
                        zone_entry_tick = pos['tick']
                
                # Save final episode (use player_end_tick instead of analysis_end)
                if current_zone and zone_entry_tick:
                    duration_ticks = player_end_tick - zone_entry_tick
                    
                    episode_damage = damage[
                        (damage['tick'] >= zone_entry_tick) &
                        (damage['tick'] <= player_end_tick) &
                        (damage['user_name'] == player)
                    ]
                    damage_taken_gun, damage_taken_util = categorize_damage(episode_damage)
                    
                    episode_damage_dealt = damage[
                        (damage['tick'] >= zone_entry_tick) &
                        (damage['tick'] <= player_end_tick) &
                        (damage['attacker_name'] == player)
                    ]
                    damage_dealt_gun, damage_dealt_util = categorize_damage(episode_damage_dealt)
                    
                    episode_deaths = deaths[
                        (deaths['tick'] >= zone_entry_tick) &
                        (deaths['tick'] <= player_end_tick) &
                        (deaths['user_name'] == player)
                    ]
                    died = len(episode_deaths) > 0
                    
                    episode_kills = deaths[
                        (deaths['tick'] >= zone_entry_tick) &
                        (deaths['tick'] <= player_end_tick) &
                        (deaths['attacker_name'] == player)
                    ]
                    got_kill = len(episode_kills) > 0
                    
                    # Calculate time remaining on bomb at episode start
                    time_since_plant = (zone_entry_tick - plant_tick) / 64.0
                    time_remaining_on_bomb = 40.0 - time_since_plant
                    
                    # Save final episode with full-window utility
                    ct_episodes.append({
                        'zone': current_zone,
                        'duration_ticks': duration_ticks,
                        'damage_taken_gun': damage_taken_gun,
                        'damage_taken_util': damage_taken_util,
                        'damage_dealt_gun': damage_dealt_gun,
                        'damage_dealt_util': damage_dealt_util,
                        'died': died,
                        'got_kill': got_kill,
                        'site': 'B' if is_b_site else 'A',
                        'plant_spot': plant_spot,
                        'round_winner': winner_team,
                        'player_name': player,
                        'utility_thrown': all_utility_thrown,  # Full window utility saved here
                        'time_remaining_on_bomb': time_remaining_on_bomb,  # Time-varying covariate
                        'ct_count_at_plant': ct_count_at_plant,
                        't_count_at_plant': t_count_at_plant,
                        'numerical_advantage': numerical_advantage,
                        'ct_count_at_retake': ct_count_at_retake,
                        't_count_at_retake': t_count_at_retake,
                        'player_hp_at_plant': ct_health_dict.get(player, 0),  # This player's HP
                        'ct_health_at_plant': ct_health_dict,  # All CT health
                        't_health_at_plant': t_health_dict,  # All T health
                        'had_kit_at_plant': player_had_kit_at_plant,
                        'kit_pickup_tick': kit_pickup_tick
                    })
            
            # Track T episodes during post-plant through retake window
            # Only track Ts who were ALIVE at plant time
            t_analysis_end = round_end_tick  # Track through actual round end
            
            t_players = ticks[
                (ticks['tick'] == plant_tick) &
                (ticks['team_num'] == 2) &
                (ticks['is_alive'] == True)
            ]['name'].unique()
            
            for player in t_players:
                # Check if player died during full post-plant window (including retake)
                player_death = deaths[
                    (deaths['tick'] >= plant_tick) &
                    (deaths['tick'] <= t_analysis_end) &
                    (deaths['user_name'] == player)
                ]
                
                # If player died, only track up to death tick; otherwise through retake
                if len(player_death) > 0:
                    player_end_tick = player_death.iloc[0]['tick']
                else:
                    player_end_tick = t_analysis_end
                
                # Track ALL utility thrown during full post-plant window (plant to actual round end)
                # This captures pre-retake utility that affects threat landscape
                all_utility_thrown = count_utility_thrown(
                    grenades_thrown, smoke_detonate, smoke_expire,
                    inferno_start, inferno_expire, flash_detonate,
                    player, plant_tick, player_end_tick,
                    classifier, ticks, damage
                )
                                player_ticks = ticks[
                    (ticks['tick'] >= plant_tick) &
                    (ticks['tick'] <= player_end_tick) &
                    (ticks['name'] == player) &
                    (ticks['is_alive'] == True)
                ].sort_values('tick')
                
                if len(player_ticks) == 0:
                    continue
                
                # Check if this CT player picks up a kit during retake window
                kit_pickup_tick = None
                player_had_kit_at_plant = ct_player in ct_kits_at_plant
                for idx in range(1, len(player_ticks)):
                    prev_row = player_ticks.iloc[idx-1]
                    curr_row = player_ticks.iloc[idx]
                    prev_kit = prev_row.get('has_defuser', False) if 'has_defuser' in prev_row.index else False
                    curr_kit = curr_row.get('has_defuser', False) if 'has_defuser' in curr_row.index else False
                    if not prev_kit and curr_kit:
                        kit_pickup_tick = curr_row['tick']
                        break
                
                current_zone = None
                zone_entry_tick = None
                
                for _, pos in player_ticks.iterrows():
                    zone = classifier.point_in_zone_A(pos['X'], pos['Y'], pos['Z'])
                    
                    if zone != current_zone:
                        if current_zone and zone_entry_tick:
                            duration_ticks = pos['tick'] - zone_entry_tick
                            
                            episode_damage = damage[
                                (damage['tick'] >= zone_entry_tick) &
                                (damage['tick'] < pos['tick']) &
                                (damage['user_name'] == player)
                            ]
                            damage_taken_gun, damage_taken_util = categorize_damage(episode_damage)
                            
                            episode_damage_dealt = damage[
                                (damage['tick'] >= zone_entry_tick) &
                                (damage['tick'] < pos['tick']) &
                                (damage['attacker_name'] == player)
                            ]
                            damage_dealt_gun, damage_dealt_util = categorize_damage(episode_damage_dealt)
                            
                            episode_deaths = deaths[
                                (deaths['tick'] >= zone_entry_tick) &
                                (deaths['tick'] < pos['tick']) &
                                (deaths['user_name'] == player)
                            ]
                            died = len(episode_deaths) > 0
                            
                            episode_kills = deaths[
                                (deaths['tick'] >= zone_entry_tick) &
                                (deaths['tick'] < pos['tick']) &
                                (deaths['attacker_name'] == player)
                            ]
                            got_kill = len(episode_kills) > 0
                            
                            # Calculate time remaining on bomb at episode start
                            time_since_plant = (zone_entry_tick - plant_tick) / 64.0
                            time_remaining_on_bomb = 40.0 - time_since_plant
                            
                            # Note: utility_thrown is tracked for full window, saved once at end
                            
                            t_episodes.append({
                                'zone': current_zone,
                                'duration_ticks': duration_ticks,
                                'damage_taken_gun': damage_taken_gun,
                                'damage_taken_util': damage_taken_util,
                                'damage_dealt_gun': damage_dealt_gun,
                                'damage_dealt_util': damage_dealt_util,
                                'died': died,
                                'got_kill': got_kill,
                                'site': 'B' if is_b_site else 'A',
                                'plant_spot': plant_spot,
                                'round_winner': winner_team,
                                'player_name': player,
                                'utility_thrown': {},  # Per-episode tracking removed, using full-window below
                                'time_remaining_on_bomb': time_remaining_on_bomb,  # Time-varying covariate
                                'ct_count_at_plant': ct_count_at_plant,
                                't_count_at_plant': t_count_at_plant,
                                'numerical_advantage': numerical_advantage,
                                'ct_count_at_retake': ct_count_at_retake,
                                't_count_at_retake': t_count_at_retake,
                                'player_hp_at_plant': t_health_dict.get(player, 0),  # This player's HP
                                'ct_health_at_plant': ct_health_dict,  # All CT health
                                't_health_at_plant': t_health_dict  # All T health
                            })
                        
                        current_zone = zone
                        zone_entry_tick = pos['tick']
                
                if current_zone and zone_entry_tick:
                    duration_ticks = player_end_tick - zone_entry_tick
                    
                    episode_damage = damage[
                        (damage['tick'] >= zone_entry_tick) &
                        (damage['tick'] <= player_end_tick) &
                        (damage['user_name'] == player)
                    ]
                    damage_taken_gun, damage_taken_util = categorize_damage(episode_damage)
                    
                    episode_damage_dealt = damage[
                        (damage['tick'] >= zone_entry_tick) &
                        (damage['tick'] <= player_end_tick) &
                        (damage['attacker_name'] == player)
                    ]
                    damage_dealt_gun, damage_dealt_util = categorize_damage(episode_damage_dealt)
                    
                    episode_deaths = deaths[
                        (deaths['tick'] >= zone_entry_tick) &
                        (deaths['tick'] <= player_end_tick) &
                        (deaths['user_name'] == player)
                    ]
                    died = len(episode_deaths) > 0
                    
                    episode_kills = deaths[
                        (deaths['tick'] >= zone_entry_tick) &
                        (deaths['tick'] <= player_end_tick) &
                        (deaths['attacker_name'] == player)
                    ]
                    got_kill = len(episode_kills) > 0
                    
                    # Calculate time remaining on bomb at episode start
                    time_since_plant = (zone_entry_tick - plant_tick) / 64.0
                    time_remaining_on_bomb = 40.0 - time_since_plant
                    
                    # Save final episode with full-window utility
                    t_episodes.append({
                        'zone': current_zone,
                        'duration_ticks': duration_ticks,
                        'damage_taken_gun': damage_taken_gun,
                        'damage_taken_util': damage_taken_util,
                        'damage_dealt_gun': damage_dealt_gun,
                        'damage_dealt_util': damage_dealt_util,
                        'died': died,
                        'got_kill': got_kill,
                        'site': 'B' if is_b_site else 'A',
                        'plant_spot': plant_spot,
                        'round_winner': winner_team,
                        'player_name': player,
                        'utility_thrown': all_utility_thrown,  # Full window utility saved here
                        'time_remaining_on_bomb': time_remaining_on_bomb,  # Time-varying covariate
                        'ct_count_at_plant': ct_count_at_plant,
                        't_count_at_plant': t_count_at_plant,
                        'numerical_advantage': numerical_advantage,
                        'ct_count_at_retake': ct_count_at_retake,
                        't_count_at_retake': t_count_at_retake,
                        'player_hp_at_plant': t_health_dict.get(player, 0),  # This player's HP
                        'ct_health_at_plant': ct_health_dict,  # All CT health
                        't_health_at_plant': t_health_dict  # All T health
                    })
        
        demos_processed += 1
        
    except Exception as e:
        print(f"  Error in {demo_path.name}: {str(e)[:50]}")
        demos_processed += 1
        continue

print(f"\n{'='*80}")
print(f"RESULTS: Analyzed {total_retakes} retakes from {demos_processed} demos")
print(f"  A-site: {a_site_retakes} retakes")
print(f"  B-site: {b_site_retakes} retakes")
print(f"  CT episodes: {len(ct_episodes)}")
print(f"  T episodes: {len(t_episodes)}")
print(f"{'='*80}\n")

# Aggregate episodes by zone for CT
ct_zone_stats = defaultdict(lambda: {
    'episodes': 0,
    'total_duration': 0,
    'total_damage_taken_gun': 0,
    'total_damage_taken_util': 0,
    'total_damage_dealt_gun': 0,
    'total_damage_dealt_util': 0,
    'deaths': 0,
    'kills': 0
})

for ep in ct_episodes:
    zone = ep['zone']
    ct_zone_stats[zone]['episodes'] += 1
    ct_zone_stats[zone]['total_duration'] += ep['duration_ticks']
    ct_zone_stats[zone]['total_damage_taken_gun'] += ep['damage_taken_gun']
    ct_zone_stats[zone]['total_damage_taken_util'] += ep['damage_taken_util']
    ct_zone_stats[zone]['total_damage_dealt_gun'] += ep['damage_dealt_gun']
    ct_zone_stats[zone]['total_damage_dealt_util'] += ep['damage_dealt_util']
    if ep['died']:
        ct_zone_stats[zone]['deaths'] += 1
    if ep['got_kill']:
        ct_zone_stats[zone]['kills'] += 1

# Aggregate episodes by zone for T
t_zone_stats = defaultdict(lambda: {
    'episodes': 0,
    'total_duration': 0,
    'total_damage_taken_gun': 0,
    'total_damage_taken_util': 0,
    'total_damage_dealt_gun': 0,
    'total_damage_dealt_util': 0,
    'deaths': 0,
    'kills': 0
})

for ep in t_episodes:
    zone = ep['zone']
    t_zone_stats[zone]['episodes'] += 1
    t_zone_stats[zone]['total_duration'] += ep['duration_ticks']
    t_zone_stats[zone]['total_damage_taken_gun'] += ep['damage_taken_gun']
    t_zone_stats[zone]['total_damage_taken_util'] += ep['damage_taken_util']
    t_zone_stats[zone]['total_damage_dealt_gun'] += ep['damage_dealt_gun']
    t_zone_stats[zone]['total_damage_dealt_util'] += ep['damage_dealt_util']
    if ep['died']:
        t_zone_stats[zone]['deaths'] += 1
    if ep['got_kill']:
        t_zone_stats[zone]['kills'] += 1

print("Top 25 CT zones during retakes:\n")
print(f"{'Zone':<30s} {'Episodes':>9s} {'AvgDur(s)':>10s} {'Deaths':>7s} {'Kills':>7s} {'GunDmg':>8s} {'UtilDmg':>8s} {'GunDlt':>8s} {'UtilDlt':>8s}")
print("-"*120)

ct_sorted = sorted(ct_zone_stats.items(), key=lambda x: x[1]['total_duration'], reverse=True)[:25]
for zone, stats in ct_sorted:
    avg_duration = stats['total_duration'] / stats['episodes'] / 64.0  # Convert to seconds
    avg_gun_taken = stats['total_damage_taken_gun'] / stats['episodes']
    avg_util_taken = stats['total_damage_taken_util'] / stats['episodes']
    avg_gun_dealt = stats['total_damage_dealt_gun'] / stats['episodes']
    avg_util_dealt = stats['total_damage_dealt_util'] / stats['episodes']
    print(f"{zone:<30s} {stats['episodes']:>9,} {avg_duration:>10.1f} {stats['deaths']:>7,} {stats['kills']:>7,} {avg_gun_taken:>8.1f} {avg_util_taken:>8.1f} {avg_gun_dealt:>8.1f} {avg_util_dealt:>8.1f}")

print(f"\n\nTop 25 T zones during post-plant:\n")
print(f"{'Zone':<30s} {'Episodes':>9s} {'AvgDur(s)':>10s} {'Deaths':>7s} {'Kills':>7s} {'GunDmg':>8s} {'UtilDmg':>8s} {'GunDlt':>8s} {'UtilDlt':>8s}")
print("-"*120)

t_sorted = sorted(t_zone_stats.items(), key=lambda x: x[1]['total_duration'], reverse=True)[:25]
for zone, stats in t_sorted:
    avg_duration = stats['total_duration'] / stats['episodes'] / 64.0
    avg_gun_taken = stats['total_damage_taken_gun'] / stats['episodes']
    avg_util_taken = stats['total_damage_taken_util'] / stats['episodes']
    avg_gun_dealt = stats['total_damage_dealt_gun'] / stats['episodes']
    avg_util_dealt = stats['total_damage_dealt_util'] / stats['episodes']
    print(f"{zone:<30s} {stats['episodes']:>9,} {avg_duration:>10.1f} {stats['deaths']:>7,} {stats['kills']:>7,} {avg_gun_taken:>8.1f} {avg_util_taken:>8.1f} {avg_gun_dealt:>8.1f} {avg_util_dealt:>8.1f}")

# Save to JSON with comprehensive episode data
output = {
    "metadata": {
        "total_retakes_analyzed": total_retakes,
        "a_site_retakes": a_site_retakes,
        "b_site_retakes": b_site_retakes,
        "total_demos_processed": demos_processed,
        "methodology": "Episode-based tracking: each player's time in each zone is tracked with damage taken/dealt, kills, deaths, utility usage, and time-varying covariates for Cox proportional hazards modeling",
        "description": "Empirical post-plant positioning episodes for hazard modeling, based on analysis of professional CS2 demos"
    },
    "ct_retake_episodes": {
        "total_episodes": len(ct_episodes),
        "description": "CT positioning episodes during retakes (from retake detection to +15s or round end). Each episode represents a player in a specific zone with associated outcomes and context.",
        "episodes": ct_episodes,
        "zone_aggregates": {zone: stats for zone, stats in ct_zone_stats.items()}
    },
    "t_postplant_episodes": {
        "total_episodes": len(t_episodes),
        "description": "T positioning episodes after plant (from plant to round end). Each episode represents a player in a specific zone with associated outcomes and context.",
        "episodes": t_episodes,
        "zone_aggregates": {zone: stats for zone, stats in t_zone_stats.items()}
    },
    "feature_documentation": {
        "time_remaining_on_bomb": "Seconds remaining on C4 timer (40s total) at episode start. Time-varying covariate that increases pressure on CTs as it decreases.",
        "ct_count_at_plant": "Number of CTs alive when bomb was planted. Initial numerical condition.",
        "t_count_at_plant": "Number of Ts alive when bomb was planted. Initial numerical condition.",
        "numerical_advantage": "ct_count - t_count at plant. Positive = CT advantage, Negative = T advantage, 0 = Even.",
        "ct_count_at_retake": "Number of CTs alive when retake was detected (e.g., 3 in a 3v3 retake).",
        "t_count_at_retake": "Number of Ts alive when retake was detected (e.g., 3 in a 3v3 retake).",
        "player_hp_at_plant": "This specific player's health points when bomb was planted (0-100).",
        "ct_health_at_plant": "Dictionary mapping all CT player names to their HP at plant time.",
        "t_health_at_plant": "Dictionary mapping all T player names to their HP at plant time.",
        "plant_spot": "Specific plant location classification (e.g., 'default', 'triple_ramp_side', 'jungle', etc.). One of 14 defined plant positions.",
        "utility_thrown": "Utility grenades thrown by this player during their tracking window (full round for final episode). Includes zone classifications and timing.",
        "damage_taken_gun": "Gun damage received during this episode.",
        "damage_taken_util": "Utility damage (HE/molly/fire) received during this episode.",
        "damage_dealt_gun": "Gun damage dealt to enemies during this episode.",
        "damage_dealt_util": "Utility damage dealt to enemies during this episode.",
        "died": "Boolean - whether player died during this episode.",
        "got_kill": "Boolean - whether player got a kill during this episode.",
        "had_kit_at_plant": "Boolean - whether this CT player had a defuse kit when bomb was planted.",
        "kit_pickup_tick": "Tick number when CT picked up a defuse kit during retake (None if no pickup or not applicable).",
        "duration_ticks": "Length of episode in ticks (64 ticks = 1 second).",
        "zone": "Zone name where player was positioned during this episode.",
        "site": "Bomb site (A or B).",
        "round_winner": "Which team won the round (CT or T)."
    },
    "integration_notes": {
        "survival_modeling": "Use time_remaining_on_bomb as time-varying covariate. Use player_hp_at_plant and numerical_advantage as baseline covariates. Zone determines baseline hazard.",
        "hazard_interpretation": "Hazard = instantaneous risk of death given survival to time t. Higher time_remaining_on_bomb increases CT hazard (must defuse faster). Numerical advantage affects both teams' baseline hazard.",
        "utility_impact": "utility_thrown can be used to model information state and zone denial effects on hazard rates."
    }
}

output_path = Path("data/player_positioning_episodes.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*80}")
print(f"Saved positioning episodes to: {output_path}")
print(f"{'='*80}")
print("\nThis data integrates with:")
print("  - Zone connectivity (threat cone pathfinding)")
print("  - Audio tracker (sound events affect hazard via information gain)")
print("  - Survival model (P(survival | position, time, enemies, info))")
print("\nNext steps:")
print("  1. Load this episode data in survival model")
print("  2. Combine with audio_tracker.py sound events")
print("  3. Use zone_connectivity.py for threat propagation")
print("  4. Calculate hazard rates conditioned on information state")
