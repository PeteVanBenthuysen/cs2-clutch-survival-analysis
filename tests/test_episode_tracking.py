"""
Test episode-based positioning tracking on mouz vs BIG game.
Shows individual player episodes with damage, kills, deaths.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from demoparser2 import DemoParser
from src.zone_classifier import MirageZoneClassifier
from src.zone_connectivity import ZoneConnectivity
from src.plant_spot_classifier import PlantSpotClassifier
from src.detect_retake import detect_retake
from collections import defaultdict

# Accept demo path and optional round number from command line
if len(sys.argv) > 1:
    demo_path = Path(sys.argv[1])
    target_round = int(sys.argv[2]) if len(sys.argv) > 2 else None
else:
    demo_path = Path("research_demos/extracted/8037_IEM_Dallas_2025/2382377/the-mongolz-vs-furia-m2-mirage.dem")
    target_round = None

print("="*80)
print(f"TESTING EPISODE TRACKING - {demo_path.name}")
print("="*80)

classifier = MirageZoneClassifier()
connectivity = ZoneConnectivity()
plant_classifier = PlantSpotClassifier()
parser = DemoParser(str(demo_path))

# Get map name from demo header
header = parser.parse_header()
map_name = header.get('map_name', 'unknown')
print(f"\nMap from demo: {map_name}")
if 'mirage' not in map_name.lower():
    print(f"WARNING: Demo map is '{map_name}', not Mirage")
print()

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
    """Get player's team (CT or T) at a given tick."""
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
                (grenades_df['weapon'] == 'hegrenade') &\
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

def get_active_utility_at_tick(smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate, tick, window_before=5*64, classifier=None, ticks=None):
    """Find smokes/mollies/flashes active at given tick (for pre-plant utility still burning/active).
    
    Args:
        smoke_detonate: DataFrame of smoke detonations
        smoke_expire: DataFrame of smoke expirations  
        inferno_start: DataFrame of molly/incendiary starts
        inferno_expire: DataFrame of molly/incendiary ends
        flash_detonate: DataFrame of flashbang detonations
        tick: The tick to check (e.g., retake start)
        window_before: How many ticks before to look for pre-existing utility (default 5s)
        classifier: ZoneClassifier for zone classification (optional)
        ticks: Tick data for team lookup (optional)
    
    Returns:
        dict with detailed utility info:
        {
            'smokes': [{user_name, detonate_tick, expire_tick, zone, team, thrower_zone}, ...],
            'mollies': [{user_name, start_tick, expire_tick, zone, team, thrower_zone}, ...],
            'recent_flashes': [{user_name, detonate_tick, effect_until_tick, zone, team}, ...]  # Last 3s
        }
    """
    # Find smokes that detonated before tick but expire after tick
    active_smokes_data = []
    active_smokes = smoke_detonate[
        (smoke_detonate['tick'] < tick) &
        (smoke_detonate['tick'] >= tick - window_before)
    ]
    
    # DEBUG
    print(f"\n[DEBUG get_active_utility_at_tick] Looking for smokes at tick {tick}, window_before={window_before/64:.1f}s")
    print(f"  Smoke detonations in window: {len(active_smokes)}")
    if len(active_smokes) > 0:
        for idx, smoke in active_smokes.head(5).iterrows():
            print(f"    {smoke['user_name']} smoke at tick {smoke['tick']} (tick-{(tick-smoke['tick'])/64:.1f}s ago)")
    
    for _, smoke in active_smokes.iterrows():
        smoke_user = smoke['user_name']
        smoke_tick = smoke['tick']
        
        # Find when this smoke expires
        expire_event = smoke_expire[
            (smoke_expire['user_name'] == smoke_user) &
            (smoke_expire['tick'] > smoke_tick)
        ].sort_values('tick')
        
        # DEBUG
        if len(expire_event) > 0:
            expire_tick = expire_event.iloc[0]['tick']
            still_active = expire_tick > tick
            print(f"    {smoke_user} smoke det@{smoke_tick} exp@{expire_tick} -> active={still_active} (exp in {(expire_tick-tick)/64:.1f}s)")
        else:
            print(f"    {smoke_user} smoke det@{smoke_tick} NO EXPIRE EVENT FOUND")
        
        if len(expire_event) > 0:
            expire_tick = expire_event.iloc[0]['tick']
            # Only include if still active at target tick
            if expire_tick > tick:
                zone = 'unknown'
                team = '?'
                thrower_zone = 'unknown'
                
                # Classify zone if classifier provided
                if classifier:
                    x, y, z = smoke.get('X', 0), smoke.get('Y', 0), smoke.get('Z', 0)
                    if x != 0 or y != 0 or z != 0:
                        zone_result = classifier.point_in_zone_A(x, y, z)
                        if zone_result:
                            zone = zone_result
                    
                    # Find thrower position at throw time (use x/y/z from grenade_thrown event)
                    if ticks is not None and 'user_X' in smoke:
                        thrower_x = smoke.get('user_X', 0)
                        thrower_y = smoke.get('user_Y', 0)
                        thrower_z = smoke.get('user_Z', 0)
                        if thrower_x != 0 or thrower_y != 0 or thrower_z != 0:
                            thrower_zone_result = classifier.point_in_zone_A(thrower_x, thrower_y, thrower_z)
                            if thrower_zone_result:
                                thrower_zone = thrower_zone_result
                
                # Get team if ticks provided
                if ticks is not None:
                    team = get_player_team(ticks, smoke_user, smoke_tick)
                
                active_smokes_data.append({
                    'user_name': smoke_user,
                    'detonate_tick': smoke_tick,
                    'expire_tick': expire_tick,
                    'zone': zone,
                    'team': team if team else '?',
                    'thrower_zone': thrower_zone
                })
    
    # Find mollies that started before tick but haven't expired
    active_mollies_data = []
    active_mollies = inferno_start[
        (inferno_start['tick'] < tick) &
        (inferno_start['tick'] >= tick - window_before)
    ]
    
    for _, molly in active_mollies.iterrows():
        molly_user = molly['user_name']
        molly_tick = molly['tick']
        
        # Find when this molly expires
        expire_event = inferno_expire[
            (inferno_expire['user_name'] == molly_user) &
            (inferno_expire['tick'] > molly_tick)
        ].sort_values('tick')
        
        if len(expire_event) > 0:
            expire_tick = expire_event.iloc[0]['tick']
            # Only include if still active at target tick
            if expire_tick > tick:
                zone = 'unknown'
                team = '?'
                thrower_zone = 'unknown'
                
                # Classify zone if classifier provided
                if classifier:
                    x, y, z = molly.get('X', 0), molly.get('Y', 0), molly.get('Z', 0)
                    if x != 0 or y != 0 or z != 0:
                        zone_result = classifier.point_in_zone_A(x, y, z)
                        if zone_result:
                            zone = zone_result
                    
                    # Find thrower position
                    if ticks is not None and 'user_X' in molly:
                        thrower_x = molly.get('user_X', 0)
                        thrower_y = molly.get('user_Y', 0)
                        thrower_z = molly.get('user_Z', 0)
                        if thrower_x != 0 or thrower_y != 0 or thrower_z != 0:
                            thrower_zone_result = classifier.point_in_zone_A(thrower_x, thrower_y, thrower_z)
                            if thrower_zone_result:
                                thrower_zone = thrower_zone_result
                
                # Get team if ticks provided
                if ticks is not None:
                    team = get_player_team(ticks, molly_user, molly_tick)
                
                active_mollies_data.append({
                    'user_name': molly_user,
                    'start_tick': molly_tick,
                    'expire_tick': expire_tick,
                    'zone': zone,
                    'team': team if team else '?',
                    'thrower_zone': thrower_zone
                })
    
    # Find recent flashes (last 3 seconds = 192 ticks)
    # Flashes don't have expire events, assume 3s effect duration
    FLASH_EFFECT_DURATION = 3 * 64  # 3 seconds in ticks
    recent_flashes_data = []
    
    recent_flashes = flash_detonate[
        (flash_detonate['tick'] < tick) &
        (flash_detonate['tick'] >= tick - FLASH_EFFECT_DURATION)
    ]
    
    for _, flash in recent_flashes.iterrows():
        effect_until = flash['tick'] + FLASH_EFFECT_DURATION
        if effect_until > tick:  # Flash effect still relevant
            recent_flashes_data.append({
                'user_name': flash['user_name'],
                'detonate_tick': flash['tick'],
                'effect_until_tick': effect_until,
                'X': flash.get('X', 0),
                'Y': flash.get('Y', 0),
                'Z': flash.get('Z', 0)
            })
    
    return {
        'smokes': active_smokes_data,
        'mollies': active_mollies_data,
        'recent_flashes': recent_flashes_data
    }

# Get all events
bomb_plants = parser.parse_event('bomb_planted')
ticks = parser.parse_ticks(['X', 'Y', 'Z', 'name', 'tick', 'team_num', 'is_alive', 'health', 'has_defuser', 'armor_value', 'has_helmet', 'active_weapon_name'])
damage = parser.parse_event('player_hurt')
deaths = parser.parse_event('player_death')
round_starts = parser.parse_event('round_start')
round_ends = parser.parse_event('round_end')
grenades_thrown = parser.parse_event('grenade_thrown')
smoke_detonate = parser.parse_event('smokegrenade_detonate')
smoke_expire = parser.parse_event('smokegrenade_expired')
inferno_start = parser.parse_event('inferno_startburn')
inferno_expire = parser.parse_event('inferno_expire')
flash_detonate = parser.parse_event('flashbang_detonate')

print(f"\nFound {len(bomb_plants)} bomb plants")

ct_episodes = []
t_episodes = []
retakes_found = 0

# Just analyze first 3 retakes for testing
for idx, plant in bomb_plants.iterrows():
    plant_tick = plant['tick']
    round_num = len(round_starts[round_starts['tick'] <= plant_tick])
    
    # Skip if filtering for specific round and this isn't it
    if target_round and round_num != target_round:
        continue
    
    if retakes_found >= 1:
        break
    
    # Find the round start for this plant
    this_round_start = round_starts[round_starts['tick'] <= plant_tick].sort_values('tick')
    if len(this_round_start) == 0:
        continue
    round_start_tick = this_round_start.iloc[-1]['tick']
    
    # Get round winner - find the NEXT round_end after plant (tick-based)
    round_end = round_ends[round_ends['tick'] > plant_tick].sort_values('tick')
    winner_team = round_end.iloc[0]['winner'] if len(round_end) > 0 else None
    
    planter_name = plant['user_name']
    planter_pos = ticks[(ticks['tick'] == plant_tick) & (ticks['name'] == planter_name)]
    
    if len(planter_pos) == 0:
        continue
        
    pos = planter_pos.iloc[0]
    bomb_zone = classifier.point_in_zone_A(pos['X'], pos['Y'], pos['Z'])
    
    if not bomb_zone:
        continue
    
    is_b_site = bomb_zone.startswith('B_') or bomb_zone.startswith('Cat_')
    site_type = "B-site" if is_b_site else "A-site"
    
    # Get specific plant spot
    site_letter = 'b' if is_b_site else 'a'
    plant_spot = plant_classifier.classify_plant(pos['X'], pos['Y'], pos['Z'], site_letter)
    
    # Get round end tick
    round_end_tick = round_end.iloc[0]['tick'] if len(round_end) > 0 else plant_tick + 40*64
    
    # Call detect_retake function
    retake_result = detect_retake(plant_tick, bomb_zone, ticks, damage, classifier, round_end_tick)
    
    if not retake_result['retake_detected']:
        continue
    
    retake_tick = retake_result['retake_tick']
    time_to_retake = retake_result['time_to_retake']
    
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
    
    # Capture armor at plant_tick - create dict mapping player name to armor info
    ct_armor_dict = {}
    for _, player_row in ct_alive_at_plant.iterrows():
        armor_value = player_row.get('armor_value', 0)
        has_helmet = player_row.get('has_helmet', False)
        
        # Classify armor type
        if armor_value > 0 and has_helmet:
            armor_type = 'Kevlar+Helmet'
        elif armor_value > 0:
            armor_type = 'Kevlar'
        else:
            armor_type = 'None'
        
        ct_armor_dict[player_row['name']] = {
            'armor_value': armor_value,
            'has_helmet': has_helmet,
            'armor_type': armor_type
        }
    
    t_armor_dict = {}
    for _, player_row in t_alive_at_plant.iterrows():
        armor_value = player_row.get('armor_value', 0)
        has_helmet = player_row.get('has_helmet', False)
        
        # Classify armor type
        if armor_value > 0 and has_helmet:
            armor_type = 'Kevlar+Helmet'
        elif armor_value > 0:
            armor_type = 'Kevlar'
        else:
            armor_type = 'None'
        
        t_armor_dict[player_row['name']] = {
            'armor_value': armor_value,
            'has_helmet': has_helmet,
            'armor_type': armor_type
        }
    
    # Capture weapons at plant_tick - create dict mapping player name to active weapon
    ct_weapons_dict = {}
    for _, player_row in ct_alive_at_plant.iterrows():
        weapon = player_row.get('active_weapon_name', 'Unknown')
        ct_weapons_dict[player_row['name']] = weapon
    
    t_weapons_dict = {}
    for _, player_row in t_alive_at_plant.iterrows():
        weapon = player_row.get('active_weapon_name', 'Unknown')
        t_weapons_dict[player_row['name']] = weapon
    
    # Skip rounds if filtering for specific round number
    if target_round and round_num != target_round:
        continue
    
    # Get round end tick (bomb explosion, defusal, or last player death)
    round_end_row = round_ends[round_ends['tick'] > plant_tick].sort_values('tick')
    round_end_tick = round_end_row.iloc[0]['tick'] if len(round_end_row) > 0 else plant_tick + 40*64
    
    # Track utility that shaped the post-plant phase (plant to round end)
    # This includes smokes/mollies that blocked zones AND utility thrown during retake fight
    # Look back 30s before plant to catch pre-plant smokes that were still active post-plant
    postplant_utility = count_utility_thrown(
        grenades_thrown, smoke_detonate, smoke_expire,
        inferno_start, inferno_expire, flash_detonate,
        None,  # All players
        plant_tick - 30*64,  # 30s before plant to catch pre-plant smokes
        round_end_tick,  # Until round ends (explosion/defusal/elimination)
        classifier, ticks, damage
    )
    
    # Also check for utility still active AT retake start
    active_utility = get_active_utility_at_tick(
        smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate,
        retake_tick, window_before=40*64,  # Look 40 seconds back to catch pre-plant smokes
        classifier=classifier, ticks=ticks
    )
    
    # Count alive players at retake_tick (critical for Cox model)
    retake_tick_state = ticks[ticks['tick'] == retake_tick]
    
    ct_alive_at_retake = retake_tick_state[
        (retake_tick_state['team_num'] == 3) &
        (retake_tick_state['is_alive'] == True)
    ]
    t_alive_at_retake = retake_tick_state[
        (retake_tick_state['team_num'] == 2) &
        (retake_tick_state['is_alive'] == True)
    ]
    
    ct_count_at_retake = len(ct_alive_at_retake['name'].unique())
    t_count_at_retake = len(t_alive_at_retake['name'].unique())
    
    # Track defuse kits at plant time
    ct_kits_at_plant = {}
    for _, ct in ct_alive_at_plant.iterrows():
        has_kit = ct.get('has_defuser', False)
        if has_kit:
            ct_kits_at_plant[ct['name']] = True
    
    kits_available = len(ct_kits_at_plant)
    kit_holders = list(ct_kits_at_plant.keys())
    
    winner_str = f"{winner_team} WIN" if winner_team else "Unknown"
    print(f"\n{'='*80}")
    print(f"Round {round_num}: {site_type} - Plant at {plant_spot} ({pos['X']:.1f}, {pos['Y']:.1f}, {pos['Z']:.1f}) - Retake at +{time_to_retake:.1f}s - {winner_str}")
    print(f"\nInitial Conditions at Plant:")
    
    # Show numerical advantage clearly
    if numerical_advantage > 0:
        adv_str = f"CT +{numerical_advantage} advantage"
    elif numerical_advantage < 0:
        adv_str = f"T +{abs(numerical_advantage)} advantage"
    else:
        adv_str = "Even"
    print(f"  Player Count: {ct_count_at_plant}v{t_count_at_plant} ({adv_str})")
    print(f"  Defuse Kits: {kits_available} ({', '.join(kit_holders) if kit_holders else 'none'})")
    
    # Show retake player counts (who's still alive when retake starts)
    retake_adv = ct_count_at_retake - t_count_at_retake
    if retake_adv > 0:
        retake_adv_str = f"CT +{retake_adv}"
    elif retake_adv < 0:
        retake_adv_str = f"T +{abs(retake_adv)}"
    else:
        retake_adv_str = "Even"
    print(f"  At Retake: {ct_count_at_retake}v{t_count_at_retake} ({retake_adv_str})")
    
    # Show per-player health
    print(f"  CT Health:")
    for player_name, hp in ct_health_dict.items():
        print(f"    {player_name}: {hp} HP")
    
    print(f"  T Health:")
    for player_name, hp in t_health_dict.items():
        print(f"    {player_name}: {hp} HP")
    
    # Debug: Show armor at plant
    print(f"\n[DEBUG] CT Armor at Plant:")
    for player_name, armor_info in ct_armor_dict.items():
        print(f"    {player_name}: {armor_info['armor_type']} (value: {armor_info['armor_value']})")
    
    print(f"[DEBUG] T Armor at Plant:")
    for player_name, armor_info in t_armor_dict.items():
        print(f"    {player_name}: {armor_info['armor_type']} (value: {armor_info['armor_value']})")
    
    # Debug: Show weapons at plant
    print(f"\n[DEBUG] CT Weapons at Plant:")
    for player_name, weapon in ct_weapons_dict.items():
        print(f"    {player_name}: {weapon}")
    
    print(f"[DEBUG] T Weapons at Plant:")
    for player_name, weapon in t_weapons_dict.items():
        print(f"    {player_name}: {weapon}")
    
    # Debug: Show ALL grenades thrown in window around retake (if available)
    if len(grenades_thrown) > 0:
        grenades_in_window = grenades_thrown[
            (grenades_thrown['tick'] >= plant_tick - 5*64) &
            (grenades_thrown['tick'] <= round_end_tick)
        ]
        print(f"\n[DEBUG] Grenades thrown (plant-5s to round end):")
        for _, nade in grenades_in_window.iterrows():
            time_rel = (nade['tick'] - plant_tick) / 64.0
            print(f"  {nade['user_name']:12s} {nade['weapon']:12s} at plant+{time_rel:+6.1f}s (tick {nade['tick']})")
    else:
        print(f"\n[DEBUG] Grenades thrown: No grenade_thrown events available (demo parsing limitation)")
    
    # DEBUG: Show ALL smokes in this round (if available)
    if len(grenades_thrown) > 0:
        round_smokes = grenades_thrown[
            (grenades_thrown['tick'] >= plant_tick - 30*64) &  # 30s before plant
            (grenades_thrown['tick'] <= retake_tick + 5*64) &
            (grenades_thrown['weapon'] == 'smokegrenade')
        ]
        print(f"\n[DEBUG] ALL smokes (plant-30s to retake+5s):")
        for _, smoke in round_smokes.iterrows():
            time_rel = (smoke['tick'] - plant_tick) / 64.0
            print(f"  {smoke['user_name']:12s} smoke at plant+{time_rel:+6.1f}s (tick {smoke['tick']})")
    else:
        print(f"\n[DEBUG] ALL smokes: Not available (grenade_thrown events missing)")
    
    # DEBUG: Show smoke detonations to verify zone detection
    smoke_dets_in_window = smoke_detonate[
        (smoke_detonate['tick'] >= plant_tick - 30*64) &
        (smoke_detonate['tick'] <= retake_tick + 5*64)
    ].sort_values('tick')
    print(f"\n[DEBUG] Smoke detonations (with zones):")
    for _, det in smoke_dets_in_window.iterrows():
        det_time = (det['tick'] - plant_tick) / 64.0
        zone = classifier.point_in_zone_A(det['x'], det['y'], det['z'])
        
        # Check expiry
        expire_event = smoke_expire[
            (smoke_expire['entityid'] == det['entityid']) &
            (smoke_expire['tick'] > det['tick'])
        ]
        if len(expire_event) > 0:
            exp_tick = expire_event.iloc[0]['tick']
            exp_time = (exp_tick - plant_tick) / 64.0
            print(f"  {det['user_name']:12s} {zone:25s} det:plant+{det_time:+6.1f}s exp:plant+{exp_time:+6.1f}s (entity {det['entityid']})")
        else:
            print(f"  {det['user_name']:12s} {zone:25s} det:plant+{det_time:+6.1f}s (entity {det['entityid']}, no expire event)")
    
    # DEBUG: Show molly starts (inferno_start events)
    molly_starts_in_window = inferno_start[
        (inferno_start['tick'] >= plant_tick - 30*64) &
        (inferno_start['tick'] <= round_end_tick)
    ].sort_values('tick')
    print(f"\n[DEBUG] Molly starts (inferno_start events):")
    for _, inf in molly_starts_in_window.iterrows():
        start_time = (inf['tick'] - plant_tick) / 64.0
        zone = classifier.point_in_zone_A(inf['x'], inf['y'], inf['z'])
        
        # Check expiry
        expire_event = inferno_expire[
            (inferno_expire['entityid'] == inf['entityid']) &
            (inferno_expire['tick'] > inf['tick'])
        ]
        if len(expire_event) > 0:
            exp_tick = expire_event.iloc[0]['tick']
            exp_time = (exp_tick - plant_tick) / 64.0
            print(f"  {inf['user_name']:12s} {zone:25s} start:plant+{start_time:+6.1f}s exp:plant+{exp_time:+6.1f}s (entity {inf['entityid']})")
        else:
            print(f"  {inf['user_name']:12s} {zone:25s} start:plant+{start_time:+6.1f}s (entity {inf['entityid']}, no expire event)")
    
    # DEBUG: Show flash detonations
    flash_dets_in_window = flash_detonate[
        (flash_detonate['tick'] >= plant_tick) &
        (flash_detonate['tick'] <= round_end_tick)
    ].sort_values('tick')
    print(f"\n[DEBUG] Flash detonations (post-plant to round end):")
    for _, det in flash_dets_in_window.iterrows():
        det_time = (det['tick'] - plant_tick) / 64.0
        zone = classifier.point_in_zone_A(det['x'], det['y'], -100)
        print(f"  {det['user_name']:12s} zone:{zone if zone else 'unknown':20s} det:plant+{det_time:+6.1f}s")
    
    # DEBUG: Show HE grenade damage events
    he_damage_in_window = damage[
        (damage['tick'] >= plant_tick) &
        (damage['tick'] <= round_end_tick) &
        (damage['weapon'] == 'hegrenade')
    ].sort_values('tick')
    print(f"\n[DEBUG] HE grenade damage (post-plant to round end):")
    for _, dmg in he_damage_in_window.iterrows():
        dmg_time = (dmg['tick'] - plant_tick) / 64.0
        victim_pos = ticks[(ticks['tick'] == dmg['tick']) & (ticks['name'] == dmg['user_name'])]
        if len(victim_pos) > 0:
            pos = victim_pos.iloc[0]
            zone = classifier.point_in_zone_A(pos['X'], pos['Y'], pos['Z'])
            print(f"  {dmg['attacker_name']:12s} -> {dmg['user_name']:12s} in {zone if zone else 'unknown':20s} dmg:{dmg['hp_damage']:3.0f} at plant+{dmg_time:+6.1f}s")
    
    # Show post-plant utility that shaped retake (blocked zones, forced delays)
    num_postplant_smokes = len(postplant_utility['smokes'])
    num_postplant_mollies = len(postplant_utility['mollies'])
    num_postplant_flashes = len(postplant_utility['flashes'])
    num_postplant_he = len(postplant_utility['he'])
    
    if num_postplant_smokes > 0 or num_postplant_mollies > 0 or num_postplant_flashes > 0 or num_postplant_he > 0:
        print(f"\n[DEBUG] Post-plant utility (plant to retake, shaped CT approach):")
        
        # Filter utility that was TACTICALLY RELEVANT:
        # 1. Smokes/mollies that were active 3+ seconds after plant (delayed CT retake)
        # 2. Utility thrown after plant (post-plant usage)
        # 3. Utility thrown near retake time (retake fight utility)
        
        active_postplant_smokes = [
            s for s in postplant_utility['smokes'] 
            if (s.get('expire_tick', 0) - plant_tick >= 3*64 or  # Active 3+ seconds after plant
                s.get('detonate_tick', 0) >= plant_tick)  # Thrown after plant
        ]
        active_postplant_mollies = [
            m for m in postplant_utility['mollies'] 
            if (m.get('expire_tick', 0) > plant_tick or  # ANY post-plant burn time (mollies only last ~7s)
                m.get('start_tick', 0) >= plant_tick)  # Thrown after plant
        ]
        active_postplant_flashes = [f for f in postplant_utility['flashes'] if f.get('detonate_tick', 0) >= plant_tick]
        active_postplant_he = [h for h in postplant_utility['he'] if h.get('damage_tick', 0) >= plant_tick]
        
        for smoke in active_postplant_smokes:
            team = smoke.get('team', '?')
            zone = smoke.get('zone', 'unknown')
            thrower_zone = smoke.get('thrower_zone', 'unknown')
            det_tick = smoke.get('detonate_tick', 0)
            exp_tick = smoke.get('expire_tick', det_tick + 18*64)
            det_time = (det_tick - plant_tick) / 64.0
            exp_time = (exp_tick - plant_tick) / 64.0
            
            if exp_tick < retake_tick:
                status = f"faded {(retake_tick - exp_tick)/64:.1f}s before retake"
            elif det_tick > retake_tick:
                status = f"thrown {(det_tick - retake_tick)/64:.1f}s after retake"
            else:
                status = f"ACTIVE at retake (fades {(exp_tick - retake_tick)/64:.1f}s after)"
            
            if thrower_zone != 'unknown' and zone != 'unknown':
                print(f"  {team}-smoke: {thrower_zone} -> {zone} (plant+{det_time:.1f}s to +{exp_time:.1f}s) [{status}]")
            elif zone != 'unknown':
                print(f"  {team}-smoke: {zone} (plant+{det_time:.1f}s to +{exp_time:.1f}s) [{status}]")
        
        for molly in active_postplant_mollies:
            team = molly.get('team', '?')
            zone = molly.get('zone', 'unknown')
            thrower_zone = molly.get('thrower_zone', 'unknown')
            start_tick = molly.get('start_tick', 0)
            exp_tick = molly.get('expire_tick', start_tick + 7*64)
            start_time = (start_tick - plant_tick) / 64.0
            exp_time = (exp_tick - plant_tick) / 64.0
            
            if exp_tick < retake_tick:
                status = f"burned out {(retake_tick - exp_tick)/64:.1f}s before retake"
            else:
                status = f"ACTIVE at retake (burns out {(exp_tick - retake_tick)/64:.1f}s after)"
            
            if thrower_zone != 'unknown' and zone != 'unknown':
                print(f"  {team}-molly: {thrower_zone} -> {zone} (plant+{start_time:.1f}s to +{exp_time:.1f}s) [{status}]")
            elif zone != 'unknown':
                print(f"  {team}-molly: {zone} (plant+{start_time:.1f}s to +{exp_time:.1f}s) [{status}]")
        
        for flash in active_postplant_flashes:
            team = flash.get('team', '?')
            zone = flash.get('zone', 'unknown')
            thrower_zone = flash.get('thrower_zone', 'unknown')
            det_tick = flash.get('detonate_tick', 0)
            det_time = (det_tick - plant_tick) / 64.0
            flash_height = flash.get('height', 0)
            flash_type = flash.get('flash_type', 'unknown')
            
            time_to_retake = (retake_tick - det_tick) / 64.0
            if det_tick < retake_tick:
                status = f"{time_to_retake:.1f}s before retake"
            else:
                status = f"{abs(time_to_retake):.1f}s after retake"
            
            height_str = f"h={flash_height:.0f} ({flash_type})"
            
            if thrower_zone != 'unknown' and zone != 'unknown':
                print(f"  {team}-flash: {thrower_zone} -> {zone} {height_str} (plant+{det_time:.1f}s) [{status}]")
            elif zone != 'unknown':
                print(f"  {team}-flash: {zone} {height_str} (plant+{det_time:.1f}s) [{status}]")
            else:
                print(f"  {team}-flash: unknown zone {height_str} (plant+{det_time:.1f}s) [{status}]")
        
        for he in active_postplant_he:
            team = he.get('team', '?')
            zone = he.get('zone', 'unknown')
            thrower_zone = he.get('thrower_zone', 'unknown')
            dmg_tick = he.get('damage_tick', 0)
            dmg_time = (dmg_tick - plant_tick) / 64.0
            
            time_to_retake = (retake_tick - dmg_tick) / 64.0
            if dmg_tick < retake_tick:
                status = f"{time_to_retake:.1f}s before retake"
            else:
                status = f"{abs(time_to_retake):.1f}s after retake"
            
            if thrower_zone != 'unknown' and zone != 'unknown':
                print(f"  {team}-HE: {thrower_zone} -> {zone} (plant+{dmg_time:.1f}s) [{status}]")
            elif zone != 'unknown':
                print(f"  {team}-HE: {zone} (plant+{dmg_time:.1f}s) [{status}]")
    
    num_smokes = len(active_utility['smokes'])
    num_mollies = len(active_utility['mollies'])
    num_flashes = len(active_utility['recent_flashes'])
    
    # DEBUG: Always show active utility status
    print(f"\n[DEBUG] Active utility check at retake:")
    print(f"  Found: {num_smokes} smokes, {num_mollies} mollies, {num_flashes} recent flashes")
    
    if num_smokes > 0 or num_mollies > 0 or num_flashes > 0:
        print(f"  Active utility at retake: {num_smokes} smokes, {num_mollies} mollies, {num_flashes} recent flashes")
        
        # Show details with expiration times
        for smoke in active_utility['smokes']:
            expire_in = (smoke['expire_tick'] - retake_tick) / 64
            print(f"    Smoke by {smoke['user_name']}: expires in {expire_in:.1f}s")
        for molly in active_utility['mollies']:
            expire_in = (molly['expire_tick'] - retake_tick) / 64
            print(f"    Molly by {molly['user_name']}: expires in {expire_in:.1f}s")
        for flash in active_utility['recent_flashes']:
            effect_remaining = (flash['effect_until_tick'] - retake_tick) / 64
            print(f"    Flash by {flash['user_name']}: effect for {effect_remaining:.1f}s more")
    
    print(f"{'='*80}")
    
    retakes_found += 1
    
    # Track CT episodes
    ct_analysis_start = retake_tick
    ct_analysis_end = retake_tick + 15*64
    
    # Track T episodes - full post-plant window for damage tracking
    t_analysis_start = plant_tick
    t_analysis_end = retake_tick + 15*64  # Track damage through full retake
    
    # Only track CTs who were ALIVE at retake start
    ct_players_alive = ticks[
        (ticks['tick'] == ct_analysis_start) & 
        (ticks['team_num'] == 3) & 
        (ticks['is_alive'] == True)
    ]['name'].unique()
    
    print(f"\nCT EPISODES (retake +15s):")
    for player in ct_players_alive:
        # Check if player died during this retake window
        player_death = deaths[
            (deaths['tick'] >= ct_analysis_start) &
            (deaths['tick'] <= ct_analysis_end) &
            (deaths['user_name'] == player)
        ]
        
        # If player died, only track up to death tick; otherwise full window
        if len(player_death) > 0:
            player_end_tick = player_death.iloc[0]['tick']
        else:
            player_end_tick = ct_analysis_end
        
        # Track ALL utility thrown during full retake window
        # For CT, we want to capture utility thrown just before/during retake
        # Use a buffer before retake start to catch setup utility
        ct_util_start = max(plant_tick, ct_analysis_start - 5*64)  # 5s buffer before retake
        all_utility_thrown = count_utility_thrown(
            grenades_thrown, smoke_detonate, smoke_expire,
            inferno_start, inferno_expire, flash_detonate,
            player, ct_util_start, player_end_tick,
            classifier, ticks, damage
        )
        
        player_ticks = ticks[
            (ticks['tick'] >= ct_analysis_start) &
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
        
        # Get CT player's HP at plant
        player_hp_at_plant = ct_health_dict.get(player, '?')
        
        current_zone = None
        zone_entry_tick = None
        episodes_this_player = []
        
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
                    
                    utility_thrown = count_utility_thrown(grenades_thrown, smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate, player, zone_entry_tick, pos['tick'], classifier, ticks, damage)
                    
                    episodes_this_player.append({
                        'zone': current_zone,
                        'duration_s': duration_ticks / 64.0,
                        'damage_taken_gun': damage_taken_gun,
                        'damage_taken_util': damage_taken_util,
                        'damage_dealt_gun': damage_dealt_gun,
                        'damage_dealt_util': damage_dealt_util,
                        'died': died,
                        'got_kill': got_kill,
                        'utility_thrown': utility_thrown
                    })
                
                current_zone = zone
                zone_entry_tick = pos['tick']
        
        # Final episode (only if player didn't die, or died in this zone)
        if current_zone and zone_entry_tick:
            duration_ticks = player_end_tick - zone_entry_tick
            episode_damage = damage[(damage['tick'] >= zone_entry_tick) & (damage['tick'] <= player_end_tick) & (damage['user_name'] == player)]
            damage_taken_gun, damage_taken_util = categorize_damage(episode_damage)
            
            episode_damage_dealt = damage[(damage['tick'] >= zone_entry_tick) & (damage['tick'] <= player_end_tick) & (damage['attacker_name'] == player)]
            damage_dealt_gun, damage_dealt_util = categorize_damage(episode_damage_dealt)
            
            episode_deaths = deaths[(deaths['tick'] >= zone_entry_tick) & (deaths['tick'] <= player_end_tick) & (deaths['user_name'] == player)]
            died = len(episode_deaths) > 0
            
            episode_kills = deaths[(deaths['tick'] >= zone_entry_tick) & (deaths['tick'] <= player_end_tick) & (deaths['attacker_name'] == player)]
            got_kill = len(episode_kills) > 0
            
            utility_thrown = count_utility_thrown(grenades_thrown, smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate, player, zone_entry_tick, player_end_tick, classifier, ticks, damage)
            
            episodes_this_player.append({
                'zone': current_zone,
                'duration_s': duration_ticks / 64.0,
                'damage_taken_gun': damage_taken_gun,
                'damage_taken_util': damage_taken_util,
                'damage_dealt_gun': damage_dealt_gun,
                'damage_dealt_util': damage_dealt_util,
                'died': died,
                'got_kill': got_kill,
                'utility_thrown': utility_thrown
            })
        
        if episodes_this_player:
            kit_info = ""
            if kit_pickup_tick:
                time_since_plant = (kit_pickup_tick - plant_tick) / 64.0
                kit_info = f" (picked up kit at plant+{time_since_plant:.1f}s)"
            elif player_had_kit_at_plant:
                kit_info = " (had kit)"
            
            print(f"\n  {player} ({player_hp_at_plant} HP){kit_info}:")
            
            # Use full-window utility tracking instead of per-episode accumulation
        # Check if player died during this retake window
        player_death = deaths[
            (deaths['tick'] >= ct_analysis_start) &
            (deaths['tick'] <= ct_analysis_end) &
            (deaths['user_name'] == player)
        ]
        
        # If player died, only track up to death tick; otherwise full window
        if len(player_death) > 0:
            player_end_tick = player_death.iloc[0]['tick']
        else:
            player_end_tick = ct_analysis_end
        
        # Track ALL utility thrown during full retake window
        # For CT, we want to capture utility thrown just before/during retake
        # Use a buffer before retake start to catch setup utility
        ct_util_start = max(plant_tick, ct_analysis_start - 5*64)  # 5s buffer before retake
        all_utility_thrown = count_utility_thrown(
            grenades_thrown, smoke_detonate, smoke_expire,
            inferno_start, inferno_expire, flash_detonate,
            player, ct_util_start, player_end_tick,
            classifier, ticks, damage
        )
        
        player_ticks = ticks[
            (ticks['tick'] >= ct_analysis_start) &
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
        
        # Get CT player's HP at plant
        player_hp_at_plant = ct_health_dict.get(player, '?')
        
        current_zone = None
        zone_entry_tick = None
        episodes_this_player = []
        
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
                    
                    utility_thrown = count_utility_thrown(grenades_thrown, smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate, player, zone_entry_tick, pos['tick'], classifier, ticks, damage)
                    
                    episodes_this_player.append({
                        'zone': current_zone,
                        'duration_s': duration_ticks / 64.0,
                        'damage_taken_gun': damage_taken_gun,
                        'damage_taken_util': damage_taken_util,
                        'damage_dealt_gun': damage_dealt_gun,
                        'damage_dealt_util': damage_dealt_util,
                        'died': died,
                        'got_kill': got_kill,
                        'utility_thrown': utility_thrown
                    })
                
                current_zone = zone
                zone_entry_tick = pos['tick']
        
        # Final episode (only if player didn't die, or died in this zone)
        if current_zone and zone_entry_tick:
            duration_ticks = player_end_tick - zone_entry_tick
            episode_damage = damage[(damage['tick'] >= zone_entry_tick) & (damage['tick'] <= player_end_tick) & (damage['user_name'] == player)]
            damage_taken_gun, damage_taken_util = categorize_damage(episode_damage)
            
            episode_damage_dealt = damage[(damage['tick'] >= zone_entry_tick) & (damage['tick'] <= player_end_tick) & (damage['attacker_name'] == player)]
            damage_dealt_gun, damage_dealt_util = categorize_damage(episode_damage_dealt)
            
            episode_deaths = deaths[(deaths['tick'] >= zone_entry_tick) & (deaths['tick'] <= player_end_tick) & (deaths['user_name'] == player)]
            died = len(episode_deaths) > 0
            
            episode_kills = deaths[(deaths['tick'] >= zone_entry_tick) & (deaths['tick'] <= player_end_tick) & (deaths['attacker_name'] == player)]
            got_kill = len(episode_kills) > 0
            
            utility_thrown = count_utility_thrown(grenades_thrown, smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate, player, zone_entry_tick, player_end_tick, classifier, ticks, damage)
            
            episodes_this_player.append({
                'zone': current_zone,
                'duration_s': duration_ticks / 64.0,
                'damage_taken_gun': damage_taken_gun,
                'damage_taken_util': damage_taken_util,
                'damage_dealt_gun': damage_dealt_gun,
                'damage_dealt_util': damage_dealt_util,
                'died': died,
                'got_kill': got_kill,
                'utility_thrown': utility_thrown
            })
        
        if episodes_this_player:
            print(f"\n  {player} ({player_hp_at_plant} HP):")
            
            # Use full-window utility tracking instead of per-episode accumulation
            util_by_zone = {}
            for util_type in ['smokes', 'flashes', 'he', 'mollies']:
                for util_item in all_utility_thrown[util_type]:
                    zone = util_item['zone']
                    key = f"{util_type}_{zone}"
                    util_by_zone[key] = util_by_zone.get(key, 0) + 1
            
            # Check if player died in ANY episode
            player_died = any(ep['died'] for ep in episodes_this_player)
            
            for idx, ep in enumerate(episodes_this_player):
                is_last_episode = (idx == len(episodes_this_player) - 1)
                
                outcome = []
                if ep['died']:
                    outcome.append("DIED")
                elif is_last_episode and not player_died:
                    outcome.append("survived")
                else:
                    outcome.append("alive")
                
                if ep['got_kill']:
                    outcome.append("KILL")
                outcome_str = ", ".join(outcome)
                
                # Show utility if any thrown in this episode (per-episode display)
                util_str = ""
                ep_util_parts = []
                for util_type in ['smokes', 'flashes', 'he', 'mollies']:
                    if len(ep['utility_thrown'][util_type]) > 0:
                        for util_item in ep['utility_thrown'][util_type]:
                            zone = util_item['zone']
                            team = util_item.get('team', '?')
                            util_name = f"{team}-{util_type[:-1]}"
                            
                            # Show FROM thrower_zone -> LANDS zone for all utility with thrower_zone
                            if 'thrower_zone' in util_item and util_item['thrower_zone'] != 'unknown':
                                thrower_zone = util_item['thrower_zone']
                                if zone != 'unknown':
                                    ep_util_parts.append(f"{util_name}:{thrower_zone}->{zone}")
                                else:
                                    ep_util_parts.append(f"{util_name}@{thrower_zone}->?")
                            else:
                                if zone != 'unknown':
                                    ep_util_parts.append(f"{util_name}@{zone}")
                                else:
                                    ep_util_parts.append(f"{util_name}@?")
                
                if ep_util_parts:
                    util_str = f"  util:[{','.join(ep_util_parts)}]"
                
                print(f"    {ep['zone']:<30s} {ep['duration_s']:>5.1f}s  gun_dmg:{ep['damage_taken_gun']:>3.0f}  util_dmg:{ep['damage_taken_util']:>3.0f}  gun_dlt:{ep['damage_dealt_gun']:>3.0f}  util_dlt:{ep['damage_dealt_util']:>3.0f}{util_str}  [{outcome_str}]")
            
            # Print detailed utility information
            if all_utility_thrown['smokes'] or all_utility_thrown['mollies'] or all_utility_thrown['he'] or all_utility_thrown['flashes']:
                print(f"    Utility thrown (full window):")
                for util_item in all_utility_thrown['smokes']:
                    team = util_item.get('team', '?')
                    thrower_zone = util_item.get('thrower_zone', 'unknown')
                    zone = util_item['zone']
                    tick = util_item.get('start_tick', util_item.get('detonate_tick', 0))
                    time_after_plant = (tick - plant_tick) / 64.0
                    if thrower_zone != 'unknown' and zone != 'unknown':
                        print(f"      {team}-smoke: {thrower_zone} -> {zone} at plant+{time_after_plant:.1f}s")
                    elif zone != 'unknown':
                        print(f"      {team}-smoke: landed {zone} at plant+{time_after_plant:.1f}s")
                    else:
                        print(f"      {team}-smoke: at plant+{time_after_plant:.1f}s")
                
                for util_item in all_utility_thrown['mollies']:
                    team = util_item.get('team', '?')
                    thrower_zone = util_item.get('thrower_zone', 'unknown')
                    zone = util_item['zone']
                    tick = util_item.get('start_tick', 0)
                    time_after_plant = (tick - plant_tick) / 64.0
                    if thrower_zone != 'unknown' and zone != 'unknown':
                        print(f"      {team}-molly: {thrower_zone} -> {zone} at plant+{time_after_plant:.1f}s")
                    elif zone != 'unknown':
                        print(f"      {team}-molly: landed {zone} at plant+{time_after_plant:.1f}s")
                    else:
                        print(f"      {team}-molly: at plant+{time_after_plant:.1f}s")
                
                for util_item in all_utility_thrown['he']:
                    team = util_item.get('team', '?')
                    thrower_zone = util_item.get('thrower_zone', 'unknown')
                    zone = util_item['zone']
                    tick = util_item.get('damage_tick', 0)
                    time_after_plant = (tick - plant_tick) / 64.0
                    if thrower_zone != 'unknown':
                        if zone != 'unknown':
                            print(f"      {team}-HE: {thrower_zone} -> {zone} at plant+{time_after_plant:.1f}s")
                        else:
                            print(f"      {team}-HE: {thrower_zone} -> (no hit) at plant+{time_after_plant:.1f}s")
                    elif zone != 'unknown':
                        print(f"      {team}-HE: landed {zone} at plant+{time_after_plant:.1f}s")
                
                for util_item in all_utility_thrown['flashes']:
                    team = util_item.get('team', '?')
                    thrower_zone = util_item.get('thrower_zone', 'unknown')
                    zone = util_item['zone']
                    tick = util_item.get('detonate_tick', 0)
                    time_after_plant = (tick - plant_tick) / 64.0
                    if thrower_zone != 'unknown' and zone != 'unknown':
                        print(f"      {team}-flash: {thrower_zone} -> {zone} at plant+{time_after_plant:.1f}s")
                    elif zone != 'unknown':
                        print(f"      {team}-flash: landed {zone} at plant+{time_after_plant:.1f}s")
            
            # Print total utility summary with zones
            if util_by_zone:
                summary_parts = []
                for k, v in util_by_zone.items():
                    util_type, zone = k.split('_', 1)
                    if zone == 'unknown':
                        summary_parts.append(f"{v} {util_type[:-1]} (no hit)")
                    else:
                        summary_parts.append(f"{v} {util_type[:-1]} @ {zone}")
                print(f"    Total utility thrown: {', '.join(summary_parts)}")
    
    # Track T episodes during post-plant (plant to retake start)
    t_players_alive = ticks[
        (ticks['tick'] == plant_tick) & 
        (ticks['team_num'] == 2) & 
        (ticks['is_alive'] == True)
    ]['name'].unique()
    
    print(f"\nT EPISODES (post-plant to retake):")
    for player in t_players_alive:
        # Check if player died during full post-plant window (including retake)
        player_death = deaths[
            (deaths['tick'] >= t_analysis_start) &
            (deaths['tick'] <= t_analysis_end) &
            (deaths['user_name'] == player)
        ]
        
        if len(player_death) > 0:
            player_end_tick = player_death.iloc[0]['tick']
        else:
            player_end_tick = t_analysis_end  # Track through full retake window
        
        # Track ALL utility thrown during full post-plant window (plant to retake end)
        # This captures pre-retake utility (e.g., torzsi molly at plant+13.4s before retake at plant+15.0s)
        all_utility_thrown = count_utility_thrown(
            grenades_thrown, smoke_detonate, smoke_expire, 
            inferno_start, inferno_expire, flash_detonate,
            player, plant_tick, player_end_tick, 
            classifier, ticks, damage
        )
        
        # Get T player's HP at plant
        player_hp_at_plant = t_health_dict.get(player, '?')
        
        player_ticks = ticks[
            (ticks['tick'] >= plant_tick) &
            (ticks['tick'] <= player_end_tick) &
            (ticks['name'] == player) &
            (ticks['is_alive'] == True)
        ].sort_values('tick')
        
        if len(player_ticks) == 0:
            continue
        
        current_zone = None
        zone_entry_tick = None
        episodes_this_player = []
        
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
                    
                    utility_thrown = count_utility_thrown(grenades_thrown, smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate, player, zone_entry_tick, pos['tick'], classifier, ticks, damage)
                    
                    episodes_this_player.append({
                        'zone': current_zone,
                        'duration_s': duration_ticks / 64.0,
                        'damage_taken_gun': damage_taken_gun,
                        'damage_taken_util': damage_taken_util,
                        'damage_dealt_gun': damage_dealt_gun,
                        'damage_dealt_util': damage_dealt_util,
                        'died': died,
                        'got_kill': got_kill,
                        'utility_thrown': utility_thrown
                    })
                
                current_zone = zone
                zone_entry_tick = pos['tick']
        
        # Final episode
        if current_zone and zone_entry_tick:
            duration_ticks = player_end_tick - zone_entry_tick
            episode_damage = damage[(damage['tick'] >= zone_entry_tick) & (damage['tick'] <= player_end_tick) & (damage['user_name'] == player)]
            damage_taken_gun, damage_taken_util = categorize_damage(episode_damage)
            
            episode_damage_dealt = damage[(damage['tick'] >= zone_entry_tick) & (damage['tick'] <= player_end_tick) & (damage['attacker_name'] == player)]
            damage_dealt_gun, damage_dealt_util = categorize_damage(episode_damage_dealt)
            
            episode_deaths = deaths[(deaths['tick'] >= zone_entry_tick) & (deaths['tick'] <= player_end_tick) & (deaths['user_name'] == player)]
            died = len(episode_deaths) > 0
            
            episode_kills = deaths[(deaths['tick'] >= zone_entry_tick) & (deaths['tick'] <= player_end_tick) & (deaths['attacker_name'] == player)]
            got_kill = len(episode_kills) > 0
            
            utility_thrown = count_utility_thrown(grenades_thrown, smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate, player, zone_entry_tick, player_end_tick, classifier, ticks, damage)
            
            episodes_this_player.append({
                'zone': current_zone,
                'duration_s': duration_ticks / 64.0,
                'damage_taken_gun': damage_taken_gun,
                'damage_taken_util': damage_taken_util,
                'damage_dealt_gun': damage_dealt_gun,
                'damage_dealt_util': damage_dealt_util,
                'died': died,
                'got_kill': got_kill,
                'utility_thrown': utility_thrown
            })
        
        if episodes_this_player:
            print(f"\n  {player} ({player_hp_at_plant} HP):")
            
            # Use full-window utility tracking instead of per-episode accumulation
            util_by_zone = {}
            for util_type in ['smokes', 'flashes', 'he', 'mollies']:
                for util_item in all_utility_thrown[util_type]:
                    zone = util_item['zone']
                    key = f"{util_type}_{zone}"
                    util_by_zone[key] = util_by_zone.get(key, 0) + 1
            
            # Check if player died in ANY episode
            player_died = any(ep['died'] for ep in episodes_this_player)
            
            for idx, ep in enumerate(episodes_this_player):
                is_last_episode = (idx == len(episodes_this_player) - 1)
                
                outcome = []
                if ep['died']:
                    outcome.append("DIED")
                elif is_last_episode and not player_died:
                    outcome.append("survived")
                else:
                    outcome.append("alive")
                
                if ep['got_kill']:
                    outcome.append("KILL")
                outcome_str = ", ".join(outcome)
                
                # Show utility if any thrown in this episode (per-episode display)
                util_str = ""
                ep_util_parts = []
                for util_type in ['smokes', 'flashes', 'he', 'mollies']:
                    if len(ep['utility_thrown'][util_type]) > 0:
                        for util_item in ep['utility_thrown'][util_type]:
                            zone = util_item['zone']
                            team = util_item.get('team', '?')
                            util_name = f"{team}-{util_type[:-1]}"
                            
                            # Show FROM thrower_zone -> LANDS zone for all utility with thrower_zone
                            if 'thrower_zone' in util_item and util_item['thrower_zone'] != 'unknown':
                                thrower_zone = util_item['thrower_zone']
                                if zone != 'unknown':
                                    ep_util_parts.append(f"{util_name}:{thrower_zone}->{zone}")
                                else:
                                    ep_util_parts.append(f"{util_name}@{thrower_zone}->?")
                            else:
                                if zone != 'unknown':
                                    ep_util_parts.append(f"{util_name}@{zone}")
                                else:
                                    ep_util_parts.append(f"{util_name}@?")
                
                if ep_util_parts:
                    util_str = f"  util:[{','.join(ep_util_parts)}]"
                
                print(f"    {ep['zone']:<30s} {ep['duration_s']:>5.1f}s  gun_dmg:{ep['damage_taken_gun']:>3.0f}  util_dmg:{ep['damage_taken_util']:>3.0f}  gun_dlt:{ep['damage_dealt_gun']:>3.0f}  util_dlt:{ep['damage_dealt_util']:>3.0f}{util_str}  [{outcome_str}]")
            
            # Print detailed utility information
            if all_utility_thrown['smokes'] or all_utility_thrown['mollies'] or all_utility_thrown['he'] or all_utility_thrown['flashes']:
                print(f"    Utility thrown (full window):")
                for util_item in all_utility_thrown['smokes']:
                    team = util_item.get('team', '?')
                    thrower_zone = util_item.get('thrower_zone', 'unknown')
                    zone = util_item['zone']
                    tick = util_item.get('start_tick', util_item.get('detonate_tick', 0))
                    time_after_plant = (tick - plant_tick) / 64.0
                    if thrower_zone != 'unknown' and zone != 'unknown':
                        print(f"      {team}-smoke: {thrower_zone} -> {zone} at plant+{time_after_plant:.1f}s")
                    elif zone != 'unknown':
                        print(f"      {team}-smoke: landed {zone} at plant+{time_after_plant:.1f}s")
                    else:
                        print(f"      {team}-smoke: at plant+{time_after_plant:.1f}s")
                
                for util_item in all_utility_thrown['mollies']:
                    team = util_item.get('team', '?')
                    thrower_zone = util_item.get('thrower_zone', 'unknown')
                    zone = util_item['zone']
                    tick = util_item.get('start_tick', 0)
                    time_after_plant = (tick - plant_tick) / 64.0
                    if thrower_zone != 'unknown' and zone != 'unknown':
                        print(f"      {team}-molly: {thrower_zone} -> {zone} at plant+{time_after_plant:.1f}s")
                    elif zone != 'unknown':
                        print(f"      {team}-molly: landed {zone} at plant+{time_after_plant:.1f}s")
                    else:
                        print(f"      {team}-molly: at plant+{time_after_plant:.1f}s")
                
                for util_item in all_utility_thrown['he']:
                    team = util_item.get('team', '?')
                    thrower_zone = util_item.get('thrower_zone', 'unknown')
                    zone = util_item['zone']
                    tick = util_item.get('damage_tick', 0)
                    time_after_plant = (tick - plant_tick) / 64.0
                    if thrower_zone != 'unknown':
                        if zone != 'unknown':
                            print(f"      {team}-HE: {thrower_zone} -> {zone} at plant+{time_after_plant:.1f}s")
                        else:
                            print(f"      {team}-HE: {thrower_zone} -> (no hit) at plant+{time_after_plant:.1f}s")
                    elif zone != 'unknown':
                        print(f"      {team}-HE: landed {zone} at plant+{time_after_plant:.1f}s")
                
                for util_item in all_utility_thrown['flashes']:
                    team = util_item.get('team', '?')
                    thrower_zone = util_item.get('thrower_zone', 'unknown')
                    zone = util_item['zone']
                    tick = util_item.get('detonate_tick', 0)
                    time_after_plant = (tick - plant_tick) / 64.0
                    if thrower_zone != 'unknown' and zone != 'unknown':
                        print(f"      {team}-flash: {thrower_zone} -> {zone} at plant+{time_after_plant:.1f}s")
                    elif zone != 'unknown':
                        print(f"      {team}-flash: landed {zone} at plant+{time_after_plant:.1f}s")
            
            # Print total utility summary with zones
            if util_by_zone:
                summary_parts = []
                for k, v in util_by_zone.items():
                    util_type, zone = k.split('_', 1)
                    if zone == 'unknown':
                        summary_parts.append(f"{v} {util_type[:-1]} (no hit)")
                    else:
                        summary_parts.append(f"{v} {util_type[:-1]} @ {zone}")
                print(f"    Total utility thrown: {', '.join(summary_parts)}")

print(f"\n{'='*80}")
print(f"Episode tracking working. Found {retakes_found} retakes")
print(f"{'='*80}")
