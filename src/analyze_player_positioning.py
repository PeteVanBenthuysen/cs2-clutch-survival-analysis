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
from collections import defaultdict
import json

print("="*80)
print("ANALYZING POST-PLANT POSITIONING EPISODES FROM ALL DEMOS")
print("="*80)

classifier = MirageZoneClassifier()
connectivity = ZoneConnectivity()

# Utility weapons for damage categorization
UTILITY_WEAPONS = {'hegrenade', 'molotov', 'incgrenade', 'inferno'}

def categorize_damage(damage_df):
    """Separate gun damage from utility damage for economy/hazard modeling."""
    if len(damage_df) == 0:
        return 0, 0
    
    gun_dmg = damage_df[~damage_df['weapon'].isin(UTILITY_WEAPONS)]['dmg_health'].sum()
    util_dmg = damage_df[damage_df['weapon'].isin(UTILITY_WEAPONS)]['dmg_health'].sum()
    return gun_dmg, util_dmg

def count_utility_thrown(grenades_df, player_name, tick_start, tick_end):
    """Count utility grenades thrown by player in tick window."""
    player_nades = grenades_df[
        (grenades_df['user_name'] == player_name) &
        (grenades_df['tick'] >= tick_start) &
        (grenades_df['tick'] < tick_end)
    ]
    
    return {
        'smoke': len(player_nades[player_nades['weapon'] == 'smokegrenade']),
        'flash': len(player_nades[player_nades['weapon'] == 'flashbang']),
        'he': len(player_nades[player_nades['weapon'] == 'hegrenade']),
        'molly': len(player_nades[player_nades['weapon'].isin(['molotov', 'incgrenade'])])
    }

def get_active_utility_at_tick(smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate, tick, window_before=5*64):
    """Find smokes/mollies/flashes active at given tick (for pre-plant utility still burning/active).
    
    Args:
        smoke_detonate: DataFrame of smoke detonations
        smoke_expire: DataFrame of smoke expirations  
        inferno_start: DataFrame of molly/incendiary starts
        inferno_expire: DataFrame of molly/incendiary ends
        flash_detonate: DataFrame of flashbang detonations
        tick: The tick to check (e.g., retake start)
        window_before: How many ticks before to look for pre-existing utility (default 5s)
    
    Returns:
        dict with detailed utility info:
        {
            'smokes': [{user_name, detonate_tick, expire_tick, zone}, ...],
            'mollies': [{user_name, start_tick, expire_tick, zone}, ...],
            'recent_flashes': [{user_name, detonate_tick, effect_until_tick}, ...]  # Last 3s
        }
    """
    # Find smokes that detonated before tick but expire after tick
    active_smokes_data = []
    active_smokes = smoke_detonate[
        (smoke_detonate['tick'] < tick) &
        (smoke_detonate['tick'] >= tick - window_before)
    ]
    
    for _, smoke in active_smokes.iterrows():
        smoke_user = smoke['user_name']
        smoke_tick = smoke['tick']
        
        # Find when this smoke expires
        expire_event = smoke_expire[
            (smoke_expire['user_name'] == smoke_user) &
            (smoke_expire['tick'] > smoke_tick)
        ].sort_values('tick')
        
        if len(expire_event) > 0:
            expire_tick = expire_event.iloc[0]['tick']
            # Only include if still active at target tick
            if expire_tick > tick:
                active_smokes_data.append({
                    'user_name': smoke_user,
                    'detonate_tick': smoke_tick,
                    'expire_tick': expire_tick,
                    'X': smoke.get('X', 0),
                    'Y': smoke.get('Y', 0),
                    'Z': smoke.get('Z', 0)
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
                active_mollies_data.append({
                    'user_name': molly_user,
                    'start_tick': molly_tick,
                    'expire_tick': expire_tick,
                    'X': molly.get('X', 0),
                    'Y': molly.get('Y', 0),
                    'Z': molly.get('Z', 0)
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

# Track player episodes: list of {player, zone, duration, damage_taken, damage_dealt, died, got_kill}
ct_episodes = []  # CT episodes during retakes
t_episodes = []   # T episodes during post-plant
a_site_retakes = 0
b_site_retakes = 0

# Define bomb site and approach zones
# B-site zones include: B_ prefix, Cat_ prefix, plus Market/Arches/Bench/Van/Dark areas
b_site_keywords = ['B_', 'Cat_', 'Market', 'Arches', 'Bench', 'Van_', 'B_Short', 'Dark', 'Backsite']

# Approach zones = zones that connect to bomb site (1-2 steps away)
# For A-site: Connector, Stairs, Jungle, CT spawn areas
a_approach_keywords = ['Connector', 'Stairs', 'Jungle', 'CT', 'Palace', 'Ramp']
# For B-site: Mid, Underpass, CT areas approaching B  
b_approach_keywords = ['Mid', 'Underpass', 'CT', 'Kitchen', 'Short']

def is_b_zone(zone):
    if not zone:
        return False
    return any(zone.startswith(kw) or kw in zone for kw in b_site_keywords)

def is_approach_zone(zone, is_b_plant):
    if not zone:
        return False
    keywords = b_approach_keywords if is_b_plant else a_approach_keywords
    return any(kw in zone for kw in keywords)

def check_ct_movement_toward_site(ticks, classifier, dmg_tick, plant_tick, is_b_plant, window_ticks=64):
    """Check if CTs are moving TOWARD bomb site (not saving).
    
    Returns: (num_moving_toward, num_moving_away, num_stationary)
    """
    # Get CT positions before and after damage event
    before_tick = max(plant_tick, dmg_tick - window_ticks)
    after_tick = dmg_tick
    
    ct_movements = []
    
    # Track each CT's movement
    ct_names = ticks[
        (ticks['tick'] == dmg_tick) &
        (ticks['team_num'] == 3) &
        (ticks['is_alive'] == True)
    ]['name'].unique()
    
    for ct_name in ct_names:
        before_pos = ticks[
            (ticks['tick'] == before_tick) &
            (ticks['name'] == ct_name)
        ]
        after_pos = ticks[
            (ticks['tick'] == after_tick) &
            (ticks['name'] == ct_name)
        ]
        
        if len(before_pos) == 0 or len(after_pos) == 0:
            continue
        
        before = before_pos.iloc[0]
        after = after_pos.iloc[0]
        
        before_zone = classifier.point_in_zone_A(before['X'], before['Y'], before['Z'])
        after_zone = classifier.point_in_zone_A(after['X'], after['Y'], after['Z'])
        
        if not before_zone or not after_zone:
            continue
        
        # Check if moved closer to or further from bomb site
        before_is_site = is_b_zone(before_zone) if is_b_plant else (before_zone and not is_b_zone(before_zone))
        after_is_site = is_b_zone(after_zone) if is_b_plant else (after_zone and not is_b_zone(after_zone))
        
        before_is_approach = is_approach_zone(before_zone, is_b_plant)
        after_is_approach = is_approach_zone(after_zone, is_b_plant)
        
        # Determine movement direction
        if not before_is_site and not before_is_approach and (after_is_site or after_is_approach):
            ct_movements.append('toward')  # Moved closer to site
        elif (before_is_site or before_is_approach) and not after_is_site and not after_is_approach:
            ct_movements.append('away')  # Moved away from site (saving)
        else:
            ct_movements.append('stationary')  # No clear direction change
    
    num_toward = ct_movements.count('toward')
    num_away = ct_movements.count('away')
    num_stationary = ct_movements.count('stationary')
    
    return num_toward, num_away, num_stationary

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
        
        # Get player positions
        ticks = parser.parse_ticks(['X', 'Y', 'Z', 'name', 'tick', 'team_num', 'is_alive'])
        
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
            
            # Get round winner - find the NEXT round_end after plant (tick-based, not round number)
            round_end = round_ends[round_ends['tick'] > plant_tick].sort_values('tick')
            winner_team = round_end.iloc[0]['winner'] if len(round_end) > 0 else None
            
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
            
            # Check if any CTs are alive at plant time (if all dead, can't have retake)
            alive_cts = ticks[
                (ticks['tick'] == plant_tick) &
                (ticks['team_num'] == 3) &
                (ticks['is_alive'] == True)
            ]
            
            if len(alive_cts) == 0:
                continue  # Skip - all CTs dead at plant, no retake possible
            
            # Find retake start: FIRST ENGAGEMENT between CTs and Ts post-plant
            # This is when retake actually happens, not just when CTs start rotating
            
            retake_tick = None
            
            # Get all damage in post-plant window
            postplant_damage = damage[
                (damage['tick'] >= plant_tick) &
                (damage['tick'] <= plant_tick + 30*64)
            ]
            
            # Find earliest damage event involving CT/T interaction in/near A-site
            earliest_tick = plant_tick + 30*64  # Default to max window
            
            for _, dmg in postplant_damage.iterrows():
                dmg_tick = dmg['tick']
                
                # Check if attacker or victim is in A-site
                attacker_pos = ticks[
                    (ticks['tick'] == dmg_tick) &
                    (ticks['name'] == dmg['attacker_name'])
                ]
                victim_pos = ticks[
                    (ticks['tick'] == dmg_tick) &
                    (ticks['name'] == dmg['user_name'])
                ]
                
                if len(attacker_pos) == 0 or len(victim_pos) == 0:
                    continue
                    
                att = attacker_pos.iloc[0]
                vic = victim_pos.iloc[0]
                
                # Check if this is CT vs T engagement
                if (att['team_num'] == 3 and vic['team_num'] == 2) or (att['team_num'] == 2 and vic['team_num'] == 3):
                    # Get zones for both players
                    att_zone = classifier.point_in_zone_A(att['X'], att['Y'], att['Z'])
                    vic_zone = classifier.point_in_zone_A(vic['X'], vic['Y'], vic['Z'])
                    
                    att_is_b = is_b_zone(att_zone)
                    vic_is_b = is_b_zone(vic_zone)
                    
                    # Check if engagement is ON bomb site
                    if is_b_site:
                        on_site = att_is_b or vic_is_b
                    else:
                        on_site = (att_zone and not att_is_b) or (vic_zone and not vic_is_b)
                    
                    # Check if engagement is in APPROACH zones
                    att_approach = is_approach_zone(att_zone, is_b_site)
                    vic_approach = is_approach_zone(vic_zone, is_b_site)
                    in_approach = att_approach or vic_approach
                    
                    # Count alive CTs at this moment
                    alive_cts_now = ticks[
                        (ticks['tick'] == dmg_tick) &
                        (ticks['team_num'] == 3) &
                        (ticks['is_alive'] == True)
                    ]
                    num_cts_alive = len(alive_cts_now['name'].unique())
                    
                    # Check CT movement direction (are they pushing toward site or saving?)
                    num_toward, num_away, num_stationary = check_ct_movement_toward_site(
                        ticks, classifier, dmg_tick, plant_tick, is_b_site
                    )
                    
                    # Retake criteria - focus on INTENT:
                    # 1. ON-SITE engagement = retake (CT is at/defending the bomb)
                    # 2. APPROACH zone engagement with CTs moving TOWARD site (not away) = retake
                    # 3. Filter out if CTs clearly saving (more moving away than toward)
                    
                    if on_site:
                        # On-site engagement is always a retake unless CTs are fleeing
                        is_retake_engagement = (num_away == 0) or (num_toward >= num_away)
                    else:
                        # Approach zone: only retake if CTs are actively pushing toward site
                        is_retake_engagement = in_approach and (num_toward > num_away)
                    
                    if is_retake_engagement:
                        if dmg_tick < earliest_tick:
                            earliest_tick = dmg_tick
                        break
            
            if earliest_tick >= plant_tick + 30*64:
                continue  # No retake detected (save round)
            
            retake_tick = earliest_tick
            
            # Get active utility at retake start (smokes/mollies/flashes already active from pre-plant)
            active_utility = get_active_utility_at_tick(
                smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate,
                retake_tick, window_before=20*64  # Look 20 seconds back (smokes last ~18s)
            )
            
            total_retakes += 1
            if is_b_site:
                b_site_retakes += 1
            else:
                a_site_retakes += 1
            
            # Track CT episodes during retake (retake start + 15s)
            analysis_start = retake_tick
            analysis_end = retake_tick + 15*64
            
            # Only track CTs who were ALIVE at retake start
            ct_players = ticks[
                (ticks['tick'] == analysis_start) &
                (ticks['team_num'] == 3) &
                (ticks['is_alive'] == True)
            ]['name'].unique()
            
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
                
                player_ticks = ticks[
                    (ticks['tick'] >= analysis_start) &
                    (ticks['tick'] <= player_end_tick) &
                    (ticks['name'] == player) &
                    (ticks['is_alive'] == True)
                ].sort_values('tick')
                
                if len(player_ticks) == 0:
                    continue
                
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
                            
                            utility_thrown = count_utility_thrown(grenades_thrown, player, zone_entry_tick, pos['tick'])
                            
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
                                'round_winner': winner_team,
                                'utility_thrown': utility_thrown
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
                    
                    utility_thrown = count_utility_thrown(grenades_thrown, player, zone_entry_tick, player_end_tick)
                    
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
                        'round_winner': winner_team,
                        'utility_thrown': utility_thrown
                    })
            
            # Track T episodes during post-plant through retake window
            # Only track Ts who were ALIVE at plant time
            t_analysis_end = retake_tick + 15*64  # Track through full retake for damage dealt
            
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
                            
                            utility_thrown = count_utility_thrown(grenades_thrown, player, zone_entry_tick, pos['tick'])
                            
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
                                'round_winner': winner_team,
                                'utility_thrown': utility_thrown
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
                    
                    utility_thrown = count_utility_thrown(grenades_thrown, player, zone_entry_tick, player_end_tick)
                    
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
                        'round_winner': winner_team,
                        'utility_thrown': utility_thrown
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

# Save to JSON
output = {
    "total_retakes_analyzed": total_retakes,
    "a_site_retakes": a_site_retakes,
    "b_site_retakes": b_site_retakes,
    "total_demos_processed": demos_processed,
    "ct_retake_episodes": {
        "total_episodes": len(ct_episodes),
        "zone_stats": {zone: stats for zone, stats in ct_zone_stats.items()},
        "description": "CT positioning episodes during retakes (15s window). Each episode = player in zone with duration, damage, outcomes."
    },
    "t_postplant_episodes": {
        "total_episodes": len(t_episodes),
        "zone_stats": {zone: stats for zone, stats in t_zone_stats.items()},
        "description": "T positioning episodes after plant (plant to retake start). Each episode = player in zone with duration, damage, outcomes."
    },
    "methodology": "Episode-based tracking: each player's time in each zone is tracked with damage taken/dealt, kills, deaths. Enables hazard rate modeling.",
    "description": "Empirical post-plant positioning episodes for hazard modeling, based on analysis of 582 pro demos"
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
