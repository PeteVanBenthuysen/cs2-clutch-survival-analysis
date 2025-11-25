"""
Test episode-based positioning tracking on mouz vs BIG game.
Shows individual player episodes with damage, kills, deaths.
"""

from pathlib import Path
from demoparser2 import DemoParser
from src.zone_classifier import MirageZoneClassifier
from src.zone_connectivity import ZoneConnectivity
from collections import defaultdict

demo_path = Path("research_demos/extracted/8036_IEM_Melbourne_2025/2381637/mouz-vs-big-m2-mirage.dem")

print("="*80)
print("TESTING EPISODE TRACKING - mouz vs BIG Map 2")
print("="*80)

classifier = MirageZoneClassifier()
connectivity = ZoneConnectivity()
parser = DemoParser(str(demo_path))

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
            expire_tick = expire_event.iloc[0]['tick']\
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

# Get all events
bomb_plants = parser.parse_event('bomb_planted')
ticks = parser.parse_ticks(['X', 'Y', 'Z', 'name', 'tick', 'team_num', 'is_alive'])
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

# Zone keywords
b_site_keywords = ['B_', 'Cat_', 'Market', 'Arches', 'Bench', 'Van_', 'B_Short', 'Dark', 'Backsite']
a_approach_keywords = ['Connector', 'Stairs', 'Jungle', 'CT', 'Palace', 'Ramp']
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

def check_ct_movement_toward_site(dmg_tick, plant_tick, is_b_plant, window_ticks=64):
    before_tick = max(plant_tick, dmg_tick - window_ticks)
    after_tick = dmg_tick
    
    ct_movements = []
    ct_names = ticks[
        (ticks['tick'] == dmg_tick) &
        (ticks['team_num'] == 3) &
        (ticks['is_alive'] == True)
    ]['name'].unique()
    
    for ct_name in ct_names:
        before_pos = ticks[(ticks['tick'] == before_tick) & (ticks['name'] == ct_name)]
        after_pos = ticks[(ticks['tick'] == after_tick) & (ticks['name'] == ct_name)]
        
        if len(before_pos) == 0 or len(after_pos) == 0:
            continue
        
        before = before_pos.iloc[0]
        after = after_pos.iloc[0]
        
        before_zone = classifier.point_in_zone_A(before['X'], before['Y'], before['Z'])
        after_zone = classifier.point_in_zone_A(after['X'], after['Y'], after['Z'])
        
        if not before_zone or not after_zone:
            continue
        
        before_is_site = is_b_zone(before_zone) if is_b_plant else (before_zone and not is_b_zone(before_zone))
        after_is_site = is_b_zone(after_zone) if is_b_plant else (after_zone and not is_b_zone(after_zone))
        
        before_is_approach = is_approach_zone(before_zone, is_b_plant)
        after_is_approach = is_approach_zone(after_zone, is_b_plant)
        
        if not before_is_site and not before_is_approach and (after_is_site or after_is_approach):
            ct_movements.append('toward')
        elif (before_is_site or before_is_approach) and not after_is_site and not after_is_approach:
            ct_movements.append('away')
        else:
            ct_movements.append('stationary')
    
    return ct_movements.count('toward'), ct_movements.count('away'), ct_movements.count('stationary')

ct_episodes = []
t_episodes = []
retakes_found = 0

# Just analyze first 3 retakes for testing
for idx, plant in bomb_plants.iterrows():
    if retakes_found >= 3:
        break
        
    plant_tick = plant['tick']
    round_num = len(round_starts[round_starts['tick'] <= plant_tick])
    
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
    
    alive_cts = ticks[(ticks['tick'] == plant_tick) & (ticks['team_num'] == 3) & (ticks['is_alive'] == True)]
    if len(alive_cts) == 0:
        continue
    
    # Find retake
    postplant_damage = damage[(damage['tick'] >= plant_tick) & (damage['tick'] <= plant_tick + 30*64)]
    earliest_tick = plant_tick + 30*64
    
    for _, dmg in postplant_damage.iterrows():
        dmg_tick = dmg['tick']
        
        attacker_pos = ticks[(ticks['tick'] == dmg_tick) & (ticks['name'] == dmg['attacker_name'])]
        victim_pos = ticks[(ticks['tick'] == dmg_tick) & (ticks['name'] == dmg['user_name'])]
        
        if len(attacker_pos) == 0 or len(victim_pos) == 0:
            continue
            
        att = attacker_pos.iloc[0]
        vic = victim_pos.iloc[0]
        
        if (att['team_num'] == 3 and vic['team_num'] == 2) or (att['team_num'] == 2 and vic['team_num'] == 3):
            att_zone = classifier.point_in_zone_A(att['X'], att['Y'], att['Z'])
            vic_zone = classifier.point_in_zone_A(vic['X'], vic['Y'], vic['Z'])
            
            att_is_b = is_b_zone(att_zone)
            vic_is_b = is_b_zone(vic_zone)
            
            if is_b_site:
                on_site = att_is_b or vic_is_b
            else:
                on_site = (att_zone and not att_is_b) or (vic_zone and not vic_is_b)
            
            att_approach = is_approach_zone(att_zone, is_b_site)
            vic_approach = is_approach_zone(vic_zone, is_b_site)
            in_approach = att_approach or vic_approach
            
            num_toward, num_away, num_stationary = check_ct_movement_toward_site(dmg_tick, plant_tick, is_b_site)
            
            if on_site:
                is_retake_engagement = (num_away == 0) or (num_toward >= num_away)
            else:
                is_retake_engagement = in_approach and (num_toward > num_away)
            
            if is_retake_engagement:
                if dmg_tick < earliest_tick:
                    earliest_tick = dmg_tick
                break
    
    if earliest_tick >= plant_tick + 30*64:
        continue
    
    retake_tick = earliest_tick
    time_to_retake = (retake_tick - plant_tick) / 64.0
    
    # Check for active utility at retake start (smokes/mollies/flashes from pre-plant still active)
    active_utility = get_active_utility_at_tick(
        smoke_detonate, smoke_expire, inferno_start, inferno_expire, flash_detonate,
        retake_tick, window_before=20*64  # Look 20 seconds back (smokes last ~18s)
    )
    
    winner_str = f"{winner_team} WIN" if winner_team else "Unknown"
    print(f"\n{'='*80}")
    print(f"Round {round_num}: {site_type} - Retake at +{time_to_retake:.1f}s - {winner_str}")
    
    num_smokes = len(active_utility['smokes'])
    num_mollies = len(active_utility['mollies'])
    num_flashes = len(active_utility['recent_flashes'])
    
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
        
        player_ticks = ticks[
            (ticks['tick'] >= ct_analysis_start) &
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
                    
                    utility_thrown = count_utility_thrown(grenades_thrown, player, zone_entry_tick, pos['tick'])
                    
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
            
            utility_thrown = count_utility_thrown(grenades_thrown, player, zone_entry_tick, player_end_tick)
            
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
            print(f"\n  {player}:")
            for ep in episodes_this_player:
                outcome = []
                if ep['died']:
                    outcome.append("DIED")
                if ep['got_kill']:
                    outcome.append("KILL")
                outcome_str = ", ".join(outcome) if outcome else "survived"
                print(f"    {ep['zone']:<30s} {ep['duration_s']:>5.1f}s  gun_dmg:{ep['damage_taken_gun']:>3.0f}  util_dmg:{ep['damage_taken_util']:>3.0f}  gun_dlt:{ep['damage_dealt_gun']:>3.0f}  util_dlt:{ep['damage_dealt_util']:>3.0f}  [{outcome_str}]")
    
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
                    
                    utility_thrown = count_utility_thrown(grenades_thrown, player, zone_entry_tick, pos['tick'])
                    
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
            
            utility_thrown = count_utility_thrown(grenades_thrown, player, zone_entry_tick, player_end_tick)
            
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
            print(f"\n  {player}:")
            for ep in episodes_this_player:
                outcome = []
                if ep['died']:
                    outcome.append("DIED")
                if ep['got_kill']:
                    outcome.append("KILL")
                outcome_str = ", ".join(outcome) if outcome else "survived"
                print(f"    {ep['zone']:<30s} {ep['duration_s']:>5.1f}s  gun_dmg:{ep['damage_taken_gun']:>3.0f}  util_dmg:{ep['damage_taken_util']:>3.0f}  gun_dlt:{ep['damage_dealt_gun']:>3.0f}  util_dlt:{ep['damage_dealt_util']:>3.0f}  [{outcome_str}]")

print(f"\n{'='*80}")
print(f"Episode tracking working! Found {retakes_found} retakes")
print(f"{'='*80}")
