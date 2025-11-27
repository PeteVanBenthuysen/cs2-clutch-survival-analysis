"""
Detect retake attempts in CS2 post-plant scenarios.
Identifies when CTs actively push toward bomb site (not exit hunting).
"""

# Zone keywords
b_site_keywords = ['B_', 'Cat_', 'Market', 'Arches', 'Bench', 'Van_', 'B_Short', 'Dark', 'Backsite']
a_site_keywords = ['A_', 'Stairs', 'Palace', 'Ramp', 'Jungle', 'Tetris', 'Shadow', 'Ninja', 'Firebox', 'Triple']
a_approach_keywords = ['Connector', 'Stairs', 'Jungle', 'CT', 'Palace', 'Ramp']
b_approach_keywords = ['Mid', 'Underpass', 'CT', 'Kitchen', 'Short']

def is_b_zone(zone):
    if not zone:
        return False
    return any(zone.startswith(kw) or kw in zone for kw in b_site_keywords)

def is_a_zone(zone):
    if not zone:
        return False
    return any(zone.startswith(kw) or kw in zone for kw in a_site_keywords)

def is_approach_zone(zone, is_b_plant):
    if not zone:
        return False
    keywords = b_approach_keywords if is_b_plant else a_approach_keywords
    return any(kw in zone for kw in keywords)

def check_ct_movement_toward_site(ticks, classifier, dmg_tick, plant_tick, is_b_plant, window_ticks=192):
    """Check if CTs are moving TOWARD bomb site (not saving).
    
    Args:
        window_ticks: Window size in ticks (default 192 = 3 seconds, increased from 64/1s to catch repositioning)
    
    Returns: (num_moving_toward, num_moving_away, num_stationary)
    """
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
        
        before_is_site = is_b_zone(before_zone) if is_b_plant else is_a_zone(before_zone)
        after_is_site = is_b_zone(after_zone) if is_b_plant else is_a_zone(after_zone)
        
        before_is_approach = is_approach_zone(before_zone, is_b_plant)
        after_is_approach = is_approach_zone(after_zone, is_b_plant)
        
        if not before_is_site and not before_is_approach and (after_is_site or after_is_approach):
            ct_movements.append('toward')
        elif (before_is_site or before_is_approach) and not after_is_site and not after_is_approach:
            ct_movements.append('away')
        else:
            ct_movements.append('stationary')
    
    return ct_movements.count('toward'), ct_movements.count('away'), ct_movements.count('stationary')


def detect_retake(plant_tick, bomb_zone, ticks, damage, classifier, round_end_tick=None):
    """Detect if a retake occurred after bomb plant.
    
    Args:
        plant_tick: Tick when bomb was planted
        bomb_zone: Zone where bomb was planted (e.g., 'A_Default', 'B_Apps')
        ticks: DataFrame with player positions
        damage: DataFrame with damage events
        classifier: Zone classifier instance
        round_end_tick: Optional round end tick (defaults to plant_tick + 30*64)
    
    Returns:
        dict with:
            - 'retake_detected': bool
            - 'retake_tick': int or None (tick when retake started)
            - 'time_to_retake': float or None (seconds after plant)
    """
    if round_end_tick is None:
        round_end_tick = plant_tick + 30*64
    
    is_b_site = bomb_zone.startswith('B_') or bomb_zone.startswith('Cat_')
    
    # Check if any CTs are alive at plant time
    alive_cts = ticks[(ticks['tick'] == plant_tick) & (ticks['team_num'] == 3) & (ticks['is_alive'] == True)]
    if len(alive_cts) == 0:
        return {'retake_detected': False, 'retake_tick': None, 'time_to_retake': None}
    
    # Find retake - look for CT vs T engagement post-plant
    postplant_damage = damage[(damage['tick'] >= plant_tick) & (damage['tick'] <= plant_tick + 30*64)]
    earliest_tick = plant_tick + 30*64
    retake_found = False
    
    # OPTIMIZATION: Pre-filter ticks to only the relevant time window
    # This prevents thousands of full-DataFrame filters in the loop below
    relevant_ticks = ticks[(ticks['tick'] >= plant_tick) & (ticks['tick'] <= plant_tick + 30*64)]
    
    for _, dmg in postplant_damage.iterrows():
        dmg_tick = dmg['tick']
        
        attacker_pos = relevant_ticks[(relevant_ticks['tick'] == dmg_tick) & (relevant_ticks['name'] == dmg['attacker_name'])]
        victim_pos = relevant_ticks[(relevant_ticks['tick'] == dmg_tick) & (relevant_ticks['name'] == dmg['user_name'])]
        
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
            
            num_toward, num_away, num_stationary = check_ct_movement_toward_site(relevant_ticks, classifier, dmg_tick, plant_tick, is_b_site)
            
            # Require at least one CT actively moving toward site (not just exit hunting)
            # BUT: If fight is already ON SITE, that's sufficient evidence of retake even without movement
            if on_site:
                # On-site fight: accept if not clearly exit hunting (allow stationary, just block pure away movement)
                is_retake_engagement = (num_away == 0) or (num_toward > 0) or (num_stationary > 0)
            else:
                # Approach zone fight: require clear movement toward site
                is_retake_engagement = in_approach and (num_toward > 0) and (num_toward > num_away)
            
            if is_retake_engagement:
                if dmg_tick < earliest_tick:
                    earliest_tick = dmg_tick
                    retake_found = True
                break
    
    # Fallback: If no clear retake detected but there was on-site damage, use first on-site engagement
    if not retake_found and len(postplant_damage) > 0:
        # Fallback: check for any on-site CT vs T engagement (silent - no print)
        for _, dmg in postplant_damage.iterrows():
            dmg_tick = dmg['tick']
            
            attacker_pos = relevant_ticks[(relevant_ticks['tick'] == dmg_tick) & (relevant_ticks['name'] == dmg['attacker_name'])]
            victim_pos = relevant_ticks[(relevant_ticks['tick'] == dmg_tick) & (relevant_ticks['name'] == dmg['user_name'])]
            
            if len(attacker_pos) == 0 or len(victim_pos) == 0:
                continue
                
            att = attacker_pos.iloc[0]
            vic = victim_pos.iloc[0]
            
            # Only consider CT vs T damage
            if (att['team_num'] == 3 and vic['team_num'] == 2) or (att['team_num'] == 2 and vic['team_num'] == 3):
                att_zone = classifier.point_in_zone_A(att['X'], att['Y'], att['Z'])
                vic_zone = classifier.point_in_zone_A(vic['X'], vic['Y'], vic['Z'])
                
                att_is_b = is_b_zone(att_zone)
                vic_is_b = is_b_zone(vic_zone)
                
                if is_b_site:
                    on_site = att_is_b or vic_is_b
                else:
                    # A-site: check if either attacker or victim is in actual A-site zone
                    att_is_a = is_a_zone(att_zone)
                    vic_is_a = is_a_zone(vic_zone)
                    on_site = att_is_a or vic_is_a
                
                if on_site:
                    print(f"  [FALLBACK] Found on-site damage at {(dmg_tick-plant_tick)/64.0:.1f}s")
                    earliest_tick = dmg_tick
                    retake_found = True
                    break
    
    if not retake_found:
        return {'retake_detected': False, 'retake_tick': None, 'time_to_retake': None}
    
    time_to_retake = (earliest_tick - plant_tick) / 64.0
    return {
        'retake_detected': True,
        'retake_tick': earliest_tick,
        'time_to_retake': time_to_retake
    }
