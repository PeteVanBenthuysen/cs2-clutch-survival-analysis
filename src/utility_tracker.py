"""
Utility tracking system for post-plant analysis.
Tracks smokes, mollys, HE nades, and flashes with zone classification and expiration times.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from src.zone_classifier import MirageZoneClassifier


class UtilityTracker:
    """Track utility usage and calculate time-varying utility state."""
    
    # Utility durations in seconds
    SMOKE_DURATION = 18.0  # CS2 smoke duration
    MOLLY_DURATION = 7.0   # Molotov/incendiary duration
    FLASH_EFFECT = 3.0     # Consider flashes relevant for 3s after detonation
    
    # Tactical relevance windows (utility that recently expired still affects decisions)
    RECENT_UTILITY_WINDOW = 10.0  # Consider utility from last 10s as tactically relevant
    
    def __init__(self):
        self.classifier = MirageZoneClassifier()
        
    def parse_utility_events(self, parser, plant_tick: int, retake_tick: Optional[int] = None) -> Dict:
        """
        Parse all utility events relevant to post-plant.
        Includes:
        - Pre-plant utility that's still active post-plant
        - Utility thrown during post-plant
        
        Returns dict with:
        - smokes: List of {tick, zone, player, team, expires_tick}
        - mollys: List of {tick, zone, player, team, expires_tick}
        - flashes: List of {tick, zone, player, team, relevant_until_tick}
        - he_nades: List of {tick, zone, player, team}
        """
        
        # Parse utility events
        smokes = parser.parse_event('smokegrenade_detonate')
        
        # Try molotov_detonate, fallback to inferno_startburn
        try:
            mollys = parser.parse_event('molotov_detonate')
        except:
            try:
                mollys = parser.parse_event('inferno_startburn')
            except:
                mollys = pd.DataFrame()
        
        flashes = parser.parse_event('flashbang_detonate')
        he_nades = parser.parse_event('hegrenade_detonate')
        
        # Parse ticks to get team info for throwers
        ticks = parser.parse_ticks(['name', 'tick', 'team_num'])
        
        # Set analysis window
        window_end = retake_tick if retake_tick else plant_tick + 1920  # 30s default
        
        # For smokes/mollys, include pre-plant utility that's still active post-plant
        # Smokes last 18s, so look back 18s before plant
        smoke_lookback = plant_tick - int(self.SMOKE_DURATION * 64)
        molly_lookback = plant_tick - int(self.MOLLY_DURATION * 64)
        
        def classify_and_extract(events_df, duration_ticks: Optional[int] = None, lookback_tick: Optional[int] = None):
            """Classify utility events by zone and add metadata."""
            results = []
            
            # Handle empty dataframe or list
            if isinstance(events_df, list) or (isinstance(events_df, pd.DataFrame) and len(events_df) == 0):
                return results
            
            # Set the start tick - either lookback or plant
            start_tick = lookback_tick if lookback_tick else plant_tick
            
            for _, event in events_df.iterrows():
                tick = event['tick']
                
                # For utility with duration (smokes/mollys), include pre-plant if still active
                if duration_ticks and lookback_tick:
                    # Check if utility would still be active at plant time
                    expires_tick = tick + duration_ticks
                    if tick < start_tick and expires_tick < plant_tick:
                        # Expired before plant, skip
                        continue
                    if tick > window_end:
                        # After analysis window, skip
                        continue
                else:
                    # For flashes/HE, only include events in post-plant window
                    if tick < plant_tick or tick > window_end:
                        continue
                
                # Classify zone
                zone = self.classifier.point_in_zone_A(event['x'], event['y'], event['z'])
                
                # Get thrower's team
                player = event['user_name']
                player_data = ticks[ticks['name'] == player]
                team = player_data.iloc[0]['team_num'] if len(player_data) > 0 else None
                team_label = 'T' if team == 2 else 'CT' if team == 3 else 'Unknown'
                
                result = {
                    'tick': tick,
                    'time_after_plant': (tick - plant_tick) / 64.0,
                    'zone': zone,
                    'player': player,
                    'team': team_label,
                    'x': event['x'],
                    'y': event['y'],
                    'z': event['z']
                }
                
                # Add expiration if duration specified
                if duration_ticks:
                    result['expires_tick'] = tick + duration_ticks
                    result['expires_time_after_plant'] = (tick + duration_ticks - plant_tick) / 64.0
                    
                    # Mark if this was pre-plant utility
                    result['thrown_pre_plant'] = tick < plant_tick
                
                results.append(result)
            
            return results
        
        # Process each utility type
        utility_data = {
            'smokes': classify_and_extract(smokes, int(self.SMOKE_DURATION * 64), smoke_lookback),
            'mollys': classify_and_extract(mollys, int(self.MOLLY_DURATION * 64), molly_lookback),
            'flashes': classify_and_extract(flashes, int(self.FLASH_EFFECT * 64), None),
            'he_nades': classify_and_extract(he_nades, None, None)
        }
        
        return utility_data
    
    def get_active_utility_at_tick(self, utility_data: Dict, query_tick: int) -> Dict:
        """
        Get all active AND recently expired utility at a specific tick.
        
        Returns dict with:
        - active_smokes: List of zones with active smokes
        - active_mollys: List of zones with active mollys
        - recent_flashes: List of zones with recent flashes (active effect)
        - recent_smokes: List of smokes that expired in last 10s
        - recent_mollys: List of mollys that expired in last 10s
        - recent_utility_all: All utility from last 10s (tactical info)
        - blocked_zones: Set of zones blocked by smoke/molly
        """
        
        recent_window_ticks = int(self.RECENT_UTILITY_WINDOW * 64)
        
        active_smokes = []
        active_mollys = []
        recent_flashes = []
        recent_smokes = []
        recent_mollys = []
        recent_utility_all = []
        
        # Check smokes
        for smoke in utility_data['smokes']:
            if smoke['tick'] <= query_tick <= smoke['expires_tick']:
                # Currently active
                active_smokes.append(smoke)
                recent_utility_all.append({**smoke, 'type': 'smoke', 'status': 'active'})
            elif smoke['expires_tick'] < query_tick <= smoke['expires_tick'] + recent_window_ticks:
                # Recently expired (last 10s)
                recent_smokes.append({**smoke, 'expired_seconds_ago': (query_tick - smoke['expires_tick']) / 64.0})
                recent_utility_all.append({**smoke, 'type': 'smoke', 'status': 'expired'})
        
        # Check mollys
        for molly in utility_data['mollys']:
            if molly['tick'] <= query_tick <= molly['expires_tick']:
                active_mollys.append(molly)
                recent_utility_all.append({**molly, 'type': 'molly', 'status': 'active'})
            elif molly['expires_tick'] < query_tick <= molly['expires_tick'] + recent_window_ticks:
                recent_mollys.append({**molly, 'expired_seconds_ago': (query_tick - molly['expires_tick']) / 64.0})
                recent_utility_all.append({**molly, 'type': 'molly', 'status': 'expired'})
        
        # Check flashes (recent only, no "active" state for flashes)
        for flash in utility_data['flashes']:
            if flash['tick'] <= query_tick <= flash['expires_tick']:
                recent_flashes.append(flash)
                recent_utility_all.append({**flash, 'type': 'flash', 'status': 'recent'})
        
        # Check HE nades (last 10s)
        for he in utility_data['he_nades']:
            if query_tick - recent_window_ticks <= he['tick'] <= query_tick:
                recent_utility_all.append({**he, 'type': 'he', 'status': 'recent'})
        
        # Get blocked zones (only from ACTIVE utility)
        blocked_zones = set()
        for smoke in active_smokes:
            if smoke['zone']:
                blocked_zones.add(smoke['zone'])
        for molly in active_mollys:
            if molly['zone']:
                blocked_zones.add(molly['zone'])
        
        return {
            'active_smokes': active_smokes,
            'active_mollys': active_mollys,
            'recent_flashes': recent_flashes,
            'recent_smokes': recent_smokes,
            'recent_mollys': recent_mollys,
            'recent_utility_all': recent_utility_all,
            'blocked_zones': blocked_zones,
            'smoke_zones': [s['zone'] for s in active_smokes if s['zone']],
            'molly_zones': [m['zone'] for m in active_mollys if m['zone']],
            'flash_zones': [f['zone'] for f in recent_flashes if f['zone']]
        }
    
    def get_utility_expiration_events(self, utility_data: Dict, plant_tick: int) -> List[Dict]:
        """
        Get timeline of utility expiration events (hazard spikes).
        When smoke/molly expires, threat cone suddenly expands.
        
        Returns:
            List of dicts with {tick, time_after_plant, zone, type, hazard_change}
            Sorted by tick ascending
        """
        expiration_events = []
        
        # Smoke expirations
        for smoke in utility_data['smokes']:
            expiration_events.append({
                'tick': smoke['expires_tick'],
                'time_after_plant': (smoke['expires_tick'] - plant_tick) / 64.0,
                'zone': smoke['zone'],
                'type': 'smoke_expires',
                'hazard_change': 'INCREASE',
                'description': f"Smoke expires at {smoke['zone']} - zone now OPEN to CT push"
            })
        
        # Molly expirations
        for molly in utility_data['mollys']:
            expiration_events.append({
                'tick': molly['expires_tick'],
                'time_after_plant': (molly['expires_tick'] - plant_tick) / 64.0,
                'zone': molly['zone'],
                'type': 'molly_expires',
                'hazard_change': 'INCREASE',
                'description': f"Molly expires at {molly['zone']} - zone now PUSHABLE"
            })
        
        # Sort by tick
        expiration_events.sort(key=lambda x: x['tick'])
        
        return expiration_events
    
    def calculate_live_threat_angles(self, blocked_zones: set, known_ct_zones: set = None) -> Dict:
        """
        Calculate which zones represent live threats vs blocked/safe.
        
        Args:
            blocked_zones: Zones blocked by smoke/molly
            known_ct_zones: Zones where CTs were detected (optional)
        
        Returns dict with threat analysis
        """
        
        # Define common retake entry zones
        retake_entry_zones = {
            'A_Ramp_Room', 'CT_Bottom_Ramp', 'Jungle', 'Deep_Jungle_Cubby',
            'Connector_Jungle_Cubby', 'Palace', 'Heaven', 'Stairs', 'CT_Side_Triple'
        }
        
        # Calculate live (unblocked) entry zones
        live_entries = retake_entry_zones - blocked_zones
        
        # Calculate known threats (if CT zones provided)
        known_threats = known_ct_zones - blocked_zones if known_ct_zones else set()
        unknown_threats = live_entries - known_threats if known_ct_zones else live_entries
        
        return {
            'all_entries': retake_entry_zones,
            'blocked_entries': blocked_zones & retake_entry_zones,
            'live_entries': live_entries,
            'known_threats': known_threats,
            'unknown_threats': unknown_threats,
            'num_live_angles': len(live_entries),
            'num_blocked_angles': len(blocked_zones & retake_entry_zones)
        }
    
    def print_utility_summary(self, utility_data: Dict, plant_tick: int, retake_tick: Optional[int] = None):
        """Print formatted summary of utility usage."""
        
        print(f"\n{'='*80}")
        print("UTILITY USAGE SUMMARY")
        print(f"{'='*80}\n")
        
        # Smokes
        print(f"SMOKES ({len(utility_data['smokes'])} total):")
        if utility_data['smokes']:
            for smoke in utility_data['smokes']:
                pre_plant = " [PRE-PLANT]" if smoke.get('thrown_pre_plant', False) else ""
                print(f"  +{smoke['time_after_plant']:5.1f}s: {smoke['player']:12s} ({smoke['team']}) → {smoke['zone']}{pre_plant}")
                print(f"          Active until +{smoke['expires_time_after_plant']:.1f}s")
        else:
            print("  None")
        
        print(f"\nMOLLYS ({len(utility_data['mollys'])} total):")
        if utility_data['mollys']:
            for molly in utility_data['mollys']:
                pre_plant = " [PRE-PLANT]" if molly.get('thrown_pre_plant', False) else ""
                print(f"  +{molly['time_after_plant']:5.1f}s: {molly['player']:12s} ({molly['team']}) → {molly['zone']}{pre_plant}")
                print(f"          Active until +{molly['expires_time_after_plant']:.1f}s")
        else:
            print("  None")
        
        print(f"\nFLASHES ({len(utility_data['flashes'])} total):")
        if utility_data['flashes']:
            for flash in utility_data['flashes']:
                print(f"  +{flash['time_after_plant']:5.1f}s: {flash['player']:12s} ({flash['team']}) → {flash['zone']}")
        else:
            print("  None")
        
        print(f"\nHE GRENADES ({len(utility_data['he_nades'])} total):")
        if utility_data['he_nades']:
            for he in utility_data['he_nades']:
                print(f"  +{he['time_after_plant']:5.1f}s: {he['player']:12s} ({he['team']}) → {he['zone']}")
        else:
            print("  None")
        
        # Show utility state at retake if provided
        if retake_tick:
            print(f"\n{'='*80}")
            print(f"UTILITY STATE AT RETAKE (tick {retake_tick}, +{(retake_tick-plant_tick)/64:.1f}s)")
            print(f"{'='*80}\n")
            
            state = self.get_active_utility_at_tick(utility_data, retake_tick)
            
            if state['active_smokes']:
                print("Active smokes:")
                for smoke in state['active_smokes']:
                    time_left = (smoke['expires_tick'] - retake_tick) / 64.0
                    print(f"  {smoke['zone']:30s} ({time_left:.1f}s remaining)")
            else:
                print("Active smokes: None")
            
            if state['active_mollys']:
                print("\nActive mollys:")
                for molly in state['active_mollys']:
                    time_left = (molly['expires_tick'] - retake_tick) / 64.0
                    print(f"  {molly['zone']:30s} ({time_left:.1f}s remaining)")
            else:
                print("\nActive mollys: None")
            
            if state['recent_flashes']:
                print("\nRecent flashes (last 3s):")
                for flash in state['recent_flashes']:
                    print(f"  {flash['zone']:30s}")
            else:
                print("\nRecent flashes: None")
            
            # Show recently expired utility (last 10s)
            if state['recent_smokes']:
                print("\nRecently expired smokes (last 10s):")
                for smoke in state['recent_smokes']:
                    print(f"  {smoke['zone']:30s} (expired {smoke['expired_seconds_ago']:.1f}s ago) - zone now OPEN")
            
            if state['recent_mollys']:
                print("\nRecently expired mollys (last 10s):")
                for molly in state['recent_mollys']:
                    print(f"  {molly['zone']:30s} (expired {molly['expired_seconds_ago']:.1f}s ago) - zone now PUSHABLE")
            
            # Show utility usage intensity
            recent_count = len(state['recent_utility_all'])
            if recent_count > 0:
                ct_util = [u for u in state['recent_utility_all'] if u['team'] == 'CT']
                t_util = [u for u in state['recent_utility_all'] if u['team'] == 'T']
                print(f"\nUtility usage (last 10s): {recent_count} total ({len(ct_util)} CT, {len(t_util)} T)")
                if len(ct_util) >= 2:
                    print(f"  ⚠ HIGH CT UTILITY USAGE - Signals committed retake")
            
            # Show threat analysis
            threat_analysis = self.calculate_live_threat_angles(state['blocked_zones'])
            
            print(f"\n{'='*80}")
            print("THREAT ANGLE ANALYSIS")
            print(f"{'='*80}\n")
            print(f"Total entry angles: {len(threat_analysis['all_entries'])}")
            print(f"Blocked by utility: {len(threat_analysis['blocked_entries'])}")
            print(f"Live threat angles: {len(threat_analysis['live_entries'])}")
            
            if threat_analysis['blocked_entries']:
                print(f"\nBlocked zones:")
                for zone in sorted(threat_analysis['blocked_entries']):
                    print(f"  - {zone}")
            
            if threat_analysis['live_entries']:
                print(f"\nLive threat zones (must be covered):")
                for zone in sorted(threat_analysis['live_entries']):
                    print(f"  - {zone}")
