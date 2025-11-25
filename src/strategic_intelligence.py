"""
Strategic intelligence analyzer - tracks team-level CT push direction.
Converts individual sound detections into actionable team intelligence.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from src.zone_classifier import MirageZoneClassifier


class StrategicIntelligence:
    """Analyzes CT push patterns and provides team-level intelligence."""
    
    # Define strategic zone groups
    CT_SPAWN_ZONES = {
        'CT_Pots', 'CT_Cubby', 'Blue_Door', 'Ticket'
    }
    
    CT_PUSH_ZONES = {
        'CT_Bottom_Ramp', 'Trashcan_CT', 'Ropz', 'CT_Side_Triple'
    }
    
    RAMP_ZONES = {
        'A_Ramp_Room', 'Tetris', 'Firebox'
    }
    
    PALACE_ZONES = {
        'Palace', 'Heaven', 'Under_Balcony'
    }
    
    CONNECTOR_ZONES = {
        'Connector_Box_Cubby', 'Connector_Window_Cubby', 'Connector_Jungle_Cubby'
    }
    
    JUNGLE_STAIRS_ZONES = {
        'Stairs', 'Top_Stairs', 'Bottom_Stairs', 'Jungle', 'Deep_Jungle_Cubby', 'Jungle_Cubby'
    }
    
    def __init__(self):
        self.classifier = MirageZoneClassifier()
    
    def analyze_ct_movements(self, audio_data: Dict, plant_tick: int) -> Dict:
        """
        Analyze CT movement patterns to determine push direction and team intel.
        
        Returns strategic intelligence about CT positioning and push direction.
        """
        ct_movements = defaultdict(list)
        
        # Group all sounds by CT player
        for event in audio_data['ct_sound_events']:
            ct_player = event['sound_maker']
            ct_movements[ct_player].append(event)
        
        # Sort each player's movements by time
        for player in ct_movements:
            ct_movements[player].sort(key=lambda x: x['tick'])
        
        # Analyze pre-plant and post-plant separately
        pre_plant_intel = self._analyze_phase(ct_movements, plant_tick, is_pre_plant=True)
        post_plant_intel = self._analyze_phase(ct_movements, plant_tick, is_pre_plant=False)
        
        # Combine into strategic assessment
        strategic_intel = {
            'pre_plant': pre_plant_intel,
            'post_plant': post_plant_intel,
            'combined_assessment': self._create_strategic_assessment(pre_plant_intel, post_plant_intel)
        }
        
        return strategic_intel
    
    def _analyze_phase(self, ct_movements: Dict, plant_tick: int, is_pre_plant: bool) -> Dict:
        """Analyze a specific phase (pre or post plant)."""
        
        # Filter movements to this phase
        phase_movements = {}
        for player, movements in ct_movements.items():
            if is_pre_plant:
                filtered = [m for m in movements if m['tick'] < plant_tick]
            else:
                filtered = [m for m in movements if m['tick'] >= plant_tick]
            
            if filtered:
                phase_movements[player] = filtered
        
        # Track which zones each CT was detected in
        ct_zone_paths = {}
        for player, movements in phase_movements.items():
            zones = []
            for m in movements:
                if m['source_zone'] and (not zones or zones[-1] != m['source_zone']):
                    zones.append(m['source_zone'])
            ct_zone_paths[player] = zones
        
        # Determine push directions
        push_directions = self._identify_push_directions(ct_zone_paths)
        
        # Count CTs by strategic area
        ct_counts_by_area = self._count_cts_by_area(ct_zone_paths)
        
        return {
            'ct_count': len(phase_movements),
            'ct_zone_paths': ct_zone_paths,
            'push_directions': push_directions,
            'area_counts': ct_counts_by_area,
            'movements': phase_movements
        }
    
    def _identify_push_directions(self, ct_zone_paths: Dict[str, List[str]]) -> Dict:
        """Identify which direction(s) CTs are pushing from."""
        
        directions = {
            'ct_spawn': [],
            'ramp': [],
            'palace': [],
            'connector': [],
            'jungle_stairs': [],
            'unknown': []
        }
        
        for player, zones in ct_zone_paths.items():
            # Check which strategic areas this CT moved through
            in_ct_spawn = any(z in self.CT_SPAWN_ZONES for z in zones)
            in_ct_push = any(z in self.CT_PUSH_ZONES for z in zones)
            in_ramp = any(z in self.RAMP_ZONES for z in zones)
            in_palace = any(z in self.PALACE_ZONES for z in zones)
            in_connector = any(z in self.CONNECTOR_ZONES for z in zones)
            in_jungle = any(z in self.JUNGLE_STAIRS_ZONES for z in zones)
            
            # Classify push direction
            if in_ct_spawn or in_ct_push:
                directions['ct_spawn'].append(player)
            elif in_ramp:
                directions['ramp'].append(player)
            elif in_palace:
                directions['palace'].append(player)
            elif in_connector:
                directions['connector'].append(player)
            elif in_jungle:
                directions['jungle_stairs'].append(player)
            else:
                directions['unknown'].append(player)
        
        return directions
    
    def _count_cts_by_area(self, ct_zone_paths: Dict[str, List[str]]) -> Dict:
        """Count how many CTs detected in each strategic area."""
        
        counts = {
            'ct_spawn': 0,
            'ct_push': 0,
            'ramp': 0,
            'palace': 0,
            'connector': 0,
            'jungle_stairs': 0
        }
        
        for player, zones in ct_zone_paths.items():
            if any(z in self.CT_SPAWN_ZONES for z in zones):
                counts['ct_spawn'] += 1
            if any(z in self.CT_PUSH_ZONES for z in zones):
                counts['ct_push'] += 1
            if any(z in self.RAMP_ZONES for z in zones):
                counts['ramp'] += 1
            if any(z in self.PALACE_ZONES for z in zones):
                counts['palace'] += 1
            if any(z in self.CONNECTOR_ZONES for z in zones):
                counts['connector'] += 1
            if any(z in self.JUNGLE_STAIRS_ZONES for z in zones):
                counts['jungle_stairs'] += 1
        
        return counts
    
    def _create_strategic_assessment(self, pre_plant: Dict, post_plant: Dict) -> Dict:
        """Create high-level strategic assessment for hazard modeling."""
        
        # Determine primary push direction
        post_dirs = post_plant['push_directions']
        primary_direction = None
        ct_count_pushing = 0
        
        for direction, players in post_dirs.items():
            if len(players) > ct_count_pushing:
                ct_count_pushing = len(players)
                primary_direction = direction
        
        # Assess information quality
        total_cts_detected = len(set(
            list(pre_plant.get('ct_zone_paths', {}).keys()) + 
            list(post_plant.get('ct_zone_paths', {}).keys())
        ))
        
        # Determine if push is coordinated (multiple CTs from same direction)
        coordinated_push = ct_count_pushing >= 2
        
        # Check for split push (CTs from multiple directions)
        active_directions = sum(1 for players in post_dirs.values() if len(players) > 0)
        split_push = active_directions >= 2
        
        return {
            'total_cts_detected': total_cts_detected,
            'primary_push_direction': primary_direction,
            'cts_in_primary_direction': ct_count_pushing,
            'coordinated_push': coordinated_push,
            'split_push': split_push,
            'information_quality': 'HIGH' if total_cts_detected >= 3 else ('MEDIUM' if total_cts_detected >= 2 else 'LOW'),
            'strategic_summary': self._generate_summary(primary_direction, ct_count_pushing, coordinated_push, split_push)
        }
    
    def _generate_summary(self, direction: str, ct_count: int, coordinated: bool, split: bool) -> str:
        """Generate human-readable strategic summary."""
        
        if split:
            return f"Split push detected - CTs attacking from multiple angles"
        elif coordinated:
            return f"{ct_count} CTs pushing from {direction.replace('_', ' ')}"
        elif ct_count > 0:
            return f"Single CT detected from {direction.replace('_', ' ')}"
        else:
            return "No clear push direction identified"
    
    def print_strategic_intelligence(self, intel: Dict, plant_tick: int):
        """Print formatted strategic intelligence report."""
        
        print(f"\n{'='*80}")
        print("STRATEGIC INTELLIGENCE REPORT")
        print(f"{'='*80}\n")
        
        # Pre-plant phase
        pre = intel['pre_plant']
        print(f"PRE-PLANT PHASE:")
        print(f"  CTs detected: {pre['ct_count']}")
        
        if pre['ct_count'] > 0:
            print(f"  Area breakdown:")
            for area, count in pre['area_counts'].items():
                if count > 0:
                    print(f"    {area.replace('_', ' ').title()}: {count} CT(s)")
            
            print(f"\n  Movement paths:")
            for player, zones in pre['ct_zone_paths'].items():
                print(f"    {player}: {' -> '.join(zones)}")
        
        # Post-plant phase
        post = intel['post_plant']
        print(f"\nPOST-PLANT PHASE:")
        print(f"  CTs detected: {post['ct_count']}")
        
        if post['ct_count'] > 0:
            print(f"  Push directions:")
            for direction, players in post['push_directions'].items():
                if players:
                    print(f"    {direction.replace('_', ' ').title()}: {len(players)} CT(s) - {', '.join(players)}")
            
            print(f"\n  Movement paths:")
            for player, zones in post['ct_zone_paths'].items():
                print(f"    {player}: {' -> '.join(zones)}")
        
        # Combined assessment
        assessment = intel['combined_assessment']
        print(f"\n{'='*80}")
        print("STRATEGIC ASSESSMENT")
        print(f"{'='*80}\n")
        print(f"Total CTs identified: {assessment['total_cts_detected']}")
        print(f"Information quality: {assessment['information_quality']}")
        print(f"Primary push direction: {assessment['primary_push_direction'].replace('_', ' ').title()}")
        print(f"CTs in primary direction: {assessment['cts_in_primary_direction']}")
        print(f"Coordinated push: {'YES' if assessment['coordinated_push'] else 'NO'}")
        print(f"Split push: {'YES' if assessment['split_push'] else 'NO'}")
        print(f"\nSummary: {assessment['strategic_summary']}")
        
        print(f"\n{'='*80}")
        print("HAZARD IMPLICATIONS")
        print(f"{'='*80}\n")
        
        if assessment['coordinated_push']:
            print(f"[!] HIGH CERTAINTY: {assessment['cts_in_primary_direction']} CTs confirmed pushing from {assessment['primary_push_direction'].replace('_', ' ')}")
            print(f"  -> Positions exposed to {assessment['primary_push_direction'].replace('_', ' ')} have ELEVATED hazard")
            print(f"  -> Positions covering other angles have REDUCED hazard (low threat)")
        elif assessment['split_push']:
            print("[!] SPLIT THREAT: CTs attacking from multiple directions")
            print("  -> All positions maintain elevated hazard")
            print("  -> Crossfire/trade opportunities limited")
        else:
            print("[!] LOW INFORMATION: Unclear CT positioning")
            print("  -> High uncertainty across all positions")
            print("  -> Hazard model relies more on historical patterns")
