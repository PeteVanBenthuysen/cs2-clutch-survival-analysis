"""
Zone connectivity system for threat cone modeling.
Calculates which zones a CT could reach from a last known position,
accounting for elapsed time and movement speed.
"""

import json
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
from collections import deque
from src.movement_speed import MovementSpeedCalculator


class ZoneConnectivity:
    """Models zone connectivity and reachability for threat analysis."""
    
    def __init__(self, connectivity_file: str = None, visibility_file: str = None):
        """Load zone connectivity graph and visibility graph."""
        if connectivity_file is None:
            connectivity_file = Path(__file__).parent.parent / 'data' / 'mirage_zone_connectivity_from_movement.json'
        
        with open(connectivity_file, 'r') as f:
            data = json.load(f)
        
        # New format: flat dict of zone -> {neighbor: count}
        # Store as zone -> {'neighbors': {neighbor: travel_time}}
        if 'connectivity' in data:
            # Old manual format
            self.connectivity = data['connectivity']
            self.speed_multipliers = data.get('speed_multipliers', {})
        else:
            # New empirical format from movement data
            # Convert counts to travel times (higher count = more traversals = faster/shorter path)
            # Use inverse of count as proxy for travel time (normalized)
            self.connectivity = {}
            for zone, neighbors in data.items():
                total_count = sum(neighbors.values())
                travel_times = {}
                for neighbor, count in neighbors.items():
                    # Higher count = shorter relative time
                    # Normalize so most frequent path = 1.0 second base time
                    max_count = max(neighbors.values())
                    travel_times[neighbor] = 1.0 * (max_count / count) if count > 0 else 10.0
                self.connectivity[zone] = {'neighbors': travel_times}
            self.speed_multipliers = {}  # Not needed with empirical data
        
        self.movement_calculator = MovementSpeedCalculator()
        
        # Load visibility graph (from damage-based empirical data)
        if visibility_file is None:
            visibility_file = Path(__file__).parent.parent / 'data' / 'mirage_visibility_from_damage.json'
        
        if Path(visibility_file).exists():
            with open(visibility_file, 'r') as f:
                vis_data = json.load(f)
            self.visibility_graph = {zone: set(visible_zones) for zone, visible_zones in vis_data['zones'].items()}
        else:
            # Fallback to neighbor-based visibility
            self.visibility_graph = None
    
    def get_reachable_zones(self, from_zone: str, time_elapsed: float, 
                           speed: str = 'running', max_zones: int = None,
                           blocked_zones: Set[str] = None,
                           weapon_name: Optional[str] = None,
                           is_walking: bool = False) -> Dict[str, float]:
        """
        Calculate which zones are reachable from a starting zone within time limit.
        
        Args:
            from_zone: Starting zone name
            time_elapsed: Time available to travel (seconds)
            speed: Movement speed ('walking', 'running', 'shift_walking') - DEPRECATED, use weapon_name
            max_zones: Limit number of zones returned (sorted by travel time)
            blocked_zones: Set of zones blocked by utility (smokes/mollies) - cannot traverse through
            weapon_name: Weapon being held (e.g., 'AWP', 'AK-47') - uses empirical speeds
            is_walking: True if shift-walking (overrides speed param)
        
        Returns:
            Dict[zone_name -> min_travel_time] for all reachable zones
        """
        if from_zone not in self.connectivity:
            return {}
        
        # Initialize blocked zones set
        if blocked_zones is None:
            blocked_zones = set()
        
        # Calculate actual movement speed
        if weapon_name:
            # Use weapon-specific speed from empirical data
            actual_speed = self.movement_calculator.calculate_speed(
                weapon_name=weapon_name,
                is_walking=is_walking
            )
            # Base travel times assume 215 units/s (rifle speed), scale accordingly
            BASE_SPEED = 215.0
            speed_mult = BASE_SPEED / actual_speed  # Slower = higher multiplier (more time)
        else:
            # Fallback to old generic categories
            speed_mult = self.speed_multipliers.get(speed, 1.0)
        
        # BFS to find shortest path to all reachable zones
        # State: (zone, cumulative_time)
        queue = deque([(from_zone, 0.0)])
        visited = {from_zone: 0.0}  # zone -> min time to reach
        
        while queue:
            current_zone, current_time = queue.popleft()
            
            # Explore neighbors
            if current_zone not in self.connectivity:
                continue
            
            neighbors = self.connectivity[current_zone]['neighbors']
            
            for next_zone, base_travel_time in neighbors.items():
                # Skip blocked zones (smoked/mollied)
                if next_zone in blocked_zones:
                    continue
                
                # Adjust travel time for speed
                adjusted_time = base_travel_time * speed_mult
                new_time = current_time + adjusted_time
                
                # Check if within time limit
                if new_time > time_elapsed:
                    continue
                
                # Check if this is a better path
                if next_zone not in visited or new_time < visited[next_zone]:
                    visited[next_zone] = new_time
                    queue.append((next_zone, new_time))
        
        # Sort by travel time and optionally limit
        reachable = dict(sorted(visited.items(), key=lambda x: x[1]))
        
        if max_zones:
            reachable = dict(list(reachable.items())[:max_zones])
        
        return reachable
    
    def get_threat_cone(self, from_zone: str, time_elapsed: float, 
                       speed: str = 'running', blocked_zones: Set[str] = None,
                       weapon_name: Optional[str] = None,
                       is_walking: bool = False) -> Dict[str, float]:
        """
        Get threat cone - zones CT could be in, with probability weights.
        
        More recent detections (lower time_elapsed) = tighter cone.
        Older detections = wider cone with lower confidence per zone.
        
        Args:
            from_zone: Last known zone
            time_elapsed: Time since last detection
            speed: Assumed movement speed - DEPRECATED, use weapon_name
            blocked_zones: Set of zones blocked by utility (smokes/mollies)
            weapon_name: Weapon being held (e.g., 'AWP', 'AK-47')
            is_walking: True if shift-walking
        
        Returns:
            Dict[zone_name -> threat_probability]
            Probabilities sum to 1.0 across all zones
        """
        reachable = self.get_reachable_zones(from_zone, time_elapsed, speed, 
                                            blocked_zones=blocked_zones,
                                            weapon_name=weapon_name,
                                            is_walking=is_walking)
        
        if not reachable:
            return {from_zone: 1.0}  # Still at last known location
        
        # Weight zones inversely by travel time
        # Closer zones = higher probability (more likely to be there)
        total_weight = 0.0
        weights = {}
        
        for zone, travel_time in reachable.items():
            # Inverse time weighting: closer zones more likely
            # Add 1.0 to avoid division by zero for starting zone
            weight = 1.0 / (travel_time + 1.0)
            weights[zone] = weight
            total_weight += weight
        
        # Normalize to probabilities
        probabilities = {
            zone: weight / total_weight 
            for zone, weight in weights.items()
        }
        
        return probabilities
    
    def calculate_position_exposure(self, position_zone: str, threat_cone: Dict[str, float],
                                    visibility_graph: Dict[str, Set[str]] = None) -> float:
        """
        Calculate exposure score for a position given a threat cone.
        
        Args:
            position_zone: Zone where T player is positioned
            threat_cone: Dict[zone -> probability] of CT locations
            visibility_graph: Dict[zone -> set of visible zones] (optional, uses loaded graph if None)
        
        Returns:
            Exposure score (0.0 = safe, 1.0 = maximum exposure)
        """
        # Use provided visibility graph or loaded one
        if visibility_graph is None:
            if self.visibility_graph is not None:
                visibility_graph = self.visibility_graph
            else:
                # Fallback to neighbor-based visibility
                visibility_graph = {}
                for zone, data in self.connectivity.items():
                    visible = set(data['neighbors'].keys())
                    visible.add(zone)  # Can see own zone
                    visibility_graph[zone] = visible
        
        # Sum threat probabilities from zones that can see your position
        exposure = 0.0
        
        for threat_zone, threat_prob in threat_cone.items():
            # Check if threat zone can see your position
            if position_zone in visibility_graph.get(threat_zone, set()):
                exposure += threat_prob
        
        return exposure
    
    def get_safest_positions(self, known_ct_locations: Dict[str, Tuple[str, float]],
                            candidate_zones: List[str], speed: str = 'running',
                            blocked_zones: Set[str] = None) -> List[Tuple[str, float]]:
        """
        Rank positions by safety given known CT locations.
        
        Args:
            known_ct_locations: Dict[player_name -> (last_zone, time_ago)]
            candidate_zones: List of zones to evaluate
            speed: Assumed CT movement speed
            blocked_zones: Set of zones blocked by utility (smokes/mollies)
        
        Returns:
            List[(zone, exposure_score)] sorted by safety (lowest exposure first)
        """
        # Build combined threat cone from all known CTs
        combined_threat = {}
        num_cts = len(known_ct_locations)
        
        for player, (last_zone, time_ago) in known_ct_locations.items():
            threat_cone = self.get_threat_cone(last_zone, time_ago, speed, blocked_zones=blocked_zones)
            
            # Weight by number of CTs (distribute probability)
            for zone, prob in threat_cone.items():
                if zone not in combined_threat:
                    combined_threat[zone] = 0.0
                combined_threat[zone] += prob / num_cts
        
        # Evaluate each candidate position
        position_scores = []
        
        for position in candidate_zones:
            exposure = self.calculate_position_exposure(position, combined_threat)
            position_scores.append((position, exposure))
        
        # Sort by exposure (lowest = safest)
        position_scores.sort(key=lambda x: x[1])
        
        return position_scores
    
    def print_threat_analysis(self, from_zone: str, time_elapsed: float, speed: str = 'running',
                             blocked_zones: Set[str] = None):
        """Print human-readable threat cone analysis."""
        print(f"\n{'='*80}")
        print(f"THREAT CONE ANALYSIS")
        print(f"{'='*80}\n")
        print(f"Last known location: {from_zone}")
        print(f"Time elapsed: {time_elapsed:.1f}s")
        print(f"Movement speed: {speed}")
        if blocked_zones:
            print(f"Blocked zones: {', '.join(sorted(blocked_zones))}")
        print()
        
        reachable = self.get_reachable_zones(from_zone, time_elapsed, speed, blocked_zones=blocked_zones)
        threat_cone = self.get_threat_cone(from_zone, time_elapsed, speed, blocked_zones=blocked_zones)
        
        print(f"Reachable zones: {len(reachable)}\n")
        print(f"{'Zone':30s} {'Travel Time':12s} {'Threat Prob':12s}")
        print("-" * 80)
        
        for zone in sorted(threat_cone.keys(), key=lambda z: threat_cone[z], reverse=True):
            travel_time = reachable[zone]
            prob = threat_cone[zone]
            print(f"{zone:30s} {travel_time:6.1f}s       {prob:6.1%}")
