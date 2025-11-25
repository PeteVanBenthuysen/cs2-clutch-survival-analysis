"""
Zone classification for CS2 Mirage A-site post-plant positions.

Hybrid approach:
- Polygon zones: point-in-polygon check with buffering
- Anchor zones: rubber-band to nearest point

Author: Pete Van Benthuysen
Date: November 2025
"""

import json
import math
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from collections import defaultdict
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


class MirageZoneClassifier:
    """Classifies player positions into A-site zones on Mirage."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize classifier with zone definitions.
        
        Args:
            config_path: Path to mirage_zones.json. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "data" / "mirage_zones.json"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.polygon_zones = config['a_site_zones']['polygon_zones']
        self.point_zones = config['a_site_zones']['point_zones']
        self.constants = config['constants']
        self.priority_order = config['priority_order']
        
        # Pre-compute buffered polygons for faster lookups
        self._build_buffered_polygons()
        
        # Build A-site master bounding polygon
        self._build_master_bounds()
    
    def _build_buffered_polygons(self):
        """Create buffered Shapely polygons for all polygon zones."""
        buffer_distance = self.constants['POLY_BUFFER']
        self.polygons = {}
        self.buffered_polygons = {}
        
        for zone_name, coords in self.polygon_zones.items():
            # Use only X,Y for 2D polygon (ignore Z for now)
            xy_coords = [(x, y) for x, y, z in coords]
            poly = Polygon(xy_coords)
            
            self.polygons[zone_name] = poly
            self.buffered_polygons[zone_name] = poly.buffer(buffer_distance)
    
    def _build_master_bounds(self):
        """Create master bounding polygon encompassing all A-site zones."""
        all_polys = list(self.buffered_polygons.values())
        self.master_bounds = unary_union(all_polys)
        
        # Extend to include anchor zones
        point_locations = [Point(x, y) for x, y, z in self.point_zones.values()]
        for pt in point_locations:
            buffered_pt = pt.buffer(self.constants['MAX_ANCHOR_DISTANCE'])
            self.master_bounds = self.master_bounds.union(buffered_pt)
    
    def _distance_3d(self, x1: float, y1: float, z1: float, 
                     x2: float, y2: float, z2: float) -> float:
        """Calculate 3D Euclidean distance."""
        return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    
    def _check_polygon_zones(self, x: float, y: float, z: float) -> Optional[str]:
        """
        Check if point is in any polygon zone.
        
        Returns zone name or None if not in any polygon.
        """
        pt = Point(x, y)
        matches = []
        
        for zone_name, poly in self.buffered_polygons.items():
            if poly.contains(pt):
                # Special handling for Palace - check z threshold
                if zone_name == "Palace":
                    # Get average z of palace polygon
                    palace_coords = self.polygon_zones["Palace"]
                    avg_z = sum(z for x, y, z in palace_coords) / len(palace_coords)
                    
                    # Palace is elevated - include balcony/heaven area
                    # Palace at ~z=24, Heaven/balcony at ~z=-40, ground at ~z=-168
                    # Accept anything above z=-100 (clearly not ground level)
                    if z > -100:
                        matches.append(zone_name)
                else:
                    matches.append(zone_name)
        
        if not matches:
            return None
        
        # If multiple matches, use priority order
        if len(matches) > 1:
            for priority_zone in self.priority_order:
                if priority_zone in matches:
                    return priority_zone
        
        return matches[0]
    
    def _check_point_zones(self, x: float, y: float, z: float) -> Tuple[Optional[str], float]:
        """
        Find nearest anchor zone.
        
        Returns (zone_name, distance) or (None, inf) if too far from all anchors.
        """
        min_dist = float('inf')
        closest_zone = None
        
        for zone_name, (ax, ay, az) in self.point_zones.items():
            dist = self._distance_3d(x, y, z, ax, ay, az)
            if dist < min_dist:
                min_dist = dist
                closest_zone = zone_name
        
        # Only return if within reasonable distance
        max_dist = self.constants['MAX_ANCHOR_DISTANCE']
        if min_dist <= max_dist:
            return closest_zone, min_dist
        
        return None, min_dist
    
    def point_in_zone_A(self, x: float, y: float, z: float) -> Optional[str]:
        """
        Classify a single point into an A-site zone.
        
        Priority:
        1. Closest distance to any zone (polygon perimeter or point center)
        2. If nothing is very close, check polygon containment
        3. Last resort: point zones within MAX_ANCHOR_DISTANCE
        
        Args:
            x, y, z: Player coordinates
        
        Returns:
            Zone name or None if outside A-site
        """
        pt = Point(x, y)
        
        # Calculate distances to all zones
        zone_distances = {}
        
        # Distance to polygon zones (use perimeter AND Z-distance)
        for zone_name, poly in self.polygons.items():
            # Distance to polygon boundary in 2D (0 if inside)
            dist_2d = poly.exterior.distance(pt)
            
            # Get average Z of the polygon
            polygon_coords = self.polygon_zones[zone_name]
            avg_z = sum(z_coord for x_coord, y_coord, z_coord in polygon_coords) / len(polygon_coords)
            
            # Calculate Z distance
            z_dist = abs(z - avg_z)
            
            # Combine 2D distance with Z distance for true 3D proximity
            # Weight Z-distance equally to horizontal distance
            dist_3d = math.sqrt(dist_2d**2 + z_dist**2)
            
            zone_distances[zone_name] = dist_3d
        
        # Distance to point zones
        for zone_name, (ax, ay, az) in self.point_zones.items():
            dist_3d = self._distance_3d(x, y, z, ax, ay, az)
            zone_distances[zone_name] = dist_3d
        
        # Find closest zone
        if zone_distances:
            closest_zone = min(zone_distances.items(), key=lambda item: item[1])
            zone_name, distance = closest_zone
            
            # If very close (within reasonable threshold), use it
            CLOSE_THRESHOLD = 200  # units - closer than this, definitely that zone
            if distance <= CLOSE_THRESHOLD:
                return zone_name
            
            # Check if inside a polygon (polygon containment as fallback)
            polygon_match = self._check_polygon_zones(x, y, z)
            if polygon_match:
                return polygon_match
            
            # Use closest zone if within MAX_ANCHOR_DISTANCE
            if distance <= self.constants['MAX_ANCHOR_DISTANCE']:
                return zone_name
        
        # Check if player is in A-site bounds but no zone matched
        if self.master_bounds.contains(pt):
            return "A_SITE_UNLABELED"
        
        return None  # Outside A-site entirely
    
    def is_in_a_site(self, x: float, y: float, z: float) -> bool:
        """Check if point is anywhere within A-site bounds."""
        pt = Point(x, y)
        return self.master_bounds.contains(pt)
    
    def assign_postplant_zone_A(self, player_track: List[Dict], 
                                plant_time: float,
                                tick_rate: int = 64,
                                damage_events: List[Dict] = None) -> Dict:
        """
        Assign A-site post-plant zone using dwell algorithm.
        
        Args:
            player_track: List of position dicts with keys: 'tick', 'x', 'y', 'z', 'alive'
            plant_time: Tick when bomb was planted
            tick_rate: Server tick rate (default 64)
        
        Returns:
            Dict with:
                - zone: Final assigned zone name
                - dwell_seconds: Time spent in final zone
                - dwell_fraction: Fraction of window in final zone
                - rotated: True if player left A-site during window
                - died: True if player died during window
                - all_zones: Debug info - all zones visited with durations
        """
        constants = self.constants
        
    def assign_postplant_zone_A(self, player_track: List[Dict], 
                                plant_time: float,
                                tick_rate: int = 64,
                                damage_events: List[Dict] = None,
                                retake_start_tick: int = None) -> Dict:
        """
        Assign A-site post-plant zone using fight-first logic.
        
        Priority:
        1. If player takes damage or dies -> use position at that moment
        2. If retake detected -> use position at retake start
        3. Otherwise -> use dwell position (most time spent)
        
        Args:
            player_track: List of position dicts with keys: 'tick', 'x', 'y', 'z', 'alive'
            plant_time: Tick when bomb was planted
            tick_rate: Server tick rate (default 64)
            damage_events: Optional list of damage events with 'tick' and 'victim_steamid'
            retake_start_tick: Optional tick when retake started (CT entered A-site)
        
        Returns:
            Dict with:
                - zone: Final assigned zone name
                - dwell_seconds: Time spent in final zone
                - dwell_fraction: Fraction of window in final zone
                - contact_tick: Tick where first damage/death occurred (None if no contact)
                - died: True if player died during window
                - method: 'fight_position', 'retake_position', or 'dwell_position'
        """
        constants = self.constants
        
        # Define time window [plant+2s, min(plant+6s, retake_start)]
        window_start = plant_time + (2.0 * tick_rate)
        default_window_end = plant_time + (6.0 * tick_rate)
        
        # Truncate window if retake started early
        if retake_start_tick and retake_start_tick < default_window_end:
            window_end = retake_start_tick
        else:
            window_end = default_window_end
        
        # Filter ticks in window
        window_ticks = [
            t for t in player_track 
            if window_start <= t['tick'] <= window_end
        ]
        
        if not window_ticks:
            return {
                'zone': None,
                'dwell_seconds': 0,
                'dwell_fraction': 0,
                'contact_tick': None,
                'died': False,
                'method': 'no_data',
                'all_zones': {}
            }
        
        # PRIORITY 1: Check for death or damage
        contact_tick = None
        
        # Find death tick
        death_tick = None
        for t in window_ticks:
            if not t['alive']:
                death_tick = t['tick']
                break
        
        # Find first damage tick (if events provided)
        first_damage_tick = None
        if damage_events:
            player_id = player_track[0].get('steamid') or player_track[0].get('player_id')
            for event in damage_events:
                if (event.get('victim_steamid') == player_id or 
                    event.get('victim_id') == player_id):
                    event_tick = event['tick']
                    if window_start <= event_tick <= window_end:
                        first_damage_tick = event_tick
                        break
        
        # Determine contact tick
        if first_damage_tick and death_tick:
            contact_tick = min(first_damage_tick, death_tick)
        elif first_damage_tick:
            contact_tick = first_damage_tick
        elif death_tick:
            contact_tick = death_tick
        
        # If we have contact, use position at contact
        if contact_tick:
            # Find position at or just before contact
            contact_position = None
            for t in window_ticks:
                if t['tick'] <= contact_tick:
                    contact_position = t
                else:
                    break
            
            if contact_position:
                zone = self.point_in_zone_A(
                    contact_position['x'], 
                    contact_position['y'], 
                    contact_position['z']
                )
                
                return {
                    'zone': zone or 'OUTSIDE_A',
                    'dwell_seconds': 0,
                    'dwell_fraction': 0,
                    'contact_tick': contact_tick,
                    'died': death_tick is not None,
                    'method': 'fight_position',
                    'all_zones': {zone: 0} if zone else {}
                }
        
        # PRIORITY 2: If retake detected, use position at retake start
        # This takes priority over dwell since it captures where Ts are when engagement begins
        if retake_start_tick:
            # Find position at or just before retake (not limited to window_end)
            retake_position = None
            for t in player_track:
                if t['tick'] <= retake_start_tick:
                    retake_position = t
                elif t['tick'] > retake_start_tick:
                    break
            
            if retake_position:
                zone = self.point_in_zone_A(
                    retake_position['x'], 
                    retake_position['y'], 
                    retake_position['z']
                )
                
                return {
                    'zone': zone or 'OUTSIDE_A',
                    'dwell_seconds': 0,
                    'dwell_fraction': 0,
                    'contact_tick': None,
                    'died': death_tick is not None,
                    'method': 'retake_position',
                    'all_zones': {zone: 0} if zone else {}
                }
        
        # PRIORITY 3: No contact, no retake - use dwell position
        # Use only ticks before any death
        if death_tick:
            analysis_ticks = [t for t in window_ticks if t['tick'] < death_tick]
        else:
            analysis_ticks = window_ticks
        
        if not analysis_ticks:
            return {
                'zone': 'DIED_IMMEDIATELY',
                'dwell_seconds': 0,
                'dwell_fraction': 0,
                'contact_tick': None,
                'died': True,
                'method': 'died_immediately',
                'all_zones': {}
            }
        
        # Label each tick with zone
        tick_labels = []
        prev_zone = None
        outside_count = 0
        
        for i, tick in enumerate(analysis_ticks):
            x, y, z = tick['x'], tick['y'], tick['z']
            
            # Get zone label
            zone = self.point_in_zone_A(x, y, z)
            
            # Check if player left A-site
            if zone is None:
                # If significantly outside, track it
                if not self.is_in_a_site(x, y, z):
                    outside_count += 1
                # Use prev zone (hysteresis)
                zone = prev_zone if prev_zone else 'A_SITE_UNLABELED'
            
            tick_labels.append(zone)
            prev_zone = zone
        
        # If player spent >50% of time outside A-site, mark as rotated
        if outside_count / len(analysis_ticks) > 0.5:
            return {
                'zone': 'ROTATED',
                'dwell_seconds': 0,
                'dwell_fraction': 0,
                'contact_tick': None,
                'died': death_tick is not None,
                'method': 'rotated',
                'all_zones': {}
            }
        
        # Calculate dwell times per zone
        zone_ticks = defaultdict(int)
        for zone in tick_labels:
            zone_ticks[zone] += 1
        
        # Convert to seconds
        tick_duration = 1.0 / tick_rate
        zone_durations = {
            zone: count * tick_duration 
            for zone, count in zone_ticks.items()
        }
        
        # Pick zone with most dwell
        if zone_durations:
            final_zone = max(zone_durations, key=zone_durations.get)
            dwell_time = zone_durations[final_zone]
        else:
            final_zone = 'A_SITE_UNLABELED'
            dwell_time = 0
        
        # Calculate fraction
        total_window = len(analysis_ticks) * tick_duration
        dwell_fraction = dwell_time / total_window if total_window > 0 else 0
        
        return {
            'zone': final_zone,
            'dwell_seconds': round(dwell_time, 2),
            'dwell_fraction': round(dwell_fraction, 3),
            'contact_tick': None,
            'died': death_tick is not None,
            'method': 'dwell_position',
            'all_zones': {k: round(v, 2) for k, v in zone_durations.items()}
        }


def example_usage():
    """Example usage of the zone classifier."""
    classifier = MirageZoneClassifier()
    
    # Test single point classification
    palace_x, palace_y, palace_z = 100.0, -2300.0, 24.0
    zone = classifier.point_in_zone_A(palace_x, palace_y, palace_z)
    print(f"Point ({palace_x}, {palace_y}, {palace_z}) -> {zone}")
    
    # Test with mock player track
    mock_track = [
        {'tick': 1000, 'x': 100, 'y': -2300, 'z': 24, 'alive': True},
        {'tick': 1001, 'x': 100, 'y': -2300, 'z': 24, 'alive': True},
        # ... more ticks
    ]
    
    plant_tick = 900
    result = classifier.assign_postplant_zone_A(mock_track, plant_tick)
    print(f"Assigned zone: {result}")


if __name__ == "__main__":
    example_usage()
