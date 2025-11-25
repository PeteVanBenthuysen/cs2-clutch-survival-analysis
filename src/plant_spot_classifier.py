"""
Plant spot classifier for Mirage A and B sites.
Determines which plant spot was used based on bomb coordinates.
"""

import json
from pathlib import Path
from typing import Tuple, Optional
import math


class PlantSpotClassifier:
    """Classify plant locations on Mirage A and B sites."""
    
    def __init__(self, plant_spots_file: str = None):
        """Load plant spot definitions."""
        if plant_spots_file is None:
            plant_spots_file = Path(__file__).parent.parent / 'data' / 'mirage_plant_spots.json'
        
        with open(plant_spots_file, 'r') as f:
            data = json.load(f)
        
        self.a_site_spots = data['a_site']
        self.b_site_spots = data['b_site']
        self.fallback = data['fallback']
    
    def classify_plant(self, x: float, y: float, z: float, site: str = 'a') -> str:
        """
        Classify plant spot based on coordinates.
        Uses only X and Y (2D distance), ignoring Z since player crouches to ground level when planting.
        
        Args:
            x, y, z: Plant coordinates
            site: 'a' for A-site or 'b' for B-site
        
        Returns:
            Plant spot name (e.g., 'default', 'ninja', 'open')
        """
        # Select the appropriate site spots
        plant_spots = self.a_site_spots if site.lower() == 'a' else self.b_site_spots
        
        # Check each defined plant spot using 2D distance only
        for spot_name, spot_data in plant_spots.items():
            center = spot_data['center']
            radius = spot_data['radius']
            
            # Calculate 2D distance (ignore Z coordinate)
            distance = math.sqrt(
                (x - center[0])**2 + 
                (y - center[1])**2
            )
            
            if distance <= radius:
                return spot_name
        
        # No match found
        return self.fallback
    
    def get_plant_from_event(self, parser, plant_tick: int, site: str = 'a') -> Tuple[str, Tuple[float, float, float]]:
        """
        Get plant spot classification from demo parser.
        
        Args:
            parser: DemoParser instance
            plant_tick: Tick when bomb was planted
            site: 'a' for A-site or 'b' for B-site
        
        Returns:
            Tuple of (plant_spot_name, (x, y, z))
        """
        # Get who planted the bomb from bomb_planted event
        bomb_events = parser.parse_event('bomb_planted')
        plant_event = bomb_events[bomb_events['tick'] == plant_tick]
        
        if len(plant_event) == 0:
            return self.fallback, (0, 0, 0)
        
        planter_name = plant_event.iloc[0]['user_name']
        
        # Get planter position at plant tick
        ticks = parser.parse_ticks(['X', 'Y', 'Z', 'name', 'tick'])
        planter_pos = ticks[(ticks['name'] == planter_name) & (ticks['tick'] == plant_tick)]
        
        if len(planter_pos) == 0:
            return self.fallback, (0, 0, 0)
        
        pos = planter_pos.iloc[0]
        x, y, z = pos['X'], pos['Y'], pos['Z']
        
        spot = self.classify_plant(x, y, z, site)
        
        return spot, (x, y, z)
    
    def print_plant_info(self, spot_name: str, coords: Tuple[float, float, float], site: str = 'a'):
        """Print formatted plant spot information."""
        site_label = "A-site" if site.lower() == 'a' else "B-site"
        plant_spots = self.a_site_spots if site.lower() == 'a' else self.b_site_spots
        
        print(f"{site_label} plant spot: {spot_name}")
        print(f"Coordinates: ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})")
        
        if spot_name in plant_spots:
            description = plant_spots[spot_name]['description']
            print(f"Description: {description}")
