"""
CS2 Movement Speed Calculator

Calculates actual player movement speed based on:
- Weapon held (active_weapon_name)
- Movement state (is_walking, in_crouch)
- Velocity data from demos

Replaces hardcoded 0.5x multiplier assumptions with empirical data.

Author: Pete Van Benthuysen
Date: November 2025
"""

import numpy as np
from typing import Optional, Dict


class MovementSpeedCalculator:
    """Calculate CS2 player movement speeds based on weapon and movement state."""
    
    # Base movement speeds (units/second) - empirically measured from demos
    WEAPON_SPEEDS = {
        # Knives
        'Butterfly Knife': 250.0,
        'Karambit': 250.0,
        'Bayonet': 250.0,
        'Knife': 250.0,
        
        # Pistols
        'Glock-18': 240.0,
        'USP-S': 240.0,
        'P2000': 240.0,
        'Desert Eagle': 230.0,
        'Dual Berettas': 240.0,
        'P250': 240.0,
        'Five-SeveN': 240.0,
        'Tec-9': 240.0,
        'CZ75-Auto': 240.0,
        'R8 Revolver': 220.0,
        
        # SMGs
        'MP9': 240.0,
        'MP7': 240.0,
        'MAC-10': 240.0,
        'MP5-SD': 235.0,
        'UMP-45': 230.0,
        'P90': 230.0,
        'PP-Bizon': 240.0,
        
        # Rifles
        'AK-47': 215.0,
        'M4A4': 215.0,
        'M4A1-S': 215.0,
        'Galil AR': 215.0,
        'FAMAS': 220.0,
        'SG 553': 210.0,
        'AUG': 220.0,
        
        # Snipers
        'AWP': 200.0,
        'SSG_08': 230.0,
        'SCAR-20': 215.0,
        'G3SG1': 215.0,
        
        # Heavy
        'Nova': 220.0,
        'XM1014': 215.0,
        'MAG-7': 215.0,
        'Sawed-Off': 210.0,
        'M249': 195.0,
        'Negev': 150.0,
        
        # Default (if weapon unknown)
        'default': 215.0,
    }
    
    # Movement state multipliers
    WALKING_MULTIPLIER = 0.52  # Shift-walking
    CROUCH_MULTIPLIER = 0.34   # Crouching
    
    def __init__(self):
        """Initialize movement speed calculator."""
        pass
    
    def get_base_speed(self, weapon_name: Optional[str]) -> float:
        """
        Get base movement speed for a weapon.
        
        Args:
            weapon_name: Name of weapon (e.g., 'AK-47', 'AWP', 'Knife')
        
        Returns:
            Base movement speed in units/second
        """
        if not weapon_name:
            return self.WEAPON_SPEEDS['default']
        
        return self.WEAPON_SPEEDS.get(weapon_name, self.WEAPON_SPEEDS['default'])
    
    def calculate_speed(self, 
                       weapon_name: Optional[str] = None,
                       is_walking: bool = False,
                       in_crouch: bool = False,
                       velocity_x: Optional[float] = None,
                       velocity_y: Optional[float] = None) -> float:
        """
        Calculate actual movement speed.
        
        Priority:
        1. If velocity data available, use it directly
        2. Otherwise, calculate from weapon + movement state
        
        Args:
            weapon_name: Active weapon name
            is_walking: True if shift-walking
            in_crouch: True if crouching
            velocity_x: X velocity component (units/second)
            velocity_y: Y velocity component (units/second)
        
        Returns:
            Movement speed in units/second
        """
        # If we have velocity data, use it directly
        if velocity_x is not None and velocity_y is not None:
            return np.sqrt(velocity_x**2 + velocity_y**2)
        
        # Otherwise calculate from weapon + state
        base_speed = self.get_base_speed(weapon_name)
        
        if in_crouch:
            return base_speed * self.CROUCH_MULTIPLIER
        elif is_walking:
            return base_speed * self.WALKING_MULTIPLIER
        else:
            return base_speed
    
    def estimate_travel_time(self,
                            distance: float,
                            weapon_name: Optional[str] = None,
                            is_walking: bool = False,
                            in_crouch: bool = False) -> float:
        """
        Estimate time to travel a distance.
        
        Args:
            distance: Distance in map units
            weapon_name: Active weapon
            is_walking: True if shift-walking
            in_crouch: True if crouching
        
        Returns:
            Travel time in seconds
        """
        speed = self.calculate_speed(weapon_name, is_walking, in_crouch)
        return distance / speed if speed > 0 else float('inf')
    
    @classmethod
    def from_demo_data(cls, demo_ticks_df) -> Dict[str, Dict[str, float]]:
        """
        Build empirical weapon speed table from demo data.
        
        Args:
            demo_ticks_df: DataFrame with columns: 
                           velocity_X, velocity_Y, active_weapon_name, is_walking, in_crouch
        
        Returns:
            Dict mapping weapon_name -> {'run': speed, 'walk': speed, 'crouch': speed}
        """
        import pandas as pd
        
        # Calculate 2D speed
        demo_ticks_df = demo_ticks_df.copy()
        demo_ticks_df['speed_2d'] = np.sqrt(
            demo_ticks_df['velocity_X']**2 + 
            demo_ticks_df['velocity_Y']**2
        )
        
        # Filter moving players (> 50 u/s)
        moving = demo_ticks_df[demo_ticks_df['speed_2d'] > 50].copy()
        
        # Group by weapon and movement state
        weapon_speeds = {}
        
        for weapon in moving['active_weapon_name'].dropna().unique():
            weapon_data = moving[moving['active_weapon_name'] == weapon]
            
            # Running speed (not walking, not crouching)
            running = weapon_data[
                (weapon_data['is_walking'] == False) & 
                (weapon_data['in_crouch'] == False)
            ]
            
            # Walking speed
            walking = weapon_data[weapon_data['is_walking'] == True]
            
            # Crouching speed
            crouching = weapon_data[weapon_data['in_crouch'] == True]
            
            weapon_speeds[weapon] = {
                'run': running['speed_2d'].median() if len(running) > 10 else None,
                'walk': walking['speed_2d'].median() if len(walking) > 10 else None,
                'crouch': crouching['speed_2d'].median() if len(crouching) > 10 else None,
                'samples': len(weapon_data)
            }
        
        return weapon_speeds
