"""
Test plant spot classification for A and B sites.
Verify that plant coordinates are correctly classified.
"""

import json
from pathlib import Path

# Load plant spots
with open('data/mirage_plant_spots.json', 'r') as f:
    plant_data = json.load(f)

print("=" * 80)
print("PLANT SPOT CLASSIFICATION TEST")
print("=" * 80)

def classify_plant(x, y, z, site='a'):
    """Classify plant based on coordinates."""
    import math
    
    site_key = 'a_site' if site == 'a' else 'b_site'
    spots = plant_data[site_key]
    
    closest_spot = None
    closest_distance = float('inf')
    
    for spot_name, spot_data in spots.items():
        center = spot_data['center']
        radius = spot_data['radius']
        
        # Calculate 2D distance (ignore Z)
        distance = math.sqrt(
            (x - center[0])**2 + 
            (y - center[1])**2
        )
        
        if distance <= radius:
            return spot_name, distance
        
        if distance < closest_distance:
            closest_distance = distance
            closest_spot = spot_name
    
    return f"UNCLASSIFIED (closest: {closest_spot})", closest_distance

# Test A-site plants
print("\nA-SITE PLANT SPOTS:")
print("-" * 80)
for spot_name, spot_data in plant_data['a_site'].items():
    center = spot_data['center']
    result, dist = classify_plant(center[0], center[1], center[2], 'a')
    status = "✓" if result == spot_name else "✗"
    print(f"{status} {spot_name:20s} -> {result:20s} (dist: {dist:.2f}m)")
    print(f"   Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"   {spot_data['description']}")

# Test B-site plants
print("\n" + "=" * 80)
print("B-SITE PLANT SPOTS:")
print("-" * 80)
for spot_name, spot_data in plant_data['b_site'].items():
    center = spot_data['center']
    result, dist = classify_plant(center[0], center[1], center[2], 'b')
    status = "✓" if result == spot_name else "✗"
    print(f"{status} {spot_name:20s} -> {result:20s} (dist: {dist:.2f}m)")
    print(f"   Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"   {spot_data['description']}")

print("\n" + "=" * 80)
print("TESTING NEARBY COORDINATES (slight variations)")
print("=" * 80)

# Test slight variations to ensure radius tolerance works
test_cases = [
    # A-site tests
    ('A-site Default +50 units X', -203.97, -2134.08, -111.0, 'a'),
    ('A-site Default +50 units Y', -253.97, -2084.08, -111.0, 'a'),
    ('A-site Ninja exact', -336.92, -2277.22, -103.97, 'a'),
    
    # B-site tests  
    ('B-site Default exact', -1969.03, 450.04, -95.97, 'b'),
    ('B-site Cat exact', -1887.32, 242.07, -95.97, 'b'),
    ('B-site Open exact', -2099.84, 244.68, -95.97, 'b'),
    ('B-site Empty exact', -2198.97, 89.03, -95.97, 'b'),
]

for test_name, x, y, z, site in test_cases:
    result, dist = classify_plant(x, y, z, site)
    print(f"{test_name:35s} -> {result:30s} (dist: {dist:.2f}m)")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
