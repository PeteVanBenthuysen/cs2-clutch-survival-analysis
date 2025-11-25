"""
Test post-plant analysis for A and B sites.
Verify plant spots, defender positions, and zone tracking.
"""

import sqlite3
from pathlib import Path
from demoparser2 import DemoParser
from src.zone_classifier import MirageZoneClassifier
from src.plant_spot_classifier import PlantSpotClassifier

# Initialize classifiers
zone_classifier = MirageZoneClassifier()
plant_classifier = PlantSpotClassifier()

print("=" * 80)
print("POST-PLANT ANALYSIS TEST")
print("=" * 80)

# Find a demo file
demo_dir = Path("research_demos/extracted")
demo_files = list(demo_dir.glob("**/*.dem"))

if not demo_files:
    print("No demo files found in research_demos/extracted")
    exit(1)

# Use first demo
demo_path = demo_files[0]
print(f"\nUsing demo: {demo_path.name}")
print(f"Path: {demo_path}")

try:
    parser = DemoParser(str(demo_path))
    
    # Get bomb plant events
    bomb_events = parser.parse_event('bomb_planted')
    print(f"\nFound {len(bomb_events)} bomb plants in demo")
    
    if len(bomb_events) == 0:
        print("No bomb plants found in this demo")
        exit(0)
    
    # Analyze first 3 plants
    for i, (idx, plant_event) in enumerate(bomb_events.head(3).iterrows()):
        if i > 0:
            print("\n" + "=" * 80)
        
        plant_tick = plant_event['tick']
        print(f"\nPlant #{i+1} at tick {plant_tick}")
        print("-" * 80)
        
        # Determine site from bomb position
        # A-site is roughly X: -600 to 200, Y: -2400 to -1800
        # B-site is roughly X: -2400 to -1600, Y: -200 to 900
        
        # Get planter position
        planter_name = plant_event['user_name']
        ticks = parser.parse_ticks(['X', 'Y', 'Z', 'name', 'tick'])
        planter_pos = ticks[(ticks['name'] == planter_name) & (ticks['tick'] == plant_tick)]
        
        if len(planter_pos) == 0:
            print(f"Could not find planter position")
            continue
        
        pos = planter_pos.iloc[0]
        x, y, z = pos['X'], pos['Y'], pos['Z']
        
        # Determine site based on coordinates
        if -600 <= x <= 200 and -2400 <= y <= -1800:
            site = 'a'
            site_name = "A-site"
        elif -2400 <= x <= -1600 and -200 <= y <= 900:
            site = 'b'
            site_name = "B-site"
        else:
            site = 'a'  # default
            site_name = "Unknown (defaulting to A-site)"
        
        print(f"Site: {site_name}")
        
        # Classify plant spot
        plant_spot = plant_classifier.classify_plant(x, y, z, site)
        print(f"\nPlant Information:")
        plant_classifier.print_plant_info(plant_spot, (x, y, z), site)
        
        # Get positions at T+2s after plant
        tick_2s_after = plant_tick + int(2.0 * 64)  # 64 tick/sec
        
        player_ticks = parser.parse_ticks(['X', 'Y', 'Z', 'name', 'tick', 'team_name', 'is_alive'])
        
        ct_positions = player_ticks[
            (player_ticks['tick'] == tick_2s_after) & 
            (player_ticks['team_name'] == 'CT') & 
            (player_ticks['is_alive'] == True)
        ]
        
        print(f"\nCT Positions at T+2.0s after plant:")
        if len(ct_positions) > 0:
            for _, ct in ct_positions.iterrows():
                zone = zone_classifier.point_in_zone_A(ct['X'], ct['Y'], ct['Z'])
                print(f"  {ct['name']:20s} -> Zone: {zone:25s} ({ct['X']:.1f}, {ct['Y']:.1f}, {ct['Z']:.1f})")
        else:
            print("  No alive CTs at T+2.0s")
        
        # Get T positions
        t_positions = player_ticks[
            (player_ticks['tick'] == tick_2s_after) & 
            (player_ticks['team_name'] == 'TERRORIST') & 
            (player_ticks['is_alive'] == True)
        ]
        
        print(f"\nT Positions at T+2.0s after plant:")
        if len(t_positions) > 0:
            for _, t in t_positions.iterrows():
                zone = zone_classifier.point_in_zone_A(t['X'], t['Y'], t['Z'])
                print(f"  {t['name']:20s} -> Zone: {zone:25s} ({t['X']:.1f}, {t['Y']:.1f}, {t['Z']:.1f})")
        else:
            print("  No alive Ts at T+2.0s")

except Exception as e:
    print(f"Error analyzing demo: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
