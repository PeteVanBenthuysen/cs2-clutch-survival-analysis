"""
Build zone connectivity graph from actual player movement in demo files.
Analyzes which zones players transition between to determine real neighbors.
"""
from demoparser2 import DemoParser
from src.zone_classifier import MirageZoneClassifier
from collections import defaultdict
import json
from pathlib import Path

def analyze_zone_transitions(demo_path, sample_interval=32):
    """
    Analyze player movement to find which zones connect to which.
    
    Args:
        demo_path: Path to demo file
        sample_interval: Sample every N ticks (32 = ~0.5s at 64 tick rate)
    
    Returns dict of zone -> {neighbor_zones with transition counts}
    """
    parser = DemoParser(demo_path)
    classifier = MirageZoneClassifier()
    
    # Verify this is actually a Mirage demo
    print(f"Parsing {Path(demo_path).name}...")
    header = parser.parse_header()
    map_name = header.get('map_name', '').lower()
    if 'mirage' not in map_name:
        print(f"  Skipping - not a Mirage demo (map: {map_name})")
        return {}
    
    # Get sampled player positions (much faster than every tick)
    ticks = parser.parse_ticks(['X', 'Y', 'Z', 'name', 'tick', 'is_alive'])
    
    # Filter to alive players only
    ticks = ticks[ticks['is_alive'] == True]
    
    # Sample positions every N ticks
    ticks = ticks[ticks['tick'] % sample_interval == 0]
    
    # Track zone transitions per player
    transitions = defaultdict(lambda: defaultdict(int))
    
    # Process each player
    for player_name in ticks['name'].unique():
        player_ticks = ticks[ticks['name'] == player_name].sort_values('tick')
        
        prev_zone = None
        for _, row in player_ticks.iterrows():
            zone = classifier.point_in_zone_A(row['X'], row['Y'], row['Z'])
            
            if zone and zone != prev_zone and prev_zone:
                # Record transition
                transitions[prev_zone][zone] += 1
                transitions[zone][prev_zone] += 1  # Bidirectional
            
            prev_zone = zone
    
    return transitions

def build_connectivity_from_demos(demo_folder, min_transitions=5):
    """
    Build connectivity graph from multiple demos.
    
    Args:
        demo_folder: Path to folder with demo files
        min_transitions: Minimum transitions needed to confirm neighbor relationship
    
    Returns:
        Dict of zone -> list of neighbor zones
    """
    demo_folder = Path(demo_folder)
    all_transitions = defaultdict(lambda: defaultdict(int))
    
    # Find all Mirage demos
    demos = list(demo_folder.rglob("*mirage*.dem"))
    print(f"Found {len(demos)} Mirage demos\n")
    
    # Analyze each demo
    num_demos = len(demos)  # Process ALL demos
    skipped = 0
    for i, demo_path in enumerate(demos[:num_demos], 1):
        print(f"[{i}/{num_demos}] Processing demo...")
        try:
            transitions = analyze_zone_transitions(str(demo_path), sample_interval=32)
            
            # Skip if not a Mirage demo
            if not transitions:
                skipped += 1
                continue
            
            # Merge transitions
            for zone, neighbors in transitions.items():
                for neighbor, count in neighbors.items():
                    all_transitions[zone][neighbor] += count
                    
            print(f"  Found {len(transitions)} zones with transitions")
        except Exception as e:
            print(f"  Error: {e}")
            skipped += 1
            continue
    
    # Build final connectivity (zones with enough transitions)
    connectivity = {}
    for zone, neighbors in sorted(all_transitions.items()):
        valid_neighbors = {
            neighbor: count 
            for neighbor, count in neighbors.items() 
            if count >= min_transitions
        }
        if valid_neighbors:
            connectivity[zone] = valid_neighbors
            print(f"\n{zone}:")
            for neighbor, count in sorted(valid_neighbors.items(), key=lambda x: x[1], reverse=True):
                print(f"  -> {neighbor}: {count} transitions")
    
    return connectivity, skipped

if __name__ == "__main__":
    # Build connectivity from demos
    demo_folder = r"research_demos\extracted"
    
    print("Building zone connectivity from actual player movement...\n")
    connectivity, skipped = build_connectivity_from_demos(demo_folder, min_transitions=10)
    
    # Save results
    output_file = "data/mirage_zone_connectivity_from_movement.json"
    with open(output_file, 'w') as f:
        json.dump(connectivity, f, indent=2)
    
    print(f"\n\nSaved to {output_file}")
    print(f"Found {len(connectivity)} zones with confirmed neighbors")
    print(f"Skipped {skipped} non-Mirage demos")
