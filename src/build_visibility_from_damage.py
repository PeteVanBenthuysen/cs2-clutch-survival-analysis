"""
Build zone visibility graph from actual damage data in demo files.
Analyzes where attackers were when they damaged victims to determine real sightlines.
Uses gun damage only (excludes grenades) for more accurate visibility mapping.
"""
from demoparser2 import DemoParser
from src.zone_classifier import MirageZoneClassifier
from collections import defaultdict
import json
from pathlib import Path
import pandas as pd

def analyze_damage_visibility(demo_path):
    """
    Analyze gun damage events to find which zones have visibility to which zones.
    
    Args:
        demo_path: Path to demo file
    
    Returns dict of attacker_zone -> {victim_zones with damage instance counts}
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
    
    # Get damage events
    try:
        damage_events = parser.parse_event('player_hurt')
    except Exception as e:
        print(f"  Error parsing events: {e}")
        return {}
    
    # Filter to gun damage only (exclude grenades and utility)
    gun_damage = damage_events[
        ~damage_events['weapon'].str.contains('grenade|inferno|flashbang|smokegrenade|molotov|incgrenade|decoy', case=False, na=False)
    ]
    
    if gun_damage.empty:
        print(f"  No gun damage events found")
        return {}
    
    print(f"  Found {len(gun_damage)} gun damage events, parsing positions for those ticks only...")
    
    # Get unique ticks where gun damage occurred - ONLY parse these ticks
    damage_ticks = gun_damage['tick'].unique().tolist()
    
    try:
        # Parse ONLY the ticks where damage happened
        ticks = parser.parse_ticks(['X', 'Y', 'Z', 'name'], ticks=damage_ticks)
    except Exception as e:
        print(f"  Error parsing positions: {e}")
        return {}
    
    # Join damage with attacker positions
    damage_with_pos = gun_damage.merge(
        ticks.rename(columns={'name': 'attacker_name', 'X': 'attacker_X', 'Y': 'attacker_Y', 'Z': 'attacker_Z'}),
        left_on=['tick', 'attacker_name'],
        right_on=['tick', 'attacker_name'],
        how='left'
    )
    
    # Join with victim positions
    damage_with_pos = damage_with_pos.merge(
        ticks.rename(columns={'name': 'user_name', 'X': 'user_X', 'Y': 'user_Y', 'Z': 'user_Z'}),
        left_on=['tick', 'user_name'],
        right_on=['tick', 'user_name'],
        how='left'
    )
    
    # Filter out rows with missing positions
    damage_with_pos = damage_with_pos.dropna(subset=['attacker_X', 'attacker_Y', 'attacker_Z', 'user_X', 'user_Y', 'user_Z'])
    
    if damage_with_pos.empty:
        print(f"  No damage events with valid positions")
        return {}
    
    print(f"  Classifying {len(damage_with_pos)} damage events into zones...")
    
    # Track visibility relationships
    visibility = defaultdict(lambda: defaultdict(int))
    
    # Classify zones for each damage event
    for _, dmg in damage_with_pos.iterrows():
        attacker_zone = classifier.point_in_zone_A(dmg['attacker_X'], dmg['attacker_Y'], dmg['attacker_Z'])
        victim_zone = classifier.point_in_zone_A(dmg['user_X'], dmg['user_Y'], dmg['user_Z'])
        
        if attacker_zone and victim_zone:
            visibility[attacker_zone][victim_zone] += 1
    
    print(f"  Found {len(visibility)} zones with damage sightlines")
    return visibility

def build_visibility_from_demos(demo_folder, min_damage_instances=20):
    """
    Build visibility graph from multiple demos using gun damage events.
    
    Args:
        demo_folder: Path to folder with demo files
        min_damage_instances: Minimum damage instances needed to confirm visibility relationship
    
    Returns:
        Dict of zone -> list of zones it can see
    """
    demo_folder = Path(demo_folder)
    all_visibility = defaultdict(lambda: defaultdict(int))
    
    # Find all Mirage demos
    demos = list(demo_folder.rglob("*mirage*.dem"))
    print(f"Found {len(demos)} Mirage demos\n")
    
    # Analyze each demo
    num_demos = len(demos)  # Process ALL demos
    skipped = 0
    for i, demo_path in enumerate(demos[:num_demos], 1):
        print(f"[{i}/{num_demos}] Processing demo...")
        try:
            visibility = analyze_damage_visibility(str(demo_path))
            
            # Skip if not a Mirage demo
            if not visibility:
                skipped += 1
                continue
            
            # Merge visibility data
            for killer_zone, victim_zones in visibility.items():
                for victim_zone, count in victim_zones.items():
                    all_visibility[killer_zone][victim_zone] += count
                    
        except Exception as e:
            print(f"  Error: {e}")
            skipped += 1
            continue
    
    # Build final visibility graph (zones with enough damage instances to confirm sightlines)
    visibility_graph = {}
    for killer_zone, victim_zones in sorted(all_visibility.items()):
        visible_zones = {
            victim_zone: count 
            for victim_zone, count in victim_zones.items() 
            if count >= min_damage_instances
        }
        if visible_zones:
            visibility_graph[killer_zone] = list(visible_zones.keys())
            print(f"\n{killer_zone} can see:")
            for victim_zone, count in sorted(visible_zones.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  → {victim_zone}: {count} damage instances")
    
    return visibility_graph, skipped

if __name__ == "__main__":
    import pandas as pd
    
    # Build visibility from demos
    demo_folder = r"research_demos\extracted"
    
    print("Building zone visibility from actual gun damage data...\n")
    visibility_graph, skipped = build_visibility_from_demos(demo_folder, min_damage_instances=20)
    
    # Save results
    output_file = "data/mirage_visibility_from_damage.json"
    output_data = {
        "description": "Empirical zone visibility graph for Mirage, derived from gun damage events across 581 professional demos. Each zone lists zones it has confirmed sightlines to (minimum 20 damage instances). Excludes grenade/utility damage.",
        "zones": visibility_graph
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n\nSaved to {output_file}")
    print(f"Found {len(visibility_graph)} zones with confirmed visibility")
    print(f"Skipped {skipped} non-Mirage demos")
