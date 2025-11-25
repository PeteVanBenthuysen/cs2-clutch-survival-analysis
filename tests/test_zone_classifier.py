"""
Demo parser test script for validating A-site zone classification.

Tests on match 2381637 (mouz vs BIG, Mirage) round 1.
Expected T post-plant positions: market door, firebox, palace, back side triple

Author: Pete Van Benthuysen
Date: November 2025
"""

import sys
from pathlib import Path
from demoparser2 import DemoParser
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.zone_classifier import MirageZoneClassifier


def parse_demo_round(demo_path: str, round_num: int = 1):
    """
    Parse a specific round from a demo file.
    
    Args:
        demo_path: Path to .dem file
        round_num: Round number to extract (1-indexed)
    
    Returns:
        Dict with round data including player tracks
    """
    print(f"\n{'='*60}")
    print(f"Parsing demo: {Path(demo_path).name}")
    print(f"Target round: {round_num}")
    print(f"{'='*60}\n")
    
    parser = DemoParser(demo_path)
    
    # Parse events we need
    print("Extracting events...")
    
    # Get bomb plant events
    bomb_plants = parser.parse_event("bomb_planted")
    print(f"Found {len(bomb_plants)} bomb plants")
    
    # Get round start/end events
    round_starts = parser.parse_event("round_start")
    round_ends = parser.parse_event("round_end_official")
    
    # Get player positions (ticks)
    print("Extracting player positions...")
    ticks = parser.parse_ticks([
        "X", "Y", "Z", "name", "steamid", "team_name", 
        "is_alive", "tick", "health"
    ])
    
    print(f"Extracted {len(ticks)} position samples")
    
    # Find the plant event for our target round
    target_plant = None
    for idx, plant in bomb_plants.iterrows():
        # Round number is typically in round_start events
        # For now, use the Nth plant
        if idx + 1 == round_num:
            target_plant = plant
            break
    
    if target_plant is None:
        print(f"ERROR: Could not find plant event for round {round_num}")
        return None
    
    plant_tick = target_plant['tick']
    site = target_plant.get('site', 'Unknown')
    
    print(f"\nRound {round_num} plant info:")
    print(f"  Tick: {plant_tick}")
    print(f"  Site: {site}")
    
    # Get tick rate from demo
    tick_rate = 64  # CS2 default, could parse from header if needed
    
    # Define the window [plant+2s, plant+4s]
    window_start = plant_tick + (2.0 * tick_rate)
    window_end = plant_tick + (4.0 * tick_rate)
    
    print(f"  Setup window: ticks {int(window_start)} to {int(window_end)}")
    
    # Filter player positions in this window
    window_ticks = ticks[
        (ticks['tick'] >= window_start) & 
        (ticks['tick'] <= window_end)
    ].copy()
    
    print(f"\nFound {len(window_ticks)} position samples in setup window")
    
    # Group by player
    player_tracks = {}
    for player_name, group in window_ticks.groupby('name'):
        # Check if this player is T side (for Mirage, figure out which team planted)
        # Simplified: assume team that planted is T side
        player_team = group['team_name'].iloc[0]
        
        track = []
        for _, row in group.iterrows():
            track.append({
                'tick': row['tick'],
                'x': row['X'],
                'y': row['Y'],
                'z': row['Z'],
                'alive': row['is_alive']
            })
        
        player_tracks[player_name] = {
            'team': player_team,
            'track': track
        }
    
    print(f"\nTracked {len(player_tracks)} unique players")
    
    return {
        'round_num': round_num,
        'plant_tick': plant_tick,
        'tick_rate': tick_rate,
        'site': site,
        'window_start': window_start,
        'window_end': window_end,
        'player_tracks': player_tracks
    }


def test_zone_classification(round_data: dict, expected_zones: dict = None):
    """
    Test zone classifier on extracted round data.
    
    Args:
        round_data: Output from parse_demo_round
        expected_zones: Optional dict of {player_name: expected_zone} for validation
    """
    print(f"\n{'='*60}")
    print("TESTING ZONE CLASSIFICATION")
    print(f"{'='*60}\n")
    
    classifier = MirageZoneClassifier()
    
    results = []
    
    for player_name, player_data in round_data['player_tracks'].items():
        track = player_data['track']
        team = player_data['team']
        
        if not track:
            continue
        
        # Assign zone
        assignment = classifier.assign_postplant_zone_A(
            track,
            round_data['plant_tick'],
            round_data['tick_rate']
        )
        
        result = {
            'player': player_name,
            'team': team,
            'zone': assignment['zone'],
            'dwell_seconds': assignment['dwell_seconds'],
            'dwell_fraction': assignment['dwell_fraction'],
            'rotated': assignment['rotated'],
            'died': assignment['died'],
            'all_zones': assignment['all_zones']
        }
        
        results.append(result)
        
        # Print result
        print(f"{player_name:20s} ({team:15s}): {assignment['zone']:20s} "
              f"(dwell: {assignment['dwell_seconds']:.2f}s, {assignment['dwell_fraction']:.1%})")
        
        if assignment['all_zones']:
            print(f"{'':38s}  All zones: {assignment['all_zones']}")
        
        # Validate against expected if provided
        if expected_zones and player_name in expected_zones:
            expected = expected_zones[player_name]
            actual = assignment['zone']
            match = "✓" if expected.lower() in actual.lower() or actual.lower() in expected.lower() else "✗"
            print(f"{'':38s}  Expected: {expected:20s} {match}")
        
        print()
    
    return results


def main():
    """Run the test on match 2381637 round 1."""
    
    demo_path = r"C:\Users\petev\OneDrive\Desktop\cs2-clutch-survival-analysis\research_demos\extracted\8036_IEM_Melbourne_2025\2381637\mouz-vs-big-m2-mirage.dem"
    
    # Parse round 1
    round_data = parse_demo_round(demo_path, round_num=1)
    
    if round_data is None:
        print("Failed to parse round data")
        return
    
    # Expected positions from user
    # Note: We don't know exact player names yet, will see from output
    expected = {
        # Will fill in after seeing player names
        # "player1": "Market_Door",
        # "player2": "Firebox",
        # "player3": "Palace", 
        # "player4": "Triple"
    }
    
    # Test classification
    results = test_zone_classification(round_data, expected)
    
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}\n")
    
    # Separate T and CT
    t_results = [r for r in results if 'terrorist' in r['team'].lower() or 'T' in r['team']]
    ct_results = [r for r in results if 'counter' in r['team'].lower() or 'CT' in r['team']]
    
    print(f"T-side players in setup window:")
    for r in t_results:
        print(f"  {r['player']:20s} -> {r['zone']}")
    
    print(f"\nExpected T positions: market door, firebox, palace, back side triple")
    print(f"\nDo the zones match? Review the output above.")
    
    # Check for issues
    issues = []
    for r in t_results:
        if r['zone'] in ['ROTATED', 'A_SITE_UNLABELED', None]:
            issues.append(f"{r['player']} -> {r['zone']}")
    
    if issues:
        print(f"\n⚠️  POTENTIAL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"\nThese might indicate polygon/anchor problems")
    else:
        print(f"\n✓ All T players got valid zone assignments")


if __name__ == "__main__":
    main()
