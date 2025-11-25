"""
Batch processing for A-site post-plant position labeling.

Processes demo files to extract post-plant positions for all rounds
where bomb is planted on A-site.

Author: Pete Van Benthuysen
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from collections import defaultdict

from .zone_classifier import MirageZoneClassifier


class PostPlantBatchProcessor:
    """Batch process demos to label T post-plant positions on A-site."""
    
    def __init__(self, zone_classifier: Optional[MirageZoneClassifier] = None):
        """
        Initialize batch processor.
        
        Args:
            zone_classifier: Optional pre-initialized classifier. Creates new if None.
        """
        self.classifier = zone_classifier or MirageZoneClassifier()
    
    def process_round(self, round_data: Dict) -> List[Dict]:
        """
        Process a single round to extract post-plant positions.
        
        Args:
            round_data: Dict containing:
                - round_id: Unique round identifier
                - tick_rate: Server tick rate
                - plant_tick: Tick when bomb was planted
                - plant_site: 'A' or 'B'
                - player_tracks: Dict[player_id -> List[position_dicts]]
                - t_side_players: List of player IDs on T side at plant time
        
        Returns:
            List of dicts, one per T player with:
                - round_id
                - tick_rate
                - plant_tick
                - plant_time_seconds
                - player_id
                - zone_A
                - dwell_seconds
                - dwell_fraction
                - rotated
                - died
                - all_zones_debug
        """
        # Skip if not A plant
        if round_data.get('plant_site') != 'A':
            return []
        
        plant_tick = round_data['plant_tick']
        tick_rate = round_data.get('tick_rate', 64)
        plant_time_seconds = plant_tick / tick_rate
        
        results = []
        
        for player_id in round_data.get('t_side_players', []):
            # Get player's position track
            track = round_data['player_tracks'].get(player_id, [])
            
            if not track:
                continue
            
            # Assign zone
            assignment = self.classifier.assign_postplant_zone_A(
                track, 
                plant_tick,
                tick_rate
            )
            
            results.append({
                'round_id': round_data['round_id'],
                'tick_rate': tick_rate,
                'plant_tick': plant_tick,
                'plant_time_seconds': round(plant_time_seconds, 2),
                'player_id': player_id,
                'zone_A': assignment['zone'],
                'dwell_seconds': assignment['dwell_seconds'],
                'dwell_fraction': assignment['dwell_fraction'],
                'rotated': assignment['rotated'],
                'died': assignment['died'],
                'all_zones_debug': str(assignment['all_zones'])
            })
        
        return results
    
    def process_multiple_rounds(self, rounds_data: List[Dict]) -> pd.DataFrame:
        """
        Process multiple rounds and return DataFrame.
        
        Args:
            rounds_data: List of round_data dicts (see process_round for schema)
        
        Returns:
            DataFrame with all labeled positions
        """
        all_results = []
        
        for round_data in tqdm(rounds_data, desc="Processing rounds"):
            round_results = self.process_round(round_data)
            all_results.extend(round_results)
        
        df = pd.DataFrame(all_results)
        return df
    
    def get_zone_distribution(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate zone distribution (counts per zone).
        
        Args:
            df: DataFrame from process_multiple_rounds
        
        Returns:
            Series with zone counts
        """
        # Exclude special labels
        valid_zones = df[
            ~df['zone_A'].isin(['ROTATED', 'DIED_IMMEDIATELY', 'A_SITE_UNLABELED', None])
        ]
        return valid_zones['zone_A'].value_counts()
    
    def get_forward_pair_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze pairs of forward Ts per round.
        
        For each round, finds the two most forward Ts (closest to A-main/default)
        and creates a pair label like "Palace+Ramp".
        
        Args:
            df: DataFrame from process_multiple_rounds
        
        Returns:
            DataFrame with columns: round_id, forward_1, forward_2, pair_label, count
        """
        # Define "forwardness" - zones closer to default/site are more forward
        # This is a simplified heuristic - adjust based on actual map knowledge
        forwardness_rank = {
            'Default': 1,
            'Firebox': 2,
            'Triple': 3,
            'Palace': 4,
            'Tetris': 5,
            'Stairs_Platform': 6,
            'Top_Stairs': 7,
            'Bottom_Stairs': 8,
            'Sandwich': 9,
            'A_Ramp_Room': 10,
            'Heaven': 11,
            'Heaven_Cubby': 12,
            'Under_Balcony': 13
        }
        
        pairs = []
        
        for round_id, group in df.groupby('round_id'):
            # Filter to valid zones only
            valid = group[
                ~group['zone_A'].isin(['ROTATED', 'DIED_IMMEDIATELY', 'A_SITE_UNLABELED', None])
            ].copy()
            
            if len(valid) < 2:
                continue
            
            # Rank by forwardness
            valid['forward_rank'] = valid['zone_A'].map(forwardness_rank)
            valid = valid.sort_values('forward_rank')
            
            # Get top 2
            top_2 = valid.head(2)
            zones = sorted(top_2['zone_A'].tolist())  # Sort for consistent pair naming
            
            pair_label = '+'.join(zones)
            
            pairs.append({
                'round_id': round_id,
                'forward_1': zones[0],
                'forward_2': zones[1] if len(zones) > 1 else None,
                'pair_label': pair_label
            })
        
        pair_df = pd.DataFrame(pairs)
        
        # Count frequency of each pair
        pair_counts = pair_df['pair_label'].value_counts().reset_index()
        pair_counts.columns = ['pair_label', 'count']
        
        return pair_counts
    
    def save_results(self, df: pd.DataFrame, output_dir: Path, 
                    prefix: str = "postplant_positions"):
        """
        Save results to CSV and generate QA artifacts.
        
        Args:
            df: DataFrame from process_multiple_rounds
            output_dir: Directory to save outputs
            prefix: Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        full_path = output_dir / f"{prefix}_full.csv"
        df.to_csv(full_path, index=False)
        print(f"Saved full results to {full_path}")
        
        # Save 50-row sample for quick QA
        sample_path = output_dir / f"{prefix}_sample.csv"
        sample = df.head(50)[['round_id', 'plant_time_seconds', 'player_id', 
                              'zone_A', 'dwell_fraction']]
        sample.to_csv(sample_path, index=False)
        print(f"Saved sample to {sample_path}")
        
        # Save zone distribution
        dist = self.get_zone_distribution(df)
        dist_path = output_dir / f"{prefix}_zone_distribution.csv"
        dist.to_csv(dist_path, header=['count'])
        print(f"Saved zone distribution to {dist_path}")
        
        # Save forward pair analysis
        pairs = self.get_forward_pair_analysis(df)
        pairs_path = output_dir / f"{prefix}_forward_pairs.csv"
        pairs.to_csv(pairs_path, index=False)
        print(f"Saved forward pair analysis to {pairs_path}")
        
        return {
            'full': full_path,
            'sample': sample_path,
            'distribution': dist_path,
            'pairs': pairs_path
        }


def example_usage():
    """Example usage with mock data."""
    
    # Mock round data
    mock_rounds = [
        {
            'round_id': 'match1_round5',
            'tick_rate': 64,
            'plant_tick': 10000,
            'plant_site': 'A',
            't_side_players': ['player1', 'player2'],
            'player_tracks': {
                'player1': [
                    {'tick': 10128, 'x': 100, 'y': -2300, 'z': 24, 'alive': True},
                    {'tick': 10129, 'x': 100, 'y': -2300, 'z': 24, 'alive': True},
                    # ... more ticks
                ],
                'player2': [
                    {'tick': 10128, 'x': -200, 'y': -2150, 'z': -105, 'alive': True},
                    {'tick': 10129, 'x': -200, 'y': -2150, 'z': -105, 'alive': True},
                    # ... more ticks
                ]
            }
        }
    ]
    
    processor = PostPlantBatchProcessor()
    df = processor.process_multiple_rounds(mock_rounds)
    
    print(df)
    print("\nZone distribution:")
    print(processor.get_zone_distribution(df))


if __name__ == "__main__":
    example_usage()
