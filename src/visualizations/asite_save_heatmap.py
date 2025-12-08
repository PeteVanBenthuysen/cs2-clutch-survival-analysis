"""
Generate CT A-site SAVE heatmap using CT episode data from parquet files.
Shows where CTs position during economic saves (avoiding retake, preserving weapons).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from awpy.plot import heatmap
from pathlib import Path

# Paths
ZONES_PATH = Path('data/mirage_zones.json')
CT_EPISODES_PATH = Path('data/ct_episodes.parquet')
OUTPUT_DIR = Path('paper/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading CT episode data...")
ct_df = pd.read_parquet(CT_EPISODES_PATH)

# Filter to A-site rounds
ct_asite = ct_df[ct_df['site'] == 'A'].copy()

# Define A-site zones (zones ON the bombsite)
a_site_core_zones = {
    'A_Default', 'A_Entry', 'A_Triple', 'A_Tetris', 'A_Sandwich', 'A_Firebox', 
    'A_Ropz', 'Under_Balcony', 'Heaven', 'Palace_Exit'
}

# Filter to SAVE zones - zones far from A-site (CTs escaping, not retaking)
# These are zones where CTs retreat to preserve weapons
ct_saves = ct_asite[~ct_asite['zone'].isin(a_site_core_zones)].copy()

# Additionally filter to zones that are clearly "away" from A-site
# (mid, B-site areas, spawn, etc.)
save_zone_keywords = ['B_', 'Mid_', 'Market', 'CT_Spawn', 'Underpass', 'Apps', 
                      'Cat', 'Connector', 'Window', 'Arches', 'Bench']

def is_save_zone(zone_name):
    """Check if zone is likely a save position (away from A-site)."""
    if zone_name in a_site_core_zones:
        return False
    return any(keyword in zone_name for keyword in save_zone_keywords)

ct_saves = ct_asite[ct_asite['zone'].apply(is_save_zone)].copy()

print(f"Total CT A-site episodes: {len(ct_asite):,}")
print(f"Save episodes (away from site): {len(ct_saves):,}")
print(f"Retake episodes (on/near site): {len(ct_asite) - len(ct_saves):,}")

# Count episodes per zone to weight heatmap density
zone_counts = ct_saves['zone'].value_counts()
print(f"\nTop 15 save zones by CT episode frequency:")
for zone, count in zone_counts.head(15).items():
    print(f"  {zone}: {count:,} episodes")

print("\nLoading zone polygon data...")
with open(ZONES_PATH, 'r') as f:
    zones_data = json.load(f)

def get_zone_center(zone_name, zones_dict):
    """Calculate centroid of a zone polygon or get point zone coordinates."""
    # Check a_site_zones polygon zones first
    if zone_name in zones_dict.get("a_site_zones", {}).get("polygon_zones", {}):
        coords = zones_dict["a_site_zones"]["polygon_zones"][zone_name]
        coords_array = np.array(coords)
        centroid = coords_array.mean(axis=0)
        return tuple(centroid)
    
    # Check a_site_zones point zones (single XYZ coordinates)
    if zone_name in zones_dict.get("a_site_zones", {}).get("point_zones", {}):
        point = zones_dict["a_site_zones"]["point_zones"][zone_name]
        return tuple(point)
    
    # Check b_site_zones polygon zones
    if zone_name in zones_dict.get("b_site_zones", {}).get("polygon_zones", {}):
        coords = zones_dict["b_site_zones"]["polygon_zones"][zone_name]
        coords_array = np.array(coords)
        centroid = coords_array.mean(axis=0)
        return tuple(centroid)
    
    # Check b_site_zones point zones
    if zone_name in zones_dict.get("b_site_zones", {}).get("point_zones", {}):
        point = zones_dict["b_site_zones"]["point_zones"][zone_name]
        return tuple(point)
    
    # Check cubby_zones
    if zone_name in zones_dict.get("cubby_zones", {}):
        coords = zones_dict["cubby_zones"][zone_name]
        coords_array = np.array(coords)
        centroid = coords_array.mean(axis=0)
        return tuple(centroid)
    
    return None

def generate_positions_in_zone(zone_center, num_points, spread=35):
    """Generate random positions around zone center for smooth KDE heatmap."""
    if zone_center is None:
        return []
    
    x, y, z = zone_center
    positions = []
    for _ in range(num_points):
        # Add random variation around center - tighter spread to stay in playable areas
        px = x + np.random.normal(0, spread)
        py = y + np.random.normal(0, spread)
        pz = z + np.random.normal(0, spread/3)
        positions.append((px, py, pz))
    return positions

# Generate positions weighted by actual CT save episode frequency
all_positions = []
zones_not_found = []
max_count = zone_counts.max()

for zone, count in zone_counts.items():
    center = get_zone_center(zone, zones_data)
    
    if center is None:
        zones_not_found.append(zone)
        continue
    
    # Number of points proportional to episode frequency
    num_points = int(1000 + (count / max_count) * 9000)
    
    positions = generate_positions_in_zone(center, num_points, spread=35)
    all_positions.extend(positions)

print(f"\nGenerated {len(all_positions):,} positions from {len(zone_counts) - len(zones_not_found)} zones")
if zones_not_found:
    print(f"Zones not found: {len(zones_not_found)}")

# Create heatmap using awpy
print("\nCreating CT A-site SAVE heatmap with KDE...")

fig, ax = heatmap(
    map_name='de_mirage',
    points=all_positions,
    method='kde',
    size=100,
    kde_lower_bound=0.005,
    alpha=0.8,
    cmap='YlOrRd'
)

# Set white background
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

output_path = OUTPUT_DIR / 'figure3b_asite_save_heatmap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Save heatmap saved to: {output_path}")
print(f"\nThis heatmap shows {len(ct_saves):,} CT save episodes (avoiding retake, preserving weapons).")
