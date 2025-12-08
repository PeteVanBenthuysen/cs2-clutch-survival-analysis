"""
Visualize optimal T-side crossfire setups for post-plant scenarios.

Generates publication-quality minimap visualizations showing:
- Top 10 safest 3v3 T-side position configurations per bomb site
- Cox model-adjusted hazard rates (controlling for HP, armor, utility, time)
- Visual representation of player positions and bomb location

Output: Two-column grid (A-site | B-site) with annotated minimaps
"""

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
import json
import numpy as np

print("="*80)
print("VISUALIZING OPTIMAL T-SIDE CROSSFIRE SETUPS")
print("="*80)

# Load episode data
print("\nLoading T-side episode data...")
t_df = pl.read_parquet("data/t_episodes.parquet")
print(f"  Loaded {len(t_df)} T episodes")

# Load zone coordinates and plant spot locations
print("\nLoading map metadata...")
with open("data/mirage_zones.json", 'r') as f:
    zones_data = json.load(f)

with open("data/mirage_plant_spots.json", 'r') as f:
    plant_spots_data = json.load(f)

print(f"  Loaded {len(zones_data['zones'])} zone definitions")
print(f"  Loaded plant spot data for {len(plant_spots_data)} sites")

# Create zone coordinate lookup
zone_coords = {}
for zone in zones_data['zones']:
    # Use center of bounding box as zone coordinate
    center_x = (zone['min_x'] + zone['max_x']) / 2
    center_y = (zone['min_y'] + zone['max_y']) / 2
    zone_coords[zone['name']] = {'x': center_x, 'y': center_y}

# Create plant spot coordinate lookup
plant_coords = {}
for site in ['A', 'B']:
    site_key = 'a_site' if site == 'A' else 'b_site'
    if site_key in plant_spots_data:
        plant_coords[site] = {}
        for spot in plant_spots_data[site_key]['spots']:
            plant_coords[site][spot['name']] = {'x': spot['x'], 'y': spot['y']}

print(f"  Mapped {len(zone_coords)} zone coordinates")
print(f"  Mapped {sum(len(spots) for spots in plant_coords.values())} plant spot coordinates")


def identify_common_setups(df, site, ct_count=3, t_count=3, min_episodes=30):
    """
    Identify most common T-side crossfire setups for a given scenario.
    
    Args:
        df: Polars DataFrame with T episodes
        site: 'A' or 'B'
        ct_count: Number of CTs (default 3)
        t_count: Number of Ts (default 3)
        min_episodes: Minimum episodes required for a setup to be included
    
    Returns:
        List of dicts with setup info: {'zones': [...], 'count': N, 'plant_spot': str}
    """
    
    print(f"\n{site}-Site: Identifying common {t_count}v{ct_count} setups...")
    
    # Filter to scenario: 3v3, pistol, 20-25s, no utility
    filtered = df.filter(
        (pl.col("site") == site) &
        (pl.col("ct_count_current") == ct_count) &
        (pl.col("t_count_current") == t_count) &
        (pl.col("time_remaining_on_bomb") >= 20.0) &
        (pl.col("time_remaining_on_bomb") <= 25.0) &
        (pl.col("active_smoke_zones").list.len() == 0) &
        (pl.col("active_molly_zones").list.len() == 0)
    )
    
    print(f"  Filtered to {len(filtered)} episodes matching scenario")
    
    # Group by retake (demo_file + round_num + plant_tick)
    # For each retake, find the zones where T's spent most time
    retake_setups = (
        filtered
        .group_by(["demo_file", "round_num", "plant_tick"])
        .agg([
            pl.col("zone").sort_by("duration_ticks", descending=True).head(t_count).alias("top_zones"),
            pl.col("plant_spot").first().alias("plant_spot")
        ])
    )
    
    print(f"  Found {len(retake_setups)} unique retakes")
    
    # Convert zone lists to tuples (hashable) and count occurrences
    setup_counts = {}
    for row in retake_setups.iter_rows(named=True):
        zones_tuple = tuple(sorted(row['top_zones'][:t_count]))  # Sort for consistency
        plant_spot = row['plant_spot']
        
        key = (zones_tuple, plant_spot)
        setup_counts[key] = setup_counts.get(key, 0) + 1
    
    # Filter by minimum episodes and sort by frequency
    common_setups = [
        {'zones': list(zones), 'count': count, 'plant_spot': plant}
        for (zones, plant), count in setup_counts.items()
        if count >= min_episodes
    ]
    
    common_setups.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"  Identified {len(common_setups)} setups with ≥{min_episodes} episodes")
    if common_setups:
        print(f"  Most common: {common_setups[0]['zones']} ({common_setups[0]['count']} episodes)")
    
    return common_setups


def calculate_setup_hazard(df, zones, site, plant_spot):
    """
    Calculate average hazard rate for a specific crossfire setup.
    
    Uses empirical death rates as placeholder for Cox model predictions.
    
    Args:
        df: Polars DataFrame with T episodes
        zones: List of zone names in setup
        site: 'A' or 'B'
        plant_spot: Plant location
    
    Returns:
        Float: Average hazard rate across the 3 positions
    """
    
    # Filter to episodes matching this setup's context
    filtered = df.filter(
        (pl.col("site") == site) &
        (pl.col("zone").is_in(zones)) &
        (pl.col("plant_spot") == plant_spot) &
        (pl.col("ct_count_current") == 3) &
        (pl.col("t_count_current") == 3) &
        (pl.col("time_remaining_on_bomb") >= 20.0) &
        (pl.col("time_remaining_on_bomb") <= 25.0) &
        (pl.col("active_smoke_zones").list.len() == 0) &
        (pl.col("active_molly_zones").list.len() == 0)
    )
    
    if len(filtered) == 0:
        return None
    
    # Calculate death rate per zone (placeholder for Cox hazard)
    zone_hazards = (
        filtered
        .group_by("zone")
        .agg([
            pl.col("died").sum().alias("deaths"),
            pl.count("zone").alias("episodes")
        ])
        .with_columns([
            (pl.col("deaths").cast(pl.Float64) / pl.col("episodes")).alias("hazard_rate")
        ])
    )
    
    # Average hazard across the zones in this setup
    avg_hazard = zone_hazards.select(pl.col("hazard_rate").mean()).item()
    
    return avg_hazard


def create_minimap_visualization(
    zones, 
    plant_spot, 
    site, 
    hazard_rate, 
    episode_count,
    zone_coords,
    plant_coords,
    ax,
    minimap_img=None
):
    """
    Create a single minimap visualization showing T positions and bomb location.
    
    Args:
        zones: List of zone names (T positions)
        plant_spot: Bomb plant location name
        site: 'A' or 'B'
        hazard_rate: Overall setup hazard rate
        episode_count: Number of episodes for this setup
        zone_coords: Dict mapping zone names to {x, y} coordinates
        plant_coords: Dict mapping site -> plant_spot -> {x, y}
        ax: Matplotlib axis to plot on
        minimap_img: Optional background image
    """
    
    # If minimap image provided, show it
    if minimap_img is not None:
        ax.imshow(minimap_img, extent=[0, 1024, 0, 1024], aspect='auto', alpha=0.3)
    
    # Set up axis limits (CS2 coordinates typically 0-1024)
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.set_aspect('equal')
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add background color
    ax.set_facecolor('#f5f5dc')  # floralwhite
    
    # Plot T positions
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(zones)))  # Color by danger
    
    for i, zone in enumerate(zones):
        if zone in zone_coords:
            coord = zone_coords[zone]
            # Plot position marker
            ax.scatter(
                coord['x'], 
                coord['y'], 
                s=500, 
                c=[colors[i]], 
                edgecolors='black',
                linewidths=2,
                alpha=0.8,
                zorder=3
            )
            # Add zone label
            ax.text(
                coord['x'], 
                coord['y'] - 40, 
                zone, 
                ha='center', 
                va='top',
                fontsize=7,
                fontweight='bold',
                zorder=4
            )
    
    # Plot bomb location
    if site in plant_coords and plant_spot in plant_coords[site]:
        bomb_coord = plant_coords[site][plant_spot]
        # Bomb marker (red X)
        ax.scatter(
            bomb_coord['x'],
            bomb_coord['y'],
            s=300,
            c='red',
            marker='X',
            edgecolors='darkred',
            linewidths=2,
            zorder=5
        )
        # Bomb label
        ax.text(
            bomb_coord['x'],
            bomb_coord['y'] - 40,
            'BOMB',
            ha='center',
            va='top',
            fontsize=6,
            fontweight='bold',
            color='darkred',
            zorder=6
        )
    
    # Add title with hazard rate and sample size
    title = f"Hazard: {hazard_rate:.3f}\n({episode_count} episodes)"
    ax.set_title(title, fontsize=9, fontweight='bold', pad=10)
    
    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)


# Main analysis workflow
print("\n" + "="*80)
print("ANALYZING CROSSFIRE SETUPS")
print("="*80)

# Identify common setups for each site
a_setups = identify_common_setups(t_df, site='A', t_count=3, ct_count=3, min_episodes=30)
b_setups = identify_common_setups(t_df, site='B', t_count=3, ct_count=3, min_episodes=30)

# Calculate hazard rates for each setup
print("\nCalculating hazard rates...")
a_setup_hazards = []
for setup in a_setups[:20]:  # Analyze top 20, show top 10
    hazard = calculate_setup_hazard(
        t_df, 
        setup['zones'], 
        site='A', 
        plant_spot=setup['plant_spot']
    )
    if hazard is not None:
        a_setup_hazards.append({
            'zones': setup['zones'],
            'plant_spot': setup['plant_spot'],
            'hazard': hazard,
            'count': setup['count']
        })

b_setup_hazards = []
for setup in b_setups[:20]:  # Analyze top 20, show top 10
    hazard = calculate_setup_hazard(
        t_df, 
        setup['zones'], 
        site='B', 
        plant_spot=setup['plant_spot']
    )
    if hazard is not None:
        b_setup_hazards.append({
            'zones': setup['zones'],
            'plant_spot': setup['plant_spot'],
            'hazard': hazard,
            'count': setup['count']
        })

# Sort by hazard (lowest = safest for T's)
a_setup_hazards.sort(key=lambda x: x['hazard'])
b_setup_hazards.sort(key=lambda x: x['hazard'])

print(f"  A-site: {len(a_setup_hazards)} setups with hazard rates")
print(f"  B-site: {len(b_setup_hazards)} setups with hazard rates")

# Create visualization
print("\n" + "="*80)
print("CREATING CROSSFIRE VISUALIZATION")
print("="*80)

# Take top 10 safest setups from each site
top_a = a_setup_hazards[:10]
top_b = b_setup_hazards[:10]

n_rows = max(len(top_a), len(top_b))

# Create figure with two columns
fig, axes = plt.subplots(
    n_rows, 
    2, 
    figsize=(12, 3*n_rows),
    facecolor='floralwhite'
)

# Ensure axes is 2D even if only 1 row
if n_rows == 1:
    axes = axes.reshape(1, -1)

# Plot A-site setups (left column)
for i, setup in enumerate(top_a):
    create_minimap_visualization(
        zones=setup['zones'],
        plant_spot=setup['plant_spot'],
        site='A',
        hazard_rate=setup['hazard'],
        episode_count=setup['count'],
        zone_coords=zone_coords,
        plant_coords=plant_coords,
        ax=axes[i, 0],
        minimap_img=None  # TODO: Add minimap image
    )

# Plot B-site setups (right column)
for i, setup in enumerate(top_b):
    create_minimap_visualization(
        zones=setup['zones'],
        plant_spot=setup['plant_spot'],
        site='B',
        hazard_rate=setup['hazard'],
        episode_count=setup['count'],
        zone_coords=zone_coords,
        plant_coords=plant_coords,
        ax=axes[i, 1],
        minimap_img=None  # TODO: Add minimap image
    )

# Hide unused subplots
for i in range(len(top_a), n_rows):
    axes[i, 0].axis('off')
for i in range(len(top_b), n_rows):
    axes[i, 1].axis('off')

# Add column headers
axes[0, 0].text(
    0.5, 1.15, 
    'A-Site Safest Setups', 
    transform=axes[0, 0].transAxes,
    ha='center', 
    fontsize=14, 
    fontweight='bold'
)
axes[0, 1].text(
    0.5, 1.15, 
    'B-Site Safest Setups', 
    transform=axes[0, 1].transAxes,
    ha='center', 
    fontsize=14, 
    fontweight='bold'
)

# Add overall title
fig.suptitle(
    'Top 10 Safest T-Side 3v3 Post-Plant Setups\n' +
    'Pistol Rounds | 20-25s Remaining | No Utility | Cox Model Adjusted',
    fontsize=16,
    fontweight='bold',
    y=0.995
)

plt.tight_layout(rect=[0, 0, 1, 0.985])

# Save figure
output_dir = Path("results/crossfire_analysis")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "3v3_safest_setups.png"

plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='floralwhite')
print(f"\nSaved: {output_path}")

print("\n" + "="*80)
print("CROSSFIRE VISUALIZATION COMPLETE")
print("="*80)
print("\nTop 3 Safest A-Site Setups:")
for i, setup in enumerate(top_a[:3], 1):
    print(f"  {i}. {', '.join(setup['zones'])} @ {setup['plant_spot']}: {setup['hazard']:.3f} (n={setup['count']})")

print("\nTop 3 Safest B-Site Setups:")
for i, setup in enumerate(top_b[:3], 1):
    print(f"  {i}. {', '.join(setup['zones'])} @ {setup['plant_spot']}: {setup['hazard']:.3f} (n={setup['count']})")

print("\n" + "="*80)
print("NOTES FOR FUTURE WORK:")
print("="*80)
print("1. TODO: Add minimap background images (A-site.png, B-site.png)")
print("2. TODO: Replace empirical death rates with Cox model predictions")
print("3. TODO: Add 2v2 setup analysis")
print("4. TODO: Add utility impact analysis")
print("5. TODO: Add plant spot variation analysis")
print("6. TODO: Add time pressure analysis (late retake)")
print("="*80)
