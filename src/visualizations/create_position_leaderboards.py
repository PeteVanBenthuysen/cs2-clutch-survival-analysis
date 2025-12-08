"""
Create publication-quality position hazard board for CS2 post-plant analysis.
Generates draft-board style tables showing how hazard rates change across scenarios.

Uses great_tables package to create clean, professional tables similar to sports analytics.
Inspired by basketball draft boards - shows zones (rows) vs scenarios (columns).
"""

import pandas as pd
import polars as pl
import great_tables as gt
from great_tables import loc, style
from pathlib import Path
import json

print("="*80)
print("CREATING POSITION HAZARD BOARD")
print("="*80)

# Load episode data
print("\nLoading episode data...")
ct_df = pl.read_parquet("data/ct_episodes.parquet")
t_df = pl.read_parquet("data/t_episodes.parquet")

print(f"  Loaded {len(ct_df)} CT episodes")
print(f"  Loaded {len(t_df)} T episodes")

# TODO: Load Cox model results to get hazard rates
# For now, we'll calculate empirical death rates per zone per scenario

def calculate_scenario_hazard(df, scenario_params):
    """
    Calculate hazard rate (death rate) for each zone under specific scenario.
    
    Args:
        df: Polars DataFrame with episode data
        scenario_params: dict with filtering criteria
    
    Returns:
        Polars DataFrame with zone -> hazard_rate mapping
    """
    
    # Filter to scenario
    filtered = filter_scenario(df, scenario_params)
    
    if len(filtered) == 0:
        return pl.DataFrame({"zone": [], "hazard_rate": []})
    
    # Calculate death rate per zone (placeholder for Cox model hazard)
    zone_hazards = (
        filtered
        .group_by("zone")
        .agg([
            pl.count("zone").alias("episodes"),
            pl.col("died").sum().alias("deaths"),
        ])
        .with_columns([
            (pl.col("deaths").cast(pl.Float64) / pl.col("episodes")).alias("hazard_rate")
        ])
        .select(["zone", "hazard_rate", "episodes"])
    )
    
    return zone_hazards


def filter_scenario(df, scenario_params):
    """
    Filter episodes to match a specific scenario.
    
    Args:
        df: Polars DataFrame with episodes
        scenario_params: dict with filtering criteria
            - ct_count: CT player count
            - t_count: T player count
            - site: 'A' or 'B'
            - plant_spot: specific plant location (optional)
            - time_min: minimum time remaining (optional)
            - time_max: maximum time remaining (optional)
            - no_utility: if True, filter to episodes with no active utility
            - required_smoke_zones: list of zones that must be in active_smoke_zones (optional)
            - required_molly_zones: list of zones that must be in active_molly_zones (optional)
    
    Returns:
        Filtered Polars DataFrame
    """
    
    filtered = df
    
    # Filter by player counts
    if 'ct_count' in scenario_params:
        filtered = filtered.filter(pl.col("ct_count_current") == scenario_params['ct_count'])
    
    if 't_count' in scenario_params:
        filtered = filtered.filter(pl.col("t_count_current") == scenario_params['t_count'])
    
    # Filter by site
    if 'site' in scenario_params:
        filtered = filtered.filter(pl.col("site") == scenario_params['site'])
    
    # Filter by plant spot
    if 'plant_spot' in scenario_params:
        filtered = filtered.filter(pl.col("plant_spot") == scenario_params['plant_spot'])
    
    # Filter by time remaining
    if 'time_min' in scenario_params:
        filtered = filtered.filter(pl.col("time_remaining_on_bomb") >= scenario_params['time_min'])
    
    if 'time_max' in scenario_params:
        filtered = filtered.filter(pl.col("time_remaining_on_bomb") <= scenario_params['time_max'])
    
    # Filter for no active utility
    if scenario_params.get('no_utility', False):
        filtered = filtered.filter(
            (pl.col("active_smoke_zones").list.len() == 0) &
            (pl.col("active_molly_zones").list.len() == 0) &
            (pl.col("in_smoke") == False) &
            (pl.col("in_molly") == False)
        )
    
    # Filter for required smoke zones (e.g., "Stairs must be smoked")
    if 'required_smoke_zones' in scenario_params:
        for zone in scenario_params['required_smoke_zones']:
            filtered = filtered.filter(pl.col("active_smoke_zones").list.contains(zone))
    
    # Filter for required molly zones (e.g., "Default must have molly")
    if 'required_molly_zones' in scenario_params:
        for zone in scenario_params['required_molly_zones']:
            filtered = filtered.filter(pl.col("active_molly_zones").list.contains(zone))
    
    return filtered


def create_hazard_board(hazard_matrix, title, subtitle, scenario_labels):
    """
    Create draft-board style table showing hazard rates across scenarios.
    
    Args:
        hazard_matrix: Pandas DataFrame with zones as rows, scenarios as columns
        title: Table title
        subtitle: Table subtitle
        scenario_labels: Dict mapping column names to display labels
    
    Returns:
        gt.GT table object
    """
    
    # Get column names for scenarios (exclude 'zone' and 'avg_hazard')
    scenario_cols = [col for col in hazard_matrix.columns if col not in ['zone', 'avg_hazard', 'total_episodes']]
    
    # Create the table
    table = (
        gt.GT(hazard_matrix)
        .tab_header(
            title=gt.md(f"**{title}**"),
            subtitle=subtitle
        )
        # Rename columns
        .cols_label(
            zone="",
            avg_hazard=gt.md("Avg."),
            **scenario_labels
        )
        # Add column spanners to group related scenarios
        # Will dynamically determine spanners based on scenario types
        
        # Format hazard rates to 2 decimals
        .fmt_number(
            columns=["avg_hazard"] + scenario_cols,
            decimals=2
        )
        # Replace missing values with "--"
        .sub_missing(missing_text="--")
        
        # Color code all hazard columns (green = safe, red = dangerous)
        .data_color(
            columns=["avg_hazard"] + scenario_cols,
            palette=["#4CAF50", "#FFC107", "#FF9800", "#F44336"],  # Green -> Yellow -> Orange -> Red
            domain=[0.3, 0.9],
            alpha=0.75,
            na_color="white"
        )
        
        # Center align all hazard columns
        .cols_align(
            columns=["avg_hazard"] + scenario_cols,
            align="center"
        )
        
        # Set column widths
        .cols_width({
            "zone": "180px",
            "avg_hazard": "60px"
        })
        
        # Add vertical borders to separate sections
        .tab_style(
            style=style.borders(sides="left", weight="2px", color="black"),
            locations=loc.body(columns="avg_hazard")
        )
        .tab_style(
            style=style.borders(sides="left", weight="2px", color="black"),
            locations=loc.column_labels(columns="avg_hazard")
        )
        
        # Reduce row padding
        .tab_options(
            data_row_padding='3px',
            table_background_color="white",
            table_font_names="Roboto",
            column_labels_font_size="11px",
            column_labels_font_weight="bold",
            table_body_hlines_color='transparent',
            column_labels_border_top_color='black',
            column_labels_border_top_width="2px",
            column_labels_border_bottom_style='none',
            heading_border_bottom_style="none",
            table_body_border_bottom_style="none",
            table_border_bottom_style='none',
            table_border_top_style='none',
        )
        
        # Adjust title font size
        .tab_style(
            style=style.text(size="28px", weight=700),
            locations=loc.title()
        )
    )
    
    return table


# Define scenarios for draft-board columns
# Separate boards for A-site and B-site with their respective plant locations

a_site_scenarios = {
    'default_3v3': {
        'label': gt.md("Default<br>3v3"),
        'params': {
            'ct_count': 3,
            't_count': 3,
            'site': 'A',
            'plant_spot': 'default',
            'time_min': 20.0,
            'time_max': 30.0,
            'no_utility': True
        }
    },
    'ninja_3v3': {
        'label': gt.md("Ninja<br>3v3"),
        'params': {
            'ct_count': 3,
            't_count': 3,
            'site': 'A',
            'plant_spot': 'ninja',
            'time_min': 20.0,
            'time_max': 30.0,
            'no_utility': True
        }
    },
    'triple_default': {
        'label': gt.md("Triple<br>Default"),
        'params': {
            'ct_count': 3,
            't_count': 3,
            'site': 'A',
            'plant_spot': 'triple_default',
            'time_min': 20.0,
            'time_max': 30.0,
            'no_utility': True
        }
    },
    'stairs_smoke': {
        'label': gt.md("Stairs<br>Smoke"),
        'params': {
            'ct_count': 3,
            't_count': 3,
            'site': 'A',
            'plant_spot': 'default',
            'time_min': 20.0,
            'time_max': 30.0,
            'required_smoke_zones': ['Stairs']  # Filter to episodes where Stairs is smoked
        }
    },
    'jungle_smoke': {
        'label': gt.md("Jungle<br>Smoke"),
        'params': {
            'ct_count': 3,
            't_count': 3,
            'site': 'A',
            'plant_spot': 'default',
            'time_min': 20.0,
            'time_max': 30.0,
            'required_smoke_zones': ['Jungle']  # Filter to episodes where Jungle is smoked
        }
    },
    'late_retake': {
        'label': gt.md("Late<br>(<15s)"),
        'params': {
            'ct_count': 3,
            't_count': 3,
            'site': 'A',
            'plant_spot': 'default',
            'time_min': 5.0,
            'time_max': 15.0,
            'no_utility': True
        }
    }
}

b_site_scenarios = {
    'default_3v3': {
        'label': gt.md("Default<br>3v3"),
        'params': {
            'ct_count': 3,
            't_count': 3,
            'site': 'B',
            'plant_spot': 'default',
            'time_min': 20.0,
            'time_max': 30.0,
            'no_utility': True
        }
    },
    'cat_plant': {
        'label': gt.md("Cat<br>Plant"),
        'params': {
            'ct_count': 3,
            't_count': 3,
            'site': 'B',
            'plant_spot': 'cat',
            'time_min': 20.0,
            'time_max': 30.0,
            'no_utility': True
        }
    },
    'empty_plant': {
        'label': gt.md("Empty<br>Plant"),
        'params': {
            'ct_count': 3,
            't_count': 3,
            'site': 'B',
            'plant_spot': 'empty',
            'time_min': 20.0,
            'time_max': 30.0,
            'no_utility': True
        }
    },
    'bench_smoke': {
        'label': gt.md("Bench<br>Smoke"),
        'params': {
            'ct_count': 3,
            't_count': 3,
            'site': 'B',
            'plant_spot': 'default',
            'time_min': 20.0,
            'time_max': 30.0,
            'required_smoke_zones': ['Bench']  # Filter to episodes where Bench is smoked
        }
    },
    'short_smoke': {
        'label': gt.md("Short<br>Smoke"),
        'params': {
            'ct_count': 3,
            't_count': 3,
            'site': 'B',
            'plant_spot': 'default',
            'time_min': 20.0,
            'time_max': 30.0,
            'required_smoke_zones': ['B_Short']  # Filter to episodes where B Short is smoked
        }
    },
    'late_retake': {
        'label': gt.md("Late<br>(<15s)"),
        'params': {
            'ct_count': 3,
            't_count': 3,
            'site': 'B',
            'plant_spot': 'default',
            'time_min': 5.0,
            'time_max': 15.0,
            'no_utility': True
        }
    }
}

# Generate hazard boards for both sites
print("\n" + "="*80)
print("GENERATING HAZARD BOARDS")
print("="*80)

output_dir = Path("results/hazard_boards")
output_dir.mkdir(parents=True, exist_ok=True)

# Function to generate board for a site
def generate_site_board(site_name, scenarios, site_label):
    """Generate hazard board for a specific site."""
    
    print(f"\n{site_label} Hazard Board:")
    print("-" * 60)
    
    # Calculate hazard rates for each scenario
    print("Calculating hazard rates across scenarios...")
    scenario_hazards = {}
    
    for scenario_name, scenario_info in scenarios.items():
        print(f"  {scenario_name}...", end=" ")
        hazards = calculate_scenario_hazard(ct_df, scenario_info['params'])
        scenario_hazards[scenario_name] = hazards
        print(f"{len(hazards)} zones")
    
    # Find common zones across scenarios (zones with sufficient data)
    zone_counts = {}
    for scenario_name, hazards in scenario_hazards.items():
        for row in hazards.iter_rows(named=True):
            zone = row['zone']
            if row['episodes'] >= 10:  # Minimum sample size threshold
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
    
    # Keep zones that appear in at least 3 scenarios
    common_zones = [zone for zone, count in zone_counts.items() if count >= 3]
    print(f"Found {len(common_zones)} zones with sufficient data across scenarios")
    
    if len(common_zones) == 0:
        print(f"WARNING: No zones with sufficient data for {site_label}")
        return
    
    # Build hazard matrix (zones x scenarios)
    hazard_data = {'zone': common_zones}
    
    for scenario_name, hazards in scenario_hazards.items():
        hazards_dict = {row['zone']: row['hazard_rate'] for row in hazards.iter_rows(named=True)}
        hazard_data[scenario_name] = [hazards_dict.get(zone) for zone in common_zones]
    
    hazard_matrix = pd.DataFrame(hazard_data)
    
    # Calculate average hazard across scenarios
    scenario_cols = list(scenarios.keys())
    hazard_matrix['avg_hazard'] = hazard_matrix[scenario_cols].mean(axis=1, skipna=True)
    
    # Sort by average hazard (most dangerous first)
    hazard_matrix = hazard_matrix.sort_values('avg_hazard', ascending=False).head(25)
    
    # Reorder columns: zone, avg_hazard, then scenarios
    hazard_matrix = hazard_matrix[['zone', 'avg_hazard'] + scenario_cols]
    
    print(f"Creating hazard board with {len(hazard_matrix)} zones...")
    
    # Create scenario labels dict
    scenario_labels = {name: info['label'] for name, info in scenarios.items()}
    
    # Create the table
    table = create_hazard_board(
        hazard_matrix,
        title=f"{site_label} CT Retake Hazard Rates",
        subtitle="Pistol Rounds: 3v3, 100 HP, No Armor | How Position Danger Changes By Context",
        scenario_labels=scenario_labels
    )
    
    # Add column spanners based on site
    if site_name == 'A':
        table = (
            table
            .tab_spanner(
                label="Plant Location",
                columns=["default_3v3", "ninja_3v3", "triple_default"]
            )
            .tab_spanner(
                label="Utility Impact",
                columns=["stairs_smoke", "jungle_smoke", "late_retake"]
            )
            # Add borders between spanner groups
            .tab_style(
                style=style.borders(sides="left", weight="2px", color="black"),
                locations=loc.body(columns="stairs_smoke")
            )
            .tab_style(
                style=style.borders(sides="left", weight="2px", color="black"),
                locations=loc.column_labels(columns="stairs_smoke")
            )
            .tab_style(
                style=style.borders(sides="left", weight="2px", color="black"),
                locations=loc.column_spanners(spanners="Utility Impact")
            )
        )
    else:  # B-site
        table = (
            table
            .tab_spanner(
                label="Plant Location",
                columns=["default_3v3", "cat_plant", "empty_plant"]
            )
            .tab_spanner(
                label="Utility Impact",
                columns=["bench_smoke", "short_smoke", "late_retake"]
            )
            # Add borders between spanner groups
            .tab_style(
                style=style.borders(sides="left", weight="2px", color="black"),
                locations=loc.body(columns="bench_smoke")
            )
            .tab_style(
                style=style.borders(sides="left", weight="2px", color="black"),
                locations=loc.column_labels(columns="bench_smoke")
            )
            .tab_style(
                style=style.borders(sides="left", weight="2px", color="black"),
                locations=loc.column_spanners(spanners="Utility Impact")
            )
        )
    
    # Save table
    output_path = output_dir / f"{site_name.lower()}_site_ct_hazard_board.png"
    table.save(str(output_path), scale=3, expand=20)
    
    print(f"Saved: {output_path}")

# Generate A-site board
generate_site_board('A', a_site_scenarios, 'A-Site')

# Generate B-site board
generate_site_board('B', b_site_scenarios, 'B-Site')

print("\n" + "="*80)
print("HAZARD BOARD GENERATION COMPLETE")
print(f"Boards saved to: {output_dir}")
print("="*80)