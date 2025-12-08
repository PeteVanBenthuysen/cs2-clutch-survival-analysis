"""
Create a publication-quality table describing Cox model features/covariates.
Generates a styled data table showing column name, data type, and description.
"""

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np

# Define Cox model features
features = [
    # Target variable
    {
        'column': 'death_time',
        'type': 'float',
        'description': 'Time (seconds) from retake start until player death or round end'
    },
    {
        'column': 'died',
        'type': 'bool',
        'description': 'Event indicator: 1 if player died, 0 if censored (survived)'
    },
    
    # Spatial features
    {
        'column': 'zone',
        'type': 'categorical',
        'description': 'Player\'s zone at episode start (291 zones on Mirage)'
    },
    {
        'column': 'plant_spot',
        'type': 'categorical',
        'description': 'Bomb plant location (default, ninja, cat, triple, open, etc.)'
    },
    {
        'column': 'site',
        'type': 'categorical',
        'description': 'Bombsite being retaken (A or B)'
    },
    
    # Player counts
    {
        'column': 'ct_count',
        'type': 'int',
        'description': 'Number of alive CTs at episode start (1-5)'
    },
    {
        'column': 't_count',
        'type': 'int',
        'description': 'Number of alive Ts at episode start (1-5)'
    },
    
    # Equipment & Economy
    {
        'column': 'has_helmet',
        'type': 'bool',
        'description': 'Player has helmet (protects against headshot multiplier)'
    },
    {
        'column': 'armor_value',
        'type': 'int',
        'description': 'Player armor points (0-100, normalized to 0-1 in model)'
    },
    {
        'column': 'has_defuse_kit',
        'type': 'bool',
        'description': 'CT has defuse kit (reduces defuse time from 10s to 5s)'
    },
    {
        'column': 'weapon_type',
        'type': 'categorical',
        'description': 'Weapon: AK-47, M4A1-S, AWP, Desert Eagle, USP-S, Glock-18, etc.'
    },
    {
        'column': 'health',
        'type': 'int',
        'description': 'Player health points at episode start (0-100, normalized to 0-1 in model)'
    },
    
    # Utility state
    {
        'column': 'active_smoke_zones',
        'type': 'list[str]',
        'description': 'List of zones obscured by smoke grenades'
    },
    {
        'column': 'active_molly_zones',
        'type': 'list[str]',
        'description': 'List of zones with active molotov/incendiary fire'
    },
    {
        'column': 'active_in_smoke',
        'type': 'bool',
        'description': 'Player is currently inside smoke grenade'
    },
    {
        'column': 'active_in_molly',
        'type': 'bool',
        'description': 'Player is currently inside molotov/incendiary fire'
    },
    {
        'column': 'flashed',
        'type': 'bool',
        'description': 'Player is currently flashbanged'
    },
    {
        'column': 'smokes_available',
        'type': 'int',
        'description': 'Number of smoke grenades available (0-1)'
    },
    {
        'column': 'mollies_available',
        'type': 'int',
        'description': 'Number of molotov/incendiary grenades available (0-1)'
    },
    {
        'column': 'flashes_available',
        'type': 'int',
        'description': 'Number of flashbang grenades available (0-2)'
    },
    {
        'column': 'he_grenades_available',
        'type': 'int',
        'description': 'Number of HE grenades available (0-1)'
    },
    {
        'column': 'total_utility',
        'type': 'int',
        'description': 'Sum of all grenades available (0-4: smoke + molly + 2 flashes)'
    },
    
    # Teammate composition
    {
        'column': 'has_crossfire_support',
        'type': 'bool',
        'description': 'Teammates positioned to cover threat zones (empirical sightlines + swing routes)'
    },
    {
        'column': 'num_teammates_total',
        'type': 'int',
        'description': 'Number of alive teammates (0-5)'
    },
    {
        'column': 'teammate_at_[zone]',
        'type': 'int',
        'description': 'Count of teammates in specific zone (e.g., teammate_at_A_Default, teammate_at_Jungle)'
    },
    {
        'column': 'numerical_advantage',
        'type': 'int',
        'description': 'Team player count difference (positive = advantage, negative = disadvantage)'
    },
    
    # Temporal features
    {
        'column': 'time_since_plant',
        'type': 'float',
        'description': 'Seconds since bomb plant (tracks defuse pressure)'
    },
    {
        'column': 'bomb_time_remaining',
        'type': 'float',
        'description': 'Seconds until bomb detonation (C4 timer: 40s)'
    },
    
    # Round context
    {
        'column': 'round_num',
        'type': 'int',
        'description': 'Round number in match (1-30+, tracks momentum/economy)'
    },
    {
        'column': 'round_time_elapsed',
        'type': 'float',
        'description': 'Seconds elapsed since round start (measures round timing)'
    },
    {
        'column': 'score_diff',
        'type': 'int',
        'description': 'Player\'s team score minus opponent score (match state)'
    }
]

def create_features_table():
    """Create styled table showing Cox model features."""
    
    # Group features by category
    categories = {
        'Survival Variables': features[0:2],
        'Spatial Features': features[2:5],
        'Equipment & Economy': features[5:10],
        'Utility State': features[10:20],
        'Teammate Composition': features[20:24],
        'Temporal Features': features[24:26],
        'Round Context': features[26:29]
    }
    
    # Create figure - more compact for paper
    fig, ax = plt.subplots(figsize=(14, 13))
    ax.axis('off')
    
    # Colors
    header_color = '#2C3E50'
    category_color = '#34495E'
    row_colors = ['#ECF0F1', '#FFFFFF']
    text_color = '#2C3E50'
    
    # Starting position - no title, start higher
    y_pos = 0.97
    row_height = 0.027
    
    # Column widths (as fraction of figure width) - centered layout
    col_widths = [0.22, 0.13, 0.55]
    col_positions = [0.10, 0.32, 0.45]
    
    # Draw main header
    header_rect = Rectangle((0.10, y_pos - row_height), 0.80, row_height,
                            facecolor=header_color, edgecolor='none')
    ax.add_patch(header_rect)
    
    ax.text(col_positions[0] + col_widths[0]/2, y_pos - row_height/2,
            'Column Name', color='white', fontsize=11, weight='bold',
            ha='center', va='center', fontfamily='sans-serif')
    ax.text(col_positions[1] + col_widths[1]/2, y_pos - row_height/2,
            'Data Type', color='white', fontsize=11, weight='bold',
            ha='center', va='center', fontfamily='sans-serif')
    ax.text(col_positions[2] + col_widths[2]/2, y_pos - row_height/2,
            'Description', color='white', fontsize=11, weight='bold',
            ha='center', va='center', fontfamily='sans-serif')
    
    y_pos -= row_height
    
    # Draw categories and rows
    row_idx = 0
    for category, cat_features in categories.items():
        # Category header
        cat_rect = Rectangle((0.10, y_pos - row_height * 0.75), 0.80, row_height * 0.75,
                             facecolor=category_color, edgecolor='none')
        ax.add_patch(cat_rect)
        
        ax.text(0.12, y_pos - row_height * 0.375,
                category, color='white', fontsize=10, weight='bold',
                ha='left', va='center', fontfamily='sans-serif', style='italic')
        
        y_pos -= row_height * 0.75
        
        # Feature rows
        for feature in cat_features:
            # Alternating row colors
            row_color = row_colors[row_idx % 2]
            row_rect = Rectangle((0.10, y_pos - row_height), 0.80, row_height,
                                facecolor=row_color, edgecolor='#BDC3C7', linewidth=0.5)
            ax.add_patch(row_rect)
            
            # Column name (monospace font for code)
            ax.text(col_positions[0] + 0.01, y_pos - row_height/2,
                    f"`{feature['column']}`", color=text_color, fontsize=10,
                    ha='left', va='center', fontfamily='monospace', weight='bold')
            
            # Data type (centered, with color coding)
            type_color = {
                'bool': '#27AE60',
                'int': '#2980B9', 
                'float': '#8E44AD',
                'categorical': '#E67E22',
                'list[str]': '#C0392B'
            }.get(feature['type'], text_color)
            
            ax.text(col_positions[1] + col_widths[1]/2, y_pos - row_height/2,
                    feature['type'], color=type_color, fontsize=10,
                    ha='center', va='center', fontfamily='monospace', weight='bold')
            
            # Description
            desc = feature['description']
            ax.text(col_positions[2] + 0.01, y_pos - row_height/2,
                    desc, color=text_color, fontsize=9,
                    ha='left', va='center', fontfamily='sans-serif')
            
            y_pos -= row_height
            row_idx += 1
    
    # Add footer with legend for data types
    y_pos -= 0.03
    legend_y = y_pos - 0.02
    
    ax.text(0.15, legend_y, 'Data Type Legend:', 
            color=text_color, fontsize=9, weight='bold',
            ha='left', va='center', fontfamily='sans-serif')
    
    type_colors = [
        ('bool', '#27AE60'),
        ('int', '#2980B9'),
        ('float', '#8E44AD'),
        ('categorical', '#E67E22'),
        ('list[str]', '#C0392B')
    ]
    
    x_offset = 0.34
    for dtype, color in type_colors:
        ax.text(x_offset, legend_y, dtype,
                color=color, fontsize=8, weight='bold',
                ha='left', va='center', fontfamily='monospace')
        x_offset += 0.11
    
    # Title - removed for research paper (goes in caption instead)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Save
    plt.tight_layout()
    output_path = 'paper/figures/figure_02_cox_features_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved Cox features table to {output_path}")
    plt.close()

if __name__ == '__main__':
    import os
    os.makedirs('paper/figures', exist_ok=True)
    create_features_table()
    print("\n✓ Cox model features table generated successfully!")
