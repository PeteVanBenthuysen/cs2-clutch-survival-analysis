import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import ast

# Load T-side B-site episodes
df = pd.read_parquet('data/t_episodes.parquet')
df_b = df[df['site'] == 'B'].copy()

# Convert duration from ticks to seconds (128 ticks/sec)
df_b['duration_sec'] = df_b['duration_ticks'] / 128.0

# Parse active_smoke_zones from string to list
def parse_smoke_zones(smoke_str):
    if pd.isna(smoke_str) or smoke_str == '[]':
        return []
    try:
        return ast.literal_eval(smoke_str)
    except:
        return []

df_b['smoke_zones_list'] = df_b['active_smoke_zones'].apply(parse_smoke_zones)

print("=" * 80)
print("T-SIDE B-SITE SMOKE SURVIVAL ANALYSIS")
print("=" * 80)
print(f"\nTotal T-side B-site episodes: {len(df_b)}")
print(f"Deaths: {df_b['died'].sum()}")
print(f"Death rate: {df_b['died'].mean():.1%}\n")

# Find most common smoke locations
all_smokes = []
for zones in df_b['smoke_zones_list']:
    all_smokes.extend(zones)

smoke_counts = pd.Series(all_smokes).value_counts()
print("\nMOST COMMON B-SITE SMOKE LOCATIONS:")
print(smoke_counts.head(15))
print()

# Create binary indicators for Cat/Arches area, Market Window, Market Door/Exit, Market (entry)
df_b['has_cat_smoke'] = df_b['smoke_zones_list'].apply(
    lambda x: any(zone in x for zone in ['Cat', 'Arches', 'Open_Arches', 'Arches_B'])
)
df_b['has_market_window_smoke'] = df_b['smoke_zones_list'].apply(lambda x: 'Market_Window' in x)
df_b['has_market_door_smoke'] = df_b['smoke_zones_list'].apply(lambda x: 'Market_Exit' in x or 'Market_Door' in x)
df_b['has_market_entry_smoke'] = df_b['smoke_zones_list'].apply(lambda x: 'Market' in x or 'Outer_Market_Exit' in x)

# Create mutually exclusive categories (prioritize if multiple)
def categorize_smoke(row):
    if row['has_cat_smoke']:
        return 'Cat/Arches Smoke'
    elif row['has_market_window_smoke']:
        return 'Market Window Smoke'
    elif row['has_market_door_smoke']:
        return 'Market Door Smoke'
    elif row['has_market_entry_smoke']:
        return 'Market Entry Smoke'
    else:
        return 'No Smoke'

df_b['smoke_category'] = df_b.apply(categorize_smoke, axis=1)

# Count and stats per category
category_stats = df_b.groupby('smoke_category').agg({
    'died': ['count', 'sum', 'mean']
}).round(3)

print("\n" + "=" * 80)
print("EPISODE COUNTS AND DEATH RATES BY SMOKE CATEGORY")
print("=" * 80)
print(category_stats)
print()

# Filter for analysis (Cat/Arches, Market Window, Market Door, Market Entry, No Smoke)
target_categories = ['Cat/Arches Smoke', 'Market Window Smoke', 'Market Door Smoke', 'Market Entry Smoke', 'No Smoke']
df_analysis = df_b[df_b['smoke_category'].isin(target_categories)].copy()

print(f"\nAnalyzing {len(df_analysis)} episodes across 5 smoke categories")

# Kaplan-Meier survival curves
kmf = KaplanMeierFitter()

plt.figure(figsize=(14, 8))

colors = {
    'Cat/Arches Smoke': '#e74c3c',
    'Market Window Smoke': '#3498db', 
    'Market Door Smoke': '#2ecc71',
    'Market Entry Smoke': '#f39c12',
    'No Smoke': '#95a5a6'
}

for category in target_categories:
    data = df_analysis[df_analysis['smoke_category'] == category]
    
    if len(data) > 0:
        kmf.fit(
            durations=data['duration_sec'],
            event_observed=data['died'],
            label=f'{category} (n={len(data)})'
        )
        
        kmf.plot_survival_function(
            ci_show=True,
            color=colors[category],
            alpha=0.7,
            linewidth=2.5
        )

plt.title('T-Side B-Site Post-Plant Survival by Active Smoke Location', fontsize=16, fontweight='bold')
plt.xlabel('Time Since Bomb Plant (seconds)', fontsize=13)
plt.ylabel('Survival Probability', fontsize=13)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='best', fontsize=11, framealpha=0.95)
plt.ylim([0, 1.05])
plt.tight_layout()

# Save figure
plt.savefig('paper/figures/t_b_survival_by_smoke_location.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("Saved: paper/figures/t_b_survival_by_smoke_location.png")
print("=" * 80)

plt.show()

# Pairwise log-rank tests
print("\n" + "=" * 80)
print("PAIRWISE LOG-RANK TESTS (Statistical Significance)")
print("=" * 80)

no_smoke_data = df_analysis[df_analysis['smoke_category'] == 'No Smoke']

for category in ['Cat Smoke', 'Market Window Smoke', 'Market Door Smoke']:
    smoke_data = df_analysis[df_analysis['smoke_category'] == category]
    
    if len(smoke_data) > 0 and len(no_smoke_data) > 0:
        result = logrank_test(
            smoke_data['duration_sec'],
            no_smoke_data['duration_sec'],
            smoke_data['died'],
            no_smoke_data['died']
        )
        
        print(f"\n{category} vs No Smoke:")
        print(f"  Test statistic: {result.test_statistic:.2f}")
        print(f"  p-value: {result.p_value:.4f}")
        if result.p_value < 0.001:
            print("  Significance: *** (p<0.001)")
        elif result.p_value < 0.01:
            print("  Significance: ** (p<0.01)")
        elif result.p_value < 0.05:
            print("  Significance: * (p<0.05)")
        else:
            print("  Significance: ns (not significant)")

# Cat vs Market Window
cat_data = df_analysis[df_analysis['smoke_category'] == 'Cat Smoke']
window_data = df_analysis[df_analysis['smoke_category'] == 'Market Window Smoke']

if len(cat_data) > 0 and len(window_data) > 0:
    result = logrank_test(
        cat_data['duration_sec'],
        window_data['duration_sec'],
        cat_data['died'],
        window_data['died']
    )
    
    print(f"\nCat Smoke vs Market Window Smoke:")
    print(f"  Test statistic: {result.test_statistic:.2f}")
    print(f"  p-value: {result.p_value:.4f}")
    if result.p_value < 0.05:
        print(f"  Significance: * (p<0.05)")
    else:
        print("  Significance: ns (not significant)")

# Median survival times
print("\n" + "=" * 80)
print("MEDIAN SURVIVAL TIMES BY SMOKE CATEGORY")
print("=" * 80)

for category in target_categories:
    data = df_analysis[df_analysis['smoke_category'] == category]
    
    if len(data) > 0:
        kmf.fit(durations=data['duration_sec'], event_observed=data['died'])
        median_survival = kmf.median_survival_time_
        
        print(f"\n{category}:")
        print(f"  Episodes: {len(data)}")
        print(f"  Deaths: {data['died'].sum()} ({data['died'].mean():.1%})")
        print(f"  Median survival: {median_survival:.1f} seconds")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

