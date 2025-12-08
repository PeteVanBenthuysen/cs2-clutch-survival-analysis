import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import ast

# Load T-side episodes
df = pd.read_parquet('data/t_episodes.parquet')

# Filter for A-site only
df_a = df[df['site'] == 'A'].copy()

# Convert duration to seconds
df_a['duration_sec'] = df_a['duration_ticks'] / 128.0

# Parse active smoke zones
df_a['smoke_zones_list'] = df_a['active_smoke_zones'].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else []
)

print("=" * 80)
print("T-SIDE A-SITE POST-PLANT SURVIVAL BY SMOKE LOCATION")
print("=" * 80)
print(f"\nTotal episodes: {len(df_a)}")
print(f"Deaths: {df_a['died'].sum()}")
print(f"Death rate: {df_a['died'].mean():.1%}\n")

# Find most common smoke locations
all_smokes = []
for zones in df_a['smoke_zones_list']:
    all_smokes.extend(zones)

smoke_counts = pd.Series(all_smokes).value_counts()
print("\nMOST COMMON A-SITE SMOKE LOCATIONS:")
print(smoke_counts.head(15))
print()

# Create binary indicators for key A-site smokes
df_a['has_stairs_smoke'] = df_a['smoke_zones_list'].apply(
    lambda x: 'Stairs_Platform' in x or 'Top_Stairs' in x or 'Bottom_Stairs' in x
)
df_a['has_jungle_smoke'] = df_a['smoke_zones_list'].apply(
    lambda x: 'Jungle_Cubby' in x or 'Deep_Jungle_Cubby' in x or 'Connector_Jungle_Cubby' in x
)
df_a['has_connector_smoke'] = df_a['smoke_zones_list'].apply(
    lambda x: 'Connector_Box_Cubby' in x or 'Connector_Window_Cubby' in x or 'Bottom_Connector' in x
)
df_a['has_ticket_smoke'] = df_a['smoke_zones_list'].apply(
    lambda x: 'Reverse_Ticket' in x or 'Ticket' in x or 'Trash_Can' in x
)
df_a['has_ct_smoke'] = df_a['smoke_zones_list'].apply(
    lambda x: 'CT' in x or 'CT_Side_Triple' in x
)

# Create mutually exclusive categories (prioritize if multiple)
def categorize_smoke(row):
    if row['has_stairs_smoke']:
        return 'Stairs Smoke'
    elif row['has_jungle_smoke']:
        return 'Jungle Smoke'
    elif row['has_connector_smoke']:
        return 'Connector Smoke'
    elif row['has_ticket_smoke']:
        return 'Ticket Smoke'
    elif row['has_ct_smoke']:
        return 'CT Smoke'
    else:
        return 'No Smoke'

df_a['smoke_category'] = df_a.apply(categorize_smoke, axis=1)

# Count and stats per category
category_stats = df_a.groupby('smoke_category').agg({
    'died': ['count', 'sum', 'mean']
}).round(3)

print("\n" + "=" * 80)
print("EPISODE COUNTS AND DEATH RATES BY SMOKE CATEGORY")
print("=" * 80)
print(category_stats)
print()

# Filter for analysis (top smokes + No Smoke)
target_categories = ['Stairs Smoke', 'Jungle Smoke', 'Connector Smoke', 'Ticket Smoke', 'No Smoke']
df_analysis = df_a[df_a['smoke_category'].isin(target_categories)].copy()

print(f"\nAnalyzing {len(df_analysis)} episodes across 5 smoke categories")

# Kaplan-Meier survival curves
fig, ax = plt.subplots(figsize=(14, 8))

colors = {
    'Stairs Smoke': '#e74c3c',      # Red
    'Jungle Smoke': '#2ecc71',      # Green
    'Connector Smoke': '#3498db',   # Blue
    'Ticket Smoke': '#f39c12',      # Orange
    'No Smoke': '#95a5a6'           # Gray
}

kmf = KaplanMeierFitter()

for category in target_categories:
    subset = df_analysis[df_analysis['smoke_category'] == category]
    
    if len(subset) == 0:
        continue
    
    kmf.fit(
        durations=subset['duration_sec'],
        event_observed=subset['died'],
        label=f'{category} (n={len(subset)})'
    )
    
    if category in colors:
        kmf.plot_survival_function(
            ci_show=True,
            color=colors[category],
            alpha=0.7,
            linewidth=2.5
        )

plt.title('T-Side A-Site Post-Plant Survival by Active Smoke Location', fontsize=16, fontweight='bold')
plt.xlabel('Time Since Bomb Plant (seconds)', fontsize=13)
plt.ylabel('Survival Probability', fontsize=13)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='best', fontsize=11, framealpha=0.95)
plt.ylim([0, 1.05])
plt.tight_layout()

# Save figure
plt.savefig('paper/figures/t_a_survival_by_smoke_location.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("Saved: paper/figures/t_a_survival_by_smoke_location.png")
print("=" * 80)

# Statistical testing - pairwise log-rank tests
print("\n" + "=" * 80)
print("PAIRWISE LOG-RANK TESTS (vs No Smoke)")
print("=" * 80)

no_smoke_data = df_analysis[df_analysis['smoke_category'] == 'No Smoke']

for category in ['Stairs Smoke', 'Jungle Smoke', 'Connector Smoke', 'Ticket Smoke']:
    category_data = df_analysis[df_analysis['smoke_category'] == category]
    
    if len(category_data) == 0:
        continue
    
    result = logrank_test(
        durations_A=category_data['duration_sec'],
        durations_B=no_smoke_data['duration_sec'],
        event_observed_A=category_data['died'],
        event_observed_B=no_smoke_data['died']
    )
    
    print(f"\n{category} vs No Smoke:")
    print(f"  Test statistic: {result.test_statistic:.4f}")
    print(f"  p-value: {result.p_value:.4e}")
    print(f"  Significant: {'Yes' if result.p_value < 0.05 else 'No'}")

# Median survival times
print("\n" + "=" * 80)
print("MEDIAN SURVIVAL TIMES BY SMOKE CATEGORY")
print("=" * 80)

for category in target_categories:
    subset = df_analysis[df_analysis['smoke_category'] == category]
    
    if len(subset) == 0:
        continue
    
    kmf.fit(
        durations=subset['duration_sec'],
        event_observed=subset['died']
    )
    
    median = kmf.median_survival_time_
    print(f"{category}: {median:.2f} seconds (n={len(subset)})")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
