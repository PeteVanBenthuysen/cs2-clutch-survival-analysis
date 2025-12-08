import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Load T-side and CT-side episodes
df_t = pd.read_parquet('data/t_episodes.parquet')
df_ct = pd.read_parquet('data/ct_episodes.parquet')

# Add side indicator
df_t['side'] = 'T'
df_ct['side'] = 'CT'

# Combine
df = pd.concat([df_t, df_ct], ignore_index=True)

# Convert duration to seconds
df['duration_sec'] = df['duration_ticks'] / 128.0

# Create simple crossfire proxy: has at least 1 teammate
# (More sophisticated version would use visibility graph, but this captures the core concept)
df['has_crossfire_support'] = (df['t_count_at_plant'] if 'side' == 'T' else df['ct_count_at_retake']) > 1

# Better approach: use current teammate count
for idx, row in df.iterrows():
    if row['side'] == 'T':
        teammates = row['t_count_current'] - 1  # Exclude self
    else:
        teammates = row['ct_count_current'] - 1
    
    df.loc[idx, 'has_crossfire_support'] = teammates >= 1

print("=" * 80)
print("CROSSFIRE EFFECTIVENESS BY NUMERICAL ADVANTAGE")
print("=" * 80)
print(f"\nTotal episodes: {len(df)}")
print(f"Deaths: {df['died'].sum()}")
print(f"Death rate: {df['died'].mean():.1%}\n")

# Categorize numerical advantage
def categorize_advantage(num_adv):
    if num_adv >= 1:
        return 'Advantage (≥1 extra player)'
    elif num_adv == 0:
        return 'Even (equal players)'
    else:
        return 'Disadvantage (outnumbered)'

df['advantage_category'] = df['numerical_advantage'].apply(categorize_advantage)

# Count episodes by category and crossfire
print("\nEPISODE COUNTS BY NUMERICAL ADVANTAGE AND CROSSFIRE")
print("=" * 80)
crossfire_counts = df.groupby(['advantage_category', 'has_crossfire_support']).agg({
    'died': ['count', 'sum', 'mean']
}).round(3)
print(crossfire_counts)
print()

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

categories = ['Advantage (≥1 extra player)', 'Even (equal players)', 'Disadvantage (outnumbered)']
colors = {'No Crossfire': '#e74c3c', 'With Crossfire': '#2ecc71'}

kmf = KaplanMeierFitter()

for idx, category in enumerate(categories):
    ax = axes[idx]
    
    df_cat = df[df['advantage_category'] == category]
    
    # Plot no crossfire
    no_crossfire = df_cat[df_cat['has_crossfire_support'] == False]
    if len(no_crossfire) > 0:
        kmf.fit(
            durations=no_crossfire['duration_sec'],
            event_observed=no_crossfire['died'],
            label=f'No Crossfire (n={len(no_crossfire)})'
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=colors['No Crossfire'],
            alpha=0.7,
            linewidth=2.5
        )
    
    # Plot with crossfire
    with_crossfire = df_cat[df_cat['has_crossfire_support'] == True]
    if len(with_crossfire) > 0:
        kmf.fit(
            durations=with_crossfire['duration_sec'],
            event_observed=with_crossfire['died'],
            label=f'With Crossfire (n={len(with_crossfire)})'
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=colors['With Crossfire'],
            alpha=0.7,
            linewidth=2.5
        )
    
    ax.set_title(category, fontsize=13, fontweight='bold')
    ax.set_xlabel('Time Since Bomb Plant (seconds)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('Survival Probability', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.set_ylim([0, 1.05])

plt.suptitle('Post-Plant Survival: Crossfire Support by Numerical Advantage', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
plt.savefig('paper/figures/crossfire_by_numerical_advantage.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("Saved: paper/figures/crossfire_by_numerical_advantage.png")
print("=" * 80)

# Statistical tests
print("\n" + "=" * 80)
print("PAIRWISE LOG-RANK TESTS (No Crossfire vs With Crossfire)")
print("=" * 80)

for category in categories:
    df_cat = df[df['advantage_category'] == category]
    
    no_crossfire = df_cat[df_cat['has_crossfire_support'] == False]
    with_crossfire = df_cat[df_cat['has_crossfire_support'] == True]
    
    if len(no_crossfire) > 0 and len(with_crossfire) > 0:
        result = logrank_test(
            durations_A=no_crossfire['duration_sec'],
            durations_B=with_crossfire['duration_sec'],
            event_observed_A=no_crossfire['died'],
            event_observed_B=with_crossfire['died']
        )
        
        print(f"\n{category}:")
        print(f"  Test statistic: {result.test_statistic:.4f}")
        print(f"  p-value: {result.p_value:.4e}")
        print(f"  Significant: {'Yes' if result.p_value < 0.05 else 'No'}")

# Median survival times
print("\n" + "=" * 80)
print("MEDIAN SURVIVAL TIMES BY CATEGORY")
print("=" * 80)

for category in categories:
    df_cat = df[df['advantage_category'] == category]
    
    print(f"\n{category}:")
    
    for has_crossfire in [False, True]:
        subset = df_cat[df_cat['has_crossfire_support'] == has_crossfire]
        
        if len(subset) > 0:
            kmf.fit(durations=subset['duration_sec'], event_observed=subset['died'])
            median = kmf.median_survival_time_
            
            crossfire_label = "With Crossfire" if has_crossfire else "No Crossfire"
            print(f"  {crossfire_label}: {median:.2f} seconds (n={len(subset)}, deaths={subset['died'].sum()})")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
