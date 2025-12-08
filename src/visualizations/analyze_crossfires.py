"""
Analyze CT retake crossfires and T post-plant position setups.
Identifies most effective 2-player, 3-player, and 4-player combinations.

Run after analyze_player_positioning.py completes and generates parquet files.
"""

import pandas as pd
from itertools import combinations
from collections import defaultdict

print("="*80)
print("CROSSFIRE & POSITION SETUP ANALYSIS")
print("="*80)

# Load episode data
print("\nLoading episode data...")
ct_df = pd.read_parquet('data/ct_episodes.parquet')
t_df = pd.read_parquet('data/t_episodes.parquet')

print(f"Loaded {len(ct_df)} CT episodes and {len(t_df)} T episodes")

# ============================================================================
# CT RETAKE CROSSFIRE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CT RETAKE CROSSFIRES (Most Effective Position Pairs)")
print("="*80)

# Group CT episodes by retake to get full setup
ct_retakes = ct_df.groupby(['demo_file', 'round_num', 'plant_tick']).agg({
    'zone': list,
    'round_winner': 'first',
    'site': 'first',
    'ct_count_at_plant': 'first',
    't_count_at_plant': 'first'
}).reset_index()

# Analyze 2-player crossfires
ct_crossfires = defaultdict(lambda: {'wins': 0, 'total': 0, 'a_site': 0, 'b_site': 0})

for _, retake in ct_retakes.iterrows():
    zones = retake['zone']
    winner = retake['round_winner']
    site = retake['site']
    
    # Get all unique 2-zone pairs
    unique_zones = sorted(set(zones))
    for pair in combinations(unique_zones, 2):
        ct_crossfires[pair]['total'] += 1
        if winner == 'CT':
            ct_crossfires[pair]['wins'] += 1
        if site == 'A':
            ct_crossfires[pair]['a_site'] += 1
        else:
            ct_crossfires[pair]['b_site'] += 1

# Filter to meaningful sample sizes (20+ retakes) and sort by win rate
ct_crossfires_filtered = {k: v for k, v in ct_crossfires.items() if v['total'] >= 20}
ct_sorted = sorted(ct_crossfires_filtered.items(), 
                   key=lambda x: x[1]['wins']/x[1]['total'], 
                   reverse=True)

print("\nTop 20 CT Crossfire Pairs (min 20 retakes):")
print(f"{'Zone 1':<30} {'Zone 2':<30} {'Win%':>6} {'Retakes':>8} {'A-site':>7} {'B-site':>7}")
print("-" * 120)

for (z1, z2), stats in ct_sorted[:20]:
    win_pct = stats['wins'] / stats['total'] * 100
    print(f"{z1:<30} {z2:<30} {win_pct:>5.1f}% {stats['total']:>8} {stats['a_site']:>7} {stats['b_site']:>7}")

# ============================================================================
# A-SITE SPECIFIC CT CROSSFIRES
# ============================================================================

print("\n" + "="*80)
print("A-SITE CT CROSSFIRES (Top 15)")
print("="*80)

a_site_retakes = ct_retakes[ct_retakes['site'] == 'A']
a_crossfires = defaultdict(lambda: {'wins': 0, 'total': 0})

for _, retake in a_site_retakes.iterrows():
    zones = retake['zone']
    winner = retake['round_winner']
    unique_zones = sorted(set(zones))
    for pair in combinations(unique_zones, 2):
        a_crossfires[pair]['total'] += 1
        if winner == 'CT':
            a_crossfires[pair]['wins'] += 1

a_crossfires_filtered = {k: v for k, v in a_crossfires.items() if v['total'] >= 15}
a_sorted = sorted(a_crossfires_filtered.items(), 
                  key=lambda x: x[1]['wins']/x[1]['total'], 
                  reverse=True)

print(f"{'Zone 1':<30} {'Zone 2':<30} {'Win%':>6} {'Retakes':>8}")
print("-" * 80)
for (z1, z2), stats in a_sorted[:15]:
    win_pct = stats['wins'] / stats['total'] * 100
    print(f"{z1:<30} {z2:<30} {win_pct:>5.1f}% {stats['total']:>8}")

# ============================================================================
# B-SITE SPECIFIC CT CROSSFIRES
# ============================================================================

print("\n" + "="*80)
print("B-SITE CT CROSSFIRES (Top 15)")
print("="*80)

b_site_retakes = ct_retakes[ct_retakes['site'] == 'B']
b_crossfires = defaultdict(lambda: {'wins': 0, 'total': 0})

for _, retake in b_site_retakes.iterrows():
    zones = retake['zone']
    winner = retake['round_winner']
    unique_zones = sorted(set(zones))
    for pair in combinations(unique_zones, 2):
        b_crossfires[pair]['total'] += 1
        if winner == 'CT':
            b_crossfires[pair]['wins'] += 1

b_crossfires_filtered = {k: v for k, v in b_crossfires.items() if v['total'] >= 10}
b_sorted = sorted(b_crossfires_filtered.items(), 
                  key=lambda x: x[1]['wins']/x[1]['total'], 
                  reverse=True)

print(f"{'Zone 1':<30} {'Zone 2':<30} {'Win%':>6} {'Retakes':>8}")
print("-" * 80)
for (z1, z2), stats in b_sorted[:15]:
    win_pct = stats['wins'] / stats['total'] * 100
    print(f"{z1:<30} {z2:<30} {win_pct:>5.1f}% {stats['total']:>8}")

# ============================================================================
# T POST-PLANT CROSSFIRE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("T POST-PLANT CROSSFIRES (Most Effective Position Pairs)")
print("="*80)

# Group T episodes by retake
t_retakes = t_df.groupby(['demo_file', 'round_num', 'plant_tick']).agg({
    'zone': list,
    'round_winner': 'first',
    'site': 'first',
    'ct_count_at_plant': 'first',
    't_count_at_plant': 'first'
}).reset_index()

# Analyze 2-player T crossfires
t_crossfires = defaultdict(lambda: {'wins': 0, 'total': 0, 'a_site': 0, 'b_site': 0})

for _, retake in t_retakes.iterrows():
    zones = retake['zone']
    winner = retake['round_winner']
    site = retake['site']
    
    unique_zones = sorted(set(zones))
    for pair in combinations(unique_zones, 2):
        t_crossfires[pair]['total'] += 1
        if winner == 'T':
            t_crossfires[pair]['wins'] += 1
        if site == 'A':
            t_crossfires[pair]['a_site'] += 1
        else:
            t_crossfires[pair]['b_site'] += 1

# Filter and sort
t_crossfires_filtered = {k: v for k, v in t_crossfires.items() if v['total'] >= 20}
t_sorted = sorted(t_crossfires_filtered.items(), 
                  key=lambda x: x[1]['wins']/x[1]['total'], 
                  reverse=True)

print("\nTop 20 T Post-Plant Pairs (min 20 retakes):")
print(f"{'Zone 1':<30} {'Zone 2':<30} {'Hold%':>6} {'Retakes':>8} {'A-site':>7} {'B-site':>7}")
print("-" * 120)

for (z1, z2), stats in t_sorted[:20]:
    hold_pct = stats['wins'] / stats['total'] * 100
    print(f"{z1:<30} {z2:<30} {hold_pct:>5.1f}% {stats['total']:>8} {stats['a_site']:>7} {stats['b_site']:>7}")

# ============================================================================
# A-SITE T POST-PLANT CROSSFIRES
# ============================================================================

print("\n" + "="*80)
print("A-SITE T POST-PLANT CROSSFIRES (Top 15)")
print("="*80)

a_t_retakes = t_retakes[t_retakes['site'] == 'A']
a_t_crossfires = defaultdict(lambda: {'wins': 0, 'total': 0})

for _, retake in a_t_retakes.iterrows():
    zones = retake['zone']
    winner = retake['round_winner']
    unique_zones = sorted(set(zones))
    for pair in combinations(unique_zones, 2):
        a_t_crossfires[pair]['total'] += 1
        if winner == 'T':
            a_t_crossfires[pair]['wins'] += 1

a_t_crossfires_filtered = {k: v for k, v in a_t_crossfires.items() if v['total'] >= 15}
a_t_sorted = sorted(a_t_crossfires_filtered.items(), 
                    key=lambda x: x[1]['wins']/x[1]['total'], 
                    reverse=True)

print(f"{'Zone 1':<30} {'Zone 2':<30} {'Hold%':>6} {'Retakes':>8}")
print("-" * 80)
for (z1, z2), stats in a_t_sorted[:15]:
    hold_pct = stats['wins'] / stats['total'] * 100
    print(f"{z1:<30} {z2:<30} {hold_pct:>5.1f}% {stats['total']:>8}")

# ============================================================================
# 3-PLAYER SETUPS (CT)
# ============================================================================

print("\n" + "="*80)
print("3-PLAYER CT RETAKE SETUPS (Top 15 A-site)")
print("="*80)

ct_3player = defaultdict(lambda: {'wins': 0, 'total': 0})

for _, retake in a_site_retakes.iterrows():
    zones = retake['zone']
    winner = retake['round_winner']
    unique_zones = sorted(set(zones))
    
    # Only analyze 3+ player retakes
    if len(unique_zones) >= 3:
        for triple in combinations(unique_zones, 3):
            ct_3player[triple]['total'] += 1
            if winner == 'CT':
                ct_3player[triple]['wins'] += 1

ct_3player_filtered = {k: v for k, v in ct_3player.items() if v['total'] >= 10}
ct_3player_sorted = sorted(ct_3player_filtered.items(), 
                           key=lambda x: x[1]['wins']/x[1]['total'], 
                           reverse=True)

print(f"{'Zone 1':<25} {'Zone 2':<25} {'Zone 3':<25} {'Win%':>6} {'Retakes':>8}")
print("-" * 115)
for (z1, z2, z3), stats in ct_3player_sorted[:15]:
    win_pct = stats['wins'] / stats['total'] * 100
    print(f"{z1:<25} {z2:<25} {z3:<25} {win_pct:>5.1f}% {stats['total']:>8}")

# ============================================================================
# 3-PLAYER SETUPS (T)
# ============================================================================

print("\n" + "="*80)
print("3-PLAYER T POST-PLANT SETUPS (Top 15 A-site)")
print("="*80)

t_3player = defaultdict(lambda: {'wins': 0, 'total': 0})

for _, retake in a_t_retakes.iterrows():
    zones = retake['zone']
    winner = retake['round_winner']
    unique_zones = sorted(set(zones))
    
    if len(unique_zones) >= 3:
        for triple in combinations(unique_zones, 3):
            t_3player[triple]['total'] += 1
            if winner == 'T':
                t_3player[triple]['wins'] += 1

t_3player_filtered = {k: v for k, v in t_3player.items() if v['total'] >= 10}
t_3player_sorted = sorted(t_3player_filtered.items(), 
                          key=lambda x: x[1]['wins']/x[1]['total'], 
                          reverse=True)

print(f"{'Zone 1':<25} {'Zone 2':<25} {'Zone 3':<25} {'Hold%':>6} {'Retakes':>8}")
print("-" * 115)
for (z1, z2, z3), stats in t_3player_sorted[:15]:
    hold_pct = stats['wins'] / stats['total'] * 100
    print(f"{z1:<25} {z2:<25} {z3:<25} {hold_pct:>5.1f}% {stats['total']:>8}")

# ============================================================================
# CONDITIONAL ON ADVANTAGE - ALL SCENARIOS
# ============================================================================

print("\n" + "="*80)
print("CT CROSSFIRES BY NUMERICAL ADVANTAGE (ALL SCENARIOS)")
print("="*80)

# Find all unique advantage scenarios in the data
advantage_counts = ct_retakes.groupby(['ct_count_at_plant', 't_count_at_plant', 'site']).size()
advantage_counts = advantage_counts[advantage_counts >= 10]  # Min 10 retakes

print(f"\nFound {len(advantage_counts)} scenario/site combinations with 10+ retakes")

for (ct_num, t_num, site), count in sorted(advantage_counts.items(), key=lambda x: x[1], reverse=True):
    filtered_retakes = ct_retakes[
        (ct_retakes['ct_count_at_plant'] == ct_num) & 
        (ct_retakes['t_count_at_plant'] == t_num) &
        (ct_retakes['site'] == site)
    ]
    
    adv_crossfires = defaultdict(lambda: {'wins': 0, 'total': 0})
    
    for _, retake in filtered_retakes.iterrows():
        zones = retake['zone']
        winner = retake['round_winner']
        unique_zones = sorted(set(zones))
        for pair in combinations(unique_zones, 2):
            adv_crossfires[pair]['total'] += 1
            if winner == 'CT':
                adv_crossfires[pair]['wins'] += 1
    
    adv_filtered = {k: v for k, v in adv_crossfires.items() if v['total'] >= 3}
    adv_sorted = sorted(adv_filtered.items(), 
                       key=lambda x: x[1]['wins']/x[1]['total'], 
                       reverse=True)
    
    # Calculate overall win rate for this scenario
    scenario_wins = filtered_retakes[filtered_retakes['round_winner'] == 'CT'].shape[0]
    scenario_total = len(filtered_retakes)
    scenario_win_rate = scenario_wins / scenario_total * 100
    
    advantage_type = "ADVANTAGE" if ct_num > t_num else ("DISADVANTAGE" if ct_num < t_num else "PARITY")
    
    print(f"\n{ct_num}v{t_num} {site}-site ({advantage_type}) - {scenario_total} retakes, {scenario_win_rate:.1f}% CT win rate")
    print(f"{'Zone 1':<30} {'Zone 2':<30} {'Win%':>6} {'Count':>6}")
    print("-" * 80)
    for (z1, z2), stats in adv_sorted[:10]:
        win_pct = stats['wins'] / stats['total'] * 100
        print(f"{z1:<30} {z2:<30} {win_pct:>5.1f}% {stats['total']:>6}")

# ============================================================================
# UTILITY STATE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CROSSFIRE EFFECTIVENESS WITH UTILITY STATE")
print("="*80)

# Add utility aggregation to retakes
ct_retakes_util = ct_df.groupby(['demo_file', 'round_num', 'plant_tick']).agg({
    'zone': list,
    'round_winner': 'first',
    'site': 'first',
    'ct_count_at_plant': 'first',
    't_count_at_plant': 'first',
    'smokes_available': 'sum',
    'mollies_available': 'sum',
    'flashes_available': 'sum',
    'he_grenades_available': 'sum',
    'active_smoke_zones': lambda x: list(x.iloc[0]) if len(x) > 0 else [],
    'active_molly_zones': lambda x: list(x.iloc[0]) if len(x) > 0 else []
}).reset_index()

# Analyze impact of having smokes on crossfire success
print("\n5v5 A-site Retakes WITH vs WITHOUT Smokes Available:")
print("-" * 80)

# WITH smokes
with_smokes = ct_retakes_util[
    (ct_retakes_util['ct_count_at_plant'] == 5) &
    (ct_retakes_util['t_count_at_plant'] == 5) &
    (ct_retakes_util['site'] == 'A') &
    (ct_retakes_util['smokes_available'] >= 1)
]

smoke_crossfires = defaultdict(lambda: {'wins': 0, 'total': 0})
for _, retake in with_smokes.iterrows():
    zones = retake['zone']
    winner = retake['round_winner']
    unique_zones = sorted(set(zones))
    for pair in combinations(unique_zones, 2):
        smoke_crossfires[pair]['total'] += 1
        if winner == 'CT':
            smoke_crossfires[pair]['wins'] += 1

smoke_filtered = {k: v for k, v in smoke_crossfires.items() if v['total'] >= 5}
smoke_sorted = sorted(smoke_filtered.items(), 
                     key=lambda x: x[1]['wins']/x[1]['total'], 
                     reverse=True)

with_smoke_wins = with_smokes[with_smokes['round_winner'] == 'CT'].shape[0]
with_smoke_total = len(with_smokes)
with_smoke_rate = with_smoke_wins / with_smoke_total * 100 if with_smoke_total > 0 else 0

print(f"\nWITH Smokes ({with_smoke_total} retakes, {with_smoke_rate:.1f}% win rate):")
print(f"{'Zone 1':<30} {'Zone 2':<30} {'Win%':>6} {'Count':>6}")
for (z1, z2), stats in smoke_sorted[:8]:
    win_pct = stats['wins'] / stats['total'] * 100
    print(f"{z1:<30} {z2:<30} {win_pct:>5.1f}% {stats['total']:>6}")

# WITHOUT smokes
without_smokes = ct_retakes_util[
    (ct_retakes_util['ct_count_at_plant'] == 5) &
    (ct_retakes_util['t_count_at_plant'] == 5) &
    (ct_retakes_util['site'] == 'A') &
    (ct_retakes_util['smokes_available'] == 0)
]

no_smoke_crossfires = defaultdict(lambda: {'wins': 0, 'total': 0})
for _, retake in without_smokes.iterrows():
    zones = retake['zone']
    winner = retake['round_winner']
    unique_zones = sorted(set(zones))
    for pair in combinations(unique_zones, 2):
        no_smoke_crossfires[pair]['total'] += 1
        if winner == 'CT':
            no_smoke_crossfires[pair]['wins'] += 1

no_smoke_filtered = {k: v for k, v in no_smoke_crossfires.items() if v['total'] >= 5}
no_smoke_sorted = sorted(no_smoke_filtered.items(), 
                        key=lambda x: x[1]['wins']/x[1]['total'], 
                        reverse=True)

without_smoke_wins = without_smokes[without_smokes['round_winner'] == 'CT'].shape[0]
without_smoke_total = len(without_smokes)
without_smoke_rate = without_smoke_wins / without_smoke_total * 100 if without_smoke_total > 0 else 0

print(f"\nWITHOUT Smokes ({without_smoke_total} retakes, {without_smoke_rate:.1f}% win rate):")
print(f"{'Zone 1':<30} {'Zone 2':<30} {'Win%':>6} {'Count':>6}")
for (z1, z2), stats in no_smoke_sorted[:8]:
    win_pct = stats['wins'] / stats['total'] * 100
    print(f"{z1:<30} {z2:<30} {win_pct:>5.1f}% {stats['total']:>6}")

# MOLLY analysis
print("\n\n5v5 A-site Retakes WITH vs WITHOUT Mollies Available:")
print("-" * 80)

with_mollies = ct_retakes_util[
    (ct_retakes_util['ct_count_at_plant'] == 5) &
    (ct_retakes_util['t_count_at_plant'] == 5) &
    (ct_retakes_util['site'] == 'A') &
    (ct_retakes_util['mollies_available'] >= 1)
]

without_mollies = ct_retakes_util[
    (ct_retakes_util['ct_count_at_plant'] == 5) &
    (ct_retakes_util['t_count_at_plant'] == 5) &
    (ct_retakes_util['site'] == 'A') &
    (ct_retakes_util['mollies_available'] == 0)
]

with_molly_rate = (with_mollies[with_mollies['round_winner'] == 'CT'].shape[0] / len(with_mollies) * 100) if len(with_mollies) > 0 else 0
without_molly_rate = (without_mollies[without_mollies['round_winner'] == 'CT'].shape[0] / len(without_mollies) * 100) if len(without_mollies) > 0 else 0

print(f"WITH Mollies: {len(with_mollies)} retakes, {with_molly_rate:.1f}% win rate")
print(f"WITHOUT Mollies: {len(without_mollies)} retakes, {without_molly_rate:.1f}% win rate")

# Utility-rich vs utility-poor retakes
print("\n\n5v5 A-site: UTILITY-RICH (2+ smokes/mollies) vs UTILITY-POOR (0-1 util):")
print("-" * 80)

util_rich = ct_retakes_util[
    (ct_retakes_util['ct_count_at_plant'] == 5) &
    (ct_retakes_util['t_count_at_plant'] == 5) &
    (ct_retakes_util['site'] == 'A') &
    ((ct_retakes_util['smokes_available'] + ct_retakes_util['mollies_available']) >= 2)
]

util_poor = ct_retakes_util[
    (ct_retakes_util['ct_count_at_plant'] == 5) &
    (ct_retakes_util['t_count_at_plant'] == 5) &
    (ct_retakes_util['site'] == 'A') &
    ((ct_retakes_util['smokes_available'] + ct_retakes_util['mollies_available']) <= 1)
]

util_rich_rate = (util_rich[util_rich['round_winner'] == 'CT'].shape[0] / len(util_rich) * 100) if len(util_rich) > 0 else 0
util_poor_rate = (util_poor[util_poor['round_winner'] == 'CT'].shape[0] / len(util_poor) * 100) if len(util_poor) > 0 else 0

print(f"UTILITY-RICH (2+ util): {len(util_rich)} retakes, {util_rich_rate:.1f}% win rate")
print(f"UTILITY-POOR (0-1 util): {len(util_poor)} retakes, {util_poor_rate:.1f}% win rate")
print(f"Utility Impact: {util_rich_rate - util_poor_rate:+.1f} percentage points")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
