import pandas as pd

# Load both T-side models
hr_t_a = pd.read_csv('results/cox_models/ridge/t_a_full_hazard_ratios.csv', index_col=0)
hr_t_b = pd.read_csv('results/cox_models/ridge/t_b_full_hazard_ratios.csv', index_col=0)

# Crossfire main effects
crossfire_t_a = hr_t_a.loc['has_crossfire_support', 'Hazard Ratio']
crossfire_t_b = hr_t_b.loc['has_crossfire_support', 'Hazard Ratio']

print("=" * 90)
print("TABLE: TOP 5 SAFEST POSITIONS WITH CROSSFIRE SUPPORT (T-SIDE)")
print("=" * 90)

# Get top 5 for A-site
zones_a = hr_t_a[[z.startswith('zone_') and '_x_' not in z for z in hr_t_a.index]]
zones_a_sorted = zones_a.sort_values('Hazard Ratio').head(5)

print("\n**A-SITE**\n")
print(f"{'Position':<30} {'Solo HR':<12} {'w/ Crossfire':<15} {'Risk Reduction':<15} {'p-value'}")
print("-" * 90)

for zone_name, row in zones_a_sorted.iterrows():
    zone_hr = row['Hazard Ratio']
    with_crossfire = zone_hr * crossfire_t_a
    risk_red = (1 - with_crossfire) * 100
    p_val = row['p-value']
    
    clean_name = zone_name.replace('zone_', '').replace('_', ' ')
    
    print(f"{clean_name:<30} {zone_hr:<12.3f} {with_crossfire:<15.3f} {risk_red:>6.1f}%{'':<8} {p_val:.2e}")

# Get top 5 for B-site
zones_b = hr_t_b[[z.startswith('zone_') and '_x_' not in z for z in hr_t_b.index]]
zones_b_sorted = zones_b.sort_values('Hazard Ratio').head(5)

print("\n\n**B-SITE**\n")
print(f"{'Position':<30} {'Solo HR':<12} {'w/ Crossfire':<15} {'Risk Reduction':<15} {'p-value'}")
print("-" * 90)

for zone_name, row in zones_b_sorted.iterrows():
    zone_hr = row['Hazard Ratio']
    with_crossfire = zone_hr * crossfire_t_b
    risk_red = (1 - with_crossfire) * 100
    p_val = row['p-value']
    
    clean_name = zone_name.replace('zone_', '').replace('_', ' ')
    
    print(f"{clean_name:<30} {zone_hr:<12.3f} {with_crossfire:<15.3f} {risk_red:>6.1f}%{'':<8} {p_val:.2e}")

print("\n" + "=" * 90)
print("Note: Crossfire support = teammate positioned to engage threat zones from different angle")
print(f"General crossfire effect: A-site HR={crossfire_t_a:.3f}, B-site HR={crossfire_t_b:.3f}")
print("=" * 90)

# Also show your requested zones for reference
print("\n\n" + "=" * 90)
print("YOUR REQUESTED ZONES (For Reference)")
print("=" * 90)

requested_a = {
    'A_Ropz': 14,
    'Connector_Jungle_Cubby': 47,
    'Palace_Entry': 74,
    'Outside_A_Main': 62,
    'A_Default': 30  # Not in model
}

requested_b = {
    'B_Pillar_1': 250,
    'Cat': 140,
    'Bench_B': 246,  # Changed from just 'Bench'
    'B_Apps_First_Section': 265,  # Closest to "Mid Apps"
    'Bench_Corner': 239
}

print("\n**A-SITE (Your Requested)**\n")
print(f"{'Position':<30} {'Zone #':<10} {'Solo HR':<12} {'w/ Crossfire':<15} {'Risk Reduction'}")
print("-" * 90)

for zone_name, zone_num in requested_a.items():
    zone_key = f'zone_{zone_name}'
    if zone_key in hr_t_a.index:
        zone_hr = hr_t_a.loc[zone_key, 'Hazard Ratio']
        with_crossfire = zone_hr * crossfire_t_a
        risk_red = (1 - with_crossfire) * 100
        
        clean_name = zone_name.replace('_', ' ')
        print(f"{clean_name:<30} {zone_num:<10} {zone_hr:<12.3f} {with_crossfire:<15.3f} {risk_red:>6.1f}%")
    else:
        print(f"{zone_name.replace('_', ' '):<30} {zone_num:<10} NOT FOUND IN MODEL")

print("\n**B-SITE (Your Requested)**\n")
print(f"{'Position':<30} {'Zone #':<10} {'Solo HR':<12} {'w/ Crossfire':<15} {'Risk Reduction'}")
print("-" * 90)

for zone_name, zone_num in requested_b.items():
    zone_key = f'zone_{zone_name}'
    if zone_key in hr_t_b.index:
        zone_hr = hr_t_b.loc[zone_key, 'Hazard Ratio']
        with_crossfire = zone_hr * crossfire_t_b
        risk_red = (1 - with_crossfire) * 100
        
        clean_name = zone_name.replace('_', ' ')
        print(f"{clean_name:<30} {zone_num:<10} {zone_hr:<12.3f} {with_crossfire:<15.3f} {risk_red:>6.1f}%")
    else:
        print(f"{zone_name.replace('_', ' '):<30} {zone_num:<10} NOT FOUND IN MODEL")
