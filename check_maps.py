from pathlib import Path
from collections import Counter
import re

# Find all .dem files in research_demos folder
demos_folder = Path(r"C:\Users\petev\OneDrive\Desktop\cs2-clutch-survival-analysis\research_demos")
demo_files = list(demos_folder.rglob("*.dem"))

# Extract map names from filenames (format: teamA-vs-teamB-tournament-mapname.dem)
# Map name is the last part before .dem
maps = []

for demo in demo_files:
    # Split filename by '-' and get the last part before .dem
    parts = demo.stem.split('-')
    if parts:
        map_name = parts[-1]
        maps.append(map_name)

map_counts = Counter(maps)

print('\nMap counts from demo files:\n')
print(f'Total demo files found: {len(demo_files)}')
print(f'Total demos with map names: {len(maps)}\n')

for map_name, count in sorted(map_counts.items(), key=lambda x: x[1], reverse=True):
    print(f'{map_name:15s}: {count:4d} demos')

print(f'\nTotal unique maps: {len(map_counts)}')
