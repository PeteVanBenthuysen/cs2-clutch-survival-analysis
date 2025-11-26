from pathlib import Path
from demoparser2 import DemoParser

demo_path = Path("research_demos/extracted/8036_IEM_Melbourne_2025/2381642/liquid-vs-virtuspro-m2-mirage.dem")
p = DemoParser(str(demo_path))

print("Testing event parsing:")
events_to_test = ['grenade_thrown', 'bomb_planted', 'smokegrenade_detonate', 'player_death']
for event_name in events_to_test:
    try:
        data = p.parse_event(event_name)
        data_type = type(data).__name__
        data_len = len(data) if hasattr(data, '__len__') else 'N/A'
        print(f"  {event_name:30s} -> {data_type:15s} len={data_len}")
    except Exception as e:
        print(f"  {event_name:30s} -> ERROR: {e}")
