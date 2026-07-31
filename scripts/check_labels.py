import json
from pathlib import Path

SAMPLES = 'bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/samples'
EVENTS = 'bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/events.json'

with open(EVENTS) as f:
    events = json.load(f)
sorted_events = sorted(events, key=lambda e: e['id'])
print('Sorted events (training order):')
for i, e in enumerate(sorted_events):
    print(f'  Index {i}: id={e["id"]} name={e["name"]}')

print()
samples = sorted(Path(SAMPLES).glob('*.png'))
print(f'Total sample images: {len(samples)}')

# Check label files
for sp in list(samples):
    txt = sp.with_suffix('.txt')
    if txt.exists():
        label = txt.read_text().strip()
        parts = label.split()
        raw_class = int(parts[0])
        filename_id = int(sp.stem.split('_')[0])
        evt = next((e for e in events if e['id'] == filename_id), None)
        evt_name = evt['name'] if evt else 'UNKNOWN'
        mapped_name = sorted_events[raw_class]['name'] if raw_class < len(sorted_events) else 'OUT_OF_RANGE'
        print(f'{sp.name}: event={evt_name} label_class={raw_class} -> {mapped_name}')
