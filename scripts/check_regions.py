"""Check what the regions actually capture on a video frame."""
import cv2, json, sys, numpy as np
from pathlib import Path

EVENTS = 'bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/events.json'

with open(EVENTS) as f:
    events = json.load(f)
sorted_events = sorted(events, key=lambda e: e['id'])
class_names = [e['name'] for e in sorted_events]

# Load a frame from the video
VIDEO = sys.argv[1]
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

# Seek to 10s (frame ~600) where elimination was likely happening
cap.set(cv2.CAP_PROP_POS_FRAMES, 600)
ret, frame = cap.read()
cap.release()
if not ret:
    print("Can't read frame")
    sys.exit(1)

h, w = frame.shape[:2]
print(f'Frame: {w}x{h}')

# Build region groups (same as C#)  
regions = []
for e in sorted_events:
    rw = e.get('screenRegionW')
    if rw and rw > 0:
        regions.append({
            'x': e.get('screenRegionX',0), 'y': e.get('screenRegionY',0),
            'w': rw, 'h': e.get('screenRegionH',0)
        })
    else:
        regions.append(None)

# Group them with containment + overlap (same as C#)
groups = []
has_full = False
for r in regions:
    if r is None: has_full = True
    else:
        merged = False
        for gi, g in enumerate(groups):
            # Check overlap
            if (g['x'] < r['x']+r['w'] and g['x']+g['w'] > r['x'] and
                g['y'] < r['y']+r['h'] and g['y']+g['h'] > r['y']):
                x = min(g['x'], r['x']); y = min(g['y'], r['y'])
                groups[gi] = {'x': x, 'y': y, 'w': max(g['x']+g['w'], r['x']+r['w'])-x,
                              'h': max(g['y']+g['h'], r['y']+r['h'])-y}
                merged = True; break
        if not merged: groups.append(dict(r))
if has_full or not groups: groups.append({'x':0,'y':0,'w':1,'h':1})

print(f'Region groups: {len(groups)}')
print()

# For each group, draw on the frame and print what's in the crop
for gi, g in enumerate(groups):
    rx, ry = int(g['x']*w), int(g['y']*h)
    rw_c = max(1, int(g['w']*w)); rh_c = max(1, int(g['h']*h))
    if rx+rw_c > w: rw_c = w-rx
    if ry+rh_c > h: rh_c = h-ry
    
    # Draw rectangle
    cv2.rectangle(frame, (rx, ry), (rx+rw_c, ry+rh_c), (0, 255, 0), 3)
    cv2.putText(frame, f'G{gi}', (rx+5, ry+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    crop = frame[ry:ry+rh_c, rx:rx+rw_c]
    if crop.size == 0:
        print(f'Group {gi}: EMPTY CROP at ({rx},{ry},{rw_c}x{rh_c})')
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        print(f'Group {gi}: ({rx},{ry},{rw_c}x{rh_c}) norm=({g["x"]:.4f},{g["y"]:.4f},{g["w"]:.4f},{g["h"]:.4f})')
        print(f'  pixel range: [{gray.min()}, {gray.max()}], mean={gray.mean():.1f}, std={gray.std():.1f}')
        
        # Check if there's any non-dark content (bright text)
        bright = (gray > 80).sum()
        print(f'  bright pixels (>80): {bright}/{gray.size} ({bright/gray.size*100:.1f}%)')

# Save the annotated frame
cv2.imwrite('debug_frame_with_regions.png', frame)
print(f'\nSaved debug_frame_with_regions.png')
print(f'Open it to see where the region boxes land on the frame.')
