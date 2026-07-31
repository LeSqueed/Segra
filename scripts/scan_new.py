#!/usr/bin/env python3
import sys, json, cv2, numpy as np, onnxruntime as ort

MODEL = 'bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/model.onnx'
EVENTS = 'bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/events.json'
ISZ = 640

with open(EVENTS) as f:
    events = json.load(f)
sorted_events = sorted(events, key=lambda e: e['id'])
class_names = [e['name'] for e in sorted_events]

session = ort.InferenceSession(MODEL)
inp_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(sys.argv[1])
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

regions = []
for e in sorted_events:
    w = e.get('screenRegionW')
    regions.append((e.get('screenRegionX',0), e.get('screenRegionY',0), w, e.get('screenRegionH',0)) if w and w > 0 else None)

groups, has_full = [], False
for r in regions:
    if r is None: has_full = True
    else:
        merged = False
        for gi, g in enumerate(groups):
            if (g[0] < r[0]+r[2] and g[0]+g[2] > r[0] and g[1] < r[1]+r[3] and g[1]+g[3] > r[1]):
                x, y = min(g[0],r[0]), min(g[1],r[1])
                groups[gi] = (x, y, max(g[0]+g[2],r[0]+r[2])-x, max(g[1]+g[3],r[1]+r[3])-y)
                merged = True; break
        if not merged: groups.append(r)
if has_full or not groups: groups.append((0,0,1,1))
print(f'Groups: {len(groups)}')

best = 0
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if frame_idx % 60: frame_idx += 1; continue

    h, w = frame.shape[:2]
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for gx, gy, gw, gh in groups:
        rx, ry = int(gx*w), int(gy*h)
        rw = max(1, int(gw*w)); rh = max(1, int(gh*h))
        if rx+rw > w: rw = w-rx
        if ry+rh > h: rh = h-ry
        if rw<=0 or rh<=0: continue
        crop = gray_full[ry:ry+rh, rx:rx+rw]
        if crop.size == 0: continue
        rz = cv2.resize(crop, (ISZ, ISZ)).astype(np.float32) / 255.0
        blob = np.tile(rz[np.newaxis, :, :], (3, 1, 1))[np.newaxis]
        out = session.run(None, {inp_name: blob})[0][0]
        scores = out[4:, :]  # (6, 8400)
        mc = float(scores.max())

        if mc > best:
            best = mc

        if mc > 0.1:
            ts = frame_idx / fps
            idx = scores.argmax()
            ci = idx // 8400
            print(f'  [{ts:.1f}s] {class_names[ci]}: {mc:.4f}')

    frame_idx += 1

print(f'\nBest confidence: {best:.6f}')
cap.release()
