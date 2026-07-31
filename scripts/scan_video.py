#!/usr/bin/env python3
"""Scan a recording video and dump detection confidence stats per region group.
Run from terminal: python scripts/scan_video.py <video_path> [sample_every_n_frames]
"""
import sys, json, math
import numpy as np
import onnxruntime as ort
import cv2

MODEL_PATH = r"C:\Users\Cyph_\Documents\Projects\Segra\bin\Release\net10.0-windows10.0.19041.0\data\training\Overwatch\model.onnx"
EVENTS_PATH = r"C:\Users\Cyph_\Documents\Projects\Segra\bin\Release\net10.0-windows10.0.19041.0\data\training\Overwatch\events.json"
INPUT_SIZE = 640
SAMPLE_EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 30  # every ~1 sec at 30fps

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
model_output_channels = session.get_outputs()[0].shape[1]  # e.g. 10 = 4 bbox + num_classes
num_classes = model_output_channels - 4
print(f"Model: {model_output_channels} channels, {num_classes} classes", flush=True)

with open(EVENTS_PATH) as f:
    events = json.load(f)
sorted_events = sorted(events, key=lambda e: e["id"])
class_names = [e["name"] for e in sorted_events[:num_classes]]
print(f"Classes ({num_classes}): {class_names}", flush=True)

screen_regions = []
for e in sorted_events:
    w = e.get("screenRegionW")
    if w and w > 0:
        screen_regions.append({"x": e.get("screenRegionX", 0), "y": e.get("screenRegionY", 0), "w": w, "h": e.get("screenRegionH", 0)})
    else:
        screen_regions.append(None)

def overlap(r1, r2):
    if r1 is None or r2 is None: return False
    return (r1["x"] < r2["x"] + r2["w"] and r1["x"] + r1["w"] > r2["x"] and
            r1["y"] < r2["y"] + r2["h"] and r1["y"] + r1["h"] > r2["y"])

def merge(r1, r2):
    x = min(r1["x"], r2["x"])
    y = min(r1["y"], r2["y"])
    return {"x": x, "y": y, "w": max(r1["x"]+r1["w"], r2["x"]+r2["w"])-x, "h": max(r1["y"]+r1["h"], r2["y"]+r2["h"])-y}

def build_groups(regions):
    groups = []
    has_full = False
    for r in regions:
        if r is None: has_full = True
        else:
            merged = False
            for g in groups:
                if overlap(g, r): g.update(merge(g, r)); merged = True; break
            if not merged: groups.append(dict(r))
    if has_full or not groups: groups.append({"x": 0, "y": 0, "w": 1, "h": 1})
    return groups

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: cannot open {video_path}")
    sys.exit(1)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total / fps
print(f"Video: {total} frames, {fps:.2f} fps, {duration:.1f}s, sampling every {SAMPLE_EVERY} frames", flush=True)

groups = build_groups(screen_regions)
print(f"Region groups: {len(groups)}")
for gi, g in enumerate(groups):
    print(f"  Group {gi}: x={g['x']:.3f} y={g['y']:.3f} w={g['w']:.3f} h={g['h']:.3f}")

frame_idx = 0
processed = 0
all_max_confs = []  # per group
per_class_max = [0.0] * num_classes
per_class_count = [0] * num_classes

while True:
    ret, frame = cap.read()
    if not ret: break
    if frame_idx % SAMPLE_EVERY != 0:
        frame_idx += 1
        continue

    h, w = frame.shape[:2]
    processed += 1
    if processed % 50 == 0:
        print(f"  Processed {processed} frames...", flush=True)

    for gi, group in enumerate(groups):
        rx = int(group["x"] * w)
        ry = int(group["y"] * h)
        rw = max(1, int(group["w"] * w))
        rh = max(1, int(group["h"] * h))
        if rx + rw > w: rw = w - rx
        if ry + rh > h: rh = h - ry
        if rw <= 0 or rh <= 0: continue

        # C# pipeline order: full-frame grayscale, then crop+resize
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray_full[ry:ry+rh, rx:rx+rw]
        resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        blob = (resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
        blob = np.tile(blob, (1, 3, 1, 1))

        outputs = session.run(None, {input_name: blob})
        raw = outputs[0][0]  # (10, 8400)
        scores = raw[4:, :]  # (6, 8400)
        max_per_det = scores.max(axis=0)  # max over classes per anchor
        best = max_per_det.max()
        all_max_confs.append(best)

        # per-class stats
        for ci in range(num_classes):
            pc = scores[ci].max()
            if pc > per_class_max[ci]:
                per_class_max[ci] = pc

        # count detections above 0.25 and 0.5
        for ci in range(num_classes):
            cnt = int((scores[ci] > 0.25).sum())
            per_class_count[ci] += cnt

    frame_idx += 1

cap.release()

print(f"\n=== Results ({processed} frames sampled) ===")
all_confs = np.array(all_max_confs)
print(f"Overall confidence stats across all region groups:")
print(f"  Mean:   {all_confs.mean():.6f}")
print(f"  Median: {np.median(all_confs):.6f}")
print(f"  Max:    {all_confs.max():.6f}")
print(f"  Min:    {all_confs.min():.6f}")
print(f"  > 0.25: {(all_confs > 0.25).sum()} / {len(all_confs)} ({(all_confs > 0.25).mean()*100:.2f}%)")
print(f"  > 0.50: {(all_confs > 0.50).sum()} / {len(all_confs)} ({(all_confs > 0.50).mean()*100:.2f}%)")
print()
print("Per-class best confidence:")
for ci in range(num_classes):
    print(f"  {class_names[ci]}: max={per_class_max[ci]:.6f}, "
          f"detections>0.25={per_class_count[ci]}")
print()
# Show top N frame confidences
sorted_conf = sorted(all_max_confs, reverse=True)
print(f"Top 20 confidences: {[f'{c:.6f}' for c in sorted_conf[:20]]}")
