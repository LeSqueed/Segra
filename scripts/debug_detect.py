#!/usr/bin/env python3
"""Debug tool: seek through a recording and run YOLO inference on any frame.
Press SPACE or click 'Detect' to run inference on the current frame.
Press ESC or 'q' to quit.
"""
import sys
import math
import numpy as np
import onnxruntime as ort
import cv2

# --- config ---
MODEL_PATH = r"C:\Users\Cyph_\Documents\Projects\Segra\bin\Release\net10.0-windows10.0.19041.0\data\training\Overwatch\model.onnx"
EVENTS_PATH = r"C:\Users\Cyph_\Documents\Projects\Segra\bin\Release\net10.0-windows10.0.19041.0\data\training\Overwatch\events.json"
CONF_THRESHOLD = 0.25  # lowered
INPUT_SIZE = 640

# --- load model ---
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# --- load events/class names ---
import json
with open(EVENTS_PATH) as f:
    events = json.load(f)
sorted_events = sorted(events, key=lambda e: e["id"])
class_names = [e["name"] for e in sorted_events]
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# --- screen regions (for crop+resize before inference) ---
screen_regions = []
for e in sorted_events:
    w = e.get("screenRegionW")
    if w and w > 0:
        screen_regions.append({
            "x": e.get("screenRegionX", 0),
            "y": e.get("screenRegionY", 0),
            "w": w,
            "h": e.get("screenRegionH", 0),
        })
    else:
        screen_regions.append(None)

def overlap(r1, r2):
    if r1 is None or r2 is None:
        return False
    return (r1["x"] < r2["x"] + r2["w"] and r1["x"] + r1["w"] > r2["x"] and
            r1["y"] < r2["y"] + r2["h"] and r1["y"] + r1["h"] > r2["y"])

def merge(r1, r2):
    x = min(r1["x"], r2["x"])
    y = min(r1["y"], r2["y"])
    return {"x": x, "y": y, "w": max(r1["x"]+r1["w"], r2["x"]+r2["w"])-x,
            "h": max(r1["y"]+r1["h"], r2["y"]+r2["h"])-y}

# Build region groups (same logic as C# BuildRegionGroups)
def build_region_groups(regions):
    groups = []
    has_full = False
    for r in regions:
        if r is None:
            has_full = True
        else:
            merged = False
            for g in groups:
                if overlap(g, r):
                    g.update(merge(g, r))
                    merged = True
                    break
            if not merged:
                groups.append(dict(r))
    if has_full or not groups:
        groups.append({"x": 0, "y": 0, "w": 1, "h": 1})
    return groups

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def run_inference(bgr_frame, frame_h, frame_w):
    """Run inference on frame, crop+resize per region group, return results."""
    regions = build_region_groups(screen_regions)
    all_results = []
    for group in regions:
        rx = int(group["x"] * frame_w)
        ry = int(group["y"] * frame_h)
        rw = max(1, int(group["w"] * frame_w))
        rh = max(1, int(group["h"] * frame_h))
        if rx + rw > frame_w:
            rw = frame_w - rx
        if ry + rh > frame_h:
            rh = frame_h - ry
        if rw <= 0 or rh <= 0:
            continue
        crop = bgr_frame[ry:ry+rh, rx:rx+rw]
        resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        # BGRA -> RGB, normalize, reshape
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]  # (1,3,640,640)
        outputs = session.run(None, {input_name: blob})
        raw = outputs[0][0]  # shape (10, 8400)
        num_detections = raw.shape[1]  # 8400
        for i in range(num_detections):
            cx = raw[0, i] / INPUT_SIZE
            cy = raw[1, i] / INPUT_SIZE
            bw = raw[2, i] / INPUT_SIZE
            bh = raw[3, i] / INPUT_SIZE
            # Class scores are already sigmoided (Sigmoid node in graph)
            # But check if they're really in 0-1 range
            scores = raw[4:, i]
            max_class = np.argmax(scores)
            max_conf = scores[max_class]
            if max_conf < CONF_THRESHOLD:
                continue
            # Map back to full frame coords
            full_x = (cx * rw + rx) / frame_w
            full_y = (cy * rh + ry) / frame_h
            full_w = bw * rw / frame_w
            full_h = bh * rh / frame_h
            all_results.append({
                "class_id": int(max_class),
                "class_name": class_names[int(max_class)] if int(max_class) < len(class_names) else f"cls{int(max_class)}",
                "conf": float(max_conf),
                "x": full_x,
                "y": full_y,
                "w": full_w,
                "h": full_h,
            })
    return all_results

def draw_results(frame, results):
    h, w = frame.shape[:2]
    for r in results:
        x1 = int(r["x"] * w)
        y1 = int(r["y"] * h)
        x2 = int((r["x"] + r["w"]) * w)
        y2 = int((r["y"] + r["h"]) * h)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{r['class_name']} {r['conf']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# --- main ---
video_path = sys.argv[1] if len(sys.argv) > 1 else r"F:\Recordings\Segra\Full Sessions\Overwatch\2026-07-31_20-23-45.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: cannot open {video_path}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total_frames / fps
print(f"Video: {total_frames} frames, {fps:.2f} fps, {duration:.1f}s")

window_name = "Segra Detection Debug"
cv2.namedWindow(window_name)

# Track detection results
last_results = []
current_pos = 0

def on_trackbar(pos):
    global current_pos
    current_pos = pos

cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, on_trackbar)
cv2.setTrackbarPos("Frame", window_name, 0)

print("\nControls:")
print("  SPACE / 'd'  - Run detection on current frame")
print("  ESC / 'q'    - Quit")
print("  Trackbar     - Seek through video")

running = True
while running:
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    if last_results:
        draw_results(display, last_results)
        info = f"Detections: {len(last_results)}"
        for r in last_results:
            info += f" | {r['class_name']} ({r['conf']:.3f})"
        cv2.displayOverlay(window_name, info, 1000)

    # Add timestamp overlay
    ts = current_pos / fps
    cv2.putText(display, f"{int(ts//60):02d}:{int(ts%60):02d}.{int((ts%1)*100):02d}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow(window_name, display)
    key = cv2.waitKey(30) & 0xFF

    if key in (27, ord('q'), ord('Q')):
        running = False
    elif key in (32, ord('d'), ord('D')):
        # Run detection
        h, w = frame.shape[:2]
        print(f"\n--- Detection at frame {current_pos} ({ts:.2f}s) ---")
        last_results = run_inference(frame, h, w)
        if last_results:
            for r in sorted(last_results, key=lambda x: -x["conf"]):
                print(f"  {r['class_name']}: conf={r['conf']:.4f} "
                      f"box=({r['x']:.3f},{r['y']:.3f},{r['w']:.3f},{r['h']:.3f})")
        else:
            print("  No detections above threshold")

        # Diagnostic: dump max conf per region group even below threshold
        groups = build_region_groups(screen_regions)
        for gi, group in enumerate(groups):
            rx = int(group["x"] * w)
            ry = int(group["y"] * h)
            rw = max(1, int(group["w"] * w))
            rh = max(1, int(group["h"] * h))
            if rx + rw > w: rw = w - rx
            if ry + rh > h: rh = h - ry
            if rw <= 0 or rh <= 0: continue
            crop = frame[ry:ry+rh, rx:rx+rw]
            resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]
            outputs = session.run(None, {input_name: blob})
            raw = outputs[0][0]
            scores = raw[4:, :]  # (6, 8400)
            max_class_scores = scores.max(axis=0)  # max over classes per detection
            overall_max = max_class_scores.max()
            overall_min = max_class_scores.min()
            overall_mean = max_class_scores.mean()
            top_idx = max_class_scores.argmax()
            top_class = int(scores[:, top_idx].argmax())
            top_conf = float(max_class_scores[top_idx])
            print(f"  Region {gi} (rx={rx},ry={ry},{rw}x{rh}): "
                  f"conf range [{overall_min:.6f}, {overall_mean:.6f}, {overall_max:.6f}], "
                  f"best={class_names[top_class] if top_class < len(class_names) else top_class} @ {top_conf:.6f}")
            # Also show per-class max
            per_class_max = scores.max(axis=1)
            print(f"    Per-class max: ", end="")
            for ci in range(num_classes):
                print(f"{class_names[ci][:8]}={per_class_max[ci]:.6f} ", end="")
            print()

    # sync trackbar if user dragged it
    pos = cv2.getTrackbarPos("Frame", window_name)
    if pos != current_pos:
        current_pos = pos

cap.release()
cv2.destroyAllWindows()
