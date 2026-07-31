#!/usr/bin/env python3
"""Quick diagnostic: dump model output stats on a few specific frames."""
import sys, json
import numpy as np
import onnxruntime as ort
import cv2

MODEL_PATH = r"C:\Users\Cyph_\Documents\Projects\Segra\bin\Release\net10.0-windows10.0.19041.0\data\training\Overwatch\model.onnx"
EVENTS_PATH = r"C:\Users\Cyph_\Documents\Projects\Segra\bin\Release\net10.0-windows10.0.19041.0\data\training\Overwatch\events.json"
INPUT_SIZE = 640

with open(EVENTS_PATH) as f:
    events = json.load(f)
sorted_events = sorted(events, key=lambda e: e["id"])
class_names = [e["name"] for e in sorted_events]
num_classes = len(class_names)
print(f"Classes: {class_names}")

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS={fps}, total frames={total}")

# Sample every 2 seconds (120 frames at 60fps)
step = int(fps * 2)
print(f"Sampling every {step} frames (every 2s)...\n")

for frame_idx in range(0, total, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    timestamp = frame_idx / fps

    # Run on full frame (not cropped)
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]

    outputs = session.run(None, {input_name: blob})
    raw = outputs[0][0]  # (10, 8400)

    # Box stats
    boxes = raw[:4, :]
    print(f"[{timestamp:6.1f}s frame {frame_idx:5d}] "
          f"box range: cx=[{boxes[0].min():.1f},{boxes[0].max():.1f}] "
          f"cy=[{boxes[1].min():.1f},{boxes[1].max():.1f}] "
          f"w=[{boxes[2].min():.1f},{boxes[2].max():.1f}] "
          f"h=[{boxes[3].min():.1f},{boxes[3].max():.1f}]")

    # Class score stats
    scores = raw[4:, :]  # (6, 8400)
    for ci in range(num_classes):
        s = scores[ci]
        top5 = np.sort(s)[-5:][::-1]
        print(f"  {class_names[ci]:18s}: "
              f"mean={s.mean():.6f} max={s.max():.6f} "
              f"top5={[f'{v:.6f}' for v in top5]}")

    print()

cap.release()
print("Done.")
