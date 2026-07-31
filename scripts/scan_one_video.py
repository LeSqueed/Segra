"""Scan a video at 1fps and report best confidence."""
import cv2, numpy as np, onnxruntime as ort, json, sys

MODEL = r'bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/model.onnx'
EVENTS = r'bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/events.json'

session = ort.InferenceSession(MODEL)
inp_name = session.get_inputs()[0].name

with open(EVENTS) as f:
    events = json.load(f)
sorted_events = sorted(events, key=lambda e: e['id'])
class_names = [e['name'] for e in sorted_events]

video = sys.argv[1]
cap = cv2.VideoCapture(video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'Video: {total} frames, {fps}fps, {total/fps:.1f}s')

best, best_frame, best_class = 0, 0, ''
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    if frame_idx % 60 != 0: continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rz = cv2.resize(gray, (640, 640), interpolation=cv2.INTER_LINEAR)
    blob = np.tile((rz.astype(np.float32)/255.0)[np.newaxis,:,:], (3,1,1))[np.newaxis]
    out = session.run(None, {inp_name: blob})[0][0]
    scores = out[4:, :]
    mc = float(scores.max())
    ci = int(scores.argmax() // scores.shape[1])
    if mc > best:
        best = mc
        best_frame = frame_idx
        best_class = class_names[ci] if ci < len(class_names) else '?'

cap.release()
print(f'Best: frame {best_frame} ({best_frame/fps:.1f}s) conf={best:.4f} class={best_class}')
