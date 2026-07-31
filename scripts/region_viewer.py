#!/usr/bin/env python3
"""Small video viewer with region-cropped detection on button press.
 
Usage:
    python scripts/region_viewer.py <video_path>
 
Controls:
    SPACE / 'd'  — Run detection on current frame using region groups
    ESC / 'q'    — Quit
    Trackbar     — Seek through video
"""
import sys, json, math
import numpy as np
import onnxruntime as ort
import cv2

MODEL = r"C:\Users\Cyph_\Documents\Projects\Segra\bin\Release\net10.0-windows10.0.19041.0\data\training\Overwatch\model.onnx"
EVENTS = r"C:\Users\Cyph_\Documents\Projects\Segra\bin\Release\net10.0-windows10.0.19041.0\data\training\Overwatch\events.json"
CONF = 0.15
ISZ = 640

with open(EVENTS) as f:
    events = json.load(f)
sorted_events = sorted(events, key=lambda e: e["id"])
class_names = [e["name"] for e in sorted_events]
num_classes = len(class_names)

session = ort.InferenceSession(MODEL)
inp_name = session.get_inputs()[0].name

def get_regions():
    regions = []
    for e in sorted_events:
        w = e.get("screenRegionW")
        if w and w > 0:
            regions.append((e.get("screenRegionX",0), e.get("screenRegionY",0), w, e.get("screenRegionH",0)))
        else:
            regions.append(None)
    return regions

def overlap(a, b):
    if a is None or b is None: return False
    return (a[0] < b[0]+b[2] and a[0]+a[2] > b[0] and a[1] < b[1]+b[3] and a[1]+a[3] > b[1])

def build_groups(regions):
    groups = []
    has_full = False
    for i, r in enumerate(regions):
        if r is None: has_full = True
        else:
            merged = False
            for gi, g in enumerate(groups):
                if overlap(g, r):
                    x = min(g[0], r[0]); y = min(g[1], r[1])
                    groups[gi] = (x, y, max(g[0]+g[2], r[0]+r[2])-x, max(g[1]+g[3], r[1]+r[3])-y)
                    merged = True
                    break
            if not merged: groups.append(r)
    if has_full or not groups: groups.append((0,0,1,1))
    return groups

def run_detection(frame, h, w, groups):
    # Grayscale full frame once, then crop per group
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = []
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
        for i in range(out.shape[1]):
            scores = out[4:, i]
            ci = np.argmax(scores)
            cf = float(scores[ci])
            if cf < CONF: continue
            cx = out[0,i]/ISZ; cy = out[1,i]/ISZ; bw = out[2,i]/ISZ; bh = out[3,i]/ISZ
            fx = (cx*rw + rx)/w; fy = (cy*rh + ry)/h; fw = bw*rw/w; fh = bh*rh/h
            results.append((ci, cf, fx, fy, fw, fh))
    return results

video = sys.argv[1]
cap = cv2.VideoCapture(video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

groups = build_groups(get_regions())
print(f"Region groups: {len(groups)}")
for i,(gx,gy,gw,gh) in enumerate(groups):
    print(f"  Group {i}: ({gx:.3f},{gy:.3f}) {gw:.3f}x{gh:.3f}")

WNAME = "Segra Detection Viewer"
cv2.namedWindow(WNAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WNAME, 960, 600)  # small enough to fit

cv2.createTrackbar("Frame", WNAME, 0, total-1, lambda x: None)
cv2.setTrackbarPos("Frame", WNAME, 0)

last_results = []
last_frame = None
cur_pos = [0]
print("\nControls:  SPACE/d = Detect | ESC/q = Quit | Trackbar = Seek")

while True:
    pos = cv2.getTrackbarPos("Frame", WNAME)
    if pos != cur_pos[0]:
        cur_pos[0] = pos
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret: last_frame = frame.copy()
    
    if last_frame is None:
        ret, frame = cap.read()
        if not ret: break
        last_frame = frame.copy()
    
    display = last_frame.copy()
    h, w = display.shape[:2]
    
    # Draw region group boxes (semi-transparent)
    overlay = display.copy()
    for gx, gy, gw, gh in groups:
        rx, ry = int(gx*w), int(gy*h)
        rw2, rh2 = int(gw*w), int(gh*h)
        cv2.rectangle(overlay, (rx,ry), (rx+rw2,ry+rh2), (100,100,255), -1)
    cv2.addWeighted(overlay, 0.12, display, 0.88, 0, display)
    for gx, gy, gw, gh in groups:
        rx, ry = int(gx*w), int(gy*h)
        rw2, rh2 = int(gw*w), int(gh*h)
        cv2.rectangle(display, (rx,ry), (rx+rw2,ry+rh2), (100,100,255), 1)
    
    # Draw detection boxes
    for ci, cf, fx, fy, fw, fh in last_results:
        x1 = int(fx*w); y1 = int(fy*h); x2 = int((fx+fw)*w); y2 = int((fy+fh)*h)
        cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
        lbl = f"{class_names[ci]} {cf:.2f}"
        (tw,th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(display, (x1,y1-th-4), (x1+tw+4,y1), (0,255,0), -1)
        cv2.putText(display, lbl, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    
    # Draw Detect button
    bx, by = w-150, h-50
    cv2.rectangle(display, (bx,by), (bx+140,by+40), (0,200,0), -1)
    cv2.putText(display, "DETECT", (bx+25,by+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.rectangle(display, (bx,by), (bx+140,by+40), (0,255,0), 2)
    
    # Detection count overlay
    info = f"{len(last_results)} detections" if last_results else "Press DETECT"
    cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    ts = cur_pos[0]/fps
    cv2.putText(display, f"{int(ts//60):02d}:{int(ts%60):02d}.{int((ts%1)*100):02d}",
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    
    cv2.imshow(WNAME, display)
    key = cv2.waitKey(30) & 0xFF
    
    if key in (27, ord('q'), ord('Q')):
        break
    elif key in (32, ord('d'), ord('D')):
        last_results = run_detection(last_frame, h, w, groups)
        if last_results:
            print(f"\n[{ts:.1f}s] {len(last_results)} detections:")
            for ci, cf, fx, fy, fw, fh in sorted(last_results, key=lambda x:-x[1]):
                print(f"  {class_names[ci]}: {cf:.3f} at ({fx:.3f},{fy:.3f},{fw:.3f},{fh:.3f})")
        else:
            print(f"\n[{ts:.1f}s] No detections above {CONF}")
            # Show max conf per class as diagnostic
            gray_full2 = cv2.cvtColor(cv2.resize(last_frame, (ISZ,ISZ)), cv2.COLOR_BGR2GRAY)
            eq_full2 = cv2.equalizeHist(gray_full2).astype(np.float32) / 255.0
            out = session.run(None, {inp_name: np.tile(eq_full2[np.newaxis, :, :], (3, 1, 1))[np.newaxis]})[0][0]
            for ci in range(num_classes):
                mc = out[4+ci].max()
                if mc > 0.01:
                    print(f"    {class_names[ci]}: max={mc:.4f}")

cap.release()
cv2.destroyAllWindows()
