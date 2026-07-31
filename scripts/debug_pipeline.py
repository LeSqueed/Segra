"""Reproduce the C# preprocessing pipeline step by step on a video frame."""
import onnxruntime as ort, cv2, numpy as np, json, sys
from pathlib import Path

MODEL = 'bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/model.onnx'
EVENTS = 'bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/events.json'
DS = Path('bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/dataset')

with open(EVENTS) as f:
    events = json.load(f)
sorted_events = sorted(events, key=lambda e: e['id'])
class_names = [e['name'] for e in sorted_events]

session = ort.InferenceSession(MODEL)
inp_name = session.get_inputs()[0].name

# 1. Test on a DATASET image (known good)
ds_img = cv2.imread(str(sorted((DS / 'images/train').glob('*.png'))[10]))
h, w = ds_img.shape[:2]
gray_ds = cv2.cvtColor(ds_img, cv2.COLOR_BGR2GRAY)
blob_ds = np.tile((gray_ds.astype(np.float32)/255.0)[np.newaxis,:,:], (3,1,1))[np.newaxis]
out_ds = session.run(None, {inp_name: blob_ds})[0][0]
print(f'Dataset img ({w}x{h}): best_conf={out_ds[4:].max():.4f}')

# 2. Test a video frame with the SAME processing
VIDEO = sys.argv[1]
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 480)
ret, frame = cap.read()
cap.release()
fh, fw = frame.shape[:2]
print(f'Video frame: {fw}x{fh}')

# Build region groups (same as C#)
regions = []
for e in sorted_events:
    rw = e.get('screenRegionW')
    if rw and rw > 0:
        regions.append((e.get('screenRegionX',0), e.get('screenRegionY',0), rw, e.get('screenRegionH',0)))
    else:
        regions.append(None)
groups, has_full = [], False
for r in regions:
    if r is None: has_full = True
    else:
        merged = False
        for gi, g in enumerate(groups):
            if (g[0] < r[0]+r[2] and g[0]+g[2] > r[0] and g[1] < r[1]+r[3] and g[1]+g[3] > r[1]):
                gx = min(g[0], r[0]); gy = min(g[1], r[1])
                groups[gi] = (gx, gy, max(g[0]+g[2], r[0]+r[2])-gx, max(g[1]+g[3], r[1]+r[3])-gy)
                merged = True; break
        if not merged: groups.append(r)
if has_full or not groups: groups.append((0,0,1,1))

# 2a. C# pipeline: BgraToGray (BT.601) on FULL frame, then ResizeGray (crop + bilinear)
gray_cs = np.dot(frame[:,:,:3][...,:3], [0.114, 0.587, 0.299]).astype(np.uint8)  # BGRA order: B=0,G=1,R=2
print(f'\nC# pipeline (full-frame grayscale, then crop+resize):')
for gi, (gx, gy, gw, gh) in enumerate(groups):
    rx, ry = int(gx*fw), int(gy*fh)
    rw_c = max(1, int(gw*fw)); rh_c = max(1, int(gh*fh))
    if rx+rw_c > fw: rw_c = fw-rx
    if ry+rh_c > fh: rh_c = fh-ry
    crop = gray_cs[ry:ry+rh_c, rx:rx+rw_c]
    rz = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LINEAR)
    blob = np.tile((rz.astype(np.float32)/255.0)[np.newaxis,:,:], (3,1,1))[np.newaxis]
    out = session.run(None, {inp_name: blob})[0][0]
    scores = out[4:, :]
    mc = float(scores.max())
    ci = int(scores.argmax() // scores.shape[1])
    print(f'  Group {gi}: conf={mc:.4f} class={class_names[ci]}')

# 2b. PIL pipeline (what the training data used)
from PIL import Image
pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
print(f'\nPIL pipeline (PIL grayscale on full frame, then crop+resize):')
for gi, (gx, gy, gw, gh) in enumerate(groups):
    rx, ry = int(gx*fw), int(gy*fh)
    rw_c = max(1, int(gw*fw)); rh_c = max(1, int(gh*fh))
    if rx+rw_c > fw: rw_c = fw-rx
    if ry+rh_c > fh: rh_c = fh-ry
    # PIL grayscale on full frame first
    pil_gray = pil_img.convert('L')
    crop = pil_gray.crop((rx, ry, rx+rw_c, ry+rh_c))
    rz = np.array(crop.resize((640, 640), Image.BICUBIC), dtype=np.float32) / 255.0
    blob = np.tile(rz[np.newaxis, :, :], (3,1,1))[np.newaxis]
    out = session.run(None, {inp_name: blob})[0][0]
    scores = out[4:, :]
    mc = float(scores.max())
    ci = int(scores.argmax() // scores.shape[1])
    print(f'  Group {gi}: conf={mc:.4f} class={class_names[ci]}')

# 2c. Same as C# but save the resized crop to compare visually
print(f'\nSaving cropped+resized frame to debug_output.png for visual inspection...')
for gi, (gx, gy, gw, gh) in enumerate(groups):
    rx, ry = int(gx*fw), int(gy*fh)
    rw_c = max(1, int(gw*fw)); rh_c = max(1, int(gh*fh))
    if rx+rw_c > fw: rw_c = fw-rx
    if ry+rh_c > fh: rh_c = fh-ry
    crop = gray_cs[ry:ry+rh_c, rx:rx+rw_c]
    rz = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f'debug_crop_group{gi}.png', rz)
    print(f'  Saved debug_crop_group{gi}.png')
