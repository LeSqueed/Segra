"""Compare: does the same region crop from a sample image work, but from a video frame fail?"""
import onnxruntime as ort, cv2, numpy as np, json
from pathlib import Path

MODEL = 'bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/model.onnx'
EVENTS = 'bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/events.json'
SAMPLES = Path('bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/samples')
ISZ = 640

with open(EVENTS) as f:
    events = json.load(f)
sorted_events = sorted(events, key=lambda e: e['id'])
class_names = [e['name'] for e in sorted_events]

session = ort.InferenceSession(MODEL)
inp_name = session.get_inputs()[0].name

# Build region groups same as C#
regions = []
for e in sorted_events:
    rw = e.get('screenRegionW')
    if rw and rw > 0:
        regions.append({'x': e.get('screenRegionX',0), 'y': e.get('screenRegionY',0), 'w': rw, 'h': e.get('screenRegionH',0)})
    else:
        regions.append(None)

groups, has_full = [], False
for r in regions:
    if r is None: has_full = True
    else:
        merged = False
        for gi, g in enumerate(groups):
            if (g['x'] < r['x']+r['w'] and g['x']+g['w'] > r['x'] and
                g['y'] < r['y']+r['h'] and g['y']+g['h'] > r['y']):
                x = min(g['x'], r['x']); y = min(g['y'], r['y'])
                groups[gi] = {'x':x,'y':y,'w':max(g['x']+g['w'],r['x']+r['w'])-x,'h':max(g['y']+g['h'],r['y']+r['h'])-y}
                merged = True; break
        if not merged: groups.append(dict(r))
if has_full or not groups: groups.append({'x':0,'y':0,'w':1,'h':1})

def infer_on_crop(img_bgr, crop_x, crop_y, crop_w, crop_h):
    """Match C# BgraToGray + ResizeGray + RunInferenceOnGray exactly."""
    h, w = img_bgr.shape[:2]
    rx, ry = int(crop_x*w), int(crop_y*h)
    rw_c = max(1, int(crop_w*w)); rh_c = max(1, int(crop_h*h))
    if rx+rw_c > w: rw_c = w-rx
    if ry+rh_c > h: rh_c = h-ry
    # BgraToGray: BT.601 on BGRA (B=0, G=1, R=2)
    b, g, r = img_bgr[:,:,0], img_bgr[:,:,1], img_bgr[:,:,2]
    gray = (0.299*r + 0.587*g + 0.114*b).astype(np.uint8)
    crop = gray[ry:ry+rh_c, rx:rx+rw_c]
    rz = cv2.resize(crop, (ISZ, ISZ), interpolation=cv2.INTER_LINEAR)
    blob = np.tile((rz.astype(np.float32)/255.0)[np.newaxis,:,:], (3,1,1))[np.newaxis]
    out = session.run(None, {inp_name: blob})[0][0]
    scores = out[4:, :]
    mc = float(scores.max())
    ci = int(scores.argmax() // scores.shape[1])
    return mc, ci, rz

# Test 1: on a SAMPLE image (exact same pipeline as C# would use on OBS frame)
print("=== Test on SAMPLE images (full res, C# grayscale pipeline) ===")
best_sample = 0
for sp in sorted(SAMPLES.glob('*.png'))[:10]:
    img = cv2.imread(str(sp))
    if img is None: continue
    for gi, g in enumerate(groups):
        mc, ci, rz = infer_on_crop(img, g['x'], g['y'], g['w'], g['h'])
        best_sample = max(best_sample, mc)
        if mc > 0.1:
            print(f'  {sp.name} group {gi}: {class_names[ci]} conf={mc:.4f}')
print(f'Best on samples: {best_sample:.4f}')

# Test 2: on the DATASET images (already correctly cropped+resized)
print("\n=== Test on DATASET images (already 640x640) ===")
best_ds = 0
for ip in sorted((Path('bin/Release/net10.0-windows10.0.19041.0/data/training/Overwatch/dataset') / 'images/train').glob('*.png'))[:10]:
    img = cv2.imread(str(ip))
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blob = np.tile((gray.astype(np.float32)/255.0)[np.newaxis,:,:], (3,1,1))[np.newaxis]
    out = session.run(None, {inp_name: blob})[0][0]
    mc = float(out[4:].max())
    best_ds = max(best_ds, mc)
print(f'Best on dataset: {best_ds:.4f}')

# Test 3: What's different? Save a sample crop and a dataset crop for comparison
print("\n=== Comparing EXACTLY how the sample crop differs from dataset crop ===")
gi = 0  # Compare first group (Elimination area)
g = groups[gi]
sp = sorted(SAMPLES.glob('*.png'))[0]
img = cv2.imread(str(sp))
# C# pipeline
mc, ci, rz_cs = infer_on_crop(img, g['x'], g['y'], g['w'], g['h'])

# PIL pipeline (what export_and_train.py uses)
from PIL import Image
pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
h2, w2 = img.shape[:2]
rx, ry = int(g['x']*w2), int(g['y']*h2)
rw_c = max(1, int(g['w']*w2)); rh_c = max(1, int(g['h']*h2))
if rx+rw_c > w2: rw_c = w2-rx
if ry+rh_c > h2: rh_c = h2-ry
pil_gray = pil_img.convert('L')
pil_crop = pil_gray.crop((rx, ry, rx+rw_c, ry+rh_c))
pil_rz = np.array(pil_crop.resize((ISZ, ISZ), Image.BICUBIC), dtype=np.float32)/255.0
blob_pil = np.tile(pil_rz[np.newaxis,:,:], (3,1,1))[np.newaxis]
out_pil = session.run(None, {inp_name: blob_pil})[0][0]
mc_pil = float(out_pil[4:].max())
ci_pil = int(out_pil[4:].argmax() // out_pil.shape[2])

print(f'Sample image {sp.name}:')
print(f'  C# pipeline:   conf={mc:.4f} class={class_names[ci]}')
print(f'  PIL pipeline: conf={mc_pil:.4f} class={class_names[ci_pil]}')

# Check pixel differences between C# and PIL grayscale
from PIL import Image as PILImg
# C# grayscale
b, g_, r_ = img[:,:,0], img[:,:,1], img[:,:,2]
gray_cs = (0.299*r_ + 0.587*g_ + 0.114*b_).astype(np.uint8)
# PIL grayscale
gray_pil = np.array(pil_img.convert('L'))
diff = np.abs(gray_cs.astype(np.int16) - gray_pil.astype(np.int16))
print(f'  Max pixel diff between C# and PIL grayscale: {diff.max()} (mean={diff.mean():.2f})')
