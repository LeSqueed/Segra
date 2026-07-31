import onnxruntime as ort, cv2, numpy as np, json
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

def test_set(imgs, label_dir, name):
    correct = total = 0
    best = 0
    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blob = (gray.astype(np.float32) / 255.0)[np.newaxis, :, :]
        blob = np.tile(blob, (3, 1, 1))[np.newaxis]
        out = session.run(None, {inp_name: blob})[0][0]
        scores = out[4:, :]
        mc = float(scores.max())
        best = max(best, mc)

        lp = label_dir / (ip.stem + '.txt')
        if lp.exists():
            for line in lp.read_text().strip().split('\n'):
                parts = line.split()
                if len(parts) >= 5:
                    gt_class = int(parts[0])
                    total += 1
                    pred_class = int(scores.argmax() // scores.shape[1])
                    if pred_class == gt_class:
                        correct += 1

    print(f'{name}: {len(imgs)} images, best_conf={best:.4f}, accuracy={correct}/{total} ({correct/total*100:.1f}%)')
    return best

b1 = test_set(sorted((DS / 'images/train').glob('*.png')), DS / 'labels/train', 'Train')
b2 = test_set(sorted((DS / 'images/val').glob('*.png')), DS / 'labels/val', 'Val')
print(f'Best overall: {max(b1,b2):.4f}')
