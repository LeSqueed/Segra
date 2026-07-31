"""Run dataset export + training from command line."""
import sys, os, random, shutil, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image

game_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/training/Overwatch")
samples_dir = game_dir / "samples"
dataset_dir = game_dir / "dataset"
events_file = game_dir / "events.json"

if not samples_dir.exists():
    print(f"Samples not found at {samples_dir}")
    sys.exit(1)

# Load events
with open(events_file) as f:
    events = json.load(f)

# Build event lookup by id
event_by_id = {e["id"]: e for e in events}
sorted_events = sorted(events, key=lambda e: e["id"])
class_map = {e["id"]: i for i, e in enumerate(sorted_events)}

def overlap(r1, r2):
    """Check if two normalized regions (x,y,w,h) overlap or one contains the other."""
    x1, y1, w1, h1 = r1["x"], r1["y"], r1["w"], r1["h"]
    x2, y2, w2, h2 = r2["x"], r2["y"], r2["w"], r2["h"]
    # Check overlap
    if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
        return True
    # Check containment
    if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
        return True
    if x2 >= x1 and y2 >= y1 and x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1:
        return True
    return False

def merge_regions(r1, r2):
    """Return the union/encompassing region of two overlapping regions."""
    x = min(r1["x"], r2["x"])
    y = min(r1["y"], r2["y"])
    w = max(r1["x"] + r1["w"], r2["x"] + r2["w"]) - x
    h = max(r1["y"] + r1["h"], r2["y"] + r2["h"]) - y
    return {"x": x, "y": y, "w": w, "h": h}

def get_region(event_id):
    """Get normalized region dict from event, or None."""
    ev = event_by_id.get(event_id)
    if ev and ev.get("screenRegionW"):
        return {"x": ev["screenRegionX"], "y": ev["screenRegionY"],
                "w": ev["screenRegionW"], "h": ev["screenRegionH"]}
    return None

def group_labels_by_region(labels):
    """
    Group label strings by distinct non-overlapping regions.
    Returns list of (region_or_None, [label_strings]).
    """
    items = []
    for lbl in labels:
        parts = lbl.split()
        eid_raw = int(parts[0])
        # Find original event id by reverse lookup in class_map
        orig_id = None
        for k, v in class_map.items():
            if v == eid_raw:
                orig_id = k
                break
        r = get_region(orig_id) if orig_id else None
        items.append((r, lbl))

    # Group by region: merge overlapping regions
    groups = []
    for r, lbl in items:
        merged = False
        for g in groups:
            gr = g["region"]
            if gr is None and r is None:
                g["labels"].append(lbl)
                merged = True
                break
            elif gr is not None and r is not None and overlap(gr, r):
                g["region"] = merge_regions(gr, r)
                g["labels"].append(lbl)
                merged = True
                break
            elif gr is None and r is not None:
                # No region group absorbs a with-region label into no-region
                g["labels"].append(lbl)
                merged = True
                break
            elif gr is not None and r is None:
                # With-region group absorbs a no-region label
                g["labels"].append(lbl)
                merged = True
                break
        if not merged:
            groups.append({"region": r, "labels": [lbl]})
    return [(g["region"], g["labels"]) for g in groups]

def crop_and_resize(src_path, region, dst_path, labels_for_crop, brightness=None):
    """Load full image, grayscale with optional random brightness shift, crop + resize."""
    with Image.open(src_path) as img:
        src_w, src_h = img.size
        # Grayscale (luminance only)
        gray = img.convert("L")
        if brightness is not None:
            arr = (np.array(gray, dtype=np.float32) * brightness)
            gray = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        gray = gray.convert("RGB")

        if region:
            rx = int(region["x"] * src_w)
            ry = int(region["y"] * src_h)
            rw = max(1, int(region["w"] * src_w))
            rh = max(1, int(region["h"] * src_h))
            cropped = gray.crop((rx, ry, rx + rw, ry + rh))
        else:
            cropped = gray
            rw, rh = src_w, src_h

        out_size = 640
        cropped.resize((out_size, out_size), Image.BICUBIC).save(dst_path)

        # Adjust label coordinates relative to crop
        adjusted = []
        for lbl in labels_for_crop:
            parts = lbl.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            if region:
                new_cx = (cx * src_w - rx) / rw
                new_cy = (cy * src_h - ry) / rh
                new_bw = (bw * src_w) / rw
                new_bh = (bh * src_h) / rh
            else:
                new_cx, new_cy, new_bw, new_bh = cx, cy, bw, bh
            adjusted.append(f"{parts[0]} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")
        return adjusted

# Clean and create dataset dirs
if dataset_dir.exists():
    shutil.rmtree(dataset_dir)
for d in ["images/train", "labels/train", "images/val", "labels/val"]:
    (dataset_dir / d).mkdir(parents=True)

# Shuffle and split
pngs = list(samples_dir.glob("*.png"))
random.seed(42)
random.shuffle(pngs)
val_count = max(1, len(pngs) // 5)
val_set = set(pngs[:val_count])

train_idx = val_idx = 0
for p in pngs:
    txt = p.with_suffix(".txt")
    if not txt.exists():
        continue
    lines = open(txt).read().strip().replace(",", ".").split("\n")
    orig_labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        eid = int(parts[0])
        if eid not in class_map:
            continue
        parts[0] = str(class_map[eid])
        orig_labels.append(" ".join(parts))
    if not orig_labels:
        continue

    # Group labels by region
    region_groups = group_labels_by_region(orig_labels)

    is_val = p in val_set

    for region, labels_for_crop in region_groups:
        is_train = not is_val
        img_dir = dataset_dir / ("images/train" if is_train else "images/val")
        lbl_dir = dataset_dir / ("labels/train" if is_train else "labels/val")
        idx = train_idx if is_train else val_idx
        if is_train:
            train_idx += 1
            brightness = random.uniform(0.6, 1.4)
        else:
            val_idx += 1
            brightness = None

        adjusted = crop_and_resize(p, region, img_dir / f"{idx:06d}.png", labels_for_crop, brightness)
        open(lbl_dir / f"{idx:06d}.txt", "w").write("\n".join(adjusted))

# Write dataset.yaml
class_names = [f"'{e['name']}'" for e in sorted_events]
yaml_path = dataset_dir / "dataset.yaml"
yaml_path.write_text(
    f"train: ./images/train\nval: ./images/val\nnc: {len(class_names)}\nnames: [{', '.join(class_names)}]\n"
)

print(f"Exported {train_idx} train, {val_idx} val samples")

# Run training
script_dir = Path(__file__).parent
train_script = script_dir / "train_yolo.py"
print(f"Starting training: python {train_script} {dataset_dir}")
subprocess.run([sys.executable, str(train_script), str(dataset_dir)])
