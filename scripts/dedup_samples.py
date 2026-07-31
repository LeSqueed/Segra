"""
Dry-run: python scripts/dedup_samples.py <game_dir>
Apply:   python scripts/dedup_samples.py <game_dir> --apply

Finds duplicate sample images (same SHA256) and merges their labels into one file.
"""

import sys
import os
import hashlib
from collections import defaultdict

def main():
    dry_run = "--apply" not in sys.argv
    samples_dir = None
    for arg in sys.argv[1:]:
        if arg != "--apply":
            samples_dir = arg

    if not samples_dir:
        print("Usage: python scripts/dedup_samples.py <game_dir> [--apply]")
        print("  game_dir: path to game training folder (e.g. data/training/Overwatch)")
        sys.exit(1)

    samples_path = os.path.join(samples_dir, "samples")
    if not os.path.isdir(samples_path):
        print(f"Samples directory not found: {samples_path}")
        sys.exit(1)

    # Group PNGs by hash
    by_hash = defaultdict(list)
    for f in os.listdir(samples_path):
        if not f.lower().endswith(".png"):
            continue
        path = os.path.join(samples_path, f)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        by_hash[h].append(f.replace(".png", ""))

    dup_count = 0
    merged_count = 0
    for h, groups in by_hash.items():
        if len(groups) < 2:
            continue
        dup_count += 1
        label_files = []
        for base in groups:
            txt = os.path.join(samples_path, f"{base}.txt")
            if os.path.exists(txt):
                label_files.append((base, open(txt).read().strip()))

        print(f"\n--- Duplicate image (SHA256: {h[:12]}...) ---")
        print(f"  {len(groups)} copies, {len(label_files)} with labels:")

        all_labels = []
        for base, lbl in label_files:
            print(f"    {base}.txt -> {lbl}")
            if lbl:
                all_labels.append(lbl)

        if len(all_labels) <= 1:
            print(f"  -> Only 1 label, just delete extras")
            if not dry_run:
                kept = groups[0]
                for base in groups:
                    if base != kept:
                        for ext in [".png", ".txt"]:
                            p = os.path.join(samples_path, f"{base}{ext}")
                            if os.path.exists(p):
                                os.remove(p)
                                print(f"    Deleted {base}{ext}")
            merged_count += 1
            continue

        # Merge labels: keep first image, write all labels into it
        print(f"  -> Merge {len(all_labels)} labels into {groups[0]}.txt:")
        for lbl in all_labels:
            print(f"       {lbl}")
        if not dry_run:
            kept = groups[0]
            out_txt = os.path.join(samples_path, f"{kept}.txt")
            with open(out_txt, "w") as f:
                f.write("\n".join(all_labels) + "\n")
            print(f"    Wrote {out_txt}")
            for base in groups:
                if base != kept:
                    for ext in [".png", ".txt"]:
                        p = os.path.join(samples_path, f"{base}{ext}")
                        if os.path.exists(p):
                            os.remove(p)
                            print(f"    Deleted {base}{ext}")
        merged_count += 1

    if dry_run:
        print(f"\n[Dry run] {dup_count} duplicate groups found. Run with --apply to merge.")
    else:
        print(f"\n[Applied] Merged {merged_count} duplicate groups.")

if __name__ == "__main__":
    main()
