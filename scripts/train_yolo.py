#!/usr/bin/env python3
"""
Train a YOLO nano model on a dataset exported by Segra's TrainingEventService.

Usage:
    python train_yolo.py <dataset_dir>

The dataset_dir must contain:
    dataset.yaml       # YOLO dataset config
    images/train/      # training images
    labels/train/      # YOLO-format label files

Output:
    <dataset_dir>/model.onnx  # Trained ONNX model
"""

import sys
import os
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_yolo.py <dataset_dir>")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1]).resolve()
    yaml_path = dataset_dir / "dataset.yaml"

    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found.")
        print("Export the dataset from the Segra Training tab first.")
        sys.exit(1)

    # Read class count from dataset.yaml
    with open(yaml_path) as f:
        content = f.read()

    import yaml
    cfg = yaml.safe_load(content)
    nc = cfg.get("nc", 0)
    names = cfg.get("names", [])
    print(f"Dataset: {nc} classes - {names}")

    # Count samples
    img_dir = dataset_dir / "images" / "train"
    n_images = len(list(img_dir.glob("*.[pP][nN][gG]")))
    print(f"Training samples: {n_images}")
    if n_images < 10:
        print("Warning: fewer than 10 samples per class may give poor results.")

    # Detect GPU — supports NVIDIA CUDA; AMD users can use Ultralytics HUB for cloud GPU
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
    else:
        print("No CUDA GPU found — training on CPU. For faster training with AMD GPUs,")
        print("use Ultralytics HUB (free cloud GPU): https://hub.ultralytics.com")
    print(f"Training device: {device}")

    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    batch_size = 32 if device != "cpu" else 16

    results = model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=batch_size,
        patience=20,
        device=device,
        workers=0,
        verbose=True,
    )

    model.export(format="onnx", imgsz=640)

    # Ultralytics saves ONNX in runs/detect/train-N/weights/best.onnx
    import shutil, glob
    onnx_path = dataset_dir / "model.onnx"
    runs_dirs = sorted(Path("runs/detect").glob("train-*"))
    if runs_dirs:
        latest = runs_dirs[-1]
        candidate = latest / "weights" / "best.onnx"
        if candidate.exists():
            shutil.copy2(str(candidate), str(onnx_path))

    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"\nTraining complete! Model saved to: {onnx_path} ({size_mb:.1f} MB)")
        print("Load the model in Segra from the Training tab.")
    else:
        print("Error: ONNX model was not created.")
        sys.exit(1)


if __name__ == "__main__":
    main()
