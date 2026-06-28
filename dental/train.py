#!/usr/bin/env python3
"""
Train YOLO11-M on combined DENTEX + OralXrays-9 dental dataset.

Usage:
    python3 train.py
"""

from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # 3 classes: caries, periapical_lesion, impacted_tooth

model.train(
    data="/tmp/dental_yolo/dental.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    lr0=1e-3,
    lrf=0.01,
    warmup_epochs=3,
    patience=30,
    workers=8,
    device=0,
    project="/tmp/dental_runs",
    name="yolo11m-dental-v1",
    exist_ok=True,
    # Augmentation
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    degrees=5.0,
    translate=0.1,
    scale=0.3,
    flipud=0.0,     # X-rays are always upright
    fliplr=0.5,
    hsv_h=0.0,      # X-rays are greyscale
    hsv_s=0.0,
    hsv_v=0.3,
    # Logging
    plots=True,
    save=True,
    save_period=20,
    val=True,
)
