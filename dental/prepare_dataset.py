#!/usr/bin/env python3
"""
Prepare combined DENTEX + OralXrays-9 dataset in YOLO format.

Classes:
  0  caries
  1  deep_caries
  2  periapical_lesion
  3  impacted_tooth

Usage:
    python3 prepare_dataset.py
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict, Counter

# ── Paths ─────────────────────────────────────────────────────────────────────

DENTEX_TRAIN_JSON = Path("/tmp/dentex_extracted/train/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json")
DENTEX_TRAIN_IMGS = Path("/tmp/dentex_extracted/train/training_data/quadrant-enumeration-disease/xrays")
DENTEX_VAL_JSON   = Path("/tmp/dentex/full/DENTEX/validation_triple.json")
DENTEX_VAL_IMGS   = Path("/tmp/dentex_extracted/val/validation_data/quadrant_enumeration_disease/xrays")

ORAL_TRAIN_IMGS   = Path("/tmp/oralxrays9/train2017")
ORAL_VAL_IMGS     = Path("/tmp/oralxrays9/val2017")
ORAL_TRAIN_ANN    = Path("/tmp/oralxrays9/annotations/instances_train2017.json")
ORAL_VAL_ANN      = Path("/tmp/oralxrays9/annotations/instances_val2017.json")

OUT_DIR           = Path("/tmp/dental_yolo")

# ── Class mappings ─────────────────────────────────────────────────────────────

CLASSES = ["caries", "periapical_lesion", "impacted_tooth"]

# DENTEX category_id_3: 0=Impacted, 1=Caries, 2=Periapical Lesion, 3=Deep Caries (merged into caries)
DENTEX_MAP = {0: 2, 1: 0, 2: 1, 3: 0}

# OralXrays-9 category_id → our class id (None = skip)
ORAL_MAP = {
    1: 1,    # Apical Periodontitis → periapical_lesion
    2: 0,    # Decay               → caries
    3: 2,    # Wisdom Tooth        → impacted_tooth
    # 4-9: restorations / non-pathology — skip
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def bbox_to_yolo(bbox_xywh, img_w, img_h):
    """COCO [x,y,w,h] → YOLO [cx,cy,w,h] normalised."""
    x, y, w, h = bbox_xywh
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def write_labels(label_path, rows):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for cls, cx, cy, w, h in rows:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def copy_image(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ── DENTEX extraction ─────────────────────────────────────────────────────────

def extract_dentex_zip(zip_path, extract_to):
    import zipfile
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    print(f"  → {extract_to}")


def find_dentex_images(base_dir):
    """Return dict filename_stem → Path for all jpegs under base_dir."""
    result = {}
    for p in Path(base_dir).rglob("*.jpg"):
        result[p.stem] = p
    for p in Path(base_dir).rglob("*.png"):
        result[p.stem] = p
    return result


def process_dentex_split(ann_path, img_lookup, out_img_dir, out_lbl_dir, split_name):
    ann = json.load(open(ann_path))
    img_map = {img["id"]: img for img in ann["images"]}

    # Group annotations by image
    by_image = defaultdict(list)
    for a in ann["annotations"]:
        cls_id = DENTEX_MAP.get(a.get("category_id_3"))
        if cls_id is None:
            continue
        by_image[a["image_id"]].append((cls_id, a["bbox"]))

    count_imgs = count_boxes = 0
    for iid, rows in by_image.items():
        img_info = img_map[iid]
        fname    = Path(img_info["file_name"]).name
        stem     = Path(fname).stem
        img_w    = img_info["width"]
        img_h    = img_info["height"]

        src = img_lookup.get(stem)
        if src is None:
            print(f"  [warn] image not found: {fname}")
            continue

        yolo_rows = []
        for cls_id, bbox in rows:
            cx, cy, w, h = bbox_to_yolo(bbox, img_w, img_h)
            yolo_rows.append((cls_id, cx, cy, w, h))

        copy_image(src, out_img_dir / fname)
        write_labels(out_lbl_dir / (stem + ".txt"), yolo_rows)
        count_imgs += 1
        count_boxes += len(yolo_rows)

    print(f"  DENTEX {split_name}: {count_imgs} images, {count_boxes} boxes")
    return count_imgs, count_boxes


# ── OralXrays-9 processing ────────────────────────────────────────────────────

def process_oral_split(ann_path, img_dir, out_img_dir, out_lbl_dir, split_name):
    ann     = json.load(open(ann_path))
    img_map = {img["id"]: img for img in ann["images"]}

    by_image = defaultdict(list)
    for a in ann["annotations"]:
        cls_id = ORAL_MAP.get(a.get("category_id"))
        if cls_id is None:
            continue
        by_image[a["image_id"]].append((cls_id, a["bbox"]))

    # Only process images that have at least one mapped annotation
    count_imgs = count_boxes = 0
    for iid, rows in by_image.items():
        img_info = img_map[iid]
        fname    = img_info["file_name"]
        img_w    = img_info["width"]
        img_h    = img_info["height"]

        src = img_dir / Path(fname).name
        if not src.exists():
            print(f"  [warn] image not found: {fname}")
            continue

        stem      = Path(fname).stem
        yolo_rows = []
        for cls_id, bbox in rows:
            cx, cy, w, h = bbox_to_yolo(bbox, img_w, img_h)
            yolo_rows.append((cls_id, cx, cy, w, h))

        # Prefix with "oral_" to avoid filename collisions with DENTEX
        out_fname = f"oral_{Path(fname).name}"
        copy_image(src, out_img_dir / out_fname)
        write_labels(out_lbl_dir / f"oral_{stem}.txt", yolo_rows)
        count_imgs += 1
        count_boxes += len(yolo_rows)

    print(f"  OralXrays-9 {split_name}: {count_imgs} images, {count_boxes} boxes")
    return count_imgs, count_boxes


# ── Dataset YAML ──────────────────────────────────────────────────────────────

def write_yaml(out_dir, classes):
    yaml_path = out_dir / "dental.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n")
        f.write(f"\nnc: {len(classes)}\n")
        f.write(f"names: {classes}\n")
    print(f"\nYAML: {yaml_path}")
    return yaml_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dentex_train_imgs = find_dentex_images(DENTEX_TRAIN_IMGS)
    dentex_val_imgs   = find_dentex_images(DENTEX_VAL_IMGS)

    total_train_imgs = total_val_imgs = 0
    total_train_boxes = total_val_boxes = 0

    # Train: DENTEX + OralXrays-9
    ti, tb = process_dentex_split(
        DENTEX_TRAIN_JSON, dentex_train_imgs,
        OUT_DIR / "images/train", OUT_DIR / "labels/train", "train")
    total_train_imgs += ti; total_train_boxes += tb

    ti, tb = process_oral_split(
        ORAL_TRAIN_ANN, ORAL_TRAIN_IMGS,
        OUT_DIR / "images/train", OUT_DIR / "labels/train", "train")
    total_train_imgs += ti; total_train_boxes += tb

    # Val: DENTEX only (our benchmark)
    vi, vb = process_dentex_split(
        DENTEX_VAL_JSON, dentex_val_imgs,
        OUT_DIR / "images/val", OUT_DIR / "labels/val", "val")
    total_val_imgs += vi; total_val_boxes += vb

    write_yaml(OUT_DIR, CLASSES)

    print(f"\n{'='*50}")
    print(f"Train: {total_train_imgs} images, {total_train_boxes} boxes")
    print(f"Val:   {total_val_imgs} images, {total_val_boxes} boxes")
    print(f"Output: {OUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
