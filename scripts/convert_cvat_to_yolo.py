#!/usr/bin/env python3
"""
Convert CVAT XML annotations to YOLO format with train/val/test split.

Dataset structure produced under the dataset root:
  images/train|val|test
  labels/train|val|test
  data.yaml (written if --write-yaml)
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, List


def parse_annotations(xml_path: Path) -> List[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    items = []
    for image_elem in root.findall(".//image"):
        name = image_elem.get("name")
        width = int(image_elem.get("width"))
        height = int(image_elem.get("height"))
        box = image_elem.find("box")
        if box is None:
            continue
        xtl = float(box.get("xtl"))
        ytl = float(box.get("ytl"))
        xbr = float(box.get("xbr"))
        ybr = float(box.get("ybr"))
        plate_text = ""
        for attr_elem in box.findall("attribute"):
            if attr_elem.get("name") == "plate number":
                plate_text = attr_elem.text or ""
                break
        items.append(
            {
                "name": name,
                "width": width,
                "height": height,
                "bbox": (xtl, ytl, xbr, ybr),
                "text": plate_text,
            }
        )
    return items


def xyxy_to_yolo(bbox: Tuple[float, float, float, float], w: int, h: int) -> Tuple[float, float, float, float]:
    xtl, ytl, xbr, ybr = bbox
    cx = (xtl + xbr) / 2.0 / w
    cy = (ytl + ybr) / 2.0 / h
    bw = (xbr - xtl) / w
    bh = (ybr - ytl) / h
    return cx, cy, bw, bh


def ensure_dirs(base: Path):
    for split in ["train", "val", "test"]:
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yolo_file(label_path: Path, bbox_yolo: Tuple[float, float, float, float]):
    cx, cy, bw, bh = bbox_yolo
    label_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def split_items(items: List[dict], test_ratio=0.3, val_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(items)
    n = len(items)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_items = items[:n_test]
    val_items = items[n_test : n_test + n_val]
    train_items = items[n_test + n_val :]
    return train_items, val_items, test_items


def main():
    parser = argparse.ArgumentParser(description="Convert CVAT XML to YOLO with splits")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Root of CVAT dataset (photos/, annotations.xml)")
    parser.add_argument("--out", type=Path, default=None, help="Output root (default: dataset root)")
    parser.add_argument("--test-ratio", type=float, default=0.3, help="Test split ratio (default 0.3)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio (default 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--write-yaml", action="store_true", help="Also write data.yaml for Ultralytics")
    args = parser.parse_args()

    root = args.dataset_path
    if root is None:
        import kagglehub
        root = Path(kagglehub.dataset_download("piotrstefaskiue/poland-vehicle-license-plate-dataset"))
    photos_dir = root / "photos"
    xml_path = root / "annotations.xml"
    if not photos_dir.exists() or not xml_path.exists():
        print("Invalid dataset structure. Expect photos/ and annotations.xml")
        return 1

    out_root = args.out or root
    ensure_dirs(out_root)

    items = parse_annotations(xml_path)
    train_items, val_items, test_items = split_items(items, test_ratio=args.test_ratio, val_ratio=args.val_ratio, seed=args.seed)

    def process_split(split_items, split_name):
        for item in split_items:
            img_src = photos_dir / item["name"]
            if not img_src.exists():
                continue
            img_dst = out_root / "images" / split_name / item["name"]
            shutil.copyfile(img_src, img_dst)
            bbox_yolo = xyxy_to_yolo(item["bbox"], item["width"], item["height"])
            label_dst = out_root / "labels" / split_name / f"{img_dst.stem}.txt"
            write_yolo_file(label_dst, bbox_yolo)

    process_split(train_items, "train")
    process_split(val_items, "val")
    process_split(test_items, "test")

    print(f"Done. Train/Val/Test sizes: {len(train_items)}/{len(val_items)}/{len(test_items)}")
    print(f"Output: {out_root}")

    if args.write_yaml:
        yaml_path = out_root / "data.yaml"
        yaml_path.write_text(
            f"path: {out_root}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n"
            f"nc: 1\n"
            f"names: ['plate']\n"
        )
        print(f"Wrote {yaml_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
