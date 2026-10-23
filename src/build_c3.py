"""Precompute the 3-channel inputs (src/channels.py) for all train and test images and
build a YOLO dataset that mirrors work/yolo_ds exactly (same split, same per-annotator
label files), with images replaced by the 3-channel PNGs (lossless).
"""
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2

from channels import make_channels

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
W = ROOT / "work"
OUT_IMG = W / "img_c3"


def convert(path):
    dst = OUT_IMG / (path.stem + ".png")
    if not dst.exists():
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        im, disk = make_channels(gray)
        cv2.imwrite(str(dst), im, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        disk = None
    return path.stem, disk


def main():
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    paths = sorted((DATA / "train" / "train_images").glob("*.jpeg")) + sorted((DATA / "test" / "test_images").glob("*.jpeg"))
    with Pool(16) as p:
        disks = dict(p.map(convert, paths))
    old = json.load(open(OUT_IMG / "disks.json")) if (OUT_IMG / "disks.json").exists() else {}
    old.update({k: v for k, v in disks.items() if v is not None})
    json.dump(old, open(OUT_IMG / "disks.json", "w"))

    src, dst = W / "yolo_ds", W / "yolo_ds_c3"
    for split in ("train", "val"):
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for f in (src / split / "images").iterdir():
            stem = f.stem.split("-", 1)[-1]
            link = dst / split / "images" / (f.stem + ".png")
            if not link.exists():
                os.symlink(OUT_IMG / f"{stem}.png", link)
            lab = src / split / "labels" / (f.stem + ".txt")
            (dst / split / "labels" / lab.name).write_text(lab.read_text())
    (dst / "data.yaml").write_text(f"path: {dst}\ntrain: train/images\nval: val/images\nnames:\n  0: filament\n")
    test_dir = OUT_IMG / "test"
    test_dir.mkdir(exist_ok=True)
    for p in (DATA / "test" / "test_images").glob("*.jpeg"):
        link = test_dir / (p.stem + ".png")
        if not link.exists():
            os.symlink(OUT_IMG / f"{p.stem}.png", link)
    print(f"converted {len(paths)} images; train {len(list((dst / 'train' / 'images').iterdir()))}, "
          f"val {len(list((dst / 'val' / 'images').iterdir()))}, test links {len(list(test_dir.iterdir()))}")


if __name__ == "__main__":
    main()
