"""Convert MAGFiLO COCO annotations into a single-class YOLO segmentation dataset.

Each annotator record (e.g. ``010401-20160920230134Lh``) becomes its own training
sample, linked to the shared JPEG. The train/val split is grouped by physical
file name so no image leaks across splits. Validation GT is also exported in the
competition CSV format for local PQ scoring.
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
ANN = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMG_DIR = DATA / "train" / "train_images"
SIZE = 2048


def polygon_to_rle(poly):
    from pycocotools import mask as mask_util
    rles = mask_util.frPyObjects([poly], SIZE, SIZE)
    rle = mask_util.merge(rles)
    return rle["counts"].decode("ascii")


def mask_to_polygons(mask, min_area=9):
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2) for c in contours if len(c) >= 3 and cv2.contourArea(c) >= min_area]


def write_consensus_train(coco, val_files, out, min_support, full):
    """Replace per-annotator train samples with one consensus sample per physical image."""
    from multiprocessing import Pool

    from consensus import build, group_by_file

    by_file = group_by_file(coco)
    train_dir = out / "train"
    for p in list((train_dir / "images").iterdir()) + list((train_dir / "labels").iterdir()):
        p.unlink()
    items = [(f, r) for f, r in by_file.items() if full or f not in val_files]
    with Pool(16) as pool:
        results = pool.starmap(_consensus_lines, [(f, r, min_support) for f, r in items])
    for f, lines in results:
        stem = Path(f).stem
        os.symlink(IMG_DIR / f, train_dir / "images" / f"{stem}.jpeg")
        (train_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    print(f"consensus train samples: {len(results)}, instances: {sum(len(l) for _, l in results)}")


def _consensus_lines(file_name, records, min_support):
    from consensus import build

    lines = []
    for m in build(records, link_iou=0.3, min_support=min_support):
        # one filament = one polygon; keep the largest piece so fragments aren't learned as instances
        for poly in sorted(mask_to_polygons(m), key=len)[-1:]:
            p = np.clip(poly / SIZE, 0, 1)
            lines.append("0 " + " ".join(f"{v:.6f}" for v in p.ravel()))
    return file_name, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "work" / "yolo_ds"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full", action="store_true", help="put every image in train (val kept for monitoring)")
    ap.add_argument("--consensus", type=float, default=None,
                    help="train on one consensus label set per image (value = min annotator support)")
    args = ap.parse_args()

    coco = json.load(open(ANN))
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    files = sorted({im["file_name"] for im in coco["images"]})
    rng = random.Random(args.seed)
    rng.shuffle(files)
    n_val = int(len(files) * args.val_frac)
    val_files = set(files[:n_val])

    out = Path(args.out)
    gt_rows = []
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    for im in coco["images"]:
        is_val = im["file_name"] in val_files
        splits = ["val"] if is_val else ["train"]
        if args.full and is_val:
            splits.append("train")
        stem = im["id"]  # unique per annotator record
        lines = []
        for a in anns[im["id"]]:
            poly = np.asarray(a["segmentation"][0], dtype=np.float64).reshape(-1, 2)
            if len(poly) < 3:
                continue
            p = np.clip(poly / SIZE, 0, 1)
            lines.append("0 " + " ".join(f"{v:.6f}" for v in p.ravel()))
            if is_val:
                gt_rows.append((f"{stem}_{a['id']}", polygon_to_rle(a["segmentation"][0])))
        for split in splits:
            link = out / split / "images" / f"{stem}.jpeg"
            if not link.exists():
                os.symlink(IMG_DIR / im["file_name"], link)
            (out / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    if args.consensus is not None:
        write_consensus_train(coco, val_files, out, args.consensus, args.full)

    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: train/images\nval: val/images\nnames:\n  0: filament\n"
    )
    # GT CSV: filament_id = "<annotator>-<file stem>_<ann id>" as in the official evaluator
    pd.DataFrame(gt_rows, columns=["filament_id", "segmentation_rle"]).to_csv(out / "val_gt.csv", index=False)
    print(f"files: {len(files)} (val {len(val_files)}), gt rows {len(gt_rows)}, out {out}")


if __name__ == "__main__":
    main()
