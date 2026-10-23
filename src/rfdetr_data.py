"""Build a tiled COCO-segmentation dataset for RF-DETR-Seg from MAGFiLO.

Each 2048 x 2048 image is cut into a 2 x 2 grid of overlapping TILE x TILE crops
(origins 0 and 2048 - TILE). Every annotator record becomes its own set of tile
images (same split as prepare_data.py: grouped by physical file), and each
filament is clipped to the tile and re-polygonised. Tiles are written once per
physical file; record-specific image entries point at symlinks so the loader sees
distinct images.
"""
import argparse
import json
import os
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
ANN = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMG_DIR = DATA / "train" / "train_images"
SIZE = 2048
TILE = 1100
ORIGINS = [(x, y) for y in (0, SIZE - TILE) for x in (0, SIZE - TILE)]
MIN_PIECE = 30  # px; drop slivers created by clipping at tile borders


def split_files(coco, val_frac=0.15, seed=42):
    files = sorted({im["file_name"] for im in coco["images"]})
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_frac)
    return set(files[n_val:]), set(files[:n_val])


def write_tiles(file_name, out_dir):
    img = cv2.imread(str(IMG_DIR / file_name), cv2.IMREAD_GRAYSCALE)
    stem = Path(file_name).stem
    for t, (x, y) in enumerate(ORIGINS):
        p = out_dir / f"{stem}_t{t}.jpg"
        if not p.exists():
            cv2.imwrite(str(p), img[y:y + TILE, x:x + TILE], [cv2.IMWRITE_JPEG_QUALITY, 97])


def tile_annotations(polys):
    """-> {tile index: [(polygon flat list, bbox, area), ...]}"""
    out = defaultdict(list)
    for poly in polys:
        m = np.zeros((SIZE, SIZE), np.uint8)
        cv2.fillPoly(m, [np.round(np.asarray(poly).reshape(-1, 2)).astype(np.int32)], 1)
        for t, (x, y) in enumerate(ORIGINS):
            crop = m[y:y + TILE, x:x + TILE]
            if crop.sum() < MIN_PIECE:
                continue
            cs, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cs:
                if cv2.contourArea(c) < MIN_PIECE or len(c) < 3:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                out[t].append((c.reshape(-1).astype(float).tolist(), [bx, by, bw, bh], float(cv2.contourArea(c))))
    return t, out


def _job(args):
    rec_id, polys = args
    return rec_id, tile_annotations(polys)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "work" / "rfdetr_ds"))
    ap.add_argument("--full", action="store_true", help="put validation files in train too")
    args = ap.parse_args()

    coco = json.load(open(ANN))
    train_files, val_files = split_files(coco)
    if args.full:
        train_files = train_files | val_files
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a["segmentation"][0])
    out = Path(args.out)
    tiles = out / "_tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    with Pool(12) as p:
        p.starmap(write_tiles, [(f, tiles) for f in sorted(train_files | val_files)])
        tiled = dict(p.map(_job, [(im["id"], anns[im["id"]]) for im in coco["images"]]))

    splits = {"train": train_files, "valid": val_files, "test": val_files}
    for split, files in splits.items():
        d = out / split
        d.mkdir(exist_ok=True)
        images, annotations = [], []
        for im in coco["images"]:
            if im["file_name"] not in files:
                continue
            stem = Path(im["file_name"]).stem
            for t in range(len(ORIGINS)):
                name = f"{im['id']}_t{t}.jpg"
                link = d / name
                if not link.exists():
                    os.symlink(tiles / f"{stem}_t{t}.jpg", link)
                iid = len(images)
                images.append({"id": iid, "file_name": name, "width": TILE, "height": TILE})
                for seg, bbox, area in tiled[im["id"]].get(t, []):
                    annotations.append({"id": len(annotations), "image_id": iid, "category_id": 0,
                                        "segmentation": [seg], "bbox": bbox, "area": area, "iscrowd": 0})
        json.dump({"images": images, "annotations": annotations,
                   "categories": [{"id": 0, "name": "filament", "supercategory": "filament"}]},
                  open(d / "_annotations.coco.json", "w"))
        print(f"{split}: {len(images)} tile images, {len(annotations)} instances")


if __name__ == "__main__":
    main()
