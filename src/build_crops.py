"""Crop dataset for the crop-detector pilot (1024 px crops of the 3-channel PNGs, trained at
imgsz 2048 = 2x zoom), same file-grouped split as prepare_data.py.

Per annotator record: one crop centred on a random GT filament of the record (+-150 px
jitter), one random on-disk crop, and one extra crop centred on a LOW-CONTRAST filament
when the record has one (oversampling). Low contrast = bottom tercile of the raw ring
contrast ((ring - inside) / ring) over all training-split GT. Labels are the record's
filaments clipped to the crop (pieces < 30 px dropped). Validation crops are built the
same way but are only used for training-time monitoring; the pilot is gated on full
validation images with tiled inference (src/analysis/e3_crop_pilot.py).
"""
import json
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
ANN = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
W = ROOT / "work"
OUT = W / "crops_c3"
SIZE, CROP, MIN_PIECE = 2048, 1024, 30
DISKS = json.load(open(W / "img_c3" / "disks.json"))


def split_files(coco, val_frac=0.15, seed=42):
    files = sorted({im["file_name"] for im in coco["images"]})
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_frac)
    return set(files[n_val:]), set(files[:n_val])


def ring_contrast(item):
    stem, polys = item
    g = cv2.imread(str(DATA / "train" / "train_images" / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    out = []
    for p in polys:
        m = np.zeros((SIZE, SIZE), np.uint8)
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
        ring = cv2.dilate(m, np.ones((15, 15), np.uint8)).astype(bool) & ~m.astype(bool)
        inside = g[m.astype(bool)].mean() if m.any() else 0
        around = g[ring].mean() if ring.any() else inside
        out.append((around - inside) / max(around, 1))
    return stem, out


def crop_labels(polys, x0, y0):
    lines = []
    for p in polys:
        m = np.zeros((CROP, CROP), np.uint8)
        cv2.fillPoly(m, [np.round(p - (x0, y0)).astype(np.int32)], 1)
        if m.sum() < MIN_PIECE:
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if len(c) >= 3 and cv2.contourArea(c) >= MIN_PIECE:
                q = np.clip(c.reshape(-1, 2) / CROP, 0, 1)
                lines.append("0 " + " ".join(f"{v:.6f}" for v in q.ravel()))
    return lines


def make(item):
    split, rec, stem, polys, q33, seed = item
    rng = random.Random(seed)
    low_idx = ([i for i, c in enumerate(ring_contrast((stem, polys))[1]) if c < q33]
               if split == "train" and polys else [])
    img = cv2.imread(str(W / "img_c3" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    cx, cy, R = DISKS[stem]
    centres = []
    if polys:
        p = polys[rng.randrange(len(polys))]
        centres.append((p[:, 0].mean() + rng.uniform(-150, 150), p[:, 1].mean() + rng.uniform(-150, 150), "gt"))
    a, r = rng.uniform(0, 2 * np.pi), R * np.sqrt(rng.random())
    centres.append((cx + r * np.cos(a), cy + r * np.sin(a), "rand"))
    if low_idx:
        p = polys[rng.choice(low_idx)]
        centres.append((p[:, 0].mean() + rng.uniform(-100, 100), p[:, 1].mean() + rng.uniform(-100, 100), "low"))
    n = 0
    for i, (px, py, kind) in enumerate(centres):
        x0 = int(np.clip(px - CROP / 2, 0, SIZE - CROP))
        y0 = int(np.clip(py - CROP / 2, 0, SIZE - CROP))
        name = f"{rec}_{i}{kind}"
        cv2.imwrite(str(OUT / split / "images" / f"{name}.png"), img[y0:y0 + CROP, x0:x0 + CROP], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        (OUT / split / "labels" / f"{name}.txt").write_text("\n".join(crop_labels(polys, x0, y0)) + "\n")
        n += 1
    return split, n, len(low_idx) > 0


def main():
    coco = json.load(open(ANN))
    train_files, val_files = split_files(coco)
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(np.asarray(a["segmentation"][0], np.float64).reshape(-1, 2))
    per_file = defaultdict(list)
    for im in coco["images"]:
        per_file[im["file_name"][:-5]].extend(anns[im["id"]])
    with Pool(16) as p:
        contrast = dict(p.map(ring_contrast, sorted((s, v) for s, v in per_file.items() if f"{s}.jpeg" in train_files)))
    allc = np.concatenate([np.asarray(v) for v in contrast.values()])
    q33 = float(np.percentile(allc, 33.3))
    for split in ("train", "val"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    items = []
    for j, im in enumerate(coco["images"]):
        stem = im["file_name"][:-5]
        split = "train" if im["file_name"] in train_files else "val"
        items.append((split, im["id"], stem, anns[im["id"]], q33, j))
    with Pool(16) as p:
        res = p.map(make, items)
    n_tr = sum(n for s, n, _ in res if s == "train")
    n_va = sum(n for s, n, _ in res if s == "val")
    n_low = sum(1 for s, _, l in res if s == "train" and l)
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: train/images\nval: val/images\nnames:\n  0: filament\n")
    print(f"low-contrast threshold (33rd pct of train ring contrast) {q33:.4f}; train crops {n_tr} "
          f"({n_low} low-contrast oversample crops), val crops {n_va}")


if __name__ == "__main__":
    main()
