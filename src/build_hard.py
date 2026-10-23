"""Hard-example tile dataset for the missed-filament specialist.

Source of hard examples: the 3-model ensemble's predictions on the TRAINING split
(one view per model, fused, baseline post-processing). These are in-sample for the
ensemble, so a filament it still misses there is a genuinely hard case; out-of-fold
mining would need K-fold retraining of all members. Validation stays fully held out.

Per annotator record (training split only), 768 px native-resolution tiles centred on:
  * each MISSED GT filament (best IoU with any ensemble mask < 0.3): 2 tiles (+-120 px jitter)
  * each NEAR-MISS GT (best IoU 0.3-0.5): 1 tile
  * one correctly detected GT (ordinary positive): 1 tile
  * each ensemble FALSE POSITIVE on the image (matches no record's GT at IoU > 0.5 and
    overlaps no GT of this record): 1 tile (hard negative; labels = whatever GT is in the tile)
  * one random on-disk position (background / ordinary): 1 tile
Labels: that record's filaments clipped to the tile (pieces < 30 px dropped).
Validation tiles are built the same way from validation records, for training-time
monitoring only (the gate uses full validation images with tiled inference).
"""
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from postprocess import decode, encode, fuse_views, postprocess  # noqa: E402

DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
ANN = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
W = ROOT / "work"
OUT = W / "hard_tiles"
SIZE, TILE, MIN_PIECE = 2048, 768, 30
BASE = dict(conf_thr=0.275, mode="union_cc", grow=-1, min_area=150)
DISKS = json.load(open(W / "img_c3" / "disks.json"))
ENS = None


def split_files(coco, val_frac=0.15, seed=42):
    files = sorted({im["file_name"] for im in coco["images"]})
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_frac)
    return set(files[n_val:]), set(files[:n_val])


def rle_poly(p):
    return mu.merge(mu.frPyObjects([p.ravel().tolist()], SIZE, SIZE))


def tile_labels(polys, x0, y0):
    lines = []
    for p in polys:
        m = np.zeros((TILE, TILE), np.uint8)
        cv2.fillPoly(m, [np.round(p - (x0, y0)).astype(np.int32)], 1)
        if m.sum() < MIN_PIECE:
            continue
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            if len(c) >= 3 and cv2.contourArea(c) >= MIN_PIECE:
                q = np.clip(c.reshape(-1, 2) / TILE, 0, 1)
                lines.append("0 " + " ".join(f"{v:.6f}" for v in q.ravel()))
    return lines


def ensemble_finals(stem):
    fused = fuse_views(ENS.get(stem, []), 3)
    return postprocess([(s, decode(c)) for s, c in fused if s >= BASE["conf_thr"]], **BASE)


def _init(ens):
    global ENS
    ENS = ens


def make(item):
    split, rec, stem, polys, all_recs, seed = item
    rng = random.Random(seed)
    fin = ensemble_finals(stem) if split == "train" else []
    fin_r = [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in fin]
    g_r = [rle_poly(p) for p in polys]
    centres, kinds = [], Counter()
    if split == "train" and g_r:
        best = (np.asarray(mu.iou(g_r, fin_r, [0] * len(fin_r))).reshape(len(g_r), len(fin_r)).max(1)
                if fin_r else np.zeros(len(g_r)))
        for i, b in enumerate(best):
            c = polys[i].mean(0)
            if b < 0.3:
                centres += [(c, "miss"), (c, "miss")]
            elif b < 0.5:
                centres.append((c, "near"))
        hits = [i for i, b in enumerate(best) if b >= 0.5]
        if hits:
            centres.append((polys[rng.choice(hits)].mean(0), "hit"))
        # hard negatives: ensemble masks matching no record of this image and not touching this record's GT
        all_g = [rle_poly(p) for ps in all_recs for p in ps]
        for m, r in zip(fin, fin_r):
            if all_g and (np.asarray(mu.iou([r], all_g, [0] * len(all_g))) > 0.5).any():
                continue
            if g_r and (np.asarray(mu.iou([r], g_r, [0] * len(g_r))) > 0).any():
                continue
            ys, xs = np.nonzero(m)
            centres.append((np.array([xs.mean(), ys.mean()]), "neg"))
    elif polys:
        centres.append((polys[rng.randrange(len(polys))].mean(0), "hit"))
    cx, cy, R = DISKS[stem]
    a, r = rng.uniform(0, 2 * np.pi), R * np.sqrt(rng.random())
    centres.append((np.array([cx + r * np.cos(a), cy + r * np.sin(a)]), "bg"))
    gray = cv2.imread(str(DATA / "train" / "train_images" / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    for i, (c, kind) in enumerate(centres):
        j = 120 if kind == "miss" else 60
        x0 = int(np.clip(c[0] + rng.uniform(-j, j) - TILE / 2, 0, SIZE - TILE))
        y0 = int(np.clip(c[1] + rng.uniform(-j, j) - TILE / 2, 0, SIZE - TILE))
        name = f"{rec}_{i}{kind}"
        cv2.imwrite(str(OUT / split / "images" / f"{name}.png"), gray[y0:y0 + TILE, x0:x0 + TILE], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        (OUT / split / "labels" / f"{name}.txt").write_text("\n".join(tile_labels(polys, x0, y0)) + "\n")
        kinds[kind] += 1
    return split, kinds


def main():
    coco = json.load(open(ANN))
    train_files, _ = split_files(coco)
    ens = {}
    for tag, run in (("b", "y26m_2048"), ("a", "y26m_1536"), ("c", "y26m_1536_cons")):
        for k, v in pickle.load(open(W / f"train_{run}_o.pkl", "rb")).items():
            ens.setdefault(k, []).extend((c, m, "o" if tag == "b" else f"{tag}o") for c, m, vw in v if vw == "o")
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(np.asarray(a["segmentation"][0], np.float64).reshape(-1, 2))
    recs_of = defaultdict(list)
    for im in coco["images"]:
        recs_of[im["file_name"][:-5]].append(anns[im["id"]])
    for split in ("train", "val"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    items = [("train" if im["file_name"] in train_files else "val", im["id"], im["file_name"][:-5], anns[im["id"]],
              recs_of[im["file_name"][:-5]], j) for j, im in enumerate(coco["images"])]
    with Pool(16, initializer=_init, initargs=(ens,)) as p:
        res = p.map(make, items)
    tot = defaultdict(Counter)
    for split, k in res:
        tot[split].update(k)
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: train/images\nval: val/images\nnames:\n  0: filament\n")
    print({s: dict(c) for s, c in tot.items()})


if __name__ == "__main__":
    main()
