"""Shared helpers for the PQ-loss diagnosis (no training; cached predictions only).

Conventions
-----------
* "record" = one annotator's label set for one image (e.g. ``010401-20160920230134Lh``);
  k = number of records on an image.
* "cluster" = one physical filament, i.e. the GT instances of different records on the
  same image linked by IoU > LINK_IOU (at most one instance per record); s = its support
  (number of records that drew it).
* The baseline is the 3-model ensemble (val_ens_b_a_c.pkl, 12 views) with the tuned
  post-processing (conf 0.275, union_cc, erode 1px, min area 150): local PQ 0.4765.
* Scoring replicates src/tune.py / the organizers' notebook: every pair with IoU > 0.5 is a
  hit, every record is scored separately, TP/FP/FN pooled.
"""
import json
import pickle
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import tune  # noqa: E402
from postprocess import decode, encode, postprocess  # noqa: E402

ROOT = SRC.parent
W = ROOT / "work"
OUT = W / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
ANN = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMG = DATA / "train" / "train_images"
TEST_IMG = DATA / "test" / "test_images"
SIZE = 2048
BASE = dict(conf_thr=0.275, mode="union_cc", grow=-1, min_area=150)
ENS = W / "val_ens_b_a_c.pkl"
ENS_VIEWS = 12
LINK_IOU = 0.3
SITES = {"B": "Big Bear", "C": "Cerro Tololo", "L": "Learmonth", "M": "Mauna Loa", "T": "Teide", "U": "Udaipur"}


def R(c):
    return {"size": [SIZE, SIZE], "counts": c.encode() if isinstance(c, str) else c}


def iou_matrix(a, b):
    """IoU between RLE lists a (rows) and b (cols)."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou([R(x) for x in a], [R(x) for x in b], [0] * len(b))).reshape(len(a), len(b))


def coco():
    return json.load(open(ANN))


def split_files(c, val_frac=0.15, seed=42):
    files = sorted({im["file_name"] for im in c["images"]})
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_frac)
    return set(files[n_val:]), set(files[:n_val])


def load_val_gt():
    """{stem: {record_id: [rle counts, ...]}} from the same CSV tune.py scores against."""
    df = pd.read_csv(W / "yolo_ds" / "val_gt.csv")
    gt = defaultdict(lambda: defaultdict(list))
    for fid, r in zip(df.filament_id, df.segmentation_rle):
        rec = fid.rsplit("_", 1)[0]
        gt[rec.split("-", 1)[1]][rec].append(r)
    return {k: dict(v) for k, v in gt.items()}


def gt_for_tune(gt):
    return {k: list(v.values()) for k, v in gt.items()}


def pq_score(preds, gt, stems=None):
    """preds: {stem: [counts]}; gt: {stem: {rec: [counts]}} -> (pq, dict)."""
    s = tp = fp = fn = 0
    for stem in (stems or gt):
        p = preds.get(stem, [])
        for rec, g in gt[stem].items():
            if not p:
                fn += len(g)
                continue
            iou = iou_matrix(p, g).T  # gt x pred
            hit = iou > 0.5
            s += iou[hit].sum()
            tp += hit.sum()
            fp += (hit.sum(0) == 0).sum()
            fn += (hit.sum(1) == 0).sum()
    d = tp + 0.5 * fp + 0.5 * fn
    return s / d, dict(sum_iou=float(s), tp=int(tp), fp=int(fp), fn=int(fn), D=float(d), sq=float(s / max(tp, 1)))


# --------------------------------------------------------------------------- baseline predictions
def _final_job(item):
    stem, raw = item
    _, fused = tune._fuse((stem, raw, ENS_VIEWS))
    inst = [(s, decode(c)) for s, c in fused]
    masks = postprocess([(s, m) for s, m in inst if s >= BASE["conf_thr"]], **BASE)
    out = []
    for m in masks:
        # score of a union_cc component = max score of fused instances that mostly lie in it
        sc = max([s for s, f in inst if s >= BASE["conf_thr"] and (f & m).sum() > 0.3 * min(f.sum(), m.sum())],
                 default=0.0)
        out.append((float(sc), encode(m)))
    return stem, out, fused


def baseline(cache=True):
    """-> (final {stem: [(score, counts)]}, fused {stem: [(score, counts)]})"""
    f = OUT / "_baseline_cache.pkl"
    if cache and f.exists():
        return pickle.load(open(f, "rb"))
    raw = pickle.load(open(ENS, "rb"))
    with Pool(12) as p:
        res = p.map(_final_job, sorted(raw.items()))
    final = {s: o for s, o, _ in res}
    fused = {s: fz for s, _, fz in res}
    pickle.dump((final, fused), open(f, "wb"))
    return final, fused


# --------------------------------------------------------------------------- GT clusters (physical filaments)
def clusters_for_image(recs):
    """recs: {rec: [counts]} -> list of clusters, each {rec: index} (one instance per record)."""
    items = [(r, i, c) for r, lst in recs.items() for i, c in enumerate(lst)]
    if not items:
        return []
    areas = [mu.area(R(c)) for _, _, c in items]
    order = np.argsort(areas)[::-1]  # larger instances seed clusters
    allc = [c for _, _, c in items]
    iou = iou_matrix(allc, allc)
    clusters, members = [], []
    for j in order:
        r, i, _ = items[j]
        best, best_iou = None, LINK_IOU
        for ci, cl in enumerate(clusters):
            if r in cl:
                continue
            v = max(iou[j, members[ci]])  # single linkage to any member
            if v > best_iou:
                best, best_iou = ci, v
        if best is None:
            clusters.append({r: i})
            members.append([j])
        else:
            clusters[best][r] = i
            members[best].append(j)
    return clusters


def all_clusters(gt):
    return {stem: clusters_for_image(recs) for stem, recs in gt.items()}


def majority_mask(recs, cl):
    ms = [decode(recs[r][i]) for r, i in cl.items()]
    votes = np.sum(ms, axis=0)
    m = votes * 2 >= len(ms)
    return m if m.any() else ms[0]


# --------------------------------------------------------------------------- per-image geometry / features
def image_path(stem):
    return TRAIN_IMG / f"{stem}.jpeg"


def disk_geometry(gray):
    _, th = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return (SIZE / 2, SIZE / 2), SIZE * 0.45
    (x, y), r = cv2.minEnclosingCircle(max(cs, key=cv2.contourArea))
    return (x, y), r


def mask_features(m, gray, center, radius):
    """area, contrast (ring-minus-inside over ring), r/R of centroid, elongation."""
    ys, xs = np.nonzero(m)
    area = len(xs)
    y0, y1, x0, x1 = max(ys.min() - 12, 0), min(ys.max() + 13, SIZE), max(xs.min() - 12, 0), min(xs.max() + 13, SIZE)
    sub, g = m[y0:y1, x0:x1], gray[y0:y1, x0:x1]
    ring = cv2.dilate(sub.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~sub
    inside = g[sub].mean()
    around = g[ring].mean() if ring.any() else inside
    mom = cv2.moments(sub.astype(np.uint8))
    cov = np.array([[mom["mu20"], mom["mu11"]], [mom["mu11"], mom["mu02"]]]) / max(mom["m00"], 1)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    return dict(area=area, contrast=float((around - inside) / max(around, 1)),
                r_rel=float(np.hypot(xs.mean() - center[0], ys.mean() - center[1]) / max(radius, 1)),
                elong=float(np.sqrt(ev[0] / max(ev[1], 1e-6))))


def md_table(df):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.4g}"
        return str(v)
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def write_md(name, sections):
    """sections: list of (title, DataFrame or str)."""
    lines = []
    for title, body in sections:
        lines.append(f"## {title}\n")
        lines.append(md_table(body) if isinstance(body, pd.DataFrame) else str(body))
        lines.append("")
    (OUT / name).write_text("\n".join(lines))
    print(f"wrote {OUT / name}")
