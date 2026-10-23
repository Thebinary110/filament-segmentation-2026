"""F2. Near-miss replacement with cached zoomed-tile candidates (no training, CPU).

Baseline = the 3-model ensemble's final masks (conf 0.275, union_cc, erode 1, min area 150).
Tile candidates = val_tiles_2048m_z15.pkl (2048 model on 3x3 tiles of 1024 px at 1.5x zoom).

Rule (per image):
  replace: a tile candidate with conf >= tc whose IoU with an existing final mask e is in
           [lo, 0.8) and whose conf >= score(e) + delta replaces e's mask (best-conf candidate
           per e; optional 1 px erosion; clipped so it does not overlap other final masks);
  add:     optionally, candidates with conf >= ta that overlap no final mask (<= 10% of their
           area) are added as new instances (clipped, eroded the same way).
Every (tc, lo, delta, erode, ta) is scored with per-image PQ components, so the split-half
selection is exact: choose on one half of the validation images, score on the other half,
5 halvings x 2, against the unchanged baseline on the same halves.
"""
import itertools
import pickle
import random
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from common import W, baseline, decode, encode, iou_matrix, load_val_gt, write_md

K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
TC = [0.3, 0.4, 0.5, 0.6]
LO = [0.2, 0.3]
DELTA = [-0.1, 0.0, 0.1, 0.2]
ERODE = [0, 1]
TA = [None, 0.5, 0.6, 0.7]
GT = FINAL = TILES = None


def _init(gt, final, tiles):
    global GT, FINAL, TILES
    GT, FINAL, TILES = gt, final, tiles


def image_stats(stem, preds):
    s = tp = fp = fn = 0
    for g in GT[stem].values():
        if not preds:
            fn += len(g)
            continue
        iou = iou_matrix(g, preds)
        hit = iou > 0.5
        s += iou[hit].sum()
        tp += hit.sum()
        fp += (hit.sum(0) == 0).sum()
        fn += (hit.sum(1) == 0).sum()
    return float(s), int(tp), int(fp), int(fn)


def run_image(stem):
    fin = FINAL.get(stem, [])
    fin_c = [c for _, c in fin]
    fin_s = [s for s, _ in fin]
    fin_m = [decode(c) for c in fin_c]
    cands = [(conf, decode(c)) for conf, c, _ in TILES.get(stem, []) if conf >= min(TC + [t for t in TA if t])]
    if fin_c and cands:
        iou = iou_matrix([encode(m) for _, m in cands], fin_c)
    else:
        iou = np.zeros((len(cands), len(fin_c)))
    union = np.zeros((2048, 2048), bool)
    for m in fin_m:
        union |= m
    contained = [((m & union).sum() / max(1, m.sum())) for _, m in cands]
    results, n_rep = {}, {}
    base = image_stats(stem, fin_c)
    for tc, lo, delta, er, ta in itertools.product(TC, LO, DELTA, ERODE, TA):
        masks = list(fin_m)
        replaced = 0
        for j in range(len(fin_m)):
            best = None
            for i, (conf, m) in enumerate(cands):
                if conf >= tc and lo <= iou[i, j] < 0.8 and conf >= fin_s[j] + delta and (best is None or conf > cands[best][0]):
                    best = i
            if best is not None:
                m = cands[best][1]
                if er:
                    m = cv2.erode(m.astype(np.uint8), K3).astype(bool)
                others = np.zeros((2048, 2048), bool)
                for k, mk in enumerate(masks):
                    if k != j:
                        others |= mk
                m = m & ~others
                if m.sum() >= 150:
                    masks[j] = m
                    replaced += 1
        if ta is not None:
            taken = np.zeros((2048, 2048), bool)
            for mk in masks:
                taken |= mk
            for i, (conf, m) in enumerate(cands):
                if conf >= ta and contained[i] <= 0.1:
                    if er:
                        m = cv2.erode(m.astype(np.uint8), K3).astype(bool)
                    m = m & ~taken
                    if m.sum() >= 150:
                        masks.append(m)
                        taken |= m
        key = (tc, lo, delta, er, ta)
        results[key] = image_stats(stem, [encode(m) for m in masks]) if (replaced or ta is not None) else base
        n_rep[key] = replaced
    return stem, base, results, n_rep


def pq(stats, stems):
    s, tp, fp, fn = (sum(stats[x][i] for x in stems) for i in range(4))
    return s / (tp + 0.5 * fp + 0.5 * fn)


def main():
    gt = load_val_gt()
    final, _ = baseline()
    tiles = pickle.load(open(W / "val_tiles_2048m_z15.pkl", "rb"))
    stems = sorted(gt)
    with Pool(12, initializer=_init, initargs=(gt, final, tiles)) as p:
        res = p.map(run_image, stems)
    base = {s: b for s, b, _, _ in res}
    keys = list(res[0][2])
    per = {k: {s: r[k] for s, _, r, _ in res} for k in keys}
    reps = {k: sum(n[k] for _, _, _, n in res) for k in keys}
    ins = pd.DataFrame([dict(tc=k[0], lo=k[1], delta=k[2], erode=k[3], add_ta=k[4] if k[4] else "none",
                             replacements=reps[k], PQ_in_sample=pq(per[k], stems)) for k in keys])
    base_pq = pq(base, stems)
    ins["dPQ_in_sample"] = ins.PQ_in_sample - base_pq
    top = ins.sort_values("PQ_in_sample", ascending=False).head(12)
    rep_only = ins[ins.add_ta == "none"].sort_values("PQ_in_sample", ascending=False).head(6)

    held = []
    for seed in range(5):
        sh = stems[:]
        random.Random(seed).shuffle(sh)
        A, B = sh[:len(sh) // 2], sh[len(sh) // 2:]
        for tr, te in ((A, B), (B, A)):
            for scope, cand_keys in (("replace only", [k for k in keys if k[4] is None]), ("replace + add", keys)):
                k = max(cand_keys, key=lambda k: pq(per[k], tr))
                held.append(dict(seed=seed, scope=scope, chosen=str(k), held_out_PQ=pq(per[k], te), baseline_held_out=pq(base, te)))
    H = pd.DataFrame(held)
    H["delta"] = H.held_out_PQ - H.baseline_held_out
    summ = H.groupby("scope").agg(held_out_PQ=("held_out_PQ", "mean"), baseline=("baseline_held_out", "mean"),
                                  delta_mean=("delta", "mean"), delta_sd=("delta", "std"),
                                  halves_positive=("delta", lambda d: int((d > 0).sum()))).reset_index()
    write_md("f2_replace.md", [
        ("Baseline", f"PQ {base_pq:.4f} (in-sample, all 106 validation images)"),
        ("Held-out (split-half: parameters chosen on one half, scored on the other; 10 halves)", summ),
        ("Top in-sample settings (all rules)", top),
        ("Top in-sample settings, replacement only", rep_only),
        ("Split-half detail", H),
    ])


if __name__ == "__main__":
    main()
