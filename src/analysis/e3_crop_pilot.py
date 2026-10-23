"""E3. Crop-detector pilot gate (add-only).

The pilot model (trained on 1024 px crops at imgsz 2048, training split only) is run on
full validation images as a 3 x 3 tiled pass (src/tiled_infer.py --color). Its detections
that do not overlap the baseline ensemble's final masks (<= 10% of their area) are the
"added" candidates. An added detection is real if its mean matched IoU across the
image's annotator records exceeds PQ/2. Gate: continue to a full run only if >= 40% of
added held-out detections are real at some confidence threshold that adds >= 20
detections. PQ after adding is reported in-sample and split-half.
"""
import pickle
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import BASE, W, baseline, decode, encode, iou_matrix, load_val_gt, pq_score, write_md

PQ0, GATE, MIN_ADDED = 0.4765, 0.40, 20
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def added_for_image(item):
    stem, final, tiles, recs = item
    taken = np.zeros((2048, 2048), bool)
    for c in final:
        taken |= decode(c)
    k = len(recs)
    out = []
    for conf, counts, _ in tiles:
        m = decode(counts)
        if (m & taken).sum() > 0.1 * m.sum():
            continue
        new = m & ~taken
        if not new.any():
            continue
        c_new = encode(new)
        tot = 0.0
        for g in recs.values():
            iou = iou_matrix([c_new], g)
            if iou.size and (iou > 0.5).any():
                tot += float(iou.max())
        out.append((conf, c_new, tot / k > PQ0 / 2, int(new.sum())))
    return stem, out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=str(W / "val_crop_pilot_tiles.pkl"))
    ap.add_argument("--name", default="e3_crop_pilot")
    args = ap.parse_args()
    gt = load_val_gt()
    final, _ = baseline()
    base = {k: [c for _, c in v] for k, v in final.items()}
    tiles = pickle.load(open(args.tiles, "rb"))
    with Pool(12) as p:
        added = dict(p.map(added_for_image, [(s, base.get(s, []), tiles.get(s, []), gt[s]) for s in sorted(gt)]))
    stems = sorted(gt)
    rows = []
    for t in THRESHOLDS:
        sel = [(s, c, r) for s, lst in added.items() for conf, c, r, a in lst if conf >= t]
        preds = {s: list(v) for s, v in base.items()}
        for s, c, _ in sel:
            preds.setdefault(s, []).append(c)
        pq, st = pq_score(preds, gt)
        rows.append(dict(threshold=t, added=len(sel), real=sum(r for _, _, r in sel),
                         share_real=(sum(r for _, _, r in sel) / len(sel)) if sel else np.nan,
                         PQ_in_sample=pq, dPQ_in_sample=pq - PQ0, TP=st["tp"], FP=st["fp"], FN=st["fn"]))
    T = pd.DataFrame(rows)

    # split-half: choose threshold (or "add nothing") on one half, score on the other
    def pq_sub(t, sub):
        preds = {s: list(base.get(s, [])) for s in sub}
        if t is not None:
            for s in sub:
                preds[s] += [c for conf, c, r, a in added[s] if conf >= t]
        return pq_score(preds, {s: gt[s] for s in sub})[0]
    held = []
    for seed in range(5):
        sh = stems[:]
        random.Random(seed).shuffle(sh)
        A, B = sh[:len(sh) // 2], sh[len(sh) // 2:]
        for tr, te in ((A, B), (B, A)):
            t = max([None] + THRESHOLDS, key=lambda t: pq_sub(t, tr))
            held.append(dict(chosen=t, held_out_PQ=pq_sub(t, te), baseline_held_out=pq_sub(None, te)))
    H = pd.DataFrame(held)
    ok = T[(T.added >= MIN_ADDED) & (T.share_real >= GATE)]
    verdict = (f"PASS: share real >= {GATE:.0%} with >= {MIN_ADDED} added at thresholds {list(ok.threshold)}"
               if len(ok) else f"STOP: no threshold reaches {GATE:.0%} real among >= {MIN_ADDED} added detections")
    write_md(f"{args.name}.md", [
        ("Added detections (not overlapping the baseline ensemble) by pilot confidence", T),
        ("Split-half: threshold chosen on one half, PQ on the other", H),
        ("Summary", f"held-out PQ with added detections {H.held_out_PQ.mean():.4f} vs baseline {H.baseline_held_out.mean():.4f} "
                    f"(delta {(H.held_out_PQ - H.baseline_held_out).mean():+.4f})\n\nGate: {verdict}"),
    ])
    (W / f"{args.name}_gate.txt").write_text(verdict + "\n")


if __name__ == "__main__":
    main()
