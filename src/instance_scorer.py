"""PQ-aware instance scorer (keep/discard rule) for the YOLO ensemble.

Under the competition PQ, keeping one prediction adds its matched IoU (if > 0.5) to
the numerator for every annotator record it matches, and 0.5 to the denominator
for every record either way. A prediction therefore helps iff its expected matched
IoU across an image's annotator records exceeds PQ / 2.

This script
  1. builds candidates from the fused ensemble (union + connected components at a
     low confidence, same erosion/min-area as the submission pipeline);
  2. computes per-candidate features (ensemble score, view/model agreement, size,
     shape, contrast, position on the disk, observatory);
  3. learns y = mean over annotator records of (IoU if IoU > 0.5 else 0) with
     LightGBM, cross-fitted over validation images (5 folds, images never seen by
     the model that scores them);
  4. reports PQ for "keep if predicted y > tau" vs the plain confidence threshold.
With --test it trains on all validation candidates and writes a submission CSV.

Runs in the project's .venv-gbm (lightgbm), which has no torch.
"""
import argparse
import json
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from pycocotools import mask as mu

from postprocess import SIZE, decode, encode, fuse_views, postprocess

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "work"
SITES = "BCLMTU"
CAND = {"conf_thr": 0.12, "mode": "union_cc", "grow": -1, "min_area": 150}
FEATS = None  # filled at runtime
ARGS = None


def rle(c):
    return {"size": [SIZE, SIZE], "counts": c.encode() if isinstance(c, str) else c}


def disk_geometry(gray):
    _, th = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return (SIZE / 2, SIZE / 2), SIZE * 0.45
    (x, y), r = cv2.minEnclosingCircle(max(cs, key=cv2.contourArea))
    return (x, y), r


def candidates_for_image(args):
    """-> list of (counts, feature dict) for one image."""
    stem, raw, n_views, img_path = args
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    (cx, cy), rad = disk_geometry(gray)
    fused = fuse_views(raw, n_views)
    comps = postprocess(fused, **CAND)
    raw_masks = [(c, decode(m), v) for c, m, v in raw if c >= 0.05]
    out = []
    for comp in comps:
        ys, xs = np.nonzero(comp)
        area = len(xs)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        sub = comp[y0:y1 + 1, x0:x1 + 1]
        # fused instances overlapping this component
        ov = [(s, (m[y0:y1 + 1, x0:x1 + 1] & sub).sum() / max(1, m.sum())) for s, m in fused
              if m[y0:y1 + 1, x0:x1 + 1].any()]
        ov = [(s, f) for s, f in ov if f > 0.3]
        smax = max([s for s, _ in ov], default=0.0)
        ssum = sum(s * f for s, f in ov)
        # raw-view agreement (IoU > 0.3 with the component), per model letter
        agree = defaultdict(float)
        n_agree = 0
        for c, m, v in raw_masks:
            mm = m[y0:y1 + 1, x0:x1 + 1]
            inter = (mm & sub).sum()
            if inter == 0:
                continue
            iou = inter / (m.sum() + area - inter)
            if iou > 0.3:
                n_agree += 1
                agree[v[0] if v[0] in "ac" else "b"] = max(agree[v[0] if v[0] in "ac" else "b"], c)
        mom = cv2.moments(sub.astype(np.uint8))
        cov = np.array([[mom["mu20"], mom["mu11"]], [mom["mu11"], mom["mu02"]]]) / max(mom["m00"], 1)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = float(np.sqrt(ev[0] / max(ev[1], 1e-6)))
        cs, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(max(cs, key=cv2.contourArea))) if cs else area
        ring = cv2.dilate(comp.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~comp
        inside = gray[comp].mean()
        around = gray[ring].mean() if ring.any() else inside
        mx, my = xs.mean(), ys.mean()
        r_rel = float(np.hypot(mx - cx, my - cy) / max(rad, 1))
        f = {
            "smax": smax, "ssum": ssum, "n_fused": len(ov), "n_agree": n_agree / n_views,
            "agree_b": agree["b"], "agree_a": agree["a"], "agree_c": agree["c"],
            "log_area": np.log(area), "elong": elong, "solidity": area / max(hull, 1),
            "bbox_fill": area / ((x1 - x0 + 1) * (y1 - y0 + 1)),
            "contrast": (around - inside) / max(around, 1), "inside": inside, "r_rel": r_rel,
            "site": SITES.index(stem[-2]) if stem[-2] in SITES else -1,
        }
        out.append((encode(comp), f))
    return stem, out


def target(cands, gt_records):
    """y per candidate: mean over records of IoU if > 0.5 else 0."""
    if not cands:
        return []
    pr = [rle(c) for c, _ in cands]
    ys = np.zeros(len(cands))
    for g in gt_records:
        gr = [rle(c) for c in g]
        iou = np.asarray(mu.iou(pr, gr, [0] * len(gr))).reshape(len(pr), len(gr))
        best = iou.max(1) if len(gr) else np.zeros(len(pr))
        ys += np.where(best > 0.5, best, 0)
    return ys / max(1, len(gt_records))


def pq_of(kept_by_img, gt):
    s = tp = fp = fn = 0
    for stem, recs in gt.items():
        pr = [rle(c) for c in kept_by_img.get(stem, [])]
        for g in recs:
            gr = [rle(c) for c in g]
            if not pr:
                fn += len(gr)
                continue
            iou = np.asarray(mu.iou(pr, gr, [0] * len(gr))).reshape(len(pr), len(gr)).T
            hit = iou > 0.5
            s += iou[hit].sum()
            tp += hit.sum()
            fp += (hit.sum(0) == 0).sum()
            fn += (hit.sum(1) == 0).sum()
    return s / (tp + 0.5 * fp + 0.5 * fn), dict(tp=int(tp), fp=int(fp), fn=int(fn))


def load_gt(path):
    gt_df = pd.read_csv(path)
    gt = defaultdict(lambda: defaultdict(list))
    for fid, r in zip(gt_df.filament_id, gt_df.segmentation_rle):
        rec = fid.rsplit("_", 1)[0]
        gt[rec.split("-", 1)[1]][rec].append(r)
    return {k: list(v.values()) for k, v in gt.items()}


def build(preds, images, n_views, procs):
    raw = pickle.load(open(preds, "rb"))
    paths = {}
    for p in Path(images).iterdir():
        paths.setdefault(p.stem.split("-", 1)[-1], p)
    with Pool(procs) as p:
        return dict(p.map(candidates_for_image, [(k, v, n_views, paths[k]) for k, v in raw.items()]))


PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-preds", default=str(W / "val_ens_b_a_c.pkl"))
    ap.add_argument("--test-preds", default=str(W / "test_ens_b_a_c.pkl"))
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--gt", default=str(W / "yolo_ds" / "val_gt.csv"))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out", default=str(ROOT / "submission_ens_b_a_c_gbm.csv"))
    args = ap.parse_args()

    gt = load_gt(args.gt)
    val = build(args.val_preds, W / "yolo_ds" / "val" / "images", args.views, args.procs)
    rows = []
    for stem, cands in val.items():
        y = target(cands, gt.get(stem, []))
        for (c, f), yy in zip(cands, y):
            rows.append({"stem": stem, "counts": c, "y": yy, **f})
    df = pd.DataFrame(rows)
    feats = [c for c in df.columns if c not in ("stem", "counts", "y")]
    print(f"val candidates {len(df)}, mean y {df.y.mean():.3f}, positive {(df.y > 0).mean():.2f}")

    stems = sorted(gt)
    folds = {s: i % 5 for i, s in enumerate(random.Random(0).sample(stems, len(stems)))}
    df["fold"] = df.stem.map(folds)
    df["pred"] = 0.0
    for k in range(5):
        tr, te = df[df.fold != k], df[df.fold == k]
        m = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.y), num_boost_round=args.rounds)
        df.loc[te.index, "pred"] = m.predict(te[feats])

    def pq_keep(col, thr, sub=None):
        d = df if sub is None else df[df.stem.isin(sub)]
        kept = defaultdict(list)
        for s, c, v in zip(d.stem, d.counts, d[col]):
            if v > thr:
                kept[s].append(c)
        g = gt if sub is None else {s: gt[s] for s in sub}
        return pq_of(kept, g)

    res = {}
    for col, grid in (("smax", np.arange(0.15, 0.45, 0.025)), ("pred", np.arange(0.10, 0.40, 0.02))):
        scores = [(pq_keep(col, t)[0], t) for t in grid]
        best = max(scores)
        res[col] = best
        print(f"{col:5s} best PQ {best[0]:.4f} at thr {best[1]:.3f}   curve: " +
              " ".join(f"{t:.2f}:{p:.3f}" for p, t in scores[::2]), flush=True)
    # honest comparison: choose the threshold on half of the images, score on the other half
    for col, grid in (("smax", np.arange(0.15, 0.45, 0.025)), ("pred", np.arange(0.10, 0.40, 0.02))):
        outs = []
        for seed in range(5):
            s = stems[:]
            random.Random(seed).shuffle(s)
            A, B = s[:len(s) // 2], s[len(s) // 2:]
            for tr, te in ((A, B), (B, A)):
                t = max(grid, key=lambda t: pq_keep(col, t, tr)[0])
                outs.append(pq_keep(col, t, te)[0])
        print(f"{col:5s} held-out-half PQ {np.mean(outs):.4f} ± {np.std(outs):.4f}")
    m_all = lgb.train(PARAMS, lgb.Dataset(df[feats], df.y), num_boost_round=args.rounds)
    imp = sorted(zip(m_all.feature_importance("gain"), feats), reverse=True)
    print("feature importance:", ", ".join(f"{n} {g:.0f}" for g, n in imp[:8]))
    json.dump({k: [float(v[0]), float(v[1])] for k, v in res.items()}, open(W / "gbm_scorer_val.json", "w"))

    if args.test:
        tau = args.tau if args.tau is not None else res["pred"][1]
        test = build(args.test_preds, ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images",
                     args.views, args.procs)
        out = []
        for stem, cands in sorted(test.items()):
            if not cands:
                continue
            p = m_all.predict(pd.DataFrame([f for _, f in cands])[feats])
            keep = [c for (c, _), v in zip(cands, p) if v > tau]
            out += [(f"{stem}_{i}", c) for i, c in enumerate(keep, 1)]
        pd.DataFrame(out, columns=["filament_id", "segmentation_rle"]).to_csv(args.out, index=False)
        print(f"test: tau {tau:.3f}, {len(out)} filaments over {len({o[0].rsplit('_', 1)[0] for o in out})} images -> {args.out}")


if __name__ == "__main__":
    main()
