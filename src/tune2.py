"""Staged search over ensemble-fusion and post-processing options (no retraining).

Stage A: fusion (pixel vote fraction x per-model weights) with a short conf sweep.
Stage B: mode / min_area / hole-filling / size-aware erosion on the best fusion.
Stage C: conf refinement.
Finally a split-half check: params chosen on one half of the validation images are
scored on the other half, for both the new best and the current baseline params.
"""
import argparse
import itertools
import json
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

import tune
from postprocess import encode, fuse_views

BASE = {"conf_thr": 0.275, "mode": "union_cc", "grow": -1, "min_area": 150, "fill": False, "grow_min_area": 0}


def _fuse(item):
    stem, raw, n_views, vf, weights = item
    return stem, [(s, encode(m)) for s, m in fuse_views(raw, n_views, vote_frac=vf, weights=weights)]


def load_gt(path):
    gt_df = pd.read_csv(path)
    gt = defaultdict(lambda: defaultdict(list))
    for fid, rle in zip(gt_df.filament_id, gt_df.segmentation_rle):
        rec = fid.rsplit("_", 1)[0]
        gt[rec.split("-", 1)[1]][rec].append(rle)
    return {k: list(v.values()) for k, v in gt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--views", type=int, required=True)
    ap.add_argument("--models", default="ac", help="letters of non-reference models in the view tags")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = pickle.load(open(args.preds, "rb"))
    gt = load_gt(args.gt)
    stems = sorted(gt)
    letters = list(args.models)
    fusions = {"vf0.50_eq": (0.5, None)}
    for vf in (0.34, 0.5, 0.67):
        for name, w in (("eq", None), ("ref2x", {l: 0.5 for l in letters})):
            fusions[f"vf{vf:.2f}_{name}"] = (vf, w)
    results = []

    def run_grid(fused, fname, grid):
        best = None
        with Pool(args.procs, initializer=tune._init, initargs=(fused, gt)) as pool:
            for params in grid:
                pq, st = tune.evaluate(pool, stems, params)
                results.append({"fusion": fname, **params, "pq": pq, **st})
                print(f"{pq:.4f} {fname} {params}", flush=True)
                if best is None or pq > best[0]:
                    best = (pq, params)
        return best

    fused_cache = {}
    best_a = None
    for fname, (vf, w) in fusions.items():
        with Pool(args.procs) as p:
            fused_cache[fname] = dict(p.map(_fuse, [(k, v, args.views, vf, w) for k, v in raw.items()]))
        b = run_grid(fused_cache[fname], fname, [{**BASE, "conf_thr": c} for c in (0.25, 0.275, 0.3, 0.35)])
        if best_a is None or b[0] > best_a[0]:
            best_a = (b[0], b[1], fname)
    _, pa, fbest = best_a
    grid_b = [{**pa, "mode": m, "min_area": a, "fill": f, "grow": g, "grow_min_area": gm}
              for m, a, f, (g, gm) in itertools.product(["union_cc", "paint"], [0, 150, 300], [False, True],
                                                        [(-1, 0), (-1, 800), (-1, 1500), (0, 0)])]
    pb = run_grid(fused_cache[fbest], fbest, grid_b)[1]
    pc = run_grid(fused_cache[fbest], fbest, [{**pb, "conf_thr": round(c, 3)} for c in np.arange(pb["conf_thr"] - 0.075, pb["conf_thr"] + 0.08, 0.025)])[1]

    # split-half check (5 random halvings): tune conf on half A among a few options, score on half B
    def half_scores(fused, params_list):
        out = []
        with Pool(args.procs, initializer=tune._init, initargs=(fused, gt)) as pool:
            for seed in range(5):
                s = stems[:]
                random.Random(seed).shuffle(s)
                A, B = s[: len(s) // 2], s[len(s) // 2:]
                for tr, te in ((A, B), (B, A)):
                    pick = max(params_list, key=lambda p: tune.evaluate(pool, tr, p)[0])
                    out.append(tune.evaluate(pool, te, pick)[0])
        return float(np.mean(out)), float(np.std(out))

    conf_opts = lambda p: [{**p, "conf_thr": round(p["conf_thr"] + d, 3)} for d in (-0.05, -0.025, 0, 0.025, 0.05)]
    cv_new = half_scores(fused_cache[fbest], conf_opts(pc))
    cv_base = half_scores(fused_cache["vf0.50_eq"], conf_opts(BASE))
    top = max(results, key=lambda r: r["pq"])
    summary = {"best": {**top, "vote_frac": fusions[top["fusion"]][0], "weights": fusions[top["fusion"]][1]},
               "cv_new_mean_std": cv_new, "cv_base_mean_std": cv_base, "all": results}
    json.dump(summary, open(args.out, "w"), indent=1, default=str)
    print("BEST", json.dumps(summary["best"], default=str))
    print(f"held-out halves: new {cv_new[0]:.4f}±{cv_new[1]:.4f} vs baseline {cv_base[0]:.4f}±{cv_base[1]:.4f}")


if __name__ == "__main__":
    main()
