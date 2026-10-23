"""Grid-search post-processing parameters on cached validation predictions (local PQ)."""
import argparse
import itertools
import json
import pickle
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from postprocess import SIZE, decode, encode, fuse_views, postprocess

_FUSED = {}
_GT = {}


def _rle(c):
    return {"size": [SIZE, SIZE], "counts": c.encode("ascii") if isinstance(c, str) else c}


def _fuse(item):
    stem, raw, n_views = item
    return stem, [(s, encode(m)) for s, m in fuse_views(raw, n_views)]


def _score_image(args):
    stem, params = args
    inst = [(s, decode(c)) for s, c in _FUSED[stem] if s >= params["conf_thr"]]
    preds = [_rle(encode(m)) for m in postprocess(inst, **params)]
    tp_iou, fp, fn = [], 0, 0
    for gts in _GT.get(stem, []):
        g = [_rle(c) for c in gts]
        if not preds:
            fn += len(g)
            continue
        iou = np.asarray(mask_util.iou(preds, g, [0] * len(g))).reshape(len(preds), len(g)).T
        hit = iou > 0.5
        tp_iou.extend(iou[hit].tolist())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    return sum(tp_iou), len(tp_iou), fp, fn


def evaluate(pool, stems, params):
    res = pool.map(_score_image, [(s, params) for s in stems])
    s, tp, fp, fn = map(sum, zip(*res))
    den = tp + 0.5 * fp + 0.5 * fn
    return (s / den if den else 0.0), dict(tp=tp, fp=fp, fn=fn, sq=s / tp if tp else 0)


def _init(fused, gt):
    _FUSED.update(fused)
    _GT.update(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--views", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = pickle.load(open(args.preds, "rb"))
    if args.views == 1:
        raw = {k: [r for r in v if r[2] == "o"] for k, v in raw.items()}
    gt_df = pd.read_csv(args.gt)
    gt = defaultdict(lambda: defaultdict(list))
    for fid, rle in zip(gt_df.filament_id, gt_df.segmentation_rle):
        rec = fid.rsplit("_", 1)[0]
        gt[rec.split("-", 1)[1]][rec].append(rle)
    gt = {k: list(v.values()) for k, v in gt.items()}

    with Pool(16) as p:
        fused = dict(p.map(_fuse, [(k, v, args.views) for k, v in raw.items()]))
    stems = sorted(gt)
    results = []
    with Pool(16, initializer=_init, initargs=(fused, gt)) as pool:
        def run(params):
            pq, st = evaluate(pool, stems, params)
            results.append({**params, "pq": pq, **st})
            print(f"{pq:.4f} {st} {params}", flush=True)
            return pq

        # stage 1: threshold x overlap mode
        best = max(
            ({"conf_thr": c, "mode": m, "grow": 0, "min_area": 50}
             for c, m in itertools.product([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["paint", "union_cc"])),
            key=run)
        # stage 2: grow x min_area around best
        best2 = max(
            ({**best, "grow": g, "min_area": a}
             for g, a in itertools.product([-1, 0, 1, 2], [0, 50, 150, 300])),
            key=run)
        # stage 3: refine threshold
        c0 = best2["conf_thr"]
        best3 = max(({**best2, "conf_thr": round(c, 3)} for c in np.arange(max(0.05, c0 - 0.1), c0 + 0.11, 0.025)),
                    key=run)
    best_row = max(results, key=lambda r: r["pq"])
    print("BEST", json.dumps(best_row))
    if args.out:
        json.dump({"best": best_row, "all": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
