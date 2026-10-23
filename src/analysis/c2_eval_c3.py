"""C2. Evaluate the 3-channel model (y26m_2048_c3) against the baseline.

* Single models: 2048 grayscale (y26m_2048) vs 2048 3-channel (y26m_2048_c3), 4-view TTA.
* Ensembles: baseline 3-model (b+a+c), c3 replacing b (c3+a+c), c3 added (b+a+c+c3).
For each: post-processing tuned over the usual grid (IN-SAMPLE) and a split-half check
(threshold chosen on half the images, scored on the other half; 5 random halvings).
Recall by GT contrast and by r/R uses the same feature definitions as a5_patterns.
"""
import itertools
import pickle
import random
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from common import W, decode, disk_geometry, encode, gt_for_tune, image_path, iou_matrix, load_val_gt, mask_features, write_md
import tune  # noqa: E402
from postprocess import postprocess

GRID = [dict(conf_thr=c, mode=m, grow=g, min_area=a)
        for c, m, g, a in itertools.product([0.15, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.4, 0.45, 0.5],
                                            ["union_cc", "paint"], [-1, 0], [0, 150, 300])]


def combine(runs):
    """runs: list of (letter, run name); the first is the reference (plain flip tags)."""
    out = {}
    ref = runs[0][0]
    for tag, run in runs:
        for k, v in pickle.load(open(W / f"val_{run}.pkl", "rb")).items():
            out.setdefault(k, []).extend((c, m, vw if tag == ref else f"{tag}{vw}") for c, m, vw in v)
    return out


def evaluate(raw, views, gt):
    gtt = gt_for_tune(gt)
    stems = sorted(gtt)
    with Pool(12) as p:
        fused = dict(p.map(tune._fuse, [(k, v, views) for k, v in raw.items()]))
    with Pool(12, initializer=tune._init, initargs=(fused, gtt)) as pool:
        res = [(tune.evaluate(pool, stems, prm), prm) for prm in GRID]
        (pq, st), best = max(res, key=lambda x: x[0][0])
        conf_opts = [dict(best, conf_thr=c) for c in (0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.4)]
        cv = []
        for seed in range(5):
            sh = stems[:]
            random.Random(seed).shuffle(sh)
            A, B = sh[:len(sh) // 2], sh[len(sh) // 2:]
            for tr, te in ((A, B), (B, A)):
                pick = max(conf_opts, key=lambda q: tune.evaluate(pool, tr, q)[0])
                cv.append(tune.evaluate(pool, te, pick)[0])
    return dict(PQ_in_sample=pq, PQ_split_half=float(np.mean(cv)), split_half_sd=float(np.std(cv)),
                TP=st["tp"], FP=st["fp"], FN=st["fn"], SQ=st["sq"], params=best), fused


def finals(fused, params):
    out = {}
    for s, inst in fused.items():
        ms = postprocess([(sc, decode(c)) for sc, c in inst if sc >= params["conf_thr"]], **params)
        out[s] = [encode(m) for m in ms]
    return out


def gt_hits(item):
    stem, recs, pred_sets = item
    gray = cv2.imread(str(image_path(stem)), cv2.IMREAD_GRAYSCALE)
    center, radius = disk_geometry(gray)
    rows = []
    for r, g in recs.items():
        ms = {name: (iou_matrix(g, p) if p else np.zeros((len(g), 0))) for name, p in pred_sets.items()}
        for i, c in enumerate(g):
            f = mask_features(decode(c), gray, center, radius)
            row = dict(stem=stem, rec=r, contrast=f["contrast"], r_rel=f["r_rel"], area=f["area"])
            for name, m in ms.items():
                row[name] = bool(m.shape[1] and (m[i] > 0.5).any())
            rows.append(row)
    return rows


def main():
    gt = load_val_gt()
    configs = {
        "2048 gray (y26m_2048)": ([("b", "y26m_2048")], 4),
        "2048 3-channel (y26m_2048_c3)": ([("x", "y26m_2048_c3")], 4),
        "ensemble b+a+c (baseline)": ([("b", "y26m_2048"), ("a", "y26m_1536"), ("c", "y26m_1536_cons")], 12),
        "ensemble c3+a+c (c3 replaces b)": ([("x", "y26m_2048_c3"), ("a", "y26m_1536"), ("c", "y26m_1536_cons")], 12),
        "ensemble b+a+c+c3 (c3 added)": ([("b", "y26m_2048"), ("a", "y26m_1536"), ("c", "y26m_1536_cons"), ("x", "y26m_2048_c3")], 16),
    }
    rows, pred_sets = [], {}
    for name, (runs, views) in configs.items():
        r, fused = evaluate(combine(runs), views, gt)
        rows.append(dict(model=name, **{k: v for k, v in r.items() if k != "params"},
                         best=f"conf {r['params']['conf_thr']} {r['params']['mode']} grow {r['params']['grow']} min {r['params']['min_area']}"))
        pred_sets[name] = finals(fused, r["params"])
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows)

    with Pool(12) as p:
        H = pd.DataFrame([x for rs in p.map(gt_hits, [(s, gt[s], {n: ps.get(s, []) for n, ps in pred_sets.items()})
                                                      for s in sorted(gt)]) for x in rs])
    con_b, con_l = [-1, 0.05, 0.1, 0.15, 0.2, 1], ["<0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", ">0.20"]
    r_b, r_l = [0, 0.5, 0.7, 0.85, 2], ["<0.5", "0.5-0.7", "0.7-0.85", ">0.85"]
    a_b, a_l = [0, 500, 1000, 2000, 4000, 1e9], ["<500", "500-1k", "1k-2k", "2k-4k", ">4k"]

    def by(col, bins, labels):
        d = H.assign(bin=pd.cut(H[col], bins, labels=labels))
        t = d.groupby("bin", observed=True).agg(GT=("stem", "size"), **{n: (n, "mean") for n in configs}).reset_index()
        return t.rename(columns={"bin": col})

    H.to_csv(W / "analysis" / "c2_gt_hits.csv", index=False)
    write_md("c2_eval_c3.md", [
        ("PQ: single models and ensembles (best = in-sample tuned; split-half = threshold chosen on held-out halves)", T),
        ("Recall by GT contrast", by("contrast", con_b, con_l)),
        ("Recall by GT distance from disk centre (r/R)", by("r_rel", r_b, r_l)),
        ("Recall by GT area", by("area", a_b, a_l)),
    ])


if __name__ == "__main__":
    main()
