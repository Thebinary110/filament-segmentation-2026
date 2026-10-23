"""A4. Does the ensemble/TTA fusion score (summed confidence / 12 views) penalise small
filaments that appear in only a few views?

Fusion is re-run with the same algorithm as postprocess.fuse_views (IoU>0.5 clustering,
one member per view, pixel majority) but keeping each cluster's members, so every fused
instance has: views present, models present (b = 2048, a = 1536, c = 1536-consensus),
mean / max confidence when present, area, and its value under the metric
(records hit, sum IoU - PQ*k/2). Alternative scores are then compared with the same
post-processing; best-threshold numbers are IN-SAMPLE, the split-half column is not.
"""
import random
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import ENS, ENS_VIEWS, R, decode, encode, iou_matrix, load_val_gt, gt_for_tune, write_md
import pickle
import tune  # importable once common has put src/ on sys.path

PQ0 = 0.4765


def model_of(view):
    return view[0] if view[0] in "ac" else "b"


def fuse_keep(item):
    stem, raw = item
    raw = sorted(raw, key=lambda r: -r[0])
    clusters = []  # [members(list of (conf, view)), votes, ref, bbox]
    for c, counts, view in raw:
        m = decode(counts)
        ys, xs = np.nonzero(m)
        bb = (xs.min(), ys.min(), xs.max(), ys.max())
        best, best_iou = None, 0.5
        for cl in clusters:
            if view in {v for _, v in cl[0]}:
                continue
            b = cl[3]
            if bb[2] < b[0] or b[2] < bb[0] or bb[3] < b[1] or b[3] < bb[1]:
                continue
            inter = np.logical_and(m, cl[2]).sum()
            iou = inter / (m.sum() + cl[2].sum() - inter)
            if iou > best_iou:
                best, best_iou = cl, iou
        if best is None:
            clusters.append([[(c, view)], m.astype(np.uint8), m, bb])
        else:
            best[0].append((c, view))
            best[1] += m
    out = []
    for members, votes, _, _ in clusters:
        mask = votes >= max(1, int(np.ceil(len(members) / 2)))
        if mask.any():
            confs = [c for c, _ in members]
            out.append(dict(counts=encode(mask), area=int(mask.sum()), n_views=len(members),
                            n_models=len({model_of(v) for _, v in members}),
                            sum_score=sum(confs) / ENS_VIEWS, mean_conf=float(np.mean(confs)), max_conf=max(confs)))
    return stem, out


def label(item):
    stem, inst, recs = item
    k = len(recs)
    rows = []
    for d in inst:
        n_hit, s = 0, 0.0
        for g in recs.values():
            m = iou_matrix([d["counts"]], g)
            if m.size and (m > 0.5).any():
                n_hit += 1
                s += float(m.max())
        rows.append(dict(stem=stem, k=k, n_hit=n_hit, sum_iou=s, value=s - PQ0 * k / 2, **d))
    return rows


def main():
    raw = pickle.load(open(ENS, "rb"))
    gt = load_val_gt()
    with Pool(12) as p:
        fused = dict(p.map(fuse_keep, sorted(raw.items())))
        rows = [r for rs in p.map(label, [(s, fused[s], gt[s]) for s in sorted(gt)]) for r in rs]
    F = pd.DataFrame(rows)
    F["true"] = F.n_hit > 0
    F["area_bin"] = pd.cut(F.area, [0, 500, 1000, 2000, 4000, 1e9], labels=["<500", "500-1k", "1k-2k", "2k-4k", ">4k"])
    F["views_bin"] = pd.cut(F.n_views, [0, 3, 6, 9, 12], labels=["1-3", "4-6", "7-9", "10-12"])
    cand = F[F.max_conf >= 0.1]

    # 1) views present vs size, for real (hit >=1 record) instances
    t1 = (cand[cand.true].groupby("area_bin", observed=True)
          .agg(true_instances=("n_views", "size"), mean_views=("n_views", "mean"), mean_models=("n_models", "mean"),
               mean_conf_present=("mean_conf", "mean"), mean_fused_score=("sum_score", "mean"),
               share_dropped_at_0_275=("sum_score", lambda v: (v < 0.275).mean()))
          .reset_index())
    # 2) precision / value by area and views
    t2 = (cand.groupby(["area_bin", "views_bin"], observed=True)
          .agg(instances=("true", "size"), share_true=("true", "mean"), mean_value=("value", "mean"),
               mean_conf_present=("mean_conf", "mean"))
          .reset_index())
    # 3) the specific worry: real filaments dropped although confident where present
    dropped = cand[(cand.sum_score < 0.275)]
    t3 = (dropped.assign(confident_when_present=dropped.mean_conf >= 0.4)
          .groupby(["confident_when_present", "area_bin"], observed=True)
          .agg(dropped_instances=("true", "size"), of_which_true=("true", "sum"), share_true=("true", "mean"),
               mean_views=("n_views", "mean"), mean_value=("value", "mean"))
          .reset_index())

    # 4) alternative scores, same post-processing grid
    gtt = gt_for_tune(gt)
    stems = sorted(gt)
    variants = {
        "baseline: sum conf / 12": lambda d: d["sum_score"],
        "mean conf when present": lambda d: d["mean_conf"],
        "max conf": lambda d: d["max_conf"],
        "sqrt(sum/12 * mean)": lambda d: float(np.sqrt(d["sum_score"] * d["mean_conf"])),
        "sum / 12, small (<1k px) boosted x1.3": lambda d: d["sum_score"] * (1.3 if d["area"] < 1000 else 1.0),
        "sum / (4 x models present)": lambda d: sum([d["mean_conf"]] * d["n_views"]) / (4 * d["n_models"]),
    }
    grid = [0.15, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.4, 0.45, 0.5]
    alt = []
    for name, fn in variants.items():
        fz = {s: [(fn(d), d["counts"]) for d in fused[s]] for s in fused}
        with Pool(10, initializer=tune._init, initargs=(fz, gtt)) as pool:
            res = {c: tune.evaluate(pool, stems, dict(conf_thr=c, mode="union_cc", grow=-1, min_area=150)) for c in grid}
            best_c = max(res, key=lambda c: res[c][0])
            cv = []
            for seed in range(5):
                sh = stems[:]
                random.Random(seed).shuffle(sh)
                A, B = sh[:len(sh) // 2], sh[len(sh) // 2:]
                for tr, te in ((A, B), (B, A)):
                    c = max(grid, key=lambda c: tune.evaluate(pool, tr, dict(conf_thr=c, mode="union_cc", grow=-1, min_area=150))[0])
                    cv.append(tune.evaluate(pool, te, dict(conf_thr=c, mode="union_cc", grow=-1, min_area=150))[0])
        st = res[best_c][1]
        alt.append(dict(score=name, best_thr=best_c, PQ_in_sample=res[best_c][0], PQ_split_half=np.mean(cv),
                        split_half_sd=np.std(cv), TP=st["tp"], FP=st["fp"], FN=st["fn"]))
    A = pd.DataFrame(alt)

    F.drop(columns=["counts"]).to_csv("../../work/analysis/a4_fused_instances.csv", index=False)
    write_md("a4_fusion_small.md", [
        ("Real filaments (fused instance hits >=1 record): views present and fused score by size", t1),
        ("All fused candidates (max conf >= 0.1): share real and mean metric value, by size x views present", t2),
        ("Candidates dropped by the 0.275 threshold: are they confident where present?", t3),
        ("Alternative fused scores (union_cc, erode 1, min area 150; threshold tuned; in-sample vs split-half)", A),
    ])


if __name__ == "__main__":
    main()
