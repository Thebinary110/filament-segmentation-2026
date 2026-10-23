"""A5. Systematic patterns in the baseline's errors: size, contrast, distance from disk
centre, observatory, annotation group / annotator.

GT side: recall per GT instance (hit = some prediction at IoU > 0.5), with the instance's
own features. Prediction side: per final prediction, share of the image's records it hits
and its metric value. Per annotator: pooled PQ over only that annotator's records.
"""
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from common import (SITES, baseline, decode, disk_geometry, image_path, iou_matrix, load_val_gt, mask_features,
                    pq_score, write_md)

PQ0 = 0.4765


def per_image(item):
    stem, recs, preds = item
    gray = cv2.imread(str(image_path(stem)), cv2.IMREAD_GRAYSCALE)
    center, radius = disk_geometry(gray)
    k = len(recs)
    g_rows, p_rows = [], []
    for r, g in recs.items():
        m = iou_matrix(g, preds) if preds else np.zeros((len(g), 0))
        for i, c in enumerate(g):
            f = mask_features(decode(c), gray, center, radius)
            g_rows.append(dict(stem=stem, rec=r, k=k, hit=bool(m.shape[1] and (m[i] > 0.5).any()), **f))
    for c in preds:
        n_hit, s = 0, 0.0
        for g in recs.values():
            m = iou_matrix([c], g)
            if m.size and (m > 0.5).any():
                n_hit += 1
                s += float(m.max())
        f = mask_features(decode(c), gray, center, radius)
        p_rows.append(dict(stem=stem, k=k, n_hit=n_hit, hit_share=n_hit / k, value=s - PQ0 * k / 2, **f))
    return g_rows, p_rows


def binned(df, col, bins, labels, target, extra=None):
    d = df.assign(bin=pd.cut(df[col], bins, labels=labels))
    agg = dict(n=(target, "size"), rate=(target, "mean"))
    if extra:
        agg.update(extra)
    return d.groupby("bin", observed=True).agg(**agg).reset_index().rename(columns={"bin": col})


def main():
    gt = load_val_gt()
    final, _ = baseline()
    preds = {k: [c for _, c in v] for k, v in final.items()}
    with Pool(12) as p:
        res = p.map(per_image, [(s, gt[s], preds.get(s, [])) for s in sorted(gt)])
    G = pd.DataFrame([r for g, _ in res for r in g])
    P = pd.DataFrame([r for _, p in res for r in p])
    G["site"] = G.stem.str[14]
    P["site"] = P.stem.str[14]
    G["group"] = G.rec.str[:2]
    G["annotator"] = G.rec.str[:6]

    area_b, area_l = [0, 500, 1000, 2000, 4000, 1e9], ["<500", "500-1k", "1k-2k", "2k-4k", ">4k"]
    con_b, con_l = [-1, 0.05, 0.1, 0.15, 0.2, 1], ["<0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", ">0.20"]
    r_b, r_l = [0, 0.5, 0.7, 0.85, 0.95, 2], ["<0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", ">0.95 (limb)"]
    vext = dict(mean_value=("value", "mean"), total_value=("value", "sum"))
    sections = [
        ("Recall of GT instances by area (px)", binned(G, "area", area_b, area_l, "hit")),
        ("Recall of GT instances by contrast ((ring - inside) / ring)", binned(G, "contrast", con_b, con_l, "hit")),
        ("Recall of GT instances by distance from disk centre (r/R)", binned(G, "r_rel", r_b, r_l, "hit")),
        ("Predictions: share of records hit and metric value, by area", binned(P, "area", area_b, area_l, "hit_share", vext)),
        ("Predictions: by contrast", binned(P, "contrast", con_b, con_l, "hit_share", vext)),
        ("Predictions: by r/R", binned(P, "r_rel", r_b, r_l, "hit_share", vext)),
    ]
    # recall by area x contrast (where are the never-found filaments?)
    ac = (G.assign(area_bin=pd.cut(G.area, area_b, labels=area_l), con_bin=pd.cut(G.contrast, con_b, labels=con_l))
          .pivot_table(index="area_bin", columns="con_bin", values="hit", aggfunc="mean", observed=True).reset_index())
    ac.columns = [str(c) for c in ac.columns]
    sections.append(("Recall by area (rows) x contrast (columns)", ac))

    site_rows = []
    for s in "BCLMTU":
        stems = [x for x in gt if x[14] == s]
        pq, st = pq_score(preds, gt, stems)
        site_rows.append(dict(site=f"{s} {SITES[s]}", images=len(stems), PQ=pq, SQ=st["sq"], TP=st["tp"], FP=st["fp"], FN=st["fn"],
                              GT_per_record=(st["tp"] + st["fn"]) / sum(len(gt[x]) for x in stems)))
    sections.append(("PQ by observatory", pd.DataFrame(site_rows)))

    def pq_subset(sel):
        sub = {}
        for stem, recs in gt.items():
            rr = {r: g for r, g in recs.items() if sel(r)}
            if rr:
                sub[stem] = rr
        return pq_score(preds, sub)

    grp = []
    for g in sorted(G.group.unique()):
        pq, st = pq_subset(lambda r, g=g: r[:2] == g)
        grp.append(dict(group=g, records=int(G[G.group == g].rec.nunique()), PQ=pq, SQ=st["sq"], recall=st["tp"] / (st["tp"] + st["fn"]),
                        TP=st["tp"], FP=st["fp"], FN=st["fn"],
                        GT_per_record=(st["tp"] + st["fn"]) / max(1, G[G.group == g].rec.nunique()),
                        median_GT_area=G[G.group == g].area.median()))
    sections.append(("PQ by annotation group (first 2 digits of record id), scoring only that group's records", pd.DataFrame(grp)))
    ann = []
    for a in sorted(G.annotator.unique()):
        n = G[G.annotator == a].rec.nunique()
        if n < 4:
            continue
        pq, st = pq_subset(lambda r, a=a: r[:6] == a)
        ann.append(dict(record_prefix=a, records=n, PQ=pq, recall=st["tp"] / (st["tp"] + st["fn"]),
                        precision=st["tp"] / max(1, st["tp"] + st["fp"]), GT_per_record=(st["tp"] + st["fn"]) / n))
    A = pd.DataFrame(ann).sort_values("PQ")
    sections.append(("PQ per record-id prefix (>= 4 records), sorted", A))
    sections.append(("Spread of per-prefix PQ", A.PQ.describe()[["count", "min", "25%", "50%", "75%", "max"]].rename("PQ").reset_index()))

    G.to_csv("../../work/analysis/a5_gt_features.csv", index=False)
    P.to_csv("../../work/analysis/a5_pred_features.csv", index=False)
    write_md("a5_patterns.md", sections)


if __name__ == "__main__":
    main()
