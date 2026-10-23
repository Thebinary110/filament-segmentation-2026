"""F1. How much does overlap-only association lose? (cached tile predictions; no training)

Tile candidates = the zoomed 3x3 tiled pass of the 2048 model (val_tiles_2048m_z15.pkl).
Each candidate at conf >= T is related to the baseline ensemble's final masks by
  containment = |cand & existing| / |cand|   and   iou = max IoU with one existing mask,
and to the annotations by whether it matches (IoU > 0.5) a GT instance of some record that
the ensemble did NOT already hit. Counts:
  * wrongly discarded: overlap rule (containment > 0.1) drops it, yet it is a NEW true
    match (a distinct small filament next to / touching an existing one);
  * wrongly kept: overlap rule keeps it (containment <= 0.1), yet it matches nothing new
    and sits next to an existing mask (dilated by 8 px): a fragment / extension of an
    existing filament rather than a new one.
Also: how often annotated filaments of one record touch each other at all (adjacency),
which bounds the "distinct but touching" case.
"""
import pickle
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from common import W, baseline, decode, iou_matrix, load_val_gt, write_md

T_LIST = [0.3, 0.4, 0.5]
K8 = np.ones((17, 17), np.uint8)


def main():
    gt = load_val_gt()
    final, _ = baseline()
    tiles = pickle.load(open(W / "val_tiles_2048m_z15.pkl", "rb"))
    rows = []
    for stem, recs in gt.items():
        fin = [c for _, c in final.get(stem, [])]
        fm = [decode(c) for c in fin]
        union = np.zeros((2048, 2048), bool)
        for m in fm:
            union |= m
        near = cv2.dilate(union.astype(np.uint8), K8).astype(bool)
        # GT instances already hit by the ensemble, per record
        hit = {}
        for r, g in recs.items():
            m = iou_matrix(g, fin) if fin else np.zeros((len(g), 0))
            hit[r] = [bool(m.shape[1] and (m[i] > 0.5).any()) for i in range(len(g))]
        for conf, c, _ in tiles.get(stem, []):
            if conf < min(T_LIST):
                continue
            m = decode(c)
            a = m.sum()
            cont = (m & union).sum() / a
            iou_ex = float(iou_matrix([c], fin).max()) if fin else 0.0
            new_match = old_match = False
            for r, g in recs.items():
                mm = iou_matrix([c], g)[0] if g else np.zeros(0)
                for i in np.flatnonzero(mm > 0.5):
                    if hit[r][i]:
                        old_match = True
                    else:
                        new_match = True
            rows.append(dict(stem=stem, conf=conf, area=int(a), containment=cont, iou_existing=iou_ex,
                             new_match=new_match, old_match=old_match, adjacent=bool((m & near).any()) and cont <= 0.1))
    D = pd.DataFrame(rows)
    out = []
    for t in T_LIST:
        d = D[D.conf >= t]
        dropped, kept = d[d.containment > 0.1], d[d.containment <= 0.1]
        out.append(dict(conf_threshold=t, candidates=len(d),
                        dropped_by_overlap=len(dropped),
                        dropped_but_new_true_match=int(dropped.new_match.sum()),
                        dropped_duplicates_of_existing=int((dropped.iou_existing > 0.5).sum()),
                        kept_as_new=len(kept), kept_new_true=int(kept.new_match.sum()),
                        kept_adjacent_no_match=int((kept.adjacent & ~kept.new_match).sum()),
                        kept_isolated_no_match=int((~kept.adjacent & ~kept.new_match).sum())))
    S = pd.DataFrame(out)
    lost = D[(D.containment > 0.1) & D.new_match]
    lost_tab = (lost.assign(cont_bin=pd.cut(lost.containment, [0.1, 0.3, 0.5, 0.8, 1.0001], labels=["0.1-0.3", "0.3-0.5", "0.5-0.8", ">0.8"]))
                .groupby("cont_bin", observed=True).agg(n=("conf", "size"), median_area=("area", "median"),
                                                       median_iou_existing=("iou_existing", "median")).reset_index())

    # adjacency of annotated filaments within one record
    touch = tot = 0
    for recs in gt.values():
        for g in recs.values():
            ms = [cv2.dilate(decode(c).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) for c in g]
            for i, m in enumerate(ms):
                tot += 1
                touch += any((m & ms[j]).any() for j in range(len(ms)) if j != i)
    adj = pd.DataFrame([dict(GT_instances=tot, touching_another_GT_of_same_record_within_2px=touch, share=touch / tot)])
    write_md("f1_association_check.md", [
        ("Overlap-only association (drop if > 10% of the candidate lies inside an existing mask)", S),
        ("Wrongly discarded candidates (new true matches) by containment", lost_tab),
        ("How often annotated filaments touch each other (same annotator record)", adj),
    ])


if __name__ == "__main__":
    main()
