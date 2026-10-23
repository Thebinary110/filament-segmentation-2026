"""B3. The near-threshold band: size, share real, and the ceiling of a perfect re-ranker.

Candidates = fused ensemble instances (12 views, before union/erosion). A candidate is
"real" if its mean matched IoU across the image's annotator records exceeds PQ/2, i.e.
it has positive marginal value under the pooled metric. Ceiling: keep everything with
conf >= 0.45, drop everything < 0.15, and inside [0.15, 0.45) keep exactly the real
candidates (oracle labels), then apply the unchanged post-processing (union_cc, erode
1 px, min area 150). This bounds what any re-ranker of that band can reach.
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import baseline, gt_for_tune, iou_matrix, load_val_gt, write_md
import tune  # noqa: E402

PQ0 = 0.4765
LO, HI = 0.15, 0.45


def label(item):
    stem, inst, recs = item
    k = len(recs)
    out = []
    for s, c in inst:
        tot = 0.0
        for g in recs.values():
            m = iou_matrix([c], g)
            if m.size and (m > 0.5).any():
                tot += float(m.max())
        out.append((s, c, tot / k))
    return stem, out


def main():
    gt = load_val_gt()
    _, fused = baseline()
    with Pool(12) as p:
        lab = dict(p.map(label, [(s, fused[s], gt[s]) for s in sorted(gt)]))
    rows = [dict(stem=s, score=sc, mean_iou=m, real=m > PQ0 / 2) for s, lst in lab.items() for sc, _, m in lst]
    D = pd.DataFrame(rows)
    D["band"] = pd.cut(D.score, [0, 0.05, 0.15, 0.25, 0.275, 0.35, 0.45, 1.01],
                       labels=["<0.05", "0.05-0.15", "0.15-0.25", "0.25-0.275", "0.275-0.35", "0.35-0.45", ">=0.45"],
                       right=False)
    tab = D.groupby("band", observed=True).agg(candidates=("real", "size"), real=("real", "sum"),
                                              share_real=("real", "mean")).reset_index()

    gtt = gt_for_tune(gt)
    stems = sorted(gtt)
    P = dict(mode="union_cc", grow=-1, min_area=150)

    def run(select):
        fz = {s: [(1.0, c) for sc, c, m in lst if select(sc, m)] for s, lst in lab.items()}
        with Pool(12, initializer=tune._init, initargs=(fz, gtt)) as pool:
            return tune.evaluate(pool, stems, dict(conf_thr=0.5, **P))

    exps = [
        ("baseline (conf >= 0.275)", lambda sc, m: sc >= 0.275),
        (f"perfect re-ranker of [{LO}, {HI}): keep >= {HI} + real band candidates", lambda sc, m: sc >= HI or (LO <= sc < HI and m > PQ0 / 2)),
        (f"perfect re-ranker of [{LO}, {HI}) applied only above 0.275 (pure FP removal)", lambda sc, m: sc >= HI or (0.275 <= sc < HI and m > PQ0 / 2)),
        (f"perfect re-ranker of [{LO}, 0.275) only (pure recall add)", lambda sc, m: sc >= 0.275 or (LO <= sc < 0.275 and m > PQ0 / 2)),
        ("perfect re-ranker of all candidates >= 0.05", lambda sc, m: sc >= 0.05 and m > PQ0 / 2),
    ]
    rows = []
    for name, sel in exps:
        pq, st = run(sel)
        rows.append(dict(experiment=name, PQ=pq, dPQ=pq - PQ0, TP=st["tp"], FP=st["fp"], FN=st["fn"]))
    E = pd.DataFrame(rows)
    band = D[(D.score >= LO) & (D.score < HI)]
    summary = (f"Band [{LO}, {HI}): {len(band)} candidates, {int(band.real.sum())} real ({band.real.mean():.1%}). "
               f"Of these, {int((band.score >= 0.275).sum())} are currently kept (>= 0.275), "
               f"{int(((band.score >= 0.275) & band.real).sum())} of them real; "
               f"{int((band.score < 0.275).sum())} are currently dropped, {int(((band.score < 0.275) & band.real).sum())} of them real.")
    write_md("b3_band_ceiling.md", [
        ("Band summary", summary),
        ("Fused candidates by confidence band (real = mean matched IoU across annotators > PQ/2)", tab),
        ("Oracle re-ranking ceilings (labels used; upper bounds)", E),
    ])


if __name__ == "__main__":
    main()
