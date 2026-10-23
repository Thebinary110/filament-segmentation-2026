"""A1. Where PQ is lost, stratified by annotator structure, and how much is recoverable.

Marginal value of one prediction under the pooled per-record PQ
----------------------------------------------------------------
On an image with k records, a prediction that hits n of them with IoUs q_1..q_n changes
the pooled sums by  dS = sum q_i  and  dD = 0.5 * k  (each hit record: TP+1, FN-1;
each other record: FP+1). Its effect on PQ = S/D is  (dS - PQ*dD) / D, so it is worth
keeping iff  sum q_i > PQ * k / 2,  i.e. mean matched IoU over ALL k records > PQ/2.
For a filament drawn by s of k annotators, matched at IoU q: worth predicting iff
s/k > PQ / (2q).
"""
import numpy as np
import pandas as pd

from common import (BASE, all_clusters, baseline, decode, encode, gt_for_tune, iou_matrix, load_val_gt,
                    majority_mask, pq_score, write_md)


def main():
    gt = load_val_gt()
    final, _ = baseline()
    preds = {k: [c for _, c in v] for k, v in final.items()}
    pq0, st0 = pq_score(preds, gt)
    q = st0["sq"]
    clusters = all_clusters(gt)

    # ---------------- theory table
    rows = []
    for k in (1, 2, 3):
        for s in range(1, k + 1):
            dS, dD = s * q, 0.5 * k
            rows.append(dict(k=k, s=s, frac=f"{s}/{k}", dS=dS, dD=dD,
                             net_per_pred=dS - pq0 * dD, dPQ_x1000=1000 * (dS - pq0 * dD) / st0["D"],
                             worth_it="yes" if dS > pq0 * dD else "no"))
    theory = pd.DataFrame(rows)

    # ---------------- per GT instance (record level)
    g_rows = []
    for stem, recs in gt.items():
        k = len(recs)
        p = preds.get(stem, [])
        cl = clusters[stem]
        member_of = {(r, i): ci for ci, c in enumerate(cl) for r, i in c.items()}
        hit_rec = {}
        ious = {}
        for r, g in recs.items():
            m = iou_matrix(g, p) if p else np.zeros((len(g), 0))
            ious[r] = m
            for i in range(len(g)):
                hit_rec[(r, i)] = bool(m.shape[1] and (m[i] > 0.5).any())
        for r, g in recs.items():
            m = ious[r]
            for i in range(len(g)):
                ci = member_of[(r, i)]
                s = len(cl[ci])
                cl_hits = sum(hit_rec[(rr, ii)] for rr, ii in cl[ci].items())
                mx = float(m[i].max()) if m.shape[1] else 0.0
                n_ov = int((m[i] > 0.02).sum()) if m.shape[1] else 0
                if hit_rec[(r, i)]:
                    cat = "hit"
                elif cl_hits > 0:
                    cat = "FN: filament hit for another annotator (shape disagreement)"
                elif mx >= 0.3:
                    cat = "FN: near miss 0.3-0.5"
                elif n_ov >= 2:
                    cat = "FN: split over >=2 preds"
                elif n_ov == 1 and mx >= 0.02:
                    pj = int(m[i].argmax())
                    covers = sum(int((ious[rr][:, pj] > 0.02).sum()) for rr in [r])
                    cat = "FN: inside a pred covering >=2 GT" if covers >= 2 else "FN: weak overlap <0.3"
                else:
                    cat = "FN: never detected"
                g_rows.append(dict(stem=stem, rec=r, k=k, s=s, cluster=ci, hit=hit_rec[(r, i)], cat=cat,
                                   area=int(decode(g[i]).sum())))
    G = pd.DataFrame(g_rows)
    G["s/k"] = G.s.astype(str) + "/" + G.k.astype(str)

    by_sk = G.groupby(["k", "s"]).agg(GT=("hit", "size"), hits=("hit", "sum")).reset_index()
    by_sk["FN"] = by_sk.GT - by_sk.hits
    by_sk["recall"] = by_sk.hits / by_sk.GT
    by_sk["median_area"] = G.groupby(["k", "s"]).area.median().values
    fn_cat = G[~G.hit].groupby(["cat"]).size().rename("FN").reset_index().sort_values("FN", ascending=False)
    fn_cat_sk = (G[~G.hit].pivot_table(index="cat", columns="s/k", values="hit", aggfunc="size", fill_value=0)
                 .reset_index())

    # ---------------- per final prediction
    p_rows = []
    for stem, recs in gt.items():
        k = len(recs)
        for j, (sc, c) in enumerate(final.get(stem, [])):
            n_hit, sum_iou, best_any = 0, 0.0, 0.0
            for r, g in recs.items():
                m = iou_matrix([c], g)
                if m.size:
                    best_any = max(best_any, float(m.max()))
                    if (m > 0.5).any():
                        n_hit += 1
                        sum_iou += float(m[m > 0.5].sum())
            p_rows.append(dict(stem=stem, j=j, k=k, n_hit=n_hit, sum_iou=sum_iou, score=sc,
                               value=sum_iou - pq0 * k / 2, best_iou_any=best_any))
    P = pd.DataFrame(p_rows)
    P["FP_generated"] = P.k - P.n_hit
    by_hit = (P.groupby(["k", "n_hit"]).agg(preds=("value", "size"), mean_sum_iou=("sum_iou", "mean"),
                                            FP_generated=("FP_generated", "sum"), total_value=("value", "sum"))
              .reset_index())
    by_hit["dPQ_if_removed"] = -by_hit.total_value / st0["D"]

    def fp_kind(row):
        if row.n_hit > 0:
            return "pred hits some records, FP in the others (partial support)"
        if row.best_iou_any >= 0.3:
            return "pred hits nobody: near miss (0.3-0.5) to some GT"
        if row.best_iou_any >= 0.02:
            return "pred hits nobody: weak overlap / fragment"
        return "pred hits nobody: no annotator drew anything there"
    P["fp_kind"] = P.apply(fp_kind, axis=1)
    fp_tax = P.groupby("fp_kind").agg(preds=("FP_generated", "size"), FP=("FP_generated", "sum")).reset_index()

    # ---------------- oracle experiments (real scorer)
    def score_with(pmap):
        return pq_score(pmap, gt)

    oracle = []
    oracle.append(("baseline", pq0, st0))
    keep = {s: [c for (sc, c), v in zip(final.get(s, []), P[P.stem == s].value)] for s in gt}
    pr = {s: [c for (sc, c), (_, row) in zip(final.get(s, []), P[P.stem == s].iterrows()) if row.value > 0] for s in gt}
    oracle.append(("oracle prune: drop every pred with negative marginal value", *score_with(pr)))
    pr2 = {s: [c for (sc, c), (_, row) in zip(final.get(s, []), P[P.stem == s].iterrows()) if row.n_hit > 0] for s in gt}
    oracle.append(("oracle prune: drop preds that hit no record", *score_with(pr2)))

    # add missed clusters (no record hit) as oracle majority masks, by support fraction
    missed = []
    for stem, recs in gt.items():
        k = len(recs)
        hits = set(zip(G[(G.stem == stem) & G.hit].rec, G[(G.stem == stem) & G.hit].cluster))
        for ci, cl in enumerate(clusters[stem]):
            if not any((r, ci) in hits for r in cl):
                missed.append((stem, ci, len(cl), k))
    M = pd.DataFrame(missed, columns=["stem", "cluster", "s", "k"])
    M["frac"] = M.s / M.k
    masks = {(s, ci): encode(majority_mask(gt[s], clusters[s][ci])) for s, ci, _, _ in missed}
    for label, sel in [("s/k = 1", M.frac == 1), ("s/k >= 2/3", M.frac >= 2 / 3 - 1e-9),
                       ("s/k >= 1/2", M.frac >= 0.5), ("all missed clusters", M.frac > 0),
                       ("only s/k = 1/3", (M.s == 1) & (M.k == 3)), ("only s/k = 1/2", (M.s == 1) & (M.k == 2))]:
        add = {s: list(v) for s, v in preds.items()}
        for _, row in M[sel].iterrows():
            add.setdefault(row.stem, []).append(masks[(row.stem, row.cluster)])
        pq, st = score_with(add)
        # realistic: the added prediction matches each member at the model's typical SQ instead of ~1.0
        n_mem, n_oth = int(M[sel].s.sum()), int((M[sel].k - M[sel].s).sum())
        S_r = st0["sum_iou"] + q * n_mem
        D_r = st0["D"] + 0.5 * (n_mem + n_oth)
        oracle.append((f"oracle add missed clusters, {label} ({int(sel.sum())} clusters)", pq, st, S_r / D_r))
    O = pd.DataFrame([dict(experiment=o[0], PQ=o[1], dPQ=o[1] - pq0, TP=o[2]["tp"], FP=o[2]["fp"], FN=o[2]["fn"],
                           SQ=o[2]["sq"], realistic_PQ_at_model_SQ=(o[3] if len(o) > 3 else np.nan))
                      for o in oracle])
    missed_tab = M.groupby(["k", "s"]).size().rename("missed_clusters").reset_index()

    P.to_csv("../../work/analysis/a1_predictions.csv", index=False)
    G.to_csv("../../work/analysis/a1_gt_instances.csv", index=False)
    write_md("a1_metric_strata.md", [
        ("Baseline", f"PQ {pq0:.4f}; TP {st0['tp']}, FP {st0['fp']}, FN {st0['fn']}, SQ {q:.4f}, D {st0['D']:.1f}"),
        ("Theory: is a prediction for a filament drawn by s of k annotators worth making? (q = model SQ)", theory),
        ("GT instances by (k, s): recall", by_sk),
        ("FN taxonomy (all)", fn_cat),
        ("FN taxonomy by s/k", fn_cat_sk),
        ("Missed clusters (no annotator's version hit) by (k, s)", missed_tab),
        ("Final predictions by (k, records hit): value = sum IoU - PQ*k/2", by_hit),
        ("Where the 291 FP come from", fp_tax),
        ("Oracle experiments (real scorer; upper bounds, NOT achievable as-is)", O),
    ])


if __name__ == "__main__":
    main()
