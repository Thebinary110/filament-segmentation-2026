"""D1. Re-rank the near-threshold band with the semantic agreement feature.

Keep rule (2 parameters, (w, t)):
  conf >= 0.45 -> keep;  conf < 0.15 -> drop;
  0.15 <= conf < 0.45 -> keep iff conf + w * (agree - 0.5) >= t
w = 0 is the baseline threshold rule. Kept candidates go through the unchanged
post-processing (union_cc, erode 1 px, min area 150).

Per-image PQ components are computed once per (w, t), so split-half selection is exact:
choose (w, t) on one half of the validation images, score on the other (both
directions, 5 random halvings); the baseline (w = 0, t chosen on the same half) is scored
identically. In-sample gains under +0.01 are treated as noise. A test submission is
written only if the held-out re-ranked PQ beats the held-out baseline.
"""
import argparse
import itertools
import pickle
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import ROOT, W, gt_for_tune, iou_matrix, load_val_gt, write_md
import tune  # noqa: E402
from postprocess import decode, encode, postprocess

LO, HI, PQ0 = 0.15, 0.45, 0.4765
POST = dict(mode="union_cc", grow=-1, min_area=150)
WS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
TS = [0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35]


def keep(s, a, w, t):
    return s >= HI or (LO <= s < HI and s + w * (a - 0.5) >= t)


def per_image_stats(feats, gtt, w, t):
    fz = {s: [(1.0 if keep(sc, a, w, t) else 0.0, c) for sc, c, a in lst] for s, lst in feats.items()}
    stems = sorted(gtt)
    with Pool(12, initializer=tune._init, initargs=(fz, gtt)) as pool:
        res = pool.map(tune._score_image, [(s, dict(conf_thr=0.5, **POST)) for s in stems])
    return dict(zip(stems, res))


def pq(stats, stems):
    s = tp = fp = fn = 0
    for st in stems:
        a, b, c, d = stats[st]
        s, tp, fp, fn = s + a, tp + b, fp + c, fn + d
    return s / (tp + 0.5 * fp + 0.5 * fn)


def auc(score, label):
    score, label = np.asarray(score), np.asarray(label).astype(bool)
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return np.nan
    ranks = pd.Series(np.r_[pos, neg]).rank().values
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=str(W / "semantic_val_feats.pkl"))
    ap.add_argument("--test", default=str(W / "semantic_test_feats.pkl"))
    args = ap.parse_args()
    feats = pickle.load(open(args.val, "rb"))
    gt = load_val_gt()
    gtt = gt_for_tune(gt)
    stems = sorted(gtt)

    # separability inside the band (labels: mean matched IoU across annotators > PQ/2)
    rows = []
    for stem, lst in feats.items():
        recs = gt[stem]
        for sc, c, a in lst:
            if not (LO <= sc < HI):
                continue
            tot = 0.0
            for g in recs.values():
                m = iou_matrix([c], g)
                if m.size and (m > 0.5).any():
                    tot += float(m.max())
            rows.append(dict(conf=sc, agree=a, real=tot / len(recs) > PQ0 / 2))
    B = pd.DataFrame(rows)
    sep = pd.DataFrame([dict(band_candidates=len(B), share_real=B.real.mean(), AUC_conf=auc(B.conf, B.real),
                             AUC_agree=auc(B.agree, B.real),
                             AUC_conf_plus_agree=auc(B.conf + 0.5 * (B.agree - 0.5), B.real),
                             mean_agree_real=B.agree[B.real].mean(), mean_agree_not_real=B.agree[~B.real].mean())])

    grid = {(w, t): per_image_stats(feats, gtt, w, t) for w, t in itertools.product(WS, TS)}
    ins = pd.DataFrame([dict(w=w, t=t, PQ_in_sample=pq(st, stems)) for (w, t), st in grid.items()])
    best_all = ins.loc[ins.PQ_in_sample.idxmax()]
    best_base = ins[ins.w == 0].loc[ins[ins.w == 0].PQ_in_sample.idxmax()]

    held = []
    for seed in range(5):
        sh = stems[:]
        random.Random(seed).shuffle(sh)
        A, Bh = sh[:len(sh) // 2], sh[len(sh) // 2:]
        for tr, te in ((A, Bh), (Bh, A)):
            r_key = max(grid, key=lambda k: pq(grid[k], tr))
            b_key = max([k for k in grid if k[0] == 0], key=lambda k: pq(grid[k], tr))
            held.append(dict(seed=seed, rerank_w=r_key[0], rerank_t=r_key[1], rerank_PQ=pq(grid[r_key], te),
                             base_t=b_key[1], base_PQ=pq(grid[b_key], te)))
    H = pd.DataFrame(held)
    H["delta"] = H.rerank_PQ - H.base_PQ
    summary = pd.DataFrame([
        dict(rule="baseline (w = 0)", in_sample_PQ=best_base.PQ_in_sample, in_sample_params=f"t {best_base.t}",
             held_out_PQ=H.base_PQ.mean(), held_out_sd=H.base_PQ.std()),
        dict(rule="re-ranked (conf + w*(agree-0.5))", in_sample_PQ=best_all.PQ_in_sample,
             in_sample_params=f"w {best_all.w}, t {best_all.t}", held_out_PQ=H.rerank_PQ.mean(), held_out_sd=H.rerank_PQ.std()),
    ])
    verdict = (f"held-out delta {H.delta.mean():+.4f} (sd {H.delta.std():.4f}, {int((H.delta > 0).sum())}/10 halves positive); "
               f"in-sample delta {best_all.PQ_in_sample - best_base.PQ_in_sample:+.4f} "
               f"({'below' if best_all.PQ_in_sample - best_base.PQ_in_sample < 0.01 else 'above'} the +0.01 noise bar)")

    notes = [verdict]
    if H.rerank_PQ.mean() > H.base_PQ.mean() and best_all.w > 0:
        test = pickle.load(open(args.test, "rb"))
        out = []
        for stem in sorted(test):
            inst = [(1.0, decode(c)) for sc, c, a in test[stem] if keep(sc, a, best_all.w, best_all.t)]
            for i, m in enumerate(postprocess(inst, conf_thr=0.5, **POST), 1):
                out.append((f"{stem}_{i}", encode(m)))
        name = f"submission_ens_b_a_c_rerank_w{best_all.w:g}_t{best_all.t:g}.csv"
        pd.DataFrame(out, columns=["filament_id", "segmentation_rle"]).to_csv(ROOT / name, index=False)
        notes.append(f"wrote {name} ({len(out)} filaments over {len({o[0].rsplit('_', 1)[0] for o in out})} images; "
                     f"params chosen on all validation images)")
    else:
        notes.append("no submission written (held-out re-ranked PQ does not beat the held-out baseline)")

    write_md("d1_rerank.md", [
        ("Separability inside the band [0.15, 0.45) (AUC for 'real'; validation, out-of-sample semantic maps)", sep),
        ("Keep-rule results (in-sample = tuned on all validation; held-out = split-half, 5 halvings x 2)", summary),
        ("Split-half detail", H),
        ("In-sample grid", ins.pivot(index="w", columns="t", values="PQ_in_sample").reset_index()),
        ("Verdict", "\n".join(f"- {n}" for n in notes)),
    ])


if __name__ == "__main__":
    main()
