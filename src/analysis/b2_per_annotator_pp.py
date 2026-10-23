"""B2. Post-processing tuned separately for hard vs easy annotators.

Annotators (record-id prefixes) are split at the median of their per-prefix baseline PQ
on validation (all prefixes, any number of records). The post-processing grid is then
tuned on each subset, scoring only that subset's records. NOTE: the split itself is
defined from the baseline's errors on the same images, so both optima are in-sample
and the gap between them is optimistic. If the two optimal thresholds differ by more
than 0.05, submission CSVs are written for both settings, plus one at conf 0.35.
"""
import itertools
import json
import pickle
from multiprocessing import Pool

import pandas as pd

from common import ENS_VIEWS, ROOT, TEST_IMG, W, baseline, gt_for_tune, load_val_gt, pq_score, write_md
import tune  # noqa: E402  (common puts src/ on sys.path)


def subset(gt, keep):
    out = {}
    for stem, recs in gt.items():
        rr = {r: g for r, g in recs.items() if keep(r)}
        if rr:
            out[stem] = rr
    return out


def main():
    gt = load_val_gt()
    final, fused = baseline()
    preds = {k: [c for _, c in v] for k, v in final.items()}
    prefixes = sorted({r[:6] for recs in gt.values() for r in recs})
    per = []
    for p in prefixes:
        pq, st = pq_score(preds, subset(gt, lambda r, p=p: r[:6] == p))
        per.append(dict(prefix=p, records=sum(r[:6] == p for recs in gt.values() for r in recs), PQ=pq))
    per = pd.DataFrame(per)
    med = per.PQ.median()
    hard = set(per[per.PQ < med].prefix)
    sets = {"hard annotators": subset(gt, lambda r: r[:6] in hard),
            "easy annotators": subset(gt, lambda r: r[:6] not in hard),
            "all": gt}

    grid = [dict(conf_thr=c, mode=m, grow=g, min_area=a)
            for c, m, g, a in itertools.product([0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5],
                                                ["union_cc", "paint"], [-1, 0], [0, 150, 300])]
    rows, best = [], {}
    for name, g in sets.items():
        gtt = gt_for_tune(g)
        stems = sorted(gtt)
        with Pool(12, initializer=tune._init, initargs=(fused, gtt)) as pool:
            res = [(tune.evaluate(pool, stems, p), p) for p in grid]
        (pq, st), p = max(res, key=lambda x: x[0][0])
        base_pq = next(r[0][0] for r in res if r[1] == dict(conf_thr=0.275, mode="union_cc", grow=-1, min_area=150))
        conf_curve = {c: max(r[0][0] for r in res if r[1]["conf_thr"] == c) for c in sorted({q["conf_thr"] for q in grid})}
        best[name] = p
        rows.append(dict(subset=name, records=sum(len(v) for v in g.values()), best_PQ_in_sample=pq,
                         PQ_at_baseline_settings=base_pq, **{f"best_{k}": v for k, v in p.items()},
                         TP=st["tp"], FP=st["fp"], FN=st["fn"]))
        rows[-1]["conf_curve"] = " ".join(f"{c}:{v:.3f}" for c, v in conf_curve.items())
    B = pd.DataFrame(rows)

    notes = []
    diff = abs(best["hard annotators"]["conf_thr"] - best["easy annotators"]["conf_thr"])
    test_raw = pickle.load(open(W / "test_ens_b_a_c.pkl", "rb"))
    to_write = [("conf035", dict(conf_thr=0.35, mode="union_cc", grow=-1, min_area=150))]
    if diff > 0.05:
        to_write += [("hardopt", best["hard annotators"]), ("easyopt", best["easy annotators"])]
    notes.append(f"Optimal conf: hard {best['hard annotators']['conf_thr']}, easy {best['easy annotators']['conf_thr']} "
                 f"(difference {diff:.3f}; threshold for writing per-subset CSVs: 0.05).")
    with Pool(12) as pool:
        test_fused = dict(pool.map(tune._fuse, [(k, v, ENS_VIEWS) for k, v in test_raw.items()]))
    from postprocess import decode, encode, postprocess
    expected = {p.stem for p in TEST_IMG.iterdir()}
    assert set(test_fused) == expected
    for tag, p in to_write:
        out = []
        for stem in sorted(test_fused):
            inst = [(s, decode(c)) for s, c in test_fused[stem] if s >= p["conf_thr"]]
            for i, m in enumerate(postprocess(inst, **p), 1):
                out.append((f"{stem}_{i}", encode(m)))
        f = ROOT / f"submission_ens_b_a_c_{tag}.csv"
        pd.DataFrame(out, columns=["filament_id", "segmentation_rle"]).to_csv(f, index=False)
        notes.append(f"wrote {f.name}: {json.dumps(p)}, {len(out)} filaments over "
                     f"{len({o[0].rsplit('_', 1)[0] for o in out})} images")

    write_md("b2_per_annotator_pp.md", [
        ("Per-prefix baseline PQ (median split defines hard/easy)", per.sort_values("PQ")),
        ("Post-processing tuned per subset (IN-SAMPLE; the hard/easy split is itself defined from these errors)", B),
        ("Submission files", "\n".join(f"- {n}" for n in notes)),
    ])


if __name__ == "__main__":
    main()
