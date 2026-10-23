"""E1. RF-DETR-Seg Large alone and as a 4th ensemble member (same TTA / fusion /
post-processing / split-half machinery as c2_eval_c3). Keep rule: the 4-member ensemble
must gain >= +0.005 held-out (split-half) over the baseline 3-member ensemble. The
decision is written to work/rfdetr_decision.json for the next queue step; a submission
CSV is written for any configuration that beats the baseline held-out.
"""
import json
import pickle
from multiprocessing import Pool

import pandas as pd

from c2_eval_c3 import combine, evaluate, finals  # same grid and split-half protocol
from common import ROOT, W, load_val_gt, write_md
import tune  # noqa: E402
from postprocess import decode, encode, postprocess

KEEP_MARGIN = 0.005


def test_combine(runs):
    out, ref = {}, runs[0][0]
    for tag, run in runs:
        for k, v in pickle.load(open(W / f"test_{run}.pkl", "rb")).items():
            out.setdefault(k, []).extend((c, m, vw if tag == ref else f"{tag}{vw}") for c, m, vw in v)
    return out


def write_submission(runs, views, params, name):
    raw = test_combine(runs)
    with Pool(12) as p:
        fused = dict(p.map(tune._fuse, [(k, v, views) for k, v in raw.items()]))
    out = []
    for stem in sorted(fused):
        inst = [(s, decode(c)) for s, c in fused[stem] if s >= params["conf_thr"]]
        out += [(f"{stem}_{i}", encode(m)) for i, m in enumerate(postprocess(inst, **params), 1)]
    pd.DataFrame(out, columns=["filament_id", "segmentation_rle"]).to_csv(ROOT / name, index=False)
    return f"wrote {name}: {len(out)} filaments over {len({o[0].rsplit('_', 1)[0] for o in out})} images, params {params}"


def main():
    gt = load_val_gt()
    configs = {
        "RF-DETR-Seg L alone": ([("r", "rfdetr_l768")], 4),
        "ensemble b+a+c (baseline)": ([("b", "y26m_2048"), ("a", "y26m_1536"), ("c", "y26m_1536_cons")], 12),
        "ensemble b+a+c+r (RF-DETR added)": ([("b", "y26m_2048"), ("a", "y26m_1536"), ("c", "y26m_1536_cons"), ("r", "rfdetr_l768")], 16),
        "ensemble r+a+c (RF-DETR replaces 2048)": ([("r", "rfdetr_l768"), ("a", "y26m_1536"), ("c", "y26m_1536_cons")], 12),
    }
    rows, res = [], {}
    for name, (runs, views) in configs.items():
        r, _ = evaluate(combine(runs), views, gt)
        res[name] = r
        rows.append(dict(model=name, **{k: v for k, v in r.items() if k != "params"},
                         best=f"conf {r['params']['conf_thr']} {r['params']['mode']} grow {r['params']['grow']} min {r['params']['min_area']}"))
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    base = res["ensemble b+a+c (baseline)"]["PQ_split_half"]
    added = res["ensemble b+a+c+r (RF-DETR added)"]
    keep = added["PQ_split_half"] - base >= KEEP_MARGIN
    decision = dict(keep_rfdetr=bool(keep), baseline_held_out=base, with_rfdetr_held_out=added["PQ_split_half"],
                    delta=added["PQ_split_half"] - base, params_with_rfdetr=added["params"])
    json.dump(decision, open(W / "rfdetr_decision.json", "w"), indent=1)
    notes = [f"held-out delta of adding RF-DETR: {decision['delta']:+.4f} (keep needs >= +{KEEP_MARGIN}) -> "
             f"{'KEEP' if keep else 'DROP'}"]
    for name, slug in (("ensemble b+a+c+r (RF-DETR added)", "submission_ens_b_a_c_r.csv"),
                       ("ensemble r+a+c (RF-DETR replaces 2048)", "submission_ens_r_a_c.csv")):
        if res[name]["PQ_split_half"] > base:
            runs, views = configs[name]
            notes.append(write_submission(runs, views, res[name]["params"], slug))
    write_md("e1_eval_rfdetr.md", [
        ("PQ (in-sample tuned; split-half = threshold chosen on held-out halves)", T),
        ("Decision", "\n".join(f"- {n}" for n in notes)),
    ])


if __name__ == "__main__":
    main()
