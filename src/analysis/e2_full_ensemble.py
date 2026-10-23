"""E2. Full-data ensemble: build submission_ens_full.csv from the full-data retrained
members (test predictions, 4-view TTA each) with the validation-tuned post-processing
carried over unchanged, and sanity-check it against the current ensemble's test output
(submission_ens_b_a_c.csv): masks per image, area distribution, IoU agreement.
No held-out data exists for the full-data models, so nothing is re-tuned here.
"""
import json
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pycocotools import mask as mu

from common import BASE, ROOT, W, write_md
import tune  # noqa: E402
from postprocess import decode, encode, postprocess

FLAGS = dict(min_iou_agreement=0.70, max_mean_abs_count_diff=1.5, median_area_ratio=(0.8, 1.25))


def load(path):
    df = pd.read_csv(path)
    df["img"] = df.filament_id.str.rsplit("_", n=1).str[0]
    return {k: list(g.segmentation_rle) for k, g in df.groupby("img")}


def agreement(a, b):
    """share of masks in a with IoU > 0.5 to some mask in b (per image, pooled)."""
    hit = tot = 0
    for img, ma in a.items():
        mb = b.get(img, [])
        tot += len(ma)
        if not mb or not ma:
            continue
        R = lambda c: {"size": [2048, 2048], "counts": c.encode()}
        iou = np.asarray(mu.iou([R(x) for x in ma], [R(x) for x in mb], [0] * len(mb))).reshape(len(ma), len(mb))
        hit += int((iou.max(1) > 0.5).sum())
    return hit / max(tot, 1)


def main():
    dec = json.load(open(W / "rfdetr_decision.json")) if (W / "rfdetr_decision.json").exists() else {"keep_rfdetr": False}
    runs = [("b", "y26m_2048_full"), ("a", "y26m_1536_full"), ("c", "y26m_1536_cons_full")]
    params, views = dict(BASE), 12
    if dec.get("keep_rfdetr"):
        runs.append(("r", "rfdetr_l768_full"))
        params, views = dec["params_with_rfdetr"], 16
    raw, ref = {}, runs[0][0]
    for tag, run in runs:
        for k, v in pickle.load(open(W / f"test_{run}.pkl", "rb")).items():
            raw.setdefault(k, []).extend((c, m, vw if tag == ref else f"{tag}{vw}") for c, m, vw in v)
    with Pool(12) as p:
        fused = dict(p.map(tune._fuse, [(k, v, views) for k, v in raw.items()]))
    out = []
    for stem in sorted(fused):
        inst = [(s, decode(c)) for s, c in fused[stem] if s >= params["conf_thr"]]
        out += [(f"{stem}_{i}", encode(m)) for i, m in enumerate(postprocess(inst, **params), 1)]
    sub = ROOT / "submission_ens_full.csv"
    pd.DataFrame(out, columns=["filament_id", "segmentation_rle"]).to_csv(sub, index=False)

    new, cur = load(sub), load(ROOT / "submission_ens_b_a_c.csv")
    imgs = sorted(set(new) | set(cur))
    cn = np.array([len(new.get(i, [])) for i in imgs])
    cc = np.array([len(cur.get(i, [])) for i in imgs])
    area = lambda d: np.array([mu.area({"size": [2048, 2048], "counts": c.encode()}) for v in d.values() for c in v])
    an, ac = area(new), area(cur)
    ag_new, ag_cur = agreement(new, cur), agreement(cur, new)
    tab = pd.DataFrame([
        dict(set="full-data ensemble (new)", masks=int(cn.sum()), images_with_masks=int((cn > 0).sum()),
             masks_per_image_mean=cn.mean(), area_p25=np.percentile(an, 25), area_median=np.median(an), area_p75=np.percentile(an, 75),
             share_matched_IoU_gt_0_5_in_other=ag_new),
        dict(set="current ensemble (LB 0.40)", masks=int(cc.sum()), images_with_masks=int((cc > 0).sum()),
             masks_per_image_mean=cc.mean(), area_p25=np.percentile(ac, 25), area_median=np.median(ac), area_p75=np.percentile(ac, 75),
             share_matched_IoU_gt_0_5_in_other=ag_cur),
    ])
    diff = pd.DataFrame([dict(count_corr=np.corrcoef(cn, cc)[0, 1], mean_abs_count_diff=np.abs(cn - cc).mean(),
                              images_with_abs_diff_ge_3=int((np.abs(cn - cc) >= 3).sum()),
                              median_area_ratio=np.median(an) / np.median(ac))])
    flags = []
    if min(ag_new, ag_cur) < FLAGS["min_iou_agreement"]:
        flags.append(f"IoU agreement {min(ag_new, ag_cur):.2f} < {FLAGS['min_iou_agreement']}")
    if diff.mean_abs_count_diff[0] > FLAGS["max_mean_abs_count_diff"]:
        flags.append(f"mean |masks per image difference| {diff.mean_abs_count_diff[0]:.2f} > {FLAGS['max_mean_abs_count_diff']}")
    lo, hi = FLAGS["median_area_ratio"]
    if not lo <= diff.median_area_ratio[0] <= hi:
        flags.append(f"median area ratio {diff.median_area_ratio[0]:.2f} outside [{lo}, {hi}]")
    write_md("e2_full_ensemble.md", [
        ("Configuration", f"members {[r for _, r in runs]}, {views} views, post-processing carried over unchanged: {params}"),
        ("submission_ens_full.csv vs current ensemble (test set)", tab),
        ("Per-image differences", diff),
        ("Flags", "\n".join(f"- {f}" for f in flags) if flags else "- none: the full-data ensemble is consistent with the current one"),
    ])


if __name__ == "__main__":
    main()
