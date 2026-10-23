"""A3. Is the number of annotators per image (k) predictable at test time?

Test-time information is only the file name (timestamp + observatory) and pixels; there is
no sample submission in the competition data. Record ids look like GGBBAA-<stem>
(annotation group / batch / annotator index). This checks how k relates to year,
observatory, month, hour and record-id prefix, how well k can be predicted
(leave-one-out), and what the test set's time/site distribution implies.
"""
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from common import DATA, TEST_IMG, baseline, coco, load_val_gt, pq_score, write_md


def parse(stem):
    t = datetime.strptime(stem[:14], "%Y%m%d%H%M%S")
    return dict(year=t.year, month=t.month, hour=t.hour, site=stem[14], t=t.timestamp())


def main():
    c = coco()
    recs = defaultdict(list)
    for im in c["images"]:
        recs[im["file_name"][:-5]].append(im["id"].split("-")[0])
    rows = []
    for stem, ids in recs.items():
        rows.append(dict(stem=stem, k=len(ids), group=ids[0][:2], batch=ids[0][:4],
                         annot_idx=",".join(sorted(i[4:] for i in ids)), **parse(stem)))
    T = pd.DataFrame(rows)
    test = pd.DataFrame([dict(stem=p.stem, **parse(p.stem)) for p in sorted(TEST_IMG.iterdir())])
    sample = [p.name for p in DATA.rglob("*") if "sample" in p.name.lower()]

    k_dist = T.k.value_counts().sort_index().rename_axis("k").rename("images").reset_index()
    by_group = pd.crosstab(T.group, T.k).reset_index()
    by_group.columns = [str(x) for x in by_group.columns]
    by_batch = T.groupby("batch").agg(images=("k", "size"), k_values=("k", lambda v: ",".join(map(str, sorted(set(v))))),
                                      years=("year", lambda v: f"{min(v)}-{max(v)}"),
                                      sites=("site", lambda v: "".join(sorted(set(v))))).reset_index()
    by_year = pd.crosstab(T.year, T.k).reset_index()
    by_year.columns = [str(x) for x in by_year.columns]
    by_site = pd.crosstab(T.site, T.k).reset_index()
    by_site.columns = [str(x) for x in by_site.columns]

    # leave-one-out predictability of k
    base = (T.k == T.k.mode()[0]).mean()
    def loo(keycols):
        hit = 0
        for i, r in T.iterrows():
            rest = T.drop(i)
            m = rest
            for kc in keycols:
                m = m[m[kc] == r[kc]]
            pred = (m if len(m) else rest).k.mode()[0]
            hit += pred == r.k
        return hit / len(T)
    tt = T.sort_values("t").reset_index(drop=True)
    nn_hit = 0
    for i in range(len(tt)):
        d = (tt.t - tt.t[i]).abs()
        d[i] = np.inf
        nn_hit += tt.k[d.idxmin()] == tt.k[i]
    pred_tab = pd.DataFrame([
        dict(predictor="always majority class (k=1)", loo_accuracy=base),
        dict(predictor="year", loo_accuracy=loo(["year"])),
        dict(predictor="site", loo_accuracy=loo(["site"])),
        dict(predictor="year + site", loo_accuracy=loo(["year", "site"])),
        dict(predictor="year + month", loo_accuracy=loo(["year", "month"])),
        dict(predictor="nearest train image in time", loo_accuracy=nn_hit / len(tt)),
    ])

    # test set: time/site distribution and the k implied by nearest train neighbour in time
    test_rows = []
    for _, r in test.iterrows():
        d = (T.t - r.t).abs()
        j = d.idxmin()
        test_rows.append(dict(stem=r.stem, nn_k=T.k[j], nn_gap_days=d[j] / 86400, nn_batch=T.batch[j]))
    TT = test.merge(pd.DataFrame(test_rows), on="stem")
    test_vs_train = pd.DataFrame([
        dict(set="train+val", images=len(T), **{f"site_{s}": (T.site == s).mean() for s in "BCLMTU"},
             year_min=T.year.min(), year_max=T.year.max(), median_year=T.year.median()),
        dict(set="test", images=len(TT), **{f"site_{s}": (TT.site == s).mean() for s in "BCLMTU"},
             year_min=TT.year.min(), year_max=TT.year.max(), median_year=TT.year.median()),
    ])
    nn_k = TT.nn_k.value_counts().sort_index().rename_axis("k_of_nearest_train_image").rename("test_images").reset_index()
    gap = TT.nn_gap_days.describe()[["min", "25%", "50%", "75%", "max"]].rename("days to nearest train image").reset_index()

    # what a different k-mix would do to our pooled PQ (val per-k sums)
    gt = load_val_gt()
    final, _ = baseline()
    preds = {k: [c for _, c in v] for k, v in final.items()}
    per_k = {}
    for kk in (1, 2, 3):
        stems = [s for s in gt if len(gt[s]) == kk]
        per_k[kk] = pq_score(preds, gt, stems)[1]
    def mix(w):
        S = sum(w[kk] * per_k[kk]["sum_iou"] / sum(1 for s in gt if len(gt[s]) == kk) for kk in w)
        D = sum(w[kk] * per_k[kk]["D"] / sum(1 for s in gt if len(gt[s]) == kk) for kk in w)
        return S / D
    val_mix = Counter(len(v) for v in gt.values())
    mixes = pd.DataFrame([
        dict(k_mix="validation (66/16/24 images)", PQ=mix({1: val_mix[1], 2: val_mix[2], 3: val_mix[3]})),
        dict(k_mix="all k=1", PQ=mix({1: 1})),
        dict(k_mix="all k=3", PQ=mix({3: 1})),
        dict(k_mix="train+val proportions", PQ=mix({kk: int((T.k == kk).sum()) for kk in (1, 2, 3)})),
        dict(k_mix="test, k of nearest train image", PQ=mix({kk: int((TT.nn_k == kk).sum()) or 0 for kk in (1, 2, 3)})),
    ])

    T.to_csv("../../work/analysis/a3_images.csv", index=False)
    TT.to_csv("../../work/analysis/a3_test_images.csv", index=False)
    write_md("a3_annotator_count.md", [
        ("Sample submission / other metadata in the competition data", f"files with 'sample' in the name: {sample or 'none'}; test folder holds only the 180 JPEGs (no labels, no record ids)."),
        ("Annotators per image (train+val, 707 images)", k_dist),
        ("k by annotation group (first 2 digits of record id)", by_group),
        ("Annotation batches (first 4 digits of record id)", by_batch),
        ("k by year", by_year),
        ("k by observatory", by_site),
        ("Leave-one-out accuracy of predicting k from test-time information", pred_tab),
        ("Test vs train: observatory shares and years", test_vs_train),
        ("Test images: k of the nearest train image in time", nn_k),
        ("Test images: time gap to nearest train image", gap),
        ("Pooled PQ of the baseline under different k mixes (re-weighting validation per-k sums)", mixes),
    ])


if __name__ == "__main__":
    main()
