"""B1. Is annotator identity (record-id prefix) predictable from a test file name?

Record ids are GGBBAA-<stem>: group GG, batch GGBB, annotator GGBBAA. All records of one
image share the batch. A test file name gives only the timestamp and observatory, so we
check how well group / batch / annotator(s) of an image can be predicted from year,
month, day-of-year, hour, observatory, and the nearest training image in time
(leave-one-out over the 707 train+val images), against the majority-class baseline.
We also check whether the "hard" annotators (lowest per-prefix PQ in a5) are predictable.
"""
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from common import coco, write_md


def main():
    c = coco()
    recs = defaultdict(list)
    for im in c["images"]:
        recs[im["file_name"][:-5]].append(im["id"].split("-")[0])
    rows = []
    for stem, ids in recs.items():
        t = datetime.strptime(stem[:14], "%Y%m%d%H%M%S")
        rows.append(dict(stem=stem, group=ids[0][:2], batch=ids[0][:4], annotators=",".join(sorted(ids)),
                         first_annotator=sorted(ids)[0], k=len(ids), year=t.year, month=t.month,
                         doy=t.timetuple().tm_yday, hour=t.hour, site=stem[14], t=t.timestamp()))
    T = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    assert all(len({i[:4] for i in recs[s]}) == 1 for s in recs), "records of one image span batches"

    # a5 per-prefix PQ -> hard/easy annotators (below/above median of prefixes with >= 4 val records)
    a5 = pd.read_csv("../../work/analysis/a5_gt_features.csv")
    hard_prefixes = {"010401", "030101", "020101", "010101", "050102", "060403", "010403", "010103", "010102"}
    T["has_hard_annotator"] = T.annotators.apply(lambda s: any(a in hard_prefixes for a in s.split(",")))

    def loo_acc(target, keys):
        hit = 0
        for i, r in T.iterrows():
            rest = T.drop(i)
            m = rest
            for kc in keys:
                m = m[m[kc] == r[kc]]
            pred = (m if len(m) else rest)[target].mode()[0]
            hit += pred == r[target]
        return hit / len(T)

    def nn_acc(target, window_days=None):
        hit = n = 0
        for i in range(len(T)):
            d = (T.t - T.t[i]).abs()
            d[i] = np.inf
            j = d.idxmin()
            if window_days is not None and d[j] > window_days * 86400:
                continue
            n += 1
            hit += T[target][j] == T[target][i]
        return hit / max(n, 1), n

    res = []
    for target in ("group", "batch", "first_annotator", "has_hard_annotator", "k"):
        base = (T[target] == T[target].mode()[0]).mean()
        nn_all, _ = nn_acc(target)
        nn_1d, n_1d = nn_acc(target, 1.0)
        res.append(dict(target=target, classes=T[target].nunique(), majority=base,
                        year=loo_acc(target, ["year"]), site=loo_acc(target, ["site"]),
                        year_site=loo_acc(target, ["year", "site"]), hour=loo_acc(target, ["hour"]),
                        nearest_in_time=nn_all, nearest_within_1day=nn_1d, images_with_neighbour_1day=n_1d))
    R = pd.DataFrame(res)

    # temporal structure: are batches contiguous in time?
    span = T.groupby("batch").agg(images=("t", "size"), first=("year", "min"), last=("year", "max"),
                                  distinct_days=("stem", lambda s: s.str[:8].nunique()),
                                  sites=("site", lambda s: "".join(sorted(set(s))))).reset_index()
    same_day = T.assign(day=T.stem.str[:8]).groupby("day").agg(images=("batch", "size"), batches=("batch", "nunique"))
    sd = same_day[same_day.images >= 2]
    sameday_tab = pd.DataFrame([dict(days_with_2plus_images=len(sd), share_single_batch=(sd.batches == 1).mean(),
                                     mean_batches_per_day=sd.batches.mean())])
    hard_by = pd.crosstab(T.year, T.has_hard_annotator, normalize="index").reset_index()
    hard_by.columns = [str(c) for c in hard_by.columns]

    write_md("b1_annotator_identity.md", [
        ("Leave-one-out accuracy of predicting annotation identity from test-visible information", R),
        ("Batches: time and observatory coverage", span),
        ("Do images from the same day share a batch?", sameday_tab),
        ("Share of images with at least one 'hard' annotator, by year", hard_by),
    ])


if __name__ == "__main__":
    main()
