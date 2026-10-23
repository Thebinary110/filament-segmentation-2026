"""A2. Is our SQ (mean IoU of matched pairs, 0.687) at the human ceiling?

On multi-annotator validation AND training images (annotations only, no model), every
ordered pair of records (A as "prediction", B as "ground truth") is scored with the same
rule as the metric. Also: our model's SQ on the same validation images, SQ by filament
support s, and a leave-one-out consensus (majority of the other annotators) as a
"best plausible" single prediction.
"""
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from pycocotools import mask as mu

from common import (all_clusters, baseline, clusters_for_image, coco, decode, encode, iou_matrix, load_val_gt,
                    majority_mask, split_files, write_md)

SIZE = 2048


def coco_records():
    c = coco()
    anns = defaultdict(list)
    for a in c["annotations"]:
        rle = mu.merge(mu.frPyObjects([a["segmentation"][0]], SIZE, SIZE))
        anns[a["image_id"]].append(rle["counts"].decode())
    recs = defaultdict(dict)
    for im in c["images"]:
        recs[im["file_name"][:-5]][im["id"]] = anns[im["id"]]
    return recs, c


def pair_stats(pred, gtl):
    iou = iou_matrix(pred, gtl).T
    hit = iou > 0.5
    return iou[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum(), iou[hit].tolist()


def agg(rows):
    s, tp, fp, fn = (sum(r[i] for r in rows) for i in range(4))
    ious = [x for r in rows for x in r[4]]
    return dict(PQ=s / (tp + 0.5 * fp + 0.5 * fn), SQ=s / max(tp, 1), RQ=tp / (tp + 0.5 * fp + 0.5 * fn),
                TP=int(tp), FP=int(fp), FN=int(fn),
                iou_p25=np.percentile(ious, 25), iou_median=np.median(ious), iou_p75=np.percentile(ious, 75))


def main():
    recs, c = coco_records()
    train_files, val_files = split_files(c)
    multi = {s: r for s, r in recs.items() if len(r) >= 2}
    rows = []
    for split_name, files in (("train", train_files), ("val", val_files)):
        stems = [s for s in multi if f"{s}.jpeg" in files]
        for k in (2, 3):
            pr = []
            for s in stems:
                if len(multi[s]) != k:
                    continue
                for a, b in itertools.permutations(multi[s], 2):
                    pr.append(pair_stats(multi[s][a], multi[s][b]))
            if pr:
                rows.append(dict(split=split_name, k=k, images=sum(len(multi[s]) == k for s in stems), pairs=len(pr), **agg(pr)))
    human = pd.DataFrame(rows)

    # our model on the same validation multi-annotator images
    gt = load_val_gt()
    final, _ = baseline()
    model_rows = []
    for k in (1, 2, 3):
        pr = []
        for s, rr in gt.items():
            if len(rr) != k:
                continue
            p = [cc for _, cc in final.get(s, [])]
            for r, g in rr.items():
                pr.append(pair_stats(p, g) if p else (0, 0, 0, len(g), []))
        model_rows.append(dict(k=k, images=sum(len(v) == k for v in gt.values()), **agg(pr)))
    model = pd.DataFrame(model_rows)

    # SQ of our matches by support s, and leave-one-out consensus vs held-out annotator (val multi images)
    clusters = all_clusters(gt)
    by_s = defaultdict(list)
    loo = defaultdict(list)
    for s, rr in gt.items():
        if len(rr) < 2:
            continue
        p = [cc for _, cc in final.get(s, [])]
        for ci, cl in enumerate(clusters[s]):
            for r, i in cl.items():
                g = rr[r][i]
                if p:
                    iou = iou_matrix([g], p)[0]
                    if (iou > 0.5).any():
                        by_s[(len(rr), len(cl))].append(iou.max())
                others = {rx: ix for rx, ix in cl.items() if rx != r}
                if others:
                    cons = encode(majority_mask(rr, others))
                    v = iou_matrix([g], [cons])[0, 0]
                    loo[len(others) + 1].append(v)
    sq_s = pd.DataFrame([dict(k=k, s=s_, matches=len(v), our_SQ=np.mean(v)) for (k, s_), v in sorted(by_s.items())])
    loo_t = pd.DataFrame([dict(cluster_support=s_, n=len(v), mean_IoU=np.mean(v), share_IoU_gt_0_5=np.mean(np.array(v) > 0.5))
                          for s_, v in sorted(loo.items())])

    write_md("a2_interannotator.md", [
        ("Annotator vs annotator (ordered pairs; annotations only, no model)", human),
        ("Our model (baseline ensemble) on validation, by number of annotators k", model),
        ("Our SQ on matched filaments, by (k, s) on multi-annotator val images", sq_s),
        ("Leave-one-out: majority of the OTHER annotators' masks vs the held-out annotator, same filament cluster (val)", loo_t),
    ])


if __name__ == "__main__":
    main()
