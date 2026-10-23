"""Panoptic Quality, replicating the organizers' self-evaluation notebook.

GT rows are keyed ``<annotator>-<image>_<suffix>``; prediction rows ``<image>_<suffix>``.
Every annotator record is scored against the same predictions for its image, and
TP/FP/FN are pooled across all records (IoU > 0.5 is a hit).
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

SIZE = 2048


def _decode(rles, device):
    if not rles:
        return torch.zeros((0, SIZE * SIZE), device=device)
    m = mask_util.decode([{"size": [SIZE, SIZE], "counts": r} for r in rles])
    return torch.from_numpy(m.transpose(2, 0, 1).reshape(len(rles), -1)).to(device, torch.float16)


def pq_score(gt_df, pred_df, device="cuda", return_stats=False):
    gt_by_rec = defaultdict(list)
    for fid, rle in zip(gt_df.filament_id, gt_df.segmentation_rle):
        gt_by_rec[fid.rsplit("_", 1)[0]].append(rle)
    pred_by_img = defaultdict(list)
    for fid, rle in zip(pred_df.filament_id, pred_df.segmentation_rle):
        pred_by_img[fid.rsplit("_", 1)[0]].append(rle)

    tp_iou, fp, fn = [], 0, 0
    pred_cache = {}
    for rec, gt_rles in gt_by_rec.items():
        img = rec.split("-", 1)[1]
        if img not in pred_cache:
            pred_cache = {img: _decode(pred_by_img.get(img, []), device)}
        p = pred_cache[img]
        g = _decode(gt_rles, device)
        if p.shape[0] == 0:
            fn += g.shape[0]
            continue
        inter = (g.float() @ p.float().t())
        union = g.float().sum(1, keepdim=True) + p.float().sum(1)[None] - inter
        iou = torch.where(union == 0, torch.zeros_like(union), inter / union)
        hit = iou > 0.5
        tp_iou.extend(iou[hit].tolist())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    tp = len(tp_iou)
    den = tp + 0.5 * fp + 0.5 * fn
    pq = sum(tp_iou) / den if den else 0.0
    if return_stats:
        return pq, dict(tp=tp, fp=fp, fn=fn, sq=(sum(tp_iou) / tp if tp else 0.0))
    return pq


if __name__ == "__main__":
    gt, pred = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
    print(pq_score(gt, pred, return_stats=True))
