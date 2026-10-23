"""Build consensus filament instances across annotators of the same image.

PQ is computed against every annotator record separately, so a predicted filament
earns one TP per annotator who drew a matching filament. The optimal target is
therefore the set of filaments that most annotators agree on. For an image with
k annotators:
  1. cluster instances across annotators (greedy, IoU > ``link_iou``, at most one
     instance per annotator per cluster);
  2. keep clusters supported by >= ``min_support`` * k annotators;
  3. consensus mask = pixels covered by >= half of the cluster's members.
Images with a single annotator are returned unchanged.
"""
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_util

SIZE = 2048


def poly_mask(poly):
    rle = mask_util.merge(mask_util.frPyObjects([poly], SIZE, SIZE))
    return mask_util.decode(rle).astype(bool)


def build(records, link_iou=0.3, min_support=0.5):
    """records: {annotator_record_id: [polygon, ...]} for one physical image.
    Returns list of bool masks (consensus instances)."""
    recs = list(records)
    k = len(recs)
    masks = [(r, poly_mask(p)) for r in recs for p in records[r]]
    if k == 1:
        return [m for _, m in masks]
    # larger instances first so they seed clusters
    masks.sort(key=lambda x: -x[1].sum())
    clusters = []  # [members(list of masks), annotators(set), ref(bool), bbox]
    for r, m in masks:
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        bb = (xs.min(), ys.min(), xs.max(), ys.max())
        best, best_iou = None, link_iou
        for cl in clusters:
            if r in cl[1]:
                continue
            b = cl[3]
            if bb[2] < b[0] or b[2] < bb[0] or bb[3] < b[1] or b[3] < bb[1]:
                continue
            inter = np.logical_and(m, cl[2]).sum()
            iou = inter / (m.sum() + cl[2].sum() - inter)
            if iou > best_iou:
                best, best_iou = cl, iou
        if best is None:
            clusters.append([[m], {r}, m, bb])
        else:
            best[0].append(m)
            best[1].add(r)
            votes = np.sum(best[0], axis=0)
            best[2] = votes * 2 >= len(best[0])
            b = best[3]
            best[3] = (min(b[0], bb[0]), min(b[1], bb[1]), max(b[2], bb[2]), max(b[3], bb[3]))
    out = []
    for members, anns, ref, _ in clusters:
        if len(anns) >= min_support * k and ref.any():
            out.append(ref)
    return out


def group_by_file(coco):
    """-> {file_name: {record_id: [polygon, ...]}}"""
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a["segmentation"][0])
    by_file = defaultdict(dict)
    for im in coco["images"]:
        by_file[im["file_name"]][im["id"]] = anns[im["id"]]
    return by_file
