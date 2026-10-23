"""Turn cached raw instance predictions into final, non-overlapping filament masks.

Steps (each controlled by a parameter, tuned on validation PQ):
  1. TTA fusion: cluster instances from flipped views (IoU > 0.5), pixel-vote masks,
     score = summed confidence / number of views.
  2. Confidence threshold.
  3. Overlap resolution: ``paint`` assigns contested pixels to the higher-scoring
     instance; ``union_cc`` merges everything and splits by connected components.
  4. Morphological grow/shrink by ``grow`` pixels.
  5. Drop instances smaller than ``min_area``.
"""
import cv2
import numpy as np
from pycocotools import mask as mask_util

SIZE = 2048


def decode(counts):
    return mask_util.decode({"size": [SIZE, SIZE], "counts": counts}).astype(bool)


def encode(mask):
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def _bbox_iou_ok(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _model_of(view, weights):
    """Ensemble view tags are '<model letter><flip>' except the reference model's plain flips."""
    return view[0] if weights and view[0] in weights else "_ref"


def fuse_views(raw, n_views, vote_frac=0.5, weights=None):
    """raw: list of (conf, counts, view). Returns list of (score, mask).

    vote_frac: fraction of a cluster's (weighted) members that must cover a pixel.
    weights:   optional {model letter: weight}; the reference model (plain o/h/v/hv tags) has weight 1.
    """
    raw = sorted(raw, key=lambda r: -r[0])
    if n_views == 1:
        return [(c, decode(m)) for c, m, _ in raw]
    w_of = lambda v: (weights or {}).get(_model_of(v, weights), 1.0)
    # expected total weight over all views: n_views split evenly across reference + weighted models
    total_w = n_views if not weights else n_views / (1 + len(weights)) * (1 + sum(weights.values()))
    clusters = []  # [conf_sum, votes, ref_mask, bbox, views, weight_sum]
    for c, counts, view in raw:
        m = decode(counts)
        ys, xs = np.nonzero(m)
        bb = (xs.min(), ys.min(), xs.max(), ys.max())
        best, best_iou = None, 0.5
        for cl in clusters:
            if view in cl[4] or not _bbox_iou_ok(bb, cl[3]):
                continue
            ref = cl[2]
            inter = np.logical_and(m, ref).sum()
            iou = inter / (m.sum() + ref.sum() - inter)
            if iou > best_iou:
                best, best_iou = cl, iou
        w = w_of(view)
        if best is None:
            clusters.append([c * w, m.astype(np.float32) * w, m, bb, {view}, w])
        else:
            best[0] += c * w
            best[1] += m * w
            best[4].add(view)
            best[5] += w
    out = []
    for conf_sum, votes, _, _, views, wsum in clusters:
        if weights is None and vote_frac == 0.5:
            mask = votes >= max(1, int(np.ceil(len(views) / 2)))  # original behaviour
        else:
            mask = votes >= vote_frac * wsum - 1e-6
        if mask.any():
            out.append((conf_sum / (n_views if weights is None else total_w), mask))
    return out


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return inter / (a.sum() + b.sum() - inter)


def tidy(m):
    """Fill holes and keep the largest connected piece (annotations are single solid polygons)."""
    m8 = m.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n > 2:
        m8 = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m8)
    cv2.drawContours(filled, cs, -1, 1, thickness=-1)
    return filled.astype(bool)


def postprocess(instances, conf_thr=0.25, mode="paint", grow=0, min_area=50, min_keep=0.5,
                fill=False, grow_min_area=0):
    """instances: list of (score, bool mask). Returns list of bool masks.

    fill: tidy each output mask (fill holes, keep largest piece).
    grow_min_area: apply ``grow`` only to instances at least this large (protects thin/small filaments).
    """
    inst = sorted([i for i in instances if i[0] >= conf_thr], key=lambda i: -i[0])
    if not inst:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(grow) + 1,) * 2) if grow else None

    def morph(m):
        if not grow or (grow_min_area and m.sum() < grow_min_area):
            return m
        op = cv2.dilate if grow > 0 else cv2.erode
        return op(m.astype(np.uint8), kernel).astype(bool)

    if mode == "union_cc":
        union = np.zeros((SIZE, SIZE), bool)
        for _, m in inst:
            union |= m
        union = morph(union) if not grow_min_area else union
        n, labels, stats, _ = cv2.connectedComponentsWithStats(union.astype(np.uint8), connectivity=8)
        comps = [labels == k for k in range(1, n)]
        if grow_min_area:
            comps = [morph(c) for c in comps]
        if fill:
            comps = [tidy(c) for c in comps]
        return [c for c in comps if c.sum() >= min_area]

    if mode.startswith("nms"):
        # score-ordered mask-NMS (no transitive chaining), then paint to remove residual overlap
        thr = float(mode[3:] or 0.25)
        kept = []
        for s_, m in inst:
            if all(_iou(m, k) <= thr for _, k in kept):
                kept.append((s_, m))
        inst = kept

    taken = np.zeros((SIZE, SIZE), bool)
    out = []
    for _, m in inst:
        m = morph(m)
        free = m & ~taken
        a = free.sum()
        if a < min_area or a < min_keep * m.sum():
            continue
        if fill:
            free = tidy(free) & ~taken
        taken |= free
        out.append(free)
    return out
