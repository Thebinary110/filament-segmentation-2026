"""Tiled RF-DETR-Seg inference with 4-view flip TTA (run with .venv-rfdetr/bin/python).

Matches training: 2 x 2 grid of 1100 px tiles (origins 0 and 948) at the model's
resolution (768). For every flip view (o, h, v, hv) of the full 2048 frame, each tile is
predicted; detections whose box touches an inner tile edge are dropped (the neighbouring
tile sees them whole); masks are pasted into the full frame, de-duplicated across tiles
by score-ordered mask NMS, and un-flipped. Output uses infer.py's cache format:
{stem: [(conf, rle counts, view tag), ...]} so the YOLO fusion / tuning code applies.
"""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import rfdetr
from pycocotools import mask as mask_util
from tqdm import tqdm

SIZE, TILE = 2048, 1100
ORIGINS = [(x, y) for y in (0, SIZE - TILE) for x in (0, SIZE - TILE)]
EDGE = 6


def encode(m):
    return mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")


def _iou_local(a, b):
    """IoU of two (mask, x0, y0) entries with tile-local masks, computed on their union box only."""
    (ma, ax, ay), (mb, bx, by) = a, b
    x0, y0 = min(ax, bx), min(ay, by)
    x1, y1 = max(ax + ma.shape[1], bx + mb.shape[1]), max(ay + ma.shape[0], by + mb.shape[0])
    ca = np.zeros((y1 - y0, x1 - x0), bool)
    cb = np.zeros_like(ca)
    ca[ay - y0:ay - y0 + ma.shape[0], ax - x0:ax - x0 + ma.shape[1]] = ma
    cb[by - y0:by - y0 + mb.shape[0], bx - x0:bx - x0 + mb.shape[1]] = mb
    inter = np.logical_and(ca, cb).sum()
    return inter / max(1, np.logical_or(ca, cb).sum())


def predict_view(model, gray, conf, nms):
    """-> list of (conf, full-frame bool mask) after inner-edge filtering and cross-tile mask NMS."""
    cands = []  # (conf, cropped mask, x0, y0, bbox)
    for x, y in ORIGINS:
        crop = np.ascontiguousarray(np.repeat(gray[y:y + TILE, x:x + TILE, None], 3, 2))
        det = model.predict(crop, threshold=conf, include_source_image=False)
        if det.mask is None or len(det) == 0:
            continue
        for m, c, b in zip(det.mask, det.confidence, det.xyxy):
            bx0, by0, bx1, by1 = b
            inner = ((bx0 < EDGE and x > 0) or (by0 < EDGE and y > 0) or
                     (bx1 > TILE - EDGE and x + TILE < SIZE) or (by1 > TILE - EDGE and y + TILE < SIZE))
            if inner or not m.any():
                continue
            ys, xs = np.nonzero(m)
            mx0, my0 = xs.min(), ys.min()
            sub = m[my0:ys.max() + 1, mx0:xs.max() + 1]
            gx, gy = x + mx0, y + my0
            cands.append((float(c), sub, gx, gy, (gx, gy, gx + sub.shape[1], gy + sub.shape[0])))
    cands.sort(key=lambda t: -t[0])
    kept = []
    for c, sub, gx, gy, bb in cands:
        ok = True
        for _, ks, kx, ky, kb in kept:
            if bb[2] <= kb[0] or kb[2] <= bb[0] or bb[3] <= kb[1] or kb[3] <= bb[1]:
                continue
            if _iou_local((sub, gx, gy), (ks, kx, ky)) > nms:
                ok = False
                break
        if ok:
            kept.append((c, sub, gx, gy, bb))
    out = []
    for c, sub, gx, gy, _ in kept:
        full = np.zeros((SIZE, SIZE), bool)
        full[gy:gy + sub.shape[0], gx:gx + sub.shape[1]] = sub
        out.append((c, full))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="Large")
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--nms", type=float, default=0.3)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model = getattr(rfdetr, f"RFDETRSeg{args.size}")(pretrain_weights=args.weights, resolution=args.resolution)
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    seen, preds = set(), {}
    for p in tqdm(paths):
        stem = p.stem.split("-", 1)[-1]
        if stem in seen:
            continue
        seen.add(stem)
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        views = [(gray, "o")]
        if args.tta:
            views += [(gray[:, ::-1], "h"), (gray[::-1], "v"), (gray[::-1, ::-1], "hv")]
        out = []
        for img, tag in views:
            for c, m in predict_view(model, np.ascontiguousarray(img), args.conf, args.nms):
                if tag in ("h", "hv"):
                    m = m[:, ::-1]
                if tag in ("v", "hv"):
                    m = m[::-1]
                out.append((c, encode(m), tag))
        preds[stem] = out
        if args.limit and len(preds) >= args.limit:
            break
    pickle.dump(preds, open(args.out, "wb"))
    print(f"saved {len(preds)} images, {sum(len(v) for v in preds.values())} detections -> {args.out}")


if __name__ == "__main__":
    main()
