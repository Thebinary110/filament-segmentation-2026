"""Prompt SAM 2.1 with YOLO-ensemble detections and compare masks on validation PQ.

For each fused ensemble instance (score >= --pre-thr) a square crop around it
(>= 256 px, 1.6x the box) is taken from the 2048 image, so SAM's 1024 input is a
zoomed-in view. SAM is prompted with the box; the box plus a positive point on the YOLO
mask (mask pixel closest to its centroid); or the box plus the YOLO mask itself
as a dense prompt. The SAM mask replaces the
YOLO mask; the instance keeps the YOLO score. Variants are tuned with the same
grid as tune.py and compared to the YOLO masks themselves.
"""
import argparse
import itertools
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM2Predictor
from ultralytics.utils import ops

import tune
from postprocess import SIZE, decode, encode

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "work"


def crop_box(mask):
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(256, 1.6 * max(x1 - x0 + 1, y1 - y0 + 1)))
    side = min(side, SIZE)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx = int(np.clip(cx - side / 2, 0, SIZE - side))
    sy = int(np.clip(cy - side / 2, 0, SIZE - side))
    return sx, sy, side, (x0 - sx, y0 - sy, x1 - sx, y1 - sy)


def anchor_point(mask, sx, sy):
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    i = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
    return [int(xs[i] - sx), int(ys[i] - sy)]


def sam_mask(pr, shape, box, point=None, mask=None):
    """Run SAM2's prompt decoder on the image set in ``pr``; returns a bool mask of ``shape``.

    Boxes/points are scaled to the 1024 encoder frame, a dense mask prompt to SAM's 256 frame.
    """
    pts, lbl, _ = pr._prepare_prompts((1024, 1024), shape, bboxes=[box],
                                      points=None if point is None else [point],
                                      labels=None if point is None else [1])
    mk = None
    if mask is not None:
        # square crop -> no letterbox padding; dense prompt as logits (+8 inside, -8 outside), (N, 1, 256, 256)
        m256 = cv2.resize(mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR) > 0.5
        mk = torch.where(torch.from_numpy(m256), 8.0, -8.0)[None, None].to(pr.device, pts.dtype)
    pm, _ = pr._inference_features(pr.features, pts, lbl, mk, False)
    if pm.shape[0] == 0:
        return None
    pm = ops.scale_masks(pm[None].float(), shape, padding=False)[0]
    return (pm[0] > pr.model.mask_threshold).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default=str(W / "val_ens_b_a_c.pkl"))
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--images", default=str(W / "yolo_ds" / "val" / "images"))
    ap.add_argument("--gt", default=str(W / "yolo_ds" / "val_gt.csv"))
    ap.add_argument("--sam", default=str(W / "weights" / "sam2.1_b.pt"))
    ap.add_argument("--pre-thr", type=float, default=0.2)
    ap.add_argument("--out", default=str(W / "sam_val.json"))
    args = ap.parse_args()

    raw = pickle.load(open(args.fused, "rb"))
    with Pool(8) as p:
        fused = dict(p.map(tune._fuse, [(k, v, args.views) for k, v in raw.items()]))
    paths = {}
    for p in Path(args.images).iterdir():
        paths.setdefault(p.stem.split("-", 1)[-1], p)

    pr = SAM2Predictor(overrides=dict(task="segment", mode="predict", imgsz=1024, model=args.sam,
                                      save=False, verbose=False))
    variants = {"yolo": {}, "sam_box": {}, "sam_box_pt": {}, "sam_box_mask": {}}
    for i, (stem, inst) in enumerate(sorted(fused.items())):
        img = cv2.cvtColor(cv2.imread(str(paths[stem]), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        for v in variants.values():
            v[stem] = []
        for score, counts in inst:
            if score < args.pre_thr:
                continue
            m = decode(counts)
            sx, sy, side, box = crop_box(m)
            crop = img[sy:sy + side, sx:sx + side]
            variants["yolo"][stem].append((score, counts))
            pr.set_image(crop)
            cm = m[sy:sy + side, sx:sx + side]
            prompts = (("sam_box", {}),
                       ("sam_box_pt", {"point": anchor_point(m, sx, sy)}),
                       # YOLO mask as a dense prompt
                       ("sam_box_mask", {"mask": cm}))
            for name, kw in prompts:
                sm = sam_mask(pr, (side, side), list(map(int, box)), **kw)
                full = np.zeros((SIZE, SIZE), bool)
                if sm is not None:
                    full[sy:sy + side, sx:sx + side] = sm
                variants[name][stem].append((score, encode(full) if full.any() else counts))
        if i % 20 == 0:
            print(f"{i}/{len(fused)} images", flush=True)

    gt = __import__("tune2").load_gt(args.gt)
    stems = sorted(gt)
    results = {}
    for name, vf in variants.items():
        best = None
        with Pool(8, initializer=tune._init, initargs=(vf, gt)) as pool:
            for c, mode, g, a in itertools.product([0.25, 0.275, 0.3, 0.35], ["union_cc", "paint"], [-1, 0], [0, 150, 300]):
                params = {"conf_thr": c, "mode": mode, "grow": g, "min_area": a}
                pq, st = tune.evaluate(pool, stems, params)
                if best is None or pq > best[0]:
                    best = (pq, params, st)
        results[name] = best
        print(f"{name:11s} best PQ {best[0]:.4f} {best[1]} {best[2]}", flush=True)
    json.dump({k: [v[0], v[1], v[2]] for k, v in results.items()}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
