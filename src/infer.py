"""Run a trained model over a folder of images and cache raw instance predictions.

Output is a pickle mapping image stem -> list of (confidence, compressed RLE counts)
at native 2048 x 2048 resolution, so post-processing can be tuned without re-running
the network.
"""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from tqdm import tqdm
from ultralytics import YOLO

SIZE = 2048


def encode(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def predict(model, path, imgsz, conf, tta, extra_sizes=(), color=False):
    """Views are tagged ``o/h/v/hv`` at the base size and ``<flip>@<size>`` for extra sizes.

    color=True reads a precomputed 3-channel input as-is (src/channels.py); otherwise the
    grayscale JPEG is replicated to three channels.
    """
    if color:
        img3 = cv2.imread(str(path), cv2.IMREAD_COLOR)
    else:
        img3 = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    flips = [(img3, None)]
    if tta:
        flips += [(img3[:, ::-1].copy(), "h"), (img3[::-1].copy(), "v"), (img3[::-1, ::-1].copy(), "hv")]
    views = [(v, f, imgsz, f or "o") for v, f in flips]
    for size in extra_sizes:
        views += [(v, f, size, f"{f or 'o'}@{size}") for v, f in flips]
    out = []
    for view, flip, size, tag in views:
        r = model.predict(view, imgsz=size, conf=conf, iou=0.6, retina_masks=True,
                          max_det=150, verbose=False, device=0)[0]
        if r.masks is None:
            continue
        masks = r.masks.data > 0.5
        if flip in ("h", "hv"):
            masks = masks.flip(-1)
        if flip in ("v", "hv"):
            masks = masks.flip(-2)
        masks = masks.cpu().numpy()
        for m, c in zip(masks, r.boxes.conf.cpu().numpy()):
            if m.shape != (SIZE, SIZE):
                m = cv2.resize(m.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            if m.any():
                out.append((float(c), encode(m), tag))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--color", action="store_true", help="read 3-channel inputs as-is (e.g. work/img_c3)")
    ap.add_argument("--extra-sizes", type=int, nargs="*", default=[], help="additional inference sizes (multi-scale TTA)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    # val folder holds one symlink per annotator record; predict each physical image once
    seen, preds = set(), {}
    for p in tqdm(paths):
        stem = p.stem.split("-", 1)[-1]
        if stem in seen:
            continue
        seen.add(stem)
        with torch.no_grad():
            preds[stem] = predict(model, p, args.imgsz, args.conf, args.tta, args.extra_sizes, args.color)
    pickle.dump(preds, open(args.out, "wb"))
    print(f"saved {len(preds)} images -> {args.out}")


if __name__ == "__main__":
    main()
