"""Zoomed, tiled YOLO inference for small filaments.

Each 2048 image is cut into a 3 x 3 grid of TILE x TILE crops (stride (2048-TILE)/2)
and every crop is predicted at ``--imgsz`` (zoom = imgsz / TILE). A detection whose
box touches an inner tile edge is dropped (a neighbouring tile sees it whole), then
detections from all tiles are de-duplicated by score-ordered mask NMS. The result is
one extra "view" per image, cached in the same format as infer.py with tag ``z``.
"""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from postprocess import encode

SIZE = 2048


def tiles(tile):
    step = (SIZE - tile) // 2
    return [(x, y) for y in (0, step, SIZE - tile) for x in (0, step, SIZE - tile)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--edge", type=int, default=6, help="px margin that counts as touching a tile edge")
    ap.add_argument("--nms", type=float, default=0.3)
    ap.add_argument("--color", action="store_true", help="read precomputed 3-channel PNGs as-is")
    args = ap.parse_args()

    model = YOLO(args.weights)
    preds, seen = {}, set()
    for p in tqdm(sorted(Path(args.images).iterdir())):
        stem = p.stem.split("-", 1)[-1]
        if stem in seen or p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        seen.add(stem)
        img = (cv2.imread(str(p), cv2.IMREAD_COLOR) if args.color
               else cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR))
        cands = []
        for x, y in tiles(args.tile):
            crop = img[y:y + args.tile, x:x + args.tile]
            r = model.predict(crop, imgsz=args.imgsz, conf=args.conf, iou=0.6, retina_masks=True,
                              max_det=100, verbose=False, device=0)[0]
            if r.masks is None:
                continue
            masks = (r.masks.data > 0.5).cpu().numpy()
            for m, c, b in zip(masks, r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                if m.shape != (args.tile, args.tile):
                    m = cv2.resize(m.astype(np.uint8), (args.tile, args.tile), interpolation=cv2.INTER_NEAREST) > 0
                x0, y0, x1, y1 = b
                inner = ((x0 < args.edge and x > 0) or (y0 < args.edge and y > 0) or
                         (x1 > args.tile - args.edge and x + args.tile < SIZE) or
                         (y1 > args.tile - args.edge and y + args.tile < SIZE))
                if inner or not m.any():
                    continue
                full = np.zeros((SIZE, SIZE), bool)
                full[y:y + args.tile, x:x + args.tile] = m
                cands.append((float(c), full))
        cands.sort(key=lambda t: -t[0])
        kept = []
        for c, m in cands:
            ok = True
            for _, k in kept:
                inter = np.logical_and(m, k).sum()
                if inter and inter / (m.sum() + k.sum() - inter) > args.nms:
                    ok = False
                    break
            if ok:
                kept.append((c, m))
        preds[stem] = [(c, encode(m), "z") for c, m in kept]
    pickle.dump(preds, open(args.out, "wb"))
    print(f"saved {len(preds)} images, {sum(len(v) for v in preds.values())} tile detections -> {args.out}")


if __name__ == "__main__":
    main()
