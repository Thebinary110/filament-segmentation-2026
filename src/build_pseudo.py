"""Build a YOLO dataset = real training records + pseudo-labelled external GONG images.

Pseudo-labels: predictions of the two 2048 models (one view each) fused by IoU>0.5
clustering; score = summed confidence / 2, so a filament needs either agreement of
both models or one very confident model. Kept instances are non-overlapping
(score-ordered painting), full size (no erosion, since that trick is for the metric,
not for training targets), and at least ``min_area`` px. Images without any kept
instance are skipped (missing filaments would be taught as background).
"""
import argparse
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from postprocess import fuse_views, postprocess
from prepare_data import mask_to_polygons

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "work"
SIZE = 2048
ARGS = None


def _labels(item):
    stem, raw = item
    inst = fuse_views(raw, 2)
    masks = postprocess(inst, conf_thr=ARGS.conf, mode="paint", grow=0, min_area=ARGS.min_area)
    lines = []
    for m in masks:
        polys = sorted(mask_to_polygons(m), key=len)[-1:]
        for poly in polys:
            p = np.clip(poly / SIZE, 0, 1)
            lines.append("0 " + " ".join(f"{v:.6f}" for v in p.ravel()))
    return stem, lines


def _init(a):
    global ARGS
    ARGS = a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min-area", type=int, default=150)
    ap.add_argument("--out", default=str(W / "yolo_ds_self"))
    args = ap.parse_args()

    a = pickle.load(open(W / "gong_y26m_2048.pkl", "rb"))
    b = pickle.load(open(W / "gong_y26l_2048.pkl", "rb"))
    raw = {k: [(c, m, "o") for c, m, v in a.get(k, []) if v == "o"] +
              [(c, m, "lo") for c, m, v in b.get(k, []) if v == "o"] for k in set(a) | set(b)}

    out = Path(args.out)
    real = W / "yolo_ds"
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)
        for f in (real / split / "images").iterdir():
            dst = out / split / "images" / f.name
            if not dst.exists():
                os.symlink(os.readlink(f), dst)
            lab = real / split / "labels" / (f.stem + ".txt")
            (out / split / "labels" / lab.name).write_text(lab.read_text())

    with Pool(16, initializer=_init, initargs=(args,)) as p:
        results = p.map(_labels, sorted(raw.items()))
    kept = n_inst = 0
    for stem, lines in results:
        if not lines:
            continue
        dst = out / "train" / "images" / f"gong-{stem}.jpeg"
        if not dst.exists():
            os.symlink(W / "gong_unlabeled" / f"{stem}.jpeg", dst)
        (out / "train" / "labels" / f"gong-{stem}.txt").write_text("\n".join(lines) + "\n")
        kept += 1
        n_inst += len(lines)
    (out / "data.yaml").write_text(f"path: {out}\ntrain: train/images\nval: val/images\nnames:\n  0: filament\n")
    print(f"pseudo images kept {kept}/{len(results)}, instances {n_inst} "
          f"(mean {n_inst / max(kept, 1):.1f}/img); real train records {len(list((real / 'train' / 'images').iterdir()))}")


if __name__ == "__main__":
    main()
