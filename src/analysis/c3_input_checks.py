"""C3. Post-hoc checks on the 3-channel input (src/channels.py, work/img_c3).

1. Test set vs disk fit: predicted filament area (baseline ensemble on test) outside the
   fitted disk, per image.
2. Per-channel distributions of the PNGs on train (on-disk pixels): mean / std, and how
   channel 2 compares with channels 0 and 1 (pretrained weights expect similar channels).
3. Channel-2 suppression of large filaments: mean decoded local contrast inside GT masks
   by area bin, next to the raw-intensity contrast of the same masks.
"""
import json
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu

from common import BASE, ENS_VIEWS, TRAIN_IMG, W, coco, decode, split_files, write_md
import tune  # noqa: E402
from postprocess import postprocess

SIZE = 2048
DISKS = json.load(open(W / "img_c3" / "disks.json"))
YY, XX = np.mgrid[:SIZE, :SIZE]


def outside(item):
    stem, raw = item
    cx, cy, R = DISKS[stem]
    off = np.hypot(XX - cx, YY - cy) >= R
    _, fused = tune._fuse((stem, raw, ENS_VIEWS))
    masks = postprocess([(s, decode(c)) for s, c in fused if s >= BASE["conf_thr"]], **BASE)
    a = sum(int(m.sum()) for m in masks)
    o = sum(int((m & off).sum()) for m in masks)
    return dict(stem=stem, predictions=len(masks), pred_pixels=a, outside_pixels=o)


def chan_stats(stem):
    im = cv2.imread(str(W / "img_c3" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    cx, cy, R = DISKS[stem]
    on = np.hypot(XX - cx, YY - cy) < R
    return [(float(im[..., i][on].mean()), float(im[..., i][on].std())) for i in range(3)]


def in_mask(item):
    stem, polys = item
    im = cv2.imread(str(W / "img_c3" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    out = []
    for p in polys:
        m = mu.decode(mu.merge(mu.frPyObjects([p], SIZE, SIZE))).astype(bool)
        a = int(m.sum())
        if a == 0:
            continue
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = max(ys.min() - 12, 0), min(ys.max() + 13, SIZE), max(xs.min() - 12, 0), min(xs.max() + 13, SIZE)
        sub = m[y0:y1, x0:x1]
        ring = cv2.dilate(sub.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~sub
        raw, flat, con = (im[y0:y1, x0:x1, i].astype(np.float32) for i in range(3))
        c2 = (con[sub].mean() - 128) / 400  # decoded local contrast inside (flattened units)
        raw_c = (raw[ring].mean() - raw[sub].mean()) / max(raw[ring].mean(), 1) if ring.any() else np.nan
        flat_in = flat[sub].mean() / 255 + 0.5
        out.append(dict(area=a, ch2_inside=c2, flattened_ratio_inside=flat_in, raw_ring_contrast=raw_c))
    return out


def main():
    test_raw = pickle.load(open(W / "test_ens_b_a_c.pkl", "rb"))
    with Pool(12) as p:
        T = pd.DataFrame(p.map(outside, sorted(test_raw.items())))
    test_tab = pd.DataFrame([dict(test_images=len(T), predictions=int(T.predictions.sum()), pred_pixels=int(T.pred_pixels.sum()),
                                  outside_pixels=int(T.outside_pixels.sum()), share=T.outside_pixels.sum() / T.pred_pixels.sum(),
                                  images_with_any_outside=int((T.outside_pixels > 0).sum()),
                                  max_outside_in_one_image=int(T.outside_pixels.max()))])
    worst = T.sort_values("outside_pixels", ascending=False).head(5)

    c = coco()
    train_files, _ = split_files(c)
    stems = sorted(f[:-5] for f in train_files)
    random.Random(0).shuffle(stems)
    with Pool(12) as p:
        st = p.map(chan_stats, stems[:120])
    arr = np.array(st)  # images x 3 x (mean, std)
    chan = pd.DataFrame([dict(channel=n, mean_of_means=arr[:, i, 0].mean(), sd_of_means_across_images=arr[:, i, 0].std(),
                              mean_within_image_std=arr[:, i, 1].mean())
                         for i, n in enumerate(["ch0 raw", "ch1 flattened", "ch2 local contrast"])])

    polys = defaultdict(list)
    fn = {im["id"]: im["file_name"][:-5] for im in c["images"]}
    for a in c["annotations"]:
        s = fn[a["image_id"]]
        if f"{s}.jpeg" in train_files:
            polys[s].append(a["segmentation"][0])
    items = sorted(polys.items())
    random.Random(1).shuffle(items)
    with Pool(12) as p:
        M = pd.DataFrame([r for rs in p.map(in_mask, items[:250]) for r in rs])
    M["area_bin"] = pd.cut(M.area, [0, 500, 1000, 2000, 4000, 8000, 1e9], labels=["<500", "500-1k", "1k-2k", "2k-4k", "4k-8k", ">8k"])
    supp = M.groupby("area_bin", observed=True).agg(GT=("area", "size"), ch2_inside_mean=("ch2_inside", "mean"),
                                                    flattened_ratio_inside=("flattened_ratio_inside", "mean"),
                                                    raw_ring_contrast=("raw_ring_contrast", "mean")).reset_index()
    supp["ch2_relative_to_1k_2k"] = supp.ch2_inside_mean / supp.loc[supp.area_bin == "1k-2k", "ch2_inside_mean"].values[0]

    write_md("c3_input_checks.md", [
        ("Test set: baseline-ensemble predicted area outside the fitted disk", test_tab),
        ("Test images with the most predicted pixels outside the disk", worst),
        ("Per-channel distribution of the PNGs on train (on-disk pixels, 120 images; uint8 units)", chan),
        ("Channel-2 local contrast inside GT masks by area (train, 250 images; negative = darker than background)", supp),
    ])


if __name__ == "__main__":
    main()
