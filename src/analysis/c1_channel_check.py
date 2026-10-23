"""C1. Sanity checks for the 3-channel input (src/channels.py) before retraining.

1. Disk fit: share of annotated filament pixels (all 707 train+val images) that fall
   outside the fitted disk and would be zeroed.
2. Limb darkening: mean on-disk intensity per r/R bin for the raw and flattened channel
   on validation images, and the limb-to-centre ratio.
3. Visual panels for validation images where low-contrast filaments were missed by the
   baseline (from a5_gt_features.csv): raw / flattened / local-contrast crops with the
   missed GT outlined, plus full-disk views.
"""
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu

from common import OUT, TRAIN_IMG, coco, decode, load_val_gt, write_md

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from channels import fit_disk, make_channels  # noqa: E402

SIZE = 2048
PANEL = OUT / "channels"
PANEL.mkdir(exist_ok=True)


def outside_share(item):
    stem, polys = item
    gray = cv2.imread(str(TRAIN_IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    cx, cy, R = fit_disk(gray)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    out_disk = np.hypot(xx - cx, yy - cy) >= R
    tot = out = 0
    worst = 0.0
    for p in polys:
        m = mu.decode(mu.merge(mu.frPyObjects([p], SIZE, SIZE))).astype(bool)
        a, o = m.sum(), (m & out_disk).sum()
        tot += a
        out += o
        worst = max(worst, o / max(a, 1))
    return stem, tot, out, worst


def radial(stem):
    gray = cv2.imread(str(TRAIN_IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    im, (cx, cy, R) = make_channels(gray)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    rr = np.hypot(xx - cx, yy - cy) / R
    bins = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    raw, flat = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (rr >= lo) & (rr < hi)
        raw.append(gray[sel].mean())
        flat.append(im[..., 1][sel].mean() / 255 * 1.0 + 0.5)  # back to ratio units
    return raw, flat


def main():
    c = coco()
    polys = defaultdict(list)
    fn = {im["id"]: im["file_name"][:-5] for im in c["images"]}
    for a in c["annotations"]:
        polys[fn[a["image_id"]]].append(a["segmentation"][0])
    with Pool(12) as p:
        res = p.map(outside_share, sorted(polys.items()))
    tot = sum(r[1] for r in res)
    out = sum(r[2] for r in res)
    worst = sorted(res, key=lambda r: -r[3])[:5]
    disk_tab = pd.DataFrame([dict(images=len(res), GT_pixels=tot, GT_pixels_outside_fitted_disk=out, share=out / tot,
                                  images_with_any_GT_outside=sum(r[2] > 0 for r in res))])
    worst_tab = pd.DataFrame([dict(stem=r[0], worst_instance_share_outside=r[3]) for r in worst])

    gt = load_val_gt()
    stems = sorted(gt)[:60]
    with Pool(12) as p:
        rad = p.map(radial, stems)
    raw = np.mean([r for r, _ in rad], 0)
    flat = np.mean([f for _, f in rad], 0)
    labels = ["0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1.0"]
    rad_tab = pd.DataFrame(dict(r_over_R=labels, raw_mean=raw, raw_rel_to_centre=raw / raw[0],
                                flattened_ratio_mean=flat, flattened_rel_to_centre=flat / flat[0]))

    # panels: missed low-contrast GT
    G = pd.read_csv(OUT / "a5_gt_features.csv")
    miss = G[(~G.hit) & (G.contrast < 0.08) & (G.area > 300)].sort_values("contrast")
    picks = miss.drop_duplicates("stem").head(6)
    notes = []
    for _, row in picks.iterrows():
        stem = row.stem
        gray = cv2.imread(str(TRAIN_IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        im, _ = make_channels(gray)
        # the missed instance: find the GT of that record whose area matches
        cand = [decode(cc) for cc in gt[stem][row.rec]]
        m = min(cand, key=lambda x: abs(int(x.sum()) - row.area))
        ys, xs = np.nonzero(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        half = 220
        y0, x0 = np.clip(cy - half, 0, SIZE - 2 * half), np.clip(cx - half, 0, SIZE - 2 * half)
        cnt, _ = cv2.findContours(m[y0:y0 + 2 * half, x0:x0 + 2 * half].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        tiles = []
        for ch, name in ((gray, "raw"), (im[..., 1], "flattened"), (im[..., 2], "local contrast")):
            t = cv2.cvtColor(ch[y0:y0 + 2 * half, x0:x0 + 2 * half], cv2.COLOR_GRAY2BGR)
            if name == "raw":  # stretch raw crop for visibility (display only)
                t = cv2.normalize(t, None, 0, 255, cv2.NORM_MINMAX)
            cv2.drawContours(t, cnt, -1, (0, 0, 255), 1)
            cv2.putText(t, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(t)
        cv2.imwrite(str(PANEL / f"crop_{stem}.png"), np.hstack(tiles))
        full = np.hstack([cv2.resize(cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), (600, 600), interpolation=cv2.INTER_AREA)
                          for x in (gray, im[..., 1], im[..., 2])])
        cv2.imwrite(str(PANEL / f"full_{stem}.png"), full)
        notes.append(f"- {stem}: missed GT area {int(row.area)} px, contrast {row.contrast:.3f}, r/R {row.r_rel:.2f} "
                     f"-> channels/crop_{stem}.png, channels/full_{stem}.png")

    write_md("c1_channel_check.md", [
        ("Disk fit: annotated pixels zeroed by the off-disk mask (all 707 images)", disk_tab),
        ("Images with the largest share of one instance outside the fitted disk", worst_tab),
        ("Limb darkening: mean on-disk intensity by r/R (60 validation images)", rad_tab),
        ("Visual panels (missed low-contrast filaments; red = missed GT outline)", "\n".join(notes)),
    ])


if __name__ == "__main__":
    main()
