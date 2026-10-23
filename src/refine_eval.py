"""Apply the crop refiner to cached YOLO predictions and tune on validation PQ.

Produces, per image, candidate instance lists under several scoring/mask variants and
grid-searches post-processing for each, so the refiner is only adopted where it helps.
With --test it instead writes refined test instances for make_submission-style use.
"""
import argparse
import itertools
import json
import pickle
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

import tune
from postprocess import SIZE, decode, encode, fuse_views
from refiner import RefinerUNet, bbox_of, crop_box, make_input, paste_back

import cv2


@torch.no_grad()
def refine_image(model, gray, instances, pre_thr, batch=32):
    inst = [(s, m) for s, m in instances if s >= pre_thr]
    if not inst:
        return []
    union = np.zeros((SIZE, SIZE), bool)
    for _, m in inst:
        union |= m
    xs, boxes = [], []
    for s, m in inst:
        box = crop_box(bbox_of(m))
        sx, sy, side = box
        win = (slice(sy, sy + side), slice(sx, sx + side))
        xs.append(make_input(gray, m[win], union[win] & ~m[win], box))
        boxes.append(box)
    probs, keeps = [], []
    for i in range(0, len(xs), batch):
        x = torch.from_numpy(np.stack(xs[i:i + batch])).cuda().to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, keep = model(x)
        probs.extend(torch.sigmoid(logit.float())[:, 0].cpu().numpy())
        keeps.extend(torch.sigmoid(keep.float()).cpu().numpy())
    out = []
    for (s, m), p, k, box in zip(inst, probs, keeps, boxes):
        refined = paste_back(p, box) > 0.5
        out.append((float(s), float(k), encode(m), encode(refined) if refined.any() else encode(m)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--images", required=True, help="folder with <stem>.jpeg or <rec>-<stem>.jpeg")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--pre-thr", type=float, default=0.05)
    ap.add_argument("--out", required=True, help="pickle of refined instances")
    ap.add_argument("--gt", default=None, help="val GT csv -> run tuning")
    ap.add_argument("--tune-out", default=None)
    args = ap.parse_args()

    raw = pickle.load(open(args.preds, "rb"))
    if args.views == 1:
        raw = {k: [r for r in v if r[2] == "o"] for k, v in raw.items()}
    paths = {}
    from pathlib import Path
    for p in Path(args.images).iterdir():
        paths.setdefault(p.stem.split("-", 1)[-1], p)

    model = RefinerUNet(pretrained=False).cuda().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(args.weights, map_location="cuda"))
    with Pool(16) as pool:
        fused = dict(pool.map(tune._fuse, [(k, v, args.views) for k, v in raw.items()]))
    refined = {}
    for i, (stem, inst) in enumerate(sorted(fused.items())):
        gray = cv2.imread(str(paths[stem]), cv2.IMREAD_GRAYSCALE)
        refined[stem] = refine_image(model, gray, [(s, decode(c)) for s, c in inst], args.pre_thr)
        if i % 20 == 0:
            print(f"refined {i}/{len(fused)}", flush=True)
    pickle.dump(refined, open(args.out, "wb"))
    if not args.gt:
        return

    gt_df = pd.read_csv(args.gt)
    gt = defaultdict(lambda: defaultdict(list))
    for fid, rle in zip(gt_df.filament_id, gt_df.segmentation_rle):
        rec = fid.rsplit("_", 1)[0]
        gt[rec.split("-", 1)[1]][rec].append(rle)
    gt = {k: list(v.values()) for k, v in gt.items()}

    variants = {
        "yolo_orig": lambda s, k, o, r: (s, o),
        "yolo_ref": lambda s, k, o, r: (s, r),
        "geo_ref": lambda s, k, o, r: (float(np.sqrt(s * k)), r),
        "keep_ref": lambda s, k, o, r: (k, r),
        "geo_orig": lambda s, k, o, r: (float(np.sqrt(s * k)), o),
    }
    results = []
    for vname, fn in variants.items():
        vf = {stem: [fn(*t) for t in lst] for stem, lst in refined.items()}
        with Pool(16, initializer=tune._init, initargs=(vf, gt)) as pool:
            best = None
            for c, mode, g, a in itertools.product([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5], ["union_cc", "paint"],
                                                   [-1, 0], [150, 300]):
                params = {"conf_thr": c, "mode": mode, "grow": g, "min_area": a}
                pq, st = tune.evaluate(pool, sorted(gt), params)
                results.append({"variant": vname, **params, "pq": pq, **st})
                if best is None or pq > best[0]:
                    best = (pq, params, st)
        print(f"{vname:10s} best PQ {best[0]:.4f} {best[1]} {best[2]}", flush=True)
    top = max(results, key=lambda r: r["pq"])
    print("BEST", json.dumps(top))
    if args.tune_out:
        json.dump({"best": top, "all": results}, open(args.tune_out, "w"), indent=1)


if __name__ == "__main__":
    main()
