"""Train the native-resolution crop refiner (see refiner.py).

Samples come from the training split only (same file-level split as prepare_data.py):
  * gt   : every GT filament of every annotator record, with a synthetically corrupted
           seed; target = that GT mask, keep = 1.
  * prop : every YOLO proposal on a training image, once per annotator record of that
           image; target = the best-matching GT filament of that record (empty if none
           overlaps), keep = 1 if IoU > 0.5 else 0. This teaches both mask correction and
           which proposals annotators actually agree with.
"""
import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader, Dataset

from refiner import CROP, RefinerUNet, augment, bbox_of, corrupt_seed, crop_box, make_input, seg_loss

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
ANN = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMG_DIR = DATA / "train" / "train_images"
SIZE = 2048


def split_files(coco, val_frac=0.15, seed=42):
    files = sorted({im["file_name"] for im in coco["images"]})
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_frac)
    return set(files[n_val:]), set(files[:n_val])


def raster(polys, box):
    """Rasterise native-coordinate polygons into the (side, side) window of box."""
    sx, sy, side = box
    m = np.zeros((side, side), np.uint8)
    pts = [np.round(np.asarray(p, np.float64).reshape(-1, 2) - (sx, sy)).astype(np.int32) for p in polys]
    if pts:
        cv2.fillPoly(m, pts, 1)
    return m.astype(bool)


def poly_bbox(p):
    a = np.asarray(p).reshape(-1, 2)
    return int(a[:, 0].min()), int(a[:, 1].min()), int(np.ceil(a[:, 0].max())), int(np.ceil(a[:, 1].max()))


class RefinerData(Dataset):
    def __init__(self, coco, files, props, prop_conf=0.1, epoch_seed=0):
        self.images = {}
        recs = defaultdict(list)  # file -> [record polygons list]
        for im in coco["images"]:
            if im["file_name"] in files:
                recs[im["file_name"]].append(im["id"])
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a["segmentation"][0])
        self.recs = {f: [anns[r] for r in rs] for f, rs in recs.items()}
        self.samples = []
        for f, rlist in self.recs.items():
            for ri, polys in enumerate(rlist):
                for ai in range(len(polys)):
                    self.samples.append(("gt", f, ri, ai))
        n_gt = len(self.samples)
        # proposals: match against every record of the image
        self.props = {}
        for f in self.recs:
            stem = Path(f).stem
            pl = [(c, m) for c, m, v in props.get(stem, []) if v == "o" and c >= prop_conf]
            if not pl:
                continue
            rles = [{"size": [SIZE, SIZE], "counts": m.encode()} for _, m in pl]
            self.props[f] = rles
            for ri, polys in enumerate(self.recs[f]):
                g = mask_util.frPyObjects([p for p in polys], SIZE, SIZE) if polys else []
                iou = np.asarray(mask_util.iou(rles, g, [0] * len(g))).reshape(len(rles), len(g)) if g else np.zeros((len(rles), 0))
                for pi in range(len(rles)):
                    best = int(iou[pi].argmax()) if iou.shape[1] else -1
                    best_iou = float(iou[pi, best]) if best >= 0 else 0.0
                    self.samples.append(("prop", f, ri, (pi, best if best_iou > 0.1 else -1, best_iou > 0.5)))
        print(f"refiner samples: gt {n_gt}, prop {len(self.samples) - n_gt}")
        for f in self.recs:
            self.images[f] = cv2.imread(str(IMG_DIR / f), cv2.IMREAD_GRAYSCALE)
        self.seed = epoch_seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rng = random.Random(hash((i, self.seed, torch.initial_seed())))
        kind, f, ri, extra = self.samples[i]
        gray = self.images[f]
        polys = self.recs[f][ri]
        if kind == "gt":
            ai = extra
            x0, y0, x1, y1 = poly_bbox(polys[ai])
            j = lambda v: v + rng.randint(-8, 8)
            box = crop_box((j(x0), j(y0), j(x1), j(y1)))
            target = raster([polys[ai]], box)
            if not target.any():
                return self.__getitem__((i + 1) % len(self))
            seed = corrupt_seed(target, rng)
            others = raster([p for k, p in enumerate(polys) if k != ai], box)
            keep = 1.0
        else:
            pi, best, hit = extra
            rles = self.props[f]
            pm = mask_util.decode(rles[pi]).astype(bool)
            box = crop_box(bbox_of(pm))
            sx, sy, side = box
            seed = pm[sy:sy + side, sx:sx + side]
            others = np.zeros_like(seed)
            for k, r in enumerate(rles):
                if k != pi:
                    others |= mask_util.decode(r)[sy:sy + side, sx:sx + side].astype(bool)
            others &= ~seed
            target = raster([polys[best]], box) if best >= 0 else np.zeros_like(seed)
            keep = float(hit)
        x = make_input(gray, seed, others, box)
        y = cv2.resize(target.astype(np.float32), (CROP, CROP), interpolation=cv2.INTER_LINEAR)
        x, y = augment(x, y, rng)
        return torch.from_numpy(x), torch.from_numpy(y)[None], torch.tensor(keep), torch.tensor(kind == "prop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", default=str(ROOT / "work" / "train_props_y26m_1536.pkl"))
    ap.add_argument("--out", default=str(ROOT / "work" / "refiner_r34.pt"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--full", action="store_true", help="train on train+val files")
    args = ap.parse_args()

    coco = json.load(open(ANN))
    train_files, val_files = split_files(coco)
    if args.full:
        train_files |= val_files
    props = pickle.load(open(args.props, "rb"))
    ds = RefinerData(coco, train_files, props)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=12, drop_last=True,
                    persistent_workers=True, pin_memory=True)

    model = RefinerUNet().cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for x, y, keep, is_prop in dl:
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            y, keep = y.cuda(non_blocking=True), keep.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, keep_logit = model(x)
            logit, keep_logit = logit.float(), keep_logit.float()
            has_t = (y.flatten(1).sum(1) > 0)
            loss = F.binary_cross_entropy_with_logits(keep_logit, keep)
            if has_t.any():
                loss = loss + seg_loss(logit[has_t], y[has_t])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
        print(f"epoch {ep + 1}/{args.epochs} loss {tot / n:.4f}", flush=True)
        torch.save(model.state_dict(), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
