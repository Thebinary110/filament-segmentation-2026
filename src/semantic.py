"""Semantic "agreement" model for re-ranking near-threshold ensemble candidates.

train : U-Net (ResNet-34 encoder, class from refiner.py) on native-resolution 768 px crops
        of the TRAINING split only (same file-grouped split as prepare_data.py, seed 42).
        Each annotator record is its own sample and its target is the union of that
        record's filament masks, so the pixel probability estimates the share of
        annotators who would draw that pixel. BCE + soft Dice.
infer : full-disk probability map per image (1024 px tiles, stride 512, cosine blending),
        reduced immediately to one feature per fused ensemble candidate:
        agree = mean probability inside the candidate mask.
        Output: {stem: [(ensemble score, counts, agree), ...]}.
Validation maps are out-of-sample (the model never saw validation labels).
"""
import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from postprocess import decode
from refiner import RefinerUNet

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
ANN = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
W = ROOT / "work"
SIZE, CROP, TILE, STRIDE = 2048, 768, 1024, 512
DISKS = json.load(open(W / "img_c3" / "disks.json"))


def split_files(coco, val_frac=0.15, seed=42):
    files = sorted({im["file_name"] for im in coco["images"]})
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_frac)
    return set(files[n_val:]), set(files[:n_val])


def norm(g):
    return (g.astype(np.float32) / 255.0 - 0.5) / 0.25


class CropData(Dataset):
    def __init__(self, coco, files, per_record=4):
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(np.asarray(a["segmentation"][0], np.float64).reshape(-1, 2))
        self.recs = [(im["file_name"][:-5], anns[im["id"]]) for im in coco["images"] if im["file_name"] in files]
        self.imgs = {s: cv2.imread(str(DATA / "train" / "train_images" / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
                     for s in {s for s, _ in self.recs}}
        self.per_record = per_record

    def __len__(self):
        return len(self.recs) * self.per_record

    def __getitem__(self, i):
        rng = random.Random(i * 7919 + torch.initial_seed() % 100000)
        stem, polys = self.recs[i // self.per_record]
        cx, cy, R = DISKS[stem]
        if polys and rng.random() < 0.7:
            p = polys[rng.randrange(len(polys))]
            px, py = p[:, 0].mean() + rng.uniform(-200, 200), p[:, 1].mean() + rng.uniform(-200, 200)
        else:
            a, r = rng.uniform(0, 2 * np.pi), R * np.sqrt(rng.random())
            px, py = cx + r * np.cos(a), cy + r * np.sin(a)
        x0 = int(np.clip(px - CROP / 2, 0, SIZE - CROP))
        y0 = int(np.clip(py - CROP / 2, 0, SIZE - CROP))
        g = self.imgs[stem][y0:y0 + CROP, x0:x0 + CROP]
        t = np.zeros((CROP, CROP), np.uint8)
        if polys:
            cv2.fillPoly(t, [np.round(p - (x0, y0)).astype(np.int32) for p in polys], 1)
        k = rng.randint(0, 3)
        g, t = np.rot90(g, k), np.rot90(t, k)
        if rng.random() < 0.5:
            g, t = g[:, ::-1], t[:, ::-1]
        x = norm(g) * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
        x = np.repeat(np.ascontiguousarray(x)[None], 3, 0)
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(t, np.float32))[None]


def loss_fn(logit, t):
    bce = F.binary_cross_entropy_with_logits(logit, t)
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * t).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def train(args):
    coco = json.load(open(ANN))
    train_files, _ = split_files(coco)
    dl = DataLoader(CropData(coco, train_files), batch_size=args.batch, shuffle=True, num_workers=12,
                    drop_last=True, persistent_workers=True, pin_memory=True)
    model = RefinerUNet().cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.05)
    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for x, t in dl:
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            t = t.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, _ = model(x)
            loss = loss_fn(logit.float(), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
        print(f"epoch {ep + 1}/{args.epochs} loss {tot / n:.4f}", flush=True)
        torch.save(model.state_dict(), args.weights)


@torch.no_grad()
def prob_map(model, gray):
    starts = list(range(0, SIZE - TILE + 1, STRIDE))
    win = np.outer(np.hanning(TILE), np.hanning(TILE)).astype(np.float32) + 1e-3
    acc = np.zeros((SIZE, SIZE), np.float32)
    wsum = np.zeros((SIZE, SIZE), np.float32)
    tiles, pos = [], []
    for y in starts:
        for x in starts:
            tiles.append(np.repeat(norm(gray[y:y + TILE, x:x + TILE])[None], 3, 0))
            pos.append((y, x))
    for i in range(0, len(tiles), 3):
        xb = torch.from_numpy(np.stack(tiles[i:i + 3])).cuda().to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, _ = model(xb)
        pr = torch.sigmoid(logit.float())[:, 0].cpu().numpy()
        for p, (y, x) in zip(pr, pos[i:i + 3]):
            acc[y:y + TILE, x:x + TILE] += p * win
            wsum[y:y + TILE, x:x + TILE] += win
    return acc / wsum


def infer(args):
    sys.path.insert(0, str(ROOT / "src"))
    import tune
    raw = pickle.load(open(args.preds, "rb"))
    with Pool(12) as p:
        fused = dict(p.map(tune._fuse, [(k, v, args.views) for k, v in raw.items()]))
    model = RefinerUNet(pretrained=False).cuda().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(args.weights, map_location="cuda"))
    img_dir = Path(args.images)
    out = {}
    for i, (stem, inst) in enumerate(sorted(fused.items())):
        gray = cv2.imread(str(img_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        pm = prob_map(model, gray)
        out[stem] = [(s, c, float(pm[decode(c)].mean())) for s, c in inst]
        if i % 20 == 0:
            print(f"infer {i}/{len(fused)}", flush=True)
    pickle.dump(out, open(args.out, "wb"))
    print("saved", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "infer"])
    ap.add_argument("--weights", default=str(W / "semantic_r34.pt"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--preds", help="cached raw ensemble predictions (infer)")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--images", help="folder with <stem>.jpeg (infer)")
    ap.add_argument("--out")
    args = ap.parse_args()
    train(args) if args.mode == "train" else infer(args)


if __name__ == "__main__":
    main()
