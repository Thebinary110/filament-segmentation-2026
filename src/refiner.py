"""Native-resolution crop refiner for filament masks.

Stage 2 of a detect-then-refine pipeline. For every filament proposed by the YOLO
model, a square crop around it is taken from the full 2048 x 2048 image (resized to
``CROP`` px) with three input channels:
  0. grayscale image (crop-normalised),
  1. the coarse YOLO mask of this instance (the "seed"),
  2. the union of all *other* coarse masks in the crop (so neighbours are not absorbed).
A ResNet-34 U-Net predicts the refined mask of the seeded filament, plus a
"keep" logit estimating whether the seed matches a real filament (IoU > 0.5).

Training uses GT filaments with synthetically corrupted seeds (shift, dilate/erode,
partial drop, polygon simplification), together with real YOLO proposals from the
training images matched to GT, so the refiner learns to correct the detector's
actual error modes.
"""
import math
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

CROP = 256          # network input size
PAD_FRAC = 0.35     # context around the seed bbox
MIN_SIDE = 96       # min crop side in native pixels


# --------------------------------------------------------------------------- model
class _Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class RefinerUNet(nn.Module):
    def __init__(self, in_ch=3, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet34(weights=w)
        if in_ch != 3:
            r.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)          # /2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # /4 64, /8 128, /16 256, /32 512
        self.u4 = _Up(512, 256, 256)
        self.u3 = _Up(256, 128, 128)
        self.u2 = _Up(128, 64, 64)
        self.u1 = _Up(64, 64, 32)
        self.u0 = _Up(32, in_ch, 16)                               # full-res skip from raw input
        self.head = nn.Conv2d(16, 1, 1)
        self.keep = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1))

    def forward(self, x):
        s0 = self.stem(x)
        e1 = self.l1(self.pool(s0))
        e2 = self.l2(e1)
        e3 = self.l3(e2)
        e4 = self.l4(e3)
        d = self.u4(e4, e3)
        d = self.u3(d, e2)
        d = self.u2(d, e1)
        d = self.u1(d, s0)
        d = self.u0(d, x)
        return self.head(d), self.keep(e4).squeeze(1)


# --------------------------------------------------------------------------- crops
def crop_box(seed_bbox, size=2048):
    """seed_bbox = (x0, y0, x1, y1) inclusive -> square crop (x0, y0, side) inside the image."""
    x0, y0, x1, y1 = seed_bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0 + 1, y1 - y0 + 1)
    side = int(max(MIN_SIDE, math.ceil(side * (1 + 2 * PAD_FRAC))))
    side = min(side, size)
    sx = int(round(min(max(cx - side / 2, 0), size - side)))
    sy = int(round(min(max(cy - side / 2, 0), size - side)))
    return sx, sy, side


def make_input(gray, seed, others, box):
    """Build the (3, CROP, CROP) float tensor for one crop.

    ``gray`` is the full image; ``seed``/``others`` are already cropped to the box window.
    """
    sx, sy, side = box
    g = gray[sy:sy + side, sx:sx + side].astype(np.float32)
    s = seed.astype(np.float32)
    o = others.astype(np.float32)
    g = cv2.resize(g, (CROP, CROP), interpolation=cv2.INTER_AREA)
    s = cv2.resize(s, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    o = cv2.resize(o, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    disk = g > 8  # ignore off-disk black background in normalisation
    mu, sd = (g[disk].mean(), g[disk].std() + 1e-3) if disk.sum() > 100 else (g.mean(), g.std() + 1e-3)
    g = (g - mu) / sd
    return np.stack([g, s * 2 - 1, o * 2 - 1]).astype(np.float32)


def paste_back(prob, box, size=2048):
    """Resize a (CROP, CROP) probability map back to native resolution and place it."""
    sx, sy, side = box
    full = np.zeros((size, size), np.float32)
    full[sy:sy + side, sx:sx + side] = cv2.resize(prob, (side, side), interpolation=cv2.INTER_LINEAR)
    return full


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()


# --------------------------------------------------------------------------- seed corruption
def corrupt_seed(gt, rng):
    """Mimic detector errors on a GT mask: shift, grow/shrink, partial drop, blur-to-blob."""
    m = gt.astype(np.uint8)
    ys, xs = np.nonzero(m)
    if rng.random() < 0.8:  # shift
        dx, dy = rng.randint(-6, 6), rng.randint(-6, 6)
        m = np.roll(np.roll(m, dy, 0), dx, 1)
    r = rng.random()
    if r < 0.35:
        k = rng.randint(1, 5)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2))
    elif r < 0.6:
        k = rng.randint(1, 3)
        e = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2))
        m = e if e.sum() > 20 else m
    if rng.random() < 0.3 and len(xs) > 0:  # drop one end of the filament (missed fragment)
        horiz = xs.max() - xs.min() > ys.max() - ys.min()
        lo, hi = (xs.min(), xs.max()) if horiz else (ys.min(), ys.max())
        cut = rng.randint(lo + (hi - lo) // 4, hi - (hi - lo) // 4) if hi - lo > 8 else None
        if cut is not None:
            dropped = m.copy()
            if horiz:
                dropped[:, cut:] = 0 if rng.random() < 0.5 else dropped[:, cut:]
                if dropped.sum() == m.sum():
                    dropped[:, :cut] = 0
            else:
                dropped[cut:, :] = 0 if rng.random() < 0.5 else dropped[cut:, :]
                if dropped.sum() == m.sum():
                    dropped[:cut, :] = 0
            if dropped.sum() >= 0.3 * m.sum():
                m = dropped
    if rng.random() < 0.3:  # coarse low-res look (YOLO masks are upsampled)
        k = rng.choice([3, 5, 7])
        m = (cv2.GaussianBlur(m.astype(np.float32), (k * 2 + 1,) * 2, 0) > 0.5).astype(np.uint8)
    return m.astype(bool) if m.any() else gt.astype(bool)


# --------------------------------------------------------------------------- loss
def seg_loss(logits, target, weight=None):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weight is not None:
        bce = bce * weight
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    union = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (union + 1)
    return bce.mean() + dice.mean()


def augment(x, y, rng):
    """Random dihedral transform (crop-level, consistent for input and target)."""
    k = rng.randint(0, 3)
    x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (0, 1))
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    if rng.random() < 0.5:  # brightness / contrast jitter on image channel only
        x = x.copy()
        x[0] = x[0] * rng.uniform(0.8, 1.2) + rng.uniform(-0.2, 0.2)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)
