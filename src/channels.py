"""Deterministic 3-channel input representation for H-alpha full-disk images.

  ch0  raw intensity
  ch1  limb-darkening-flattened: raw / azimuthal median intensity at the same radius
  ch2  local contrast: flattened minus its large-scale (sigma 24 px) on-disk background
Everything outside the fitted solar disk is set to 0 in all channels.

Disk fit: an initial circle from the thresholded disk, refined by locating the limb as
the steepest radial intensity drop along 360 rays and fitting a circle to those edge
points by least squares (two passes with outlier rejection). Thresholding alone is
biased by the bright off-limb halo of GONG frames.

Encodings (uint8, fixed for train and inference):
  ch1 = (ratio - 0.5) * 255           ratio in [0.5, 1.5]  (1.0 -> 127.5)
  ch2 = 128 + 400 * (flat - background) clipped           (+-0.32 range)
"""
import cv2
import numpy as np

SIZE = 2048
N_RAYS = 360
SIGMA_BG = 24
FLAT_LO, FLAT_HI = 0.5, 1.5
CONTRAST_GAIN = 400.0


def _initial_circle(gray):
    _, th = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return SIZE / 2, SIZE / 2, SIZE * 0.45
    (x, y), r = cv2.minEnclosingCircle(max(cs, key=cv2.contourArea))
    return x, y, r


def _fit_circle(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return cx, cy, np.sqrt(c + cx ** 2 + cy ** 2)


def fit_disk(gray):
    """-> (cx, cy, R) of the solar limb in pixel coordinates."""
    cx, cy, r0 = _initial_circle(gray)
    sm = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2)
    radii = np.arange(r0 - 160, r0 + 40, 0.5)
    ang = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    xs = cx + np.outer(np.cos(ang), radii)
    ys = cy + np.outer(np.sin(ang), radii)
    prof = cv2.remap(sm, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    grad = np.diff(prof, axis=1)
    idx = np.argmin(grad, axis=1)  # steepest drop = limb
    edge_r = radii[idx] + 0.25
    pts = np.c_[cx + np.cos(ang) * edge_r, cy + np.sin(ang) * edge_r]
    fit = _fit_circle(pts)
    for _ in range(2):
        res = np.abs(np.hypot(pts[:, 0] - fit[0], pts[:, 1] - fit[1]) - fit[2])
        keep = res < max(3.0, 2.5 * np.median(res))
        fit = _fit_circle(pts[keep])
    return fit


def make_channels(gray, disk=None):
    """gray (H, W) uint8 -> (H, W, 3) uint8 [raw, flattened, local contrast], plus (cx, cy, R)."""
    cx, cy, R = disk if disk is not None else fit_disk(gray)
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    rr = np.hypot(xx - cx, yy - cy)
    on = rr < R
    g = gray.astype(np.float32)

    # azimuthal median per radius bin (2 px bins), lightly smoothed
    nb = int(np.ceil(R / 2)) + 1
    b = np.minimum((rr[on] / 2).astype(np.int32), nb - 1)
    v = g[on]
    order = np.lexsort((v, b))
    b_s, v_s = b[order], v[order]
    starts = np.searchsorted(b_s, np.arange(nb))
    ends = np.searchsorted(b_s, np.arange(nb), side="right")
    med = np.array([v_s[(s + e) // 2] if e > s else np.nan for s, e in zip(starts, ends)], dtype=np.float32)
    good = ~np.isnan(med)
    med = np.interp(np.arange(nb), np.flatnonzero(good), med[good]).astype(np.float32)
    med = np.convolve(np.pad(med, 3, mode="edge"), np.ones(7) / 7, mode="valid").astype(np.float32)
    prof = np.interp(rr / 2, np.arange(nb), med).astype(np.float32)
    flat = np.where(on, g / np.maximum(prof, 1.0), 1.0).astype(np.float32)

    # local background of the flattened image, normalized over the disk only
    w = on.astype(np.float32)
    bg = cv2.GaussianBlur(flat * w, (0, 0), SIGMA_BG) / np.maximum(cv2.GaussianBlur(w, (0, 0), SIGMA_BG), 1e-3)
    contrast = flat - bg

    ch0 = np.where(on, gray, 0).astype(np.uint8)
    ch1 = np.where(on, np.clip((flat - FLAT_LO) / (FLAT_HI - FLAT_LO) * 255, 0, 255), 0).astype(np.uint8)
    ch2 = np.where(on, np.clip(128 + CONTRAST_GAIN * contrast, 0, 255), 0).astype(np.uint8)
    return np.dstack([ch0, ch1, ch2]), (float(cx), float(cy), float(R))
