"""Download unlabeled GONG H-alpha images (public NSO archive) for self-training.

* Sites: only the six competition observatories (B, C, L, M, T, U).
* Dates: random days between 2010-06 and 2026-09, excluding +-1 day around every test
  image (and the test images themselves), so no test observation leaks into training.
* Harmonisation to the competition JPEGs: archive frames carry NSO text overlays in the
  off-disk corners; these are blanked. On-disk intensities already match the competition
  JPEGs (mean abs diff ~1.9 grey levels on a matched frame), so no intensity remapping.
"""
import argparse
import datetime as dt
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
BASE = "https://gong2.nso.edu/HA/hag"
SITES = "BCLMTU"
# overlay boxes (y0, y1, x0, x1) in 2048 px frames: header text, site/time text, footer, logo
OVERLAYS = [(0, 160, 0, 680), (0, 160, 1540, 2048), (1880, 2048, 0, 640), (1840, 2048, 1840, 2048)]


def fetch(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def site_cdfs():
    """Mean intensity CDF per site from competition training images (on-disk + off-disk)."""
    cdfs = {}
    paths = sorted((DATA / "train" / "train_images").glob("*.jpeg"))
    for s in SITES:
        hs = []
        for p in [p for p in paths if p.stem[-2] == s][:40]:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            hs.append(np.bincount(g.ravel(), minlength=256))
        h = np.sum(hs, 0).astype(np.float64)
        cdfs[s] = np.cumsum(h) / h.sum()
    return cdfs


def harmonise(img_bgr):
    g = img_bgr.astype(np.float32).mean(2)
    g = np.clip(np.round(g), 0, 255).astype(np.uint8)
    bg = int(np.median(g[200:300, 100:200]))  # off-disk background level
    for y0, y1, x0, x1 in OVERLAYS:
        g[y0:y1, x0:x1] = bg
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "work" / "gong_unlabeled"))
    ap.add_argument("--days", type=int, default=1600)
    ap.add_argument("--per-day", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    test_stems = {p.stem for p in (DATA / "test" / "test_images").iterdir()}
    train_stems = {p.stem for p in (DATA / "train" / "train_images").iterdir()}
    banned = set()
    for s in test_stems:
        d = dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        banned |= {d + dt.timedelta(days=k) for k in (-1, 0, 1)}
    start, end = dt.date(2010, 6, 1), dt.date(2026, 9, 30)
    all_days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in all_days if d not in banned]
    rng = random.Random(args.seed)
    rng.shuffle(days)
    days = days[: args.days]
    print(f"days sampled {len(days)} (banned {len(banned)} around {len(test_stems)} test images)", flush=True)

    def do_day(d):
        try:
            url = f"{BASE}/{d:%Y%m}/{d:%Y%m%d}/"
            names = re.findall(r'href="(\d{14}([A-Z])h\.jpg)"', fetch(url).decode("latin1"))
        except Exception as e:  # missing day / network hiccup
            return 0
        r = random.Random(d.toordinal())
        by_site = {}
        for n, s in names:
            if s in SITES and n[:-4] not in test_stems and n[:-4] not in train_stems:
                by_site.setdefault(s, []).append(n)
        sites = list(by_site)
        r.shuffle(sites)
        got = 0
        for s in sites[: args.per_day]:
            n = r.choice(by_site[s])
            dst = out / (n[:-4] + ".jpeg")
            if dst.exists():
                got += 1
                continue
            try:
                buf = np.frombuffer(fetch(f"{url}{n}"), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None or img.shape[:2] != (2048, 2048):
                    continue
                cv2.imwrite(str(dst), harmonise(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
                got += 1
            except Exception:
                continue
        return got

    total = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, g in enumerate(ex.map(do_day, days)):
            total += g
            if i % 100 == 0:
                print(f"days {i}/{len(days)} images {total}", flush=True)
    print(f"DONE images {total} -> {out}")


if __name__ == "__main__":
    main()
