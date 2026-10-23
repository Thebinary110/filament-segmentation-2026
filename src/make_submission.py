"""Build the competition CSV (filament_id, segmentation_rle) from cached test predictions."""
import argparse
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from postprocess import decode, encode, fuse_views, postprocess

PARAMS = None
VIEWS = 1


def _rows(item):
    stem, raw = item
    if VIEWS == 1:
        raw = [r for r in raw if r[2] == "o"]
    inst = fuse_views(raw, VIEWS)
    masks = postprocess(inst, **PARAMS)
    return [(f"{stem}_{i}", encode(m)) for i, m in enumerate(masks, 1)]


def _init(params, views):
    global PARAMS, VIEWS
    PARAMS, VIEWS = params, views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--params", required=True, help="tune.py output json")
    ap.add_argument("--views", type=int, default=1)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    best = json.load(open(args.params))["best"]
    params = {k: best[k] for k in ("conf_thr", "mode", "grow", "min_area")}
    raw = pickle.load(open(args.preds, "rb"))
    expected = {p.stem for p in Path(args.test_dir).iterdir()}
    assert set(raw) == expected, f"missing predictions for {len(expected - set(raw))} test images"

    with Pool(16, initializer=_init, initargs=(params, args.views)) as p:
        rows = [r for rs in p.map(_rows, sorted(raw.items())) for r in rs]
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(args.out, index=False)
    n_img = df.filament_id.str.rsplit("_", n=1).str[0].nunique()
    print(f"params {params}; {len(df)} filaments over {n_img}/{len(expected)} images -> {args.out}")


if __name__ == "__main__":
    main()
