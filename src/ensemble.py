"""Combine cached per-model predictions (infer.py / rfdetr_infer.py output) into one ensemble
cache. Every view is tagged <letter><flip> (e.g. bh, av) except the reference (first) model's
unflipped view, which keeps the plain tag "o", so fusion (postprocess.fuse_views) treats
every model x flip as a separate view. This reproduces the caches behind the reported scores.

  python3 src/ensemble.py --split val  --models b:y26m_2048 a:y26m_1536 c:y26m_1536_cons --out work/val_ens_b_a_c.pkl
  python3 src/ensemble.py --split test --models b:y26m_2048 a:y26m_1536 c:y26m_1536_cons --out work/test_ens_b_a_c.pkl
"""
import argparse
import pickle
from pathlib import Path

W = Path(__file__).resolve().parents[1] / "work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--models", nargs="+", required=True, help="letter:run, reference model first")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out, ref = {}, None
    for spec in args.models:
        tag, run = spec.split(":", 1)
        ref = ref or tag
        for k, v in pickle.load(open(W / f"{args.split}_{run}.pkl", "rb")).items():
            out.setdefault(k, []).extend((c, m, "o" if (tag == ref and vw == "o") else f"{tag}{vw}") for c, m, vw in v)
    pickle.dump(out, open(args.out, "wb"))
    print(f"{len(out)} images, {sum(len(v) for v in out.values())} raw detections from {len(args.models)} models -> {args.out}")


if __name__ == "__main__":
    main()
