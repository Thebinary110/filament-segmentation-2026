"""Train RF-DETR-Seg on the tiled MAGFiLO dataset (run with .venv-rfdetr/bin/python)."""
import argparse
from pathlib import Path

import rfdetr

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="Large", help="Nano/Small/Medium/Large/XLarge/2XLarge")
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--data", default=str(ROOT / "work" / "rfdetr_ds"))
    ap.add_argument("--out", default=str(ROOT / "work" / "runs_rfdetr" / "seg_large_768"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    model = getattr(rfdetr, f"RFDETRSeg{args.size}")(resolution=args.resolution)
    model.train(
        dataset_dir=args.data,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        lr=args.lr,
        num_workers=12,
        early_stopping=False,
        checkpoint_interval=5,
    )


if __name__ == "__main__":
    main()
