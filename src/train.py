"""Train a single-class YOLO26 instance-segmentation model on MAGFiLO."""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26m-seg.pt")
    ap.add_argument("--data", default=str(ROOT / "work" / "yolo_ds" / "data.yaml"))
    ap.add_argument("--imgsz", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--name", default="y26m_2048")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=3)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        fraction=args.fraction,
        device=0,
        workers=16,
        project=str(ROOT / "work" / "runs"),
        name=args.name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=args.warmup,
        patience=30,
        # Solar disk: rotation/flip invariant; no colour; no mosaic (keeps full-disk context)
        degrees=180.0,
        fliplr=0.5,
        flipud=0.5,
        translate=0.05,
        scale=0.15,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        overlap_mask=True,
        mask_ratio=2,
        max_det=100,
        amp=True,
        plots=True,
        seed=0,
    )


if __name__ == "__main__":
    main()
