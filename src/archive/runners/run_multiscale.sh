#!/usr/bin/env bash
# Multi-scale TTA test for the 1536 model (runs after the experiment queue frees the GPU).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/work
while systemctl --user is-active -q filament-queue; do sleep 60; done
cd "$ROOT/src"
python3 infer.py --weights "$W/runs/y26m_1536/weights/best.pt" --images "$W/yolo_ds/val/images" \
  --out "$W/val_y26m_1536_ms.pkl" --imgsz 1536 --tta --extra-sizes 1792 2048
python3 tune.py --preds "$W/val_y26m_1536_ms.pkl" --gt "$W/yolo_ds/val_gt.csv" --views 12 --out "$W/tune_y26m_1536_ms_v12.json" | tail -1
echo MULTISCALE_DONE
