#!/usr/bin/env bash
# Train the crop refiner, then evaluate refined masks vs. the YOLO baseline on validation PQ.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/work
cd "$ROOT/src"
python3 train_refiner.py --props "$W/train_props_y26m_1536.pkl" --out "$W/refiner_r34.pt" --epochs 20
python3 refine_eval.py --weights "$W/refiner_r34.pt" --preds "$W/val_y26m_1536.pkl" --images "$W/yolo_ds/val/images" \
  --views 4 --out "$W/val_refined_y26m_1536.pkl" --gt "$W/yolo_ds/val_gt.csv" --tune-out "$W/tune_refined_y26m_1536.json"
echo REFINER_DONE
