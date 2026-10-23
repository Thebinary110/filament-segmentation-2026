#!/usr/bin/env bash
# YOLO26m @ 2048 on the 3-channel input (raw / limb-flattened / local contrast), same recipe and
# split as y26m_2048; then 4-view TTA inference on val and test, tuning, and a submission file.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/work
cd "$W"
python3 ../src/train.py --model yolo26m-seg.pt --data "$W/yolo_ds_c3/data.yaml" --imgsz 2048 --epochs 40 --batch 2 \
  --name y26m_2048_c3 > train_y26m_2048_c3.log 2>&1
cd "$ROOT/src"
WEIGHTS=$W/runs/y26m_2048_c3/weights/best.pt
python3 infer.py --weights "$WEIGHTS" --images "$W/yolo_ds_c3/val/images" --out "$W/val_y26m_2048_c3.pkl" --imgsz 2048 --tta --color
python3 infer.py --weights "$WEIGHTS" --images "$W/img_c3/test" --out "$W/test_y26m_2048_c3.pkl" --imgsz 2048 --tta --color
python3 tune.py --preds "$W/val_y26m_2048_c3.pkl" --gt "$W/yolo_ds/val_gt.csv" --views 4 --out "$W/tune_y26m_2048_c3_v4.json" | tail -1
python3 make_submission.py --preds "$W/test_y26m_2048_c3.pkl" --params "$W/tune_y26m_2048_c3_v4.json" --views 4 \
  --test-dir "$ROOT/data/MAGFiLO_1.0_Kaggle_2026/test/test_images" --out "$ROOT/submission_y26m_2048_c3.csv"
echo C3_DONE
