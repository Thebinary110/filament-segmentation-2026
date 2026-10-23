#!/usr/bin/env bash
# Evaluate the saved RF-DETR-Seg L checkpoint (7 epochs; best EMA = epoch 5): tiled 4-view TTA on val + test,
# then alone / as 4th ensemble member with split-half (src/analysis/e1_eval_rfdetr.py).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/work
D=$ROOT/data/MAGFiLO_1.0_Kaggle_2026
RF=$ROOT/.venv-rfdetr/bin/python
CK=$W/runs_rfdetr/seg_large_768_stopped_ep7/checkpoint_best_ema.pth
cd "$ROOT/src"
$RF rfdetr_infer.py --weights "$CK" --images "$W/yolo_ds/val/images" --out "$W/val_rfdetr_l768.pkl" --tta
$RF rfdetr_infer.py --weights "$CK" --images "$D/test/test_images" --out "$W/test_rfdetr_l768.pkl" --tta
cd analysis && nice -n 5 python3 e1_eval_rfdetr.py
echo RFDETR_EVAL_DONE
