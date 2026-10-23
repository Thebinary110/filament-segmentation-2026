#!/usr/bin/env bash
# Step 1a: RF-DETR-Seg Large, 40 epochs, tiled 2x2 x 1100 px @ 768, original 85/15 split (work/rfdetr_ds).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/src"
"$ROOT/.venv-rfdetr/bin/python" rfdetr_train.py --size Large --resolution 768 --epochs 40 --batch 4 --accum 4 --lr 1e-4 \
  --out "$ROOT/work/runs_rfdetr/seg_large_768"
echo RFDETR_TRAIN_DONE
