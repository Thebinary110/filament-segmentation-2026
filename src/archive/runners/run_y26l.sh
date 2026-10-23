#!/usr/bin/env bash
# YOLO26l-seg at full 2048 resolution, then TTA eval + submission.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/work"
python3 ../src/train.py --model yolo26l-seg.pt --imgsz 2048 --epochs 40 --batch 2 --name y26l_2048 > train_y26l_2048.log 2>&1
../src/eval_run.sh y26l_2048 2048 > eval_y26l_2048.log 2>&1
echo Y26L_DONE
