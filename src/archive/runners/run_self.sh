#!/usr/bin/env bash
# Self-training: pseudo-label dataset -> warm-started mixed training -> fine-tune on real labels -> eval.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/work
while systemctl --user is-active -q filament-pseudo; do sleep 30; done
cd "$ROOT/src"
python3 build_pseudo.py --conf 0.35 --min-area 150
cd "$W"
python3 ../src/train.py --model "$W/runs/y26m_2048/weights/best.pt" --data "$W/yolo_ds_self/data.yaml" \
  --imgsz 2048 --epochs 12 --batch 2 --name y26m_2048_self_s1 > train_y26m_2048_self_s1.log 2>&1
python3 ../src/train.py --model "$W/runs/y26m_2048_self_s1/weights/last.pt" --data "$W/yolo_ds/data.yaml" \
  --imgsz 2048 --epochs 10 --batch 2 --lr0 2e-4 --warmup 0 --name y26m_2048_self > train_y26m_2048_self.log 2>&1
../src/eval_run.sh y26m_2048_self 2048 > eval_y26m_2048_self.log 2>&1
echo SELF_DONE
