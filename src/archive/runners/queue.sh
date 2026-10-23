#!/usr/bin/env bash
# Experiment queue: runs after the current training + final pipeline complete.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/work"
while pgrep -f "src/train.py|run_final.sh" >/dev/null; do sleep 60; done
# Exp C: consensus labels, 1536 (compare with y26m_1536)
python3 ../src/train.py --model yolo26m-seg.pt --data "$ROOT/work/yolo_ds_cons/data.yaml" --imgsz 1536 --epochs 60 --batch 4 --name y26m_1536_cons > train_y26m_1536_cons.log 2>&1
../src/eval_run.sh y26m_1536_cons 1536 > eval_y26m_1536_cons.log 2>&1
# Exp B: full resolution 2048, per-annotator labels
python3 ../src/train.py --model yolo26m-seg.pt --imgsz 2048 --epochs 40 --batch 2 --name y26m_2048 > train_y26m_2048.log 2>&1
../src/eval_run.sh y26m_2048 2048 > eval_y26m_2048.log 2>&1
echo QUEUE_DONE
