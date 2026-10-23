#!/usr/bin/env bash
# Final pipeline: wait for training, cache predictions (with 4-view flip TTA), tune, write submission.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/work
RUN=${1:-y26m_1536}
IMGSZ=${2:-1536}
TEST=$ROOT/data/MAGFiLO_1.0_Kaggle_2026/test/test_images

while pgrep -f "src/train.py" >/dev/null; do sleep 60; done
WEIGHTS=$W/runs/$RUN/weights/best.pt

cd "$ROOT/src"
python3 infer.py --weights "$WEIGHTS" --images "$W/yolo_ds/val/images" --out "$W/val_$RUN.pkl" --imgsz "$IMGSZ" --tta
python3 infer.py --weights "$WEIGHTS" --images "$TEST" --out "$W/test_$RUN.pkl" --imgsz "$IMGSZ" --tta
for V in 1 4; do
  python3 tune.py --preds "$W/val_$RUN.pkl" --gt "$W/yolo_ds/val_gt.csv" --views $V --out "$W/tune_${RUN}_v$V.json" | tail -1
done
BEST_V=$(python3 -c "import json;a=[json.load(open('$W/tune_${RUN}_v%d.json'%v))['best']['pq'] for v in (1,4)];print(1 if a[0]>=a[1] else 4)")
python3 make_submission.py --preds "$W/test_$RUN.pkl" --params "$W/tune_${RUN}_v$BEST_V.json" --views $BEST_V \
  --test-dir "$TEST" --out "$ROOT/submission.csv"
echo "DONE views=$BEST_V"
