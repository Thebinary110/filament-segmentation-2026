#!/usr/bin/env bash
# Step 3: hard-example specialist. Waits for the RF-DETR evaluation (one GPU job at a time).
#  1. ensemble members on the training split (one view each) -> hard-example mining
#  2. 768 px native tiles: missed x2, near misses, one hit, ensemble FPs (hard negatives), background
#  3. YOLO26s-seg at imgsz 1536 (2x zoom), 15 epochs
#  4. tiled inference on full validation images + add-only gate and split-half PQ
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/work
while systemctl --user is-active -q filament-rfdetr-eval; do sleep 60; done
cd "$ROOT/src"
for r in y26m_2048:2048 y26m_1536:1536 y26m_1536_cons:1536; do n=${r%%:*}; s=${r##*:}
  python3 infer.py --weights "$W/runs/$n/weights/best.pt" --images "$W/yolo_ds/train/images" --out "$W/train_${n}_o.pkl" --imgsz "$s"
done
nice -n 5 python3 build_hard.py
cd "$W"
python3 ../src/train.py --model yolo26s-seg.pt --data "$W/hard_tiles/data.yaml" --imgsz 1536 --epochs 15 --batch 8 --name y26s_hard768 > train_y26s_hard768.log 2>&1
cd "$ROOT/src"
python3 tiled_infer.py --weights "$W/runs/y26s_hard768/weights/last.pt" --images "$W/yolo_ds/val/images" --out "$W/val_hard_tiles.pkl" --tile 768 --imgsz 1536
cd analysis && nice -n 5 python3 e3_crop_pilot.py --tiles "$W/val_hard_tiles.pkl" --name e4_hard_specialist
echo HARD_DONE
