#!/usr/bin/env bash
# Pseudo-label the unlabeled GONG images with the two 2048 models (single view each).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/work
cd "$ROOT/src"
for R in y26m_2048 y26l_2048; do
  python3 infer.py --weights "$W/runs/$R/weights/best.pt" --images "$W/gong_unlabeled" --out "$W/gong_$R.pkl" --imgsz 2048 --conf 0.05
done
echo PSEUDO_INFER_DONE
