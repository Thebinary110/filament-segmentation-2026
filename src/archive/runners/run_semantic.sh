#!/usr/bin/env bash
# Semantic agreement model (train split only) -> out-of-sample features for val, features for test -> re-ranker eval.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/work
D=$ROOT/data/MAGFiLO_1.0_Kaggle_2026
cd "$ROOT/src"
python3 semantic.py train --weights "$W/semantic_r34.pt" --epochs 12
python3 semantic.py infer --weights "$W/semantic_r34.pt" --preds "$W/val_ens_b_a_c.pkl" --images "$D/train/train_images" --out "$W/semantic_val_feats.pkl"
python3 semantic.py infer --weights "$W/semantic_r34.pt" --preds "$W/test_ens_b_a_c.pkl" --images "$D/test/test_images" --out "$W/semantic_test_feats.pkl"
cd analysis && nice -n 5 python3 d1_rerank.py
echo SEMANTIC_DONE
