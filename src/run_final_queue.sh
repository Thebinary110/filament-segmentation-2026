#!/usr/bin/env bash
# Final GPU queue, one GPU job at a time, after filament-rfdetr (RF-DETR training) finishes:
#   1. RF-DETR tiled TTA inference on val + test, evaluation and keep/drop decision (e1)
#   2. full-data (707 images) retrain of the ensemble members (+ RF-DETR if kept), test inference,
#      submission_ens_full.csv + sanity check (e2)
#   3. crop-detector pilot (10 epochs) + tiled val inference + add-only gate (e3)
# A failing step is logged and the queue moves on.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/work
D=$ROOT/data/MAGFiLO_1.0_Kaggle_2026
RF=$ROOT/.venv-rfdetr/bin/python
log() { echo "[$(date '+%F %T')] $*"; }
run() { local name=$1; shift; log "START $name"; if "$@" > "$W/q_$name.log" 2>&1; then log "OK $name"; else log "FAIL $name (see work/q_$name.log)"; fi; }
yolo_test() {  # weights name imgsz
  python3 "$ROOT/src/infer.py" --weights "$1" --images "$D/test/test_images" --out "$W/test_$2.pkl" --imgsz "$3" --tta
}

while systemctl --user is-active -q filament-rfdetr; do sleep 60; done
grep -q RFDETR_TRAIN_DONE "$W/rfdetr_full.log" && log "RF-DETR training finished" || log "WARNING: RF-DETR training did not report completion"
RFW=$W/runs_rfdetr/seg_large_768/checkpoint_best_ema.pth

# ---- step 1: RF-DETR inference + evaluation
cd "$ROOT/src"
run rfdetr_val  "$RF" rfdetr_infer.py --weights "$RFW" --images "$W/yolo_ds/val/images" --out "$W/val_rfdetr_l768.pkl" --tta
run rfdetr_test "$RF" rfdetr_infer.py --weights "$RFW" --images "$D/test/test_images" --out "$W/test_rfdetr_l768.pkl" --tta
(cd analysis && run e1_eval nice -n 5 python3 e1_eval_rfdetr.py)
KEEP=$(python3 -c "import json;print(json.load(open('$W/rfdetr_decision.json'))['keep_rfdetr'])" 2>/dev/null || echo False)
log "RF-DETR kept: $KEEP"

# ---- step 2: full-data retrain (original recipes; last-epoch weights since validation is now in training)
run prep_full_cons python3 prepare_data.py --out "$W/yolo_ds_cons_full" --consensus 0.6 --full
rm -rf "$W/runs/y26m_2048_full" "$W/runs/y26m_1536_full" "$W/runs/y26m_1536_cons_full"
cd "$W"
run train_1536_full python3 ../src/train.py --model yolo26m-seg.pt --data "$W/yolo_ds_full/data.yaml" --imgsz 1536 --epochs 40 --batch 4 --name y26m_1536_full
run train_1536_cons_full python3 ../src/train.py --model yolo26m-seg.pt --data "$W/yolo_ds_cons_full/data.yaml" --imgsz 1536 --epochs 60 --batch 4 --name y26m_1536_cons_full
run train_2048_full python3 ../src/train.py --model yolo26m-seg.pt --data "$W/yolo_ds_full/data.yaml" --imgsz 2048 --epochs 40 --batch 2 --name y26m_2048_full
cd "$ROOT/src"
run test_1536_full yolo_test "$W/runs/y26m_1536_full/weights/last.pt" y26m_1536_full 1536
run test_1536_cons_full yolo_test "$W/runs/y26m_1536_cons_full/weights/last.pt" y26m_1536_cons_full 1536
run test_2048_full yolo_test "$W/runs/y26m_2048_full/weights/last.pt" y26m_2048_full 2048
if [ "$KEEP" = "True" ]; then
  run prep_rfdetr_full "$RF" rfdetr_data.py --out "$W/rfdetr_ds_full" --full
  run train_rfdetr_full "$RF" rfdetr_train.py --size Large --resolution 768 --epochs 40 --batch 4 --accum 4 --lr 1e-4 \
      --data "$W/rfdetr_ds_full" --out "$W/runs_rfdetr/seg_large_768_full"
  run test_rfdetr_full "$RF" rfdetr_infer.py --weights "$W/runs_rfdetr/seg_large_768_full/last_ema.pth" \
      --images "$D/test/test_images" --out "$W/test_rfdetr_l768_full.pkl" --tta
fi
(cd analysis && run e2_full nice -n 5 python3 e2_full_ensemble.py)

# ---- step 3: crop-detector pilot (10 epochs), gated on held-out add-only precision
run build_crops nice -n 5 python3 build_crops.py
cd "$W"
run train_crop_pilot python3 ../src/train.py --model yolo26m-seg.pt --data "$W/crops_c3/data.yaml" --imgsz 2048 --epochs 10 --batch 2 --name y26m_crop1024_c3_pilot
cd "$ROOT/src"
run crop_pilot_val python3 tiled_infer.py --weights "$W/runs/y26m_crop1024_c3_pilot/weights/last.pt" --images "$W/yolo_ds_c3/val/images" \
    --out "$W/val_crop_pilot_tiles.pkl" --tile 1024 --imgsz 2048 --color
(cd analysis && run e3_gate nice -n 5 python3 e3_crop_pilot.py)
log "QUEUE DONE"
