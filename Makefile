# One command per stage for the final pipeline (3-model YOLO26m ensemble, local PQ 0.4765, LB 0.40).
# Run from the repository root with the main environment (envs/requirements-main.txt).
#   make prepare && make train && make infer && make tune && make submit
# `make full` retrains the members on all 707 images and writes submission_ens_full.csv.
PY      := python3
W       := work
TEST    := data/MAGFiLO_1.0_Kaggle_2026/test/test_images
MODELS  := b:y26m_2048 a:y26m_1536 c:y26m_1536_cons

.PHONY: data prepare train infer tune submit full figures analysis

data:      ## download + unzip the competition data (needs Kaggle credentials)
	kaggle competitions download -c filament-segmentation-2026 -p data && cd data && unzip -oq filament-segmentation-2026.zip

prepare:   ## YOLO datasets: per-annotator records and consensus labels, 85/15 split grouped by file (seed 42)
	$(PY) src/prepare_data.py --out $(W)/yolo_ds
	$(PY) src/prepare_data.py --out $(W)/yolo_ds_cons --consensus 0.6

train:     ## the three ensemble members (one GPU job at a time)
	cd $(W) && $(PY) ../src/train.py --model yolo26m-seg.pt --data $(CURDIR)/$(W)/yolo_ds/data.yaml      --imgsz 1536 --epochs 40 --batch 4 --name y26m_1536
	cd $(W) && $(PY) ../src/train.py --model yolo26m-seg.pt --data $(CURDIR)/$(W)/yolo_ds_cons/data.yaml --imgsz 1536 --epochs 60 --batch 4 --name y26m_1536_cons
	cd $(W) && $(PY) ../src/train.py --model yolo26m-seg.pt --data $(CURDIR)/$(W)/yolo_ds/data.yaml      --imgsz 2048 --epochs 40 --batch 2 --name y26m_2048

infer:     ## 4-view flip TTA on validation and test, cached per model
	for r in y26m_1536:1536 y26m_1536_cons:1536 y26m_2048:2048; do n=$${r%%:*}; s=$${r##*:}; \
	  $(PY) src/infer.py --weights $(W)/runs/$$n/weights/best.pt --images $(W)/yolo_ds/val/images --out $(W)/val_$$n.pkl  --imgsz $$s --tta; \
	  $(PY) src/infer.py --weights $(W)/runs/$$n/weights/best.pt --images $(TEST)               --out $(W)/test_$$n.pkl --imgsz $$s --tta; done

tune:      ## ensemble caches, post-processing grid on validation, split-half check
	$(PY) src/ensemble.py --split val  --models $(MODELS) --out $(W)/val_ens_b_a_c.pkl
	$(PY) src/ensemble.py --split test --models $(MODELS) --out $(W)/test_ens_b_a_c.pkl
	$(PY) src/tune.py  --preds $(W)/val_ens_b_a_c.pkl --gt $(W)/yolo_ds/val_gt.csv --views 12 --out $(W)/tune_ens_b_a_c_v12.json
	$(PY) src/tune2.py --preds $(W)/val_ens_b_a_c.pkl --gt $(W)/yolo_ds/val_gt.csv --views 12 --models ac --out $(W)/tune2_ens_b_a_c.json

submit:    ## submission CSV from the tuned ensemble
	$(PY) src/make_submission.py --preds $(W)/test_ens_b_a_c.pkl --params $(W)/tune_ens_b_a_c_v12.json --views 12 \
	  --test-dir $(TEST) --out submission_ens_b_a_c.csv

full:      ## full-data (707 images) retrain + RF-DETR evaluation + crop pilot, as run for the final entry
	bash src/run_final_queue.sh

analysis:  ## the PQ-loss diagnosis tables (work/analysis/*.md)
	cd src/analysis && for s in a1_metric_strata a2_interannotator a3_annotator_count a4_fusion_small a5_patterns \
	  b1_annotator_identity b2_per_annotator_pp b3_band_ceiling; do $(PY) $$s.py; done

figures:   ## report figures (work/report_figs)
	$(PY) src/report/figures.py
