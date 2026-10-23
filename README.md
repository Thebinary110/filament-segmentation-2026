# Solar Filament Segmentation Challenge 2026

Our entry for [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026): instance segmentation of solar filaments in GONG H-alpha full-disk images, scored by Panoptic Quality pooled over annotator records.

**Final pipeline:** an ensemble of three YOLO26m-seg models with 4-view flip TTA.
- Local validation PQ **0.4765** (split-half held-out 0.4752).
- Public LB **0.40**.

The technical report draft is `report/REPORT_DRAFT.md`; section numbers below refer to it.

## Quick start

```bash
# environments: envs/requirements-main.txt (main), envs/requirements-gbm.txt, envs/requirements-rfdetr.txt
make data      # download + unzip the competition data (Kaggle credentials required)
make prepare   # datasets (per-annotator records, consensus labels), 85/15 split grouped by file
make train     # three YOLO26m members (1536, 1536-consensus, 2048)
make infer     # 4-view TTA predictions on validation and test, cached per model
make tune      # ensemble caches, post-processing grid, split-half check
make submit    # submission_ens_b_a_c.csv
make full      # full-data retrain + RF-DETR evaluation + crop pilot (final queue)
make analysis  # PQ-loss diagnosis tables (work/analysis/)
make figures   # report figures (work/report_figs/)
```

## Repository map

| Report section | Code | Outputs |
|---|---|---|
| §1 Problem and data | `COMPETITION_INFO.md` | — |
| §2 Metric analysis (pooled PQ, PQ/2 keep rule) | `src/analysis/a1_metric_strata.py` | `work/analysis/a1_metric_strata.md` |
| §3 Validation protocol | `src/prepare_data.py` (grouped split), `src/pq_eval.py` / `src/tune.py` (organizer-equivalent scorer), `src/tune2.py` (split-half), `src/analysis/a2_interannotator.py` (human ceiling) | `work/analysis/a2_*.md` |
| §4 Final pipeline | `src/train.py`, `src/infer.py`, `src/ensemble.py`, `src/postprocess.py`, `src/make_submission.py` | `submission_ens_b_a_c.csv` |
| §5–6 What worked / didn't (see `EXPERIMENTS.md` for every run) | `src/consensus.py`, `src/tiled_infer.py`, `src/refiner.py` + `src/train_refiner.py` + `src/refine_eval.py`, `src/sam_refine.py`, `src/instance_scorer.py`, `src/gong_download.py` + `src/build_pseudo.py`, `src/channels.py` + `src/build_c3.py`, `src/semantic.py`, `src/rfdetr_*.py`, `src/build_crops.py` | `work/tune*.json`, `work/analysis/b*–e*.md` |
| §7 Error analysis | `src/analysis/a1`–`a5`, `b1`–`b3` | `work/analysis/README.md` |
| Figures (organizer plots + ours) | `src/report/figures.py` | `work/report_figs/` |

**Other paths:**
- `src/analysis/common.py`: shared loaders, scorer and baseline cache for all analyses.
- `src/run_final_queue.sh`, `src/run_rfdetr_full.sh`: the final GPU queue (one job at a time, as systemd user services).
- `src/archive/runners/`: superseded per-experiment runner scripts (kept for the record).

## Results at a glance

| Configuration | Local PQ (in-sample) | Held-out (split-half) | Public LB |
|---|---|---|---|
| **3-model ensemble (final)** | **0.4765** | **0.4752** | **0.40** |
| YOLO26m 2048, single | 0.468 | 0.465 | 0.39 |
| YOLO26m 1536, single | 0.460 | — | 0.40 |
| Annotator vs annotator (ceiling reference) | 0.338 | — | — |

Full results, including every negative result, are in `EXPERIMENTS.md` and `work/analysis/README.md`.

## Data and rules

- No test labels and no other filament annotations were used. We never accessed the public MAGFiLO release, which other teams report contains test ground truth.
- No other team's predictions were used.
- External GONG H-alpha images (public, NSO) were used only as unlabelled input for one self-training experiment. Test dates ±1 day were excluded. The experiment did not help and is not part of the final pipeline.
