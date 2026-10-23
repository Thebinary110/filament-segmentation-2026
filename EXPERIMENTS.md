# Solar Filament Segmentation 2026: Experiment Log

Every method, model and result tried so far, with the numbers measured at the time (as of 2026-10-16).

- **Competition:** https://www.kaggle.com/competitions/filament-segmentation-2026
- **Metric:** Panoptic Quality (PQ)
- **Deadline:** 2026-11-15
- **Hardware:** 1× NVIDIA GB10 (DGX Spark, ~120 GB unified memory, aarch64)

---

## 0. Summary

| | Value |
|---|---|
| **Best local validation PQ** | **0.4765**: 3-model YOLO26m ensemble (split-half held-out check: 0.475 ± 0.012) |
| **Best public LB** | **0.40** (three submissions reach it) |
| Inter-annotator agreement (human ceiling reference) | PQ ≈ 0.34 |
| Typical local → public gap | ≈ 0.06 (seen by us and by every public repo we surveyed) |

**What helped:**
- full resolution (2048);
- 4-view flip TTA;
- ensembling 3 models;
- union + connected components;
- 1 px erosion;
- min-area filter.

**What did not:**
- larger model;
- consensus labels;
- crop refiner;
- SAM;
- self-training on external GONG data;
- LightGBM keep-scorer;
- mask-NMS;
- multi-scale TTA;
- all mask-shape tweaks.

**Most promising open direction:** zoomed tiles. They recover missed small filaments, but need a model trained at that zoom.

---

## 1. Task and metric

- **Task:** instance segmentation of solar filaments in 2048×2048 grayscale GONG H-alpha JPEGs from 6 observatories: B Big Bear, C Cerro Tololo, L Learmonth, M Mauna Loa, T Teide, U Udaipur.
- **Data:** 707 training images (1,154 annotator records, 8,199 filaments) and 180 test images.
  - 296 images are labelled by 2–3 annotators independently.
  - Median filament area is about 1,228 px.
  - Categories (Left/Right/Unidentifiable/Ambiguous) are ignored: the submission is class-agnostic.
- **Submission:** CSV with `filament_id` and `segmentation_rle` (COCO RLE counts). Masks must not overlap.
- **Metric:** PQ = Σ IoU(TP) / (TP + ½FP + ½FN).
  - A pair is a TP if IoU > 0.5.
  - **Each annotator record is scored separately against the same predictions**, and TP/FP/FN are pooled over all records.
  - Implication: keeping a prediction adds ½ to the denominator for every annotator either way. It only pays off if its expected matched IoU across annotators is above PQ/2 (≈ 0.23).
- **Leaderboard:** public LB uses ~50% of the test set (~90 images) and shows 2 decimals. The bootstrap CI is about ±0.012, so changes below ~0.01–0.02 are invisible on the LB.

## 2. Validation protocol

- **Split:** 85/15, grouped by physical file name (seed 42): 601 train images / 106 val images (170 annotator records, 1,258 GT filaments). All annotator copies of one image stay on the same side.
- **Local scorer:** `src/pq_eval.py` / `src/tune.py`, a faithful port of the organizers' self-evaluation notebook (all IoU > 0.5 pairs, pooled per annotator record).
- **Sanity check:** one annotator scored against the others on validation gives PQ **0.338**, matching the public "two experts agree at 0.34" notebook.
- **Robustness check:** split-half CV. Settings are tuned on a random half of the validation images and scored on the other half, over 5 random splits.

## 3. Base pipeline (current best)

**Model:**
- Ultralytics **YOLO26m-seg**, single class "filament".
- Grayscale image replicated to 3 channels.
- Every annotator record is a separate training sample.

**Training:**
- AdamW, lr0 1e-3, cosine decay to 1% (lrf 0.01), warmup 3 epochs.
- Augmentation: rotation ±180°, horizontal and vertical flips, translate 0.05, scale 0.15, brightness 0.3. No mosaic, mixup or colour.
- `mask_ratio 2`, `overlap_mask`, `max_det` 100.

**Inference:**
- conf 0.05, NMS IoU 0.6, `retina_masks`.
- **4-view flip TTA** (original, h, v, hv).
- Views are fused by IoU > 0.5 clustering with a pixel majority vote. Score = summed confidence / number of views.

**Post-processing (tuned on validation):**
- confidence threshold;
- union of masks + 8-connected components (`union_cc`);
- 1 px erosion;
- min area.

| Model | Train set | imgsz | Epochs | Best mask mAP50 | Local PQ (TTA) |
|---|---|---|---|---|---|
| YOLO26m (`y26m_1536`) | per-annotator records | 1536 | 40 (best ep 33) | 0.692 | 0.460 |
| YOLO26m consensus (`y26m_1536_cons`) | 1 consensus label set per image | 1536 | 60 (best ep 53) | 0.685 | 0.465 |
| YOLO26m (`y26m_2048`) | per-annotator records | 2048 | 40 (best ep 37) | 0.665 | 0.468 |
| **Ensemble of the 3 above, 12 views** | | | | | **0.4765** |

**Best post-processing** for the ensemble: conf 0.275, `union_cc`, erosion 1 px, min area 150.

**Speeds** (GB10, model training alone):
- 1536: ~5 min/epoch;
- 2048: ~9.3 min/epoch;
- YOLO26l at 2048: ~11 min/epoch.

Two jobs sharing the GPU slowed the main run 2.3× (CPU data-loading contention), so all later jobs ran one at a time.

## 4. Kaggle submissions

| # | File | What | Local PQ | Public LB |
|---|---|---|---|---|
| 1 | `submission_ep4.csv` | YOLO26m 1536, epoch-4 checkpoint, TTA | 0.385 | 0.32 |
| 2 | `submission.csv` | YOLO26m 1536 best (ep 33) + TTA | 0.460 | **0.40** |
| 3 | `submission_ens_a_c.csv` | Ensemble 1536 + 1536-consensus (8 views) | 0.466 | **0.40** |
| 4 | `submission_probeA_conf015.csv` | Probe: #2 with conf 0.15 (1,463 masks) | 0.427 | 0.38 |
| 5 | `submission_probeB_nogrow.csv` | Probe: #2 without erosion | 0.427 | 0.36 |
| 6 | `submission_ens_b_a_c.csv` | **3-model ensemble (12 views)** | **0.477** | **0.40** |

**Unsubmitted candidates:**
- `submission_y26m_2048.csv` (0.468);
- `submission_y26m_1536_cons.csv` (0.465);
- `submission_y26l_2048.csv` (0.467);
- `submission_y26m_2048_self.csv` (0.463).

**What the probes showed:**
- The LB moves in the same direction as local PQ.
- The tuned threshold and erosion are near-optimal on the test set too.
- Local PQ is a reliable guide for decisions.

**Checkpoints** for #2 and #3 are uploaded as a **private** Kaggle dataset: `shhauryajaiswal004/filament-yolo26m-1536-checkpoints`.

## 5. Experiments: results

All PQ values are local validation PQ, each with its own post-processing tuned, unless stated otherwise.

### 5.1 Resolution and model size

| Experiment | Result | Verdict |
|---|---|---|
| 1536 → 2048 training (YOLO26m) | 0.460 → 0.468 | small gain |
| YOLO26l-seg @ 2048 (40 ep, best mAP50 0.684) | 0.467 alone; 4-model ensemble 0.470 (< 0.4765) | no gain |
| Multi-scale TTA (1536 model at 1536 + 1792 + 2048, 12 views) | 0.461 vs 0.460 | no gain |
| Mask mAP50 vs PQ across all runs | PQ flat at 0.46–0.47 for mAP50 0.665–0.692 | above ~0.66, mAP50 no longer predicts PQ |

### 5.2 Labels and annotator handling

**Consensus labels.** Annotator instances are clustered across records (IoU > 0.3). A cluster is kept if ≥ 60% of annotators support it, and its mask is the pixel majority vote.

| Test | PQ | Note |
|---|---|---|
| Consensus vs each annotator, biased (consensus includes the scored annotator) | 0.677 | |
| Leave-one-out: consensus of 2 annotators vs the 3rd | 0.334–0.348 | single annotator vs another: 0.333 |
| YOLO26m 1536 trained on consensus | 0.465 vs 0.460 | in-noise gain |

The leave-one-out numbers show **annotators genuinely disagree**; there is no hidden consensus to learn.

### 5.3 Ensembling

| Ensemble | Views | PQ |
|---|---|---|
| 1536 + 1536-consensus | 8 | 0.466 |
| **2048 + 1536 + 1536-consensus** | **12** | **0.4765** |
| + YOLO26l 2048 | 16 | 0.470 |
| + self-trained 2048 | 16 | 0.471 |
| self-trained 2048 replacing 2048 | 12 | 0.468 |

### 5.4 Post-processing (no retraining)

All tested on the 3-model ensemble against the baseline 0.4765.

| Variant | Best PQ |
|---|---|
| Pixel vote fraction 0.34 / 0.67 (vs 0.5) | ≤ 0.4736 |
| 2048 model weighted 2× in the vote | ≤ 0.4765 |
| Hole filling + keep largest piece per instance | 0.4761 |
| Size-aware erosion (skip erosion < 800 or < 1500 px) | < 0.4765 |
| Score-ordered painting (`paint`) instead of `union_cc` | ≤ 0.4718 |
| Mask-NMS at IoU 0.1 / 0.25 / 0.4 instead of `union_cc` | 0.471 / 0.472 / 0.470 |
| Erosion 2 px / 3 px (1536 model) | 0.413 / 0.290 (vs 0.460 at 1 px) |
| No erosion (1536 model) | 0.427 (LB 0.36) |
| Lower conf 0.15 (1536 model) | 0.427 (LB 0.38) |

**Held-out check of the tuned baseline:** 0.4752 ± 0.0115 vs 0.4765 in-sample, so the tuning is not overfitted.

### 5.5 Instance scoring

**LightGBM PQ/2 keep-scorer:**
- Predicts each candidate's mean matched IoU across annotators.
- Features: ensemble score, view and model agreement, size, elongation, solidity, contrast, distance from disk centre, observatory.
- 5-fold cross-fitted over validation images; ~925 candidates.

| Keep rule | All val | Held-out halves |
|---|---|---|
| Ensemble confidence > threshold | **0.459** | **0.457 ± 0.011** |
| LightGBM predicted IoU > τ | 0.449 | 0.444 ± 0.010 |

**Verdict:** worse. The ensemble confidence is already well calibrated (it was the top feature), and ~925 samples are too few.

### 5.6 Mask refinement

**Crop refiner (ResNet-34 U-Net):**
- Native-resolution 256 px crops around each detection.
- Input channels: image + YOLO seed + other instances.
- Output: refined mask plus a "keep" logit.
- Trained on 6,941 GT crops with corrupted seeds plus 13,225 real YOLO proposals; 20 epochs.

| Variant | PQ |
|---|---|
| YOLO masks + YOLO score (baseline) | **0.460** |
| YOLO masks + score blended with keep | 0.452 |
| Refined masks + blended score | 0.451 |
| Refined masks + YOLO score | 0.450 |
| Refined masks + keep score only | 0.427 |

**Verdict:** no gain. SQ (mean IoU of matched pairs) stayed at 0.68: annotator boundaries, not resolution, cap mask IoU.

**SAM 2.1-b, prompted with YOLO detections:**
- Square crop around each detection (≥ 256 px, 1.6× the box), zero-shot.

| Prompt | PQ | TP | FP | FN |
|---|---|---|---|---|
| YOLO masks (baseline) | **0.4765** | 822 | 291 | 436 |
| SAM: box + YOLO mask (dense prompt, ±8 logits) | 0.453 | 791 | 336 | 467 |
| SAM: box only | 0.382 | 690 | 439 | 568 |
| SAM: box + centre point | 0.361 | 645 | 487 | 613 |

**Verdict:** worse. SAM tends to outline the wrong dark structure on H-alpha, and it cannot add missed filaments.

### 5.7 External data: self-training on GONG

**Rules:**
- External data is allowed if public and free (Rules §2.6).
- "Models may not use any other ground-truth metadata for training." Pseudo-labels from our own model are not ground truth. No external annotations were used.

**Data:**
- 3,200 public GONG H-alpha JPEGs from `gong2.nso.edu`, 2010–2026, the 6 competition sites only.
- **Days within ±1 day of any test image were excluded.**
- NSO text overlays were blanked. On-disk intensities already match the Kaggle JPEGs (mean abs diff ≈ 1.9 grey levels).

**Pseudo-labels:**
- From YOLO26m 2048 + YOLO26l 2048, 1 view each, fused; score ≥ 0.35; paint; min area 150.
- 2,724 images kept, 15,458 filaments (5.7 per image vs ~7 in real labels).

**Training:**

| Stage | Data | Epochs | lr | Time | Mask mAP50 |
|---|---|---|---|---|---|
| 1, warm start from y26m_2048 | real + pseudo | 12 | 1e-3 → 1e-5 | ~33 min/epoch | 0.572 → 0.653 |
| 2 | real only | 10 | 2e-4 → 2e-6, no warmup | ~9.3 min/epoch | peak 0.677 |

**Result:** 0.463 alone (vs 0.468 without self-training); 0.471 in a 4-model ensemble.

**Verdict:** no gain. This matches two public repos that also report pseudo-labels did not help.

### 5.8 Zoomed tiles for small filaments (latest)

The error analysis (§6) shows 57% of misses are small, faint filaments that are never detected.

**Test (inference only):**
- The existing 2048 model on a 3×3 grid of overlapping 1024 px tiles at imgsz 1536 (1.5× zoom).
- Detections touching an inner tile edge were dropped, then cross-tile mask NMS.
- 1,751 tile detections on 106 val images.

| Variant | PQ | TP | FP | FN |
|---|---|---|---|---|
| Ensemble baseline | **0.4765** | 822 | 291 | 436 |
| Tiles alone | 0.428 | 820 | 543 | 438 |
| Ensemble + tiles as 13th view | 0.472 | 816 | 308 | 442 |
| Ensemble + new tile detections, conf ≥ 0.3 | 0.453 | **872** | 506 | **386** |
| … conf ≥ 0.4 | 0.466 | 839 | 380 | 419 |
| … conf ≥ 0.5 | 0.473 | 828 | 320 | 430 |
| … conf ≥ 0.6 | 0.476 | 822 | 296 | 436 |

**Reading:**
- Tiles **recover 50 missed filaments** (FN 436 → 386), but each comes with ~4 false positives. Only ~30% of added detections are real, just below the ~35% needed to pay off under PQ.
- The model was trained on full images, so its confidence at 1.5× zoom is poorly calibrated.

**Proposed next:** train a YOLO26m on zoomed tiles (~6 h).

## 6. Error analysis

On the 1536 model's validation predictions (TP 812, FP 340, FN 446):

| Missed GT (FN = 446) | Count | Share |
|---|---|---|
| Not detected at all | 252 | 57% |
| Near miss (IoU 0.3–0.5) | 129 | 29% |
| Covered by a prediction spanning ≥ 2 GT | 37 | 8% |
| Weak overlap (< 0.3) | 20 | 4% |
| Split across ≥ 2 predictions | 8 | 2% |

| False positives (FP = 340) | Count |
|---|---|
| Annotator drew nothing there | 145 |
| Near miss (IoU 0.3–0.5) | 131 |
| Fragment / weak overlap | 59 |
| Covers ≥ 2 GT | 5 |

**Size:** the median area of missed GT is 985 px, against 1,410 px for detected GT.

**Per-observatory PQ:**

| Site | B | C | L | M | T | U |
|---|---|---|---|---|---|---|
| PQ | 0.413 | 0.451 | 0.488 | 0.448 | 0.470 | 0.469 |

**Visual review:** 12 validation images with prediction boxes and annotator outlines (`work/viz/`, generated locally; not committed).
- Large dark filaments are found reliably.
- Small and faint filaments are missed. Worst case: 3 of 10 found in `20160831133714Th`.
- Some long, broken filaments are split into 2 boxes (`20150629161614Mh`, `20150112081954Uh`).

## 7. Started but not finished

| Item | Status |
|---|---|
| **RF-DETR-Seg Large** (tiled 2×2 × 1100 px @ 768, `rfdetr` 1.11.2 in isolated `.venv-rfdetr`) | 1-epoch smoke test: tile-level mask mAP50 **0.559**, box mAP50 0.644, ~15 min/epoch. Full 40-epoch run (~10 h) not started yet |
| Full-data retrain (707 images) of the ensemble members | Started for 2048, **stopped at user request** to try cheaper ideas first. Planned near the end |
| Merging split long filaments (end proximity + direction) | Proposed, not yet tested |

## 8. External research and competitor review

**Literature and research review** (agent, 2026-10-15). Main points:
- PQ/2 keep rule;
- U-Net + YOLO fusion;
- disk preprocessing (EdgeAttNet, arXiv 2509.02964);
- clDice (no filament evidence);
- MAGFiLO paper (doi 10.1038/s41597-024-03876-y).

**Competitor repo FELIPEACASTRO/solar-filament-2026** (cloned to `../refs/`):
- Their own Mask R-CNN pipeline peaked at **0.35–0.38 PQ** after the metric change.
- Their 0.54 entry is, by their own notes, a decoded pre-made CSV that ran no inference.
- Most of their 133 submissions were scored under the old Dice metric. We did not use any of their prediction files.

**Survey of ~23 public GitHub repos** for this competition. Best documented own-model results:

| Repo | Method | Local | LB |
|---|---|---|---|
| Solid-Snake-99 | RF-DETR-Seg tiles + refiner | 0.487 (undocumented split) | 0.41 |
| biconcavelens | 4 detectors + isotonic calibration + refiner + vote | 0.461 | 0.40 |
| zhang0894 | ConvNeXt U-Net + native refiner | 0.443 | 0.38 |
| BSergey2000 | U-Net + LightGBM scorer | 0.432 | 0.36 |
| KeiichiIto1978 | HRNet U-Net, 5-fold | 0.413 | 0.36 |
| PandaDev0069 | YOLO26s 2048 | 0.422 | ~0.36 |

- Every team reports a local → LB gap of ≈ 0.06.
- Several repos state that the public MAGFiLO release contains the test labels, and attribute LB scores above ~0.5 to that. This is unverified by us; **we never accessed that release**.

**Top LB teams (0.60–0.63):** no public code or notebooks. We found a cluster of three accounts at exactly 0.60 that submitted within 10 minutes of each other (an observation only).

**Ethics and rules decisions taken:**
- No use of test labels or the public MAGFiLO release.
- No submission of others' predictions.
- External GONG images used only as unlabelled training data, with test dates excluded.

## 9. Files

| Path | Purpose |
|---|---|
| `src/prepare_data.py` | YOLO dataset (per-annotator records; `--consensus`, `--full` options) |
| `src/consensus.py` | Annotator consensus builder |
| `src/train.py` | YOLO26 training wrapper |
| `src/infer.py` | Inference + flip TTA (+ `--extra-sizes`), caches raw predictions |
| `src/postprocess.py` | View fusion and post-processing (`union_cc` / `paint` / `nms*`, erosion, `fill`, size-aware erosion) |
| `src/tune.py`, `src/tune2.py` | Grid search of post-processing; split-half CV |
| `src/pq_eval.py` | Organizer-equivalent PQ |
| `src/make_submission.py` | Submission CSV writer |
| `src/eval_run.sh`, `src/run_*.sh`, `src/pipeline.sh`, `src/queue.sh` | Detached systemd pipelines |
| `src/refiner.py`, `src/train_refiner.py`, `src/refine_eval.py` | Crop refiner |
| `src/sam_refine.py` | SAM 2.1 prompting experiment |
| `src/instance_scorer.py` | LightGBM keep-scorer (runs in `.venv-gbm`) |
| `src/gong_download.py`, `src/build_pseudo.py`, `src/run_pseudo.sh`, `src/run_self.sh` | External GONG self-training |
| `src/tiled_infer.py` | Zoomed tiled inference |
| `src/rfdetr_data.py`, `src/rfdetr_train.py` | RF-DETR-Seg tiled dataset and training (runs in `.venv-rfdetr`) |
| `work/runs/*` | YOLO runs (weights, `results.csv`) |
| `work/val_*.pkl`, `work/test_*.pkl` | Cached raw predictions per model |
| `work/tune_*.json` | Tuned post-processing per model or ensemble |
| `COMPETITION_INFO.md` | Full competition page text (rules excluded) |
