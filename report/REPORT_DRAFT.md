# Solar Filament Instance Segmentation under a Pooled-Annotator Panoptic Quality: an Ensemble Baseline and a Quantified Negative-Results Study

*Draft for the 4-page technical report (Overleaf template). Numbers are local validation PQ unless marked "LB". "In-sample" means tuned on the same images; "held-out" means split-half (choose on one half, score on the other, 5 halvings × 2). Items marked ⏳ are filled when the final GPU queue finishes.*

---

## 1. Problem and data

**Task.** Instance segmentation of solar filaments in full-disk GONG H-alpha images.
- Images: 2048 × 2048, 8-bit grayscale JPEG.
- Six observatories: Big Bear, Cerro Tololo, Learmonth, Mauna Loa, Teide, Udaipur.
- Output: one non-overlapping mask per filament, as COCO RLE.

**Data (MAGFiLO 1.0, Kaggle split).**
- 707 training images, carrying 1,154 annotator records and 8,199 filament polygons. Median filament area is ~1.2 kpx.
- 296 images were annotated independently by 2–3 annotators: 411 images have one annotator, 145 two, 151 three.
- 180 test images, no labels.
- Chirality classes (Left / Right / Unidentifiable / Ambiguous) are not scored. We train class-agnostic.

## 2. Metric analysis

**The scorer.** The organizers' scorer computes PQ = Σ IoU(TP) / (TP + ½FP + ½FN).
- A pair is a hit if IoU > 0.5.
- **Every annotator record is scored separately against the same prediction set**, and TP/FP/FN are pooled over all records.

**Marginal value of one prediction.** On an image with k records, a prediction that hits n of them with IoUs qᵢ changes the numerator by Σqᵢ and the denominator by k/2, whether or not it hits. It raises PQ iff

  **Σ qᵢ > PQ · k / 2**,  i.e. its mean matched IoU over all k records exceeds PQ/2.

**The PQ/2 keep rule for a filament drawn by s of k annotators.** With our typical matched IoU q ≈ 0.69, a prediction is worth making iff s/k > PQ/(2q) ≈ 0.35. So:

| s / k | 1/1 | 1/2 | 2/2 | **1/3** | 2/3 | 3/3 |
|---|---|---|---|---|---|---|
| ΔPQ per prediction (×10⁻³) | +0.38 | +0.18 | +0.76 | **−0.02** | +0.56 | +1.14 |

**Why 1-of-3 filaments should not be predicted.** Predicting a filament only one of three annotators drew is worth nothing or slightly negative: it earns one match and two false positives.

**Some FPs are not errors.** A correct prediction on a filament that some annotators skipped generates FPs in their records. **122 of our 291 validation FPs are of this kind**, and they belong to net-positive predictions.

*Figure: PQ vs threshold (`fig_pq_vs_threshold.png`).*

## 3. Validation protocol

**Split.** 85/15, grouped by physical file name (seed 42): 601 training / 106 validation images (170 records, 1,258 GT filaments). All annotator copies of an image stay on one side.

**Scorer.** A port of the organizers' self-evaluation notebook (all IoU > 0.5 pairs, per-record pooling). Validated against the public "two experts agree at 0.34" figure: annotator vs annotator on validation gives **PQ 0.338**.

**Split-half CV.** Settings are chosen on one half of the validation images and scored on the other; 5 random halvings × 2 directions. The held-out baseline is 0.4752 ± 0.012, vs 0.4765 in-sample, so the tuning is not overfitted.

**Human ceiling.** Annotator vs annotator (1,020 ordered pairs on train):

| | PQ | SQ |
|---|---|---|
| Annotators vs each other | 0.33–0.36 | 0.63–0.64 |
| **Our ensemble vs each annotator** | | **0.68–0.69** |

**Local → LB.** Local PQ tracks the public LB with a stable gap of ≈ 0.06. The same gap is reported by every public repository we surveyed. Leaderboard probes moved in the same direction as local PQ:

| Probe | Local | LB |
|---|---|---|
| Lower threshold | 0.427 | 0.38 |
| No erosion | 0.427 | 0.36 |

## 4. Final pipeline

**Models.** Three Ultralytics YOLO26m-seg models, single class, grayscale replicated to 3 channels. Every annotator record is a separate training sample.

| Member | imgsz | Epochs | Labels |
|---|---|---|---|
| 2048 | 2048 | 40 | per-annotator |
| 1536 | 1536 | 40 | per-annotator |
| 1536-consensus | 1536 | 60 | one majority label set per image |

**Training.** AdamW (lr 1e-3, cosine to 1%, 3-epoch warmup). Augmentation: ±180° rotation, h/v flips, scale 0.15, brightness 0.3. No mosaic.

**Inference.** 4-view flip TTA per model, so 12 views.
- Views are fused by IoU > 0.5 clustering with a pixel majority vote; score = summed confidence / 12.
- Post-processing: confidence ≥ 0.275 → union + 8-connected components → 1 px erosion → min area 150 px.

**Results.**
- **Local PQ 0.4765 (held-out 0.4752); LB 0.40.**
- Single models: 0.460–0.468. The 2048 model alone: LB 0.39.
- ⏳ RF-DETR-Seg as a fourth member (kept only if held-out PQ gains ≥ +0.005).
- ⏳ Full-data retrain (`submission_ens_full.csv`).

*Organizer plots: IoU distribution (mean matched-pair IoU 0.61), Dice distribution, and many-to-one / one-to-many relations (`fig_iou_dist`, `fig_dice_dist`, `fig_m2n`). 962 of 1,258 GT-record instances are touched by exactly one prediction. Only 33 GT are touched by 2 predictions (splits), and 33 predictions touch 2 GT (merges).*

## 5. What worked

| Change | PQ |
|---|---|
| Epoch 4 → 33 | 0.385 → 0.460 (LB 0.32 → 0.40) |
| 4-view flip TTA | +0.007 (0.453 → 0.460) |
| Union + connected components, 1 px erosion, min area | ~+0.025 over raw masks. Erosion confirmed on the LB: 0.40 vs 0.36 without |
| 1536 → 2048 training | 0.460 → 0.468 |
| 3-model ensemble | 0.4765 (+0.009 over the best single model) |

## 6. What did not work, and why

All values are held-out (split-half) where available, otherwise in-sample vs the 0.4765 baseline.

| Attempt | Result | Why |
|---|---|---|
| YOLO26l @ 2048 | 0.467 alone; 0.470 in the ensemble | Capacity is not the bottleneck |
| Consensus labels (STAPLE-like majority, ≥ 60% support) | 0.465 | Annotators genuinely disagree. A 2-annotator consensus vs the 3rd scores only 0.334–0.348 |
| Multi-scale TTA (1536 + 1792 + 2048) | 0.461 | Model calibrated at one scale |
| Zoomed tiles (1.5×), add-only | Best 0.476. At conf 0.3: recovers 50 FN but adds 215 FP | ~30% of added detections real, below the ~35% break-even |
| Crop refiner (ResNet-34 U-Net, native crops) | 0.450–0.452 | Mask IoU is capped by annotator boundary disagreement (SQ already above human) |
| SAM 2.1 prompted by our detections | box 0.382; box + mask 0.453 | Off-domain on H-alpha; outlines the wrong dark structures |
| LightGBM PQ/2 keep-scorer | 0.444 held-out vs 0.457 (same candidates) | Ensemble confidence is already the best ranking signal |
| Mask-NMS / paint / vote fractions / model weights / hole filling / size-aware erosion | all ≤ 0.4765 | Post-processing is at its optimum |
| Self-training on 2,724 pseudo-labelled external GONG images (2-stage) | 0.463 alone; 0.471 in the ensemble | Pseudo-labels reproduce the model's own uncertainty |
| 3-channel input (raw / limb-flattened / local contrast) | 0.462 held-out single (vs 0.465); 0.4746 in the ensemble (vs 0.4752) | +10 TP, +33 FP; recall at limb +0.04, at contrast 0.05–0.15 +0.01–0.02; < 0.05 contrast unchanged |
| Semantic re-ranker (U-Net agreement map, out-of-sample) | 0.4745 held-out (Δ −0.0007) | AUC for "real" in the band: 0.569 vs 0.630 for confidence |
| Per-annotator post-processing | same optimum (conf 0.275) for hard (0.445) and easy (0.518) annotators | Annotators differ in recall/precision level, not threshold |
| ⏳ RF-DETR-Seg L (tiled 2×2 × 1100 @ 768) | | |
| ⏳ Crop detector pilot (1024 crops @ 2048, low-contrast oversampling) | add-only gate ≥ 40% real | |

## 7. Error analysis

**Where PQ is lost** (validation; TP 822 / FP 291 / FN 436):

**FN (436):**
- 264 never detected.
- 65 near misses (IoU 0.3–0.5).
- 48 shape disagreement: the prediction matched another annotator's version of the same filament.
- 35 inside a prediction covering ≥ 2 GT.
- 16 weak overlap.
- 8 split.
- **Half of all FNs (210) are on single-annotator images.**

**FP (291):**
- 122 partial support (inherent; see §2).
- 67 near misses.
- 64 true phantoms (nothing drawn near).
- 38 fragments.

**What recall depends on** (`fig_recall_bins.png`):

| Factor | Recall |
|---|---|
| Contrast < 0.05 / > 0.15 | 0.17 / 0.82 (strongest driver) |
| Area < 500 px / 2–4 kpx | 0.52 / 0.75 |
| r/R < 0.5 / 0.7–0.85 | 0.69 / 0.60 |

**Other factors:**
- **Observatory** (`fig_observatory_pq.png`): PQ 0.44–0.50, within noise.
- **Annotator:** PQ against individual annotators spans 0.37–0.55. The most plausible explanation for the ≈ 0.06 local-to-LB gap is the test set's annotator mix. Annotator identity is not predictable from file names: leave-one-out accuracy ≤ majority class for batch, group, annotator and annotator count.
- **Fusion score:** does not penalise small filaments. They appear in ~11 of 12 views, just with low confidence.

**Recoverable headroom:**

| Bucket | Oracle ceiling | Realistic |
|---|---|---|
| Missed filaments with s/k = 1 | +0.136 (+0.086 at model SQ) | +0.01 to +0.03 with a better detector |
| Perfect re-ranking of conf 0.15–0.45 | +0.043 | unreached by two independent re-rankers |
| 1-of-3 filaments, shape disagreement, partial-support FPs, mask quality | ≈ 0 | — |

*Qualitative panels: `fig_panels.png` (found / missed / annotator disagreement).*

## 8. External research and competitor review

**Literature.**
- MAGFiLO (Ahmadzadeh et al., *Sci. Data* 2024).
- EdgeAttNet (arXiv 2509.02964).
- MORDEN (arXiv 2607.24525).
- Mask R-CNN on BBSO (arXiv 1912.02743).
- clDice; Lemay et al. (MELBA 2023) on multi-rater label fusion, which supports per-annotator sampling over consensus fusion and matches our consensus result.

**Public competitor repositories (~25 surveyed).**
- Best documented own-model results are 0.44–0.46 local / 0.38–0.41 LB, all with the same ≈ 0.06 local→LB gap.
- One team's own Mask R-CNN reached 0.35–0.38 PQ after the metric change. Its 0.54 entry is, by its own notes, a decoded pre-made CSV.
- Several repositories state that the public MAGFiLO release contains test labels. We did not verify this and never accessed that release.

## 9. Ethics and rules decisions

- **No test labels:** we never downloaded or used the public MAGFiLO release, said to contain test ground truth, nor any other filament annotations ("models may not use any other ground-truth metadata").
- **No others' predictions:** we declined to use or submit any other team's prediction files or embedded payloads, including the public notebooks that decode pre-made CSVs.
- **External data, unlabelled only:**
  - 3,200 public NSO/GONG H-alpha images were used solely as unlabelled input for self-training.
  - Only the six competition observatories; days within ±1 day of any test image excluded.
  - NSO overlays blanked. Pseudo-labels come from our own models.
  - The approach did not help and is not in the final pipeline.
- **No metric exploits:** predictions are non-overlapping and one per filament. We did not use duplicate masks or partial-coverage submissions.
- **Reproducibility:** all results come from code in this repository. Pinned environments and one command per stage (see repository README).

## 10. Conclusion

The model's masks already agree with each annotator better than annotators agree with each other (SQ 0.68 vs 0.63). The remaining loss is dominated by which filaments annotators chose to draw.

Under the pooled-annotator PQ, about half of the remaining errors are either not worth recovering (1-of-3 filaments) or not errors at all (partial-support FPs).

We document a reproducible 3-model ensemble at **0.4765 local / LB 0.40**, and ten negative results with their held-out numbers, as the honest frontier of this dataset.
