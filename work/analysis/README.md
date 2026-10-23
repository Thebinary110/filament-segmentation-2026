# Where the remaining PQ is lost: diagnosis

## Setup

- **Baseline:** 3-model YOLO26m ensemble (`val_ens_b_a_c.pkl`, 12 views), with conf 0.275, `union_cc`, 1 px erosion, min area 150.
- **Reproduced locally:** **PQ 0.4765**. TP 822, FP 291, FN 436, SQ 0.687, pooled denominator D = 1185.5.
- **Validation set:** 106 images / 170 annotator records. Images by number of annotators k:

| k | Images |
|---|---|
| 1 | 66 |
| 2 | 16 |
| 3 | 24 |

- **Scoring:** everything is scored with the same rule as `src/tune.py`, the organizers' notebook port.
- **In-sample flag:** numbers marked *in-sample* had a threshold tuned on the same images. Oracle numbers are upper bounds that use the labels.

**Detail tables** (`work/analysis/`):

| File | Contents |
|---|---|
| `a1_metric_strata.md` | Metric structure, FN/FP taxonomy by annotator support, oracle experiments |
| `a2_interannotator.md` | Human-vs-human SQ/PQ vs ours |
| `a3_annotator_count.md` | Can k be predicted at test time; effect of the k mix on PQ |
| `a4_fusion_small.md` | Does the fusion score penalise small filaments |
| `a5_patterns.md` | Size, contrast, disk position, observatory, annotator |
| `*.csv` | Per-instance tables behind each of the above |

**Scripts:** `src/analysis/a1_*.py` to `a5_*.py`, plus `src/analysis/common.py`.

## 1. What the metric implies

Take a prediction on an image with k records that hits n of them with IoUs q_i. It changes the pooled sums as follows:

| Sum | Change |
|---|---|
| Numerator S | + Σ q_i |
| Denominator D | + k/2, whether or not it hits |

So it raises PQ iff **Σ q_i > PQ · k / 2**.

For a filament drawn by s of k annotators, matched at our typical IoU q ≈ 0.69, it is worth predicting iff **s/k > PQ / (2q) ≈ 0.35**.

| s/k | Worth predicting? | ΔPQ per prediction (×10⁻³) |
|---|---|---|
| 1/1 | yes | +0.38 |
| 1/2 | yes | +0.18 |
| 2/2 | yes | +0.76 |
| **1/3** | **no (≈ 0)** | **−0.02** |
| 2/3 | yes | +0.56 |
| 3/3 | yes | +1.14 |

Consequences:
- A filament only one of three annotators drew is worthless to recover.
- Every correct prediction on a filament that some annotators skipped automatically produces FPs in the other records. **122 of the 291 FPs are of this kind.** They belong to predictions that are net-positive, so they are not errors to fix.

## 2. Where the 436 FN and 291 FP come from

**FN by annotator support** (a1):

| s/k | GT instances | Recall | FN | Of which never detected |
|---|---|---|---|---|
| 1/1 (single-annotator images) | 556 | 0.62 | **210** | 144 |
| 1/2 | 65 | 0.29 | 46 | 37 |
| 2/2 | 162 | 0.80 | 33 | 12 |
| 1/3 | 77 | 0.19 | 62 | 41 |
| 2/3 | 80 | 0.59 | 33 | 18 |
| 3/3 | 318 | 0.84 | 52 | 12 |

- The model already behaves almost optimally with respect to annotator support. It finds 84% of filaments all three annotators drew and only 19% of those one of three drew; the latter are not worth finding anyway.
- **Half of all FNs (210) are on single-annotator images**, where every miss counts in full.
- **48 FNs are "shape disagreement":** our prediction matched another annotator's version of the same filament but not this one's. These cannot be recovered without losing the other match.
- **FN categories overall:**

| FN category | Count |
|---|---|
| Never detected | 264 |
| Near miss (IoU 0.3–0.5) | 65 |
| Shape disagreement | 48 |
| Inside a prediction covering ≥ 2 GT | 35 |
| Weak overlap | 16 |
| Split | 8 |

**FP by kind:**

| FP kind | FP | Comment |
|---|---|---|
| Partial support: prediction hits some records, FP in the others | **122** | Inherent to the metric; these predictions are net-positive |
| Prediction hits no record: near miss (IoU 0.3–0.5) to some GT | 67 | Shape problem, not a phantom |
| Prediction hits no record: nothing drawn anywhere near it | 64 | True false alarms |
| Prediction hits no record: fragment / weak overlap | 38 | |

The earlier figure of "145 FP where no annotator drew anything" overstated true phantoms. Only **64** FPs have no GT anywhere near them.

## 3. Is our SQ at the human ceiling? Yes, and above it (a2)

| Comparison | SQ | PQ |
|---|---|---|
| Annotator vs annotator, val | 0.627–0.643 | 0.34 |
| Annotator vs annotator, train (1,020 ordered pairs) | 0.633–0.639 | 0.33–0.36 |
| Majority of the *other* annotators vs the held-out one, same filament | mean IoU 0.57–0.58 (only 71–77% above 0.5) | |
| **Our model vs each annotator** | **0.681–0.694** | 0.47–0.48 |

Our matched masks agree with each annotator better than annotators agree with each other. **Mask quality is not where PQ is lost**, which also explains why the crop refiner and SAM could not help.

## 4. Is the number of annotators predictable at test time? No, and it would not matter much (a3)

- **No sample submission or other metadata** exists. The test folder holds only 180 JPEGs.
- **k is set by the annotation batch** (record-id prefix). For example, group 01 has 108 three-annotator images, while groups 03/04/06 are almost all single-annotator. That prefix is not available at test time.
- **Leave-one-out accuracy of predicting k:**

| Predictor | Accuracy |
|---|---|
| Always "1" (majority class) | 0.58 |
| Year | 0.58 |
| Observatory | 0.58 |
| Year + observatory | 0.56 |
| Nearest train image in time | 0.40 |

- **The test set matches train** in observatory shares and years (2011–2022, median 2014).
- **Re-weighting validation to any k mix** moves our pooled PQ only from **0.469 (all k=3) to 0.483 (all k=1)**. The k mix cannot explain the ~0.06 local → LB gap.

## 5. Does the fusion score penalise small filaments? No (a4)

- **Real small filaments appear in nearly every view.** Those under 500 px are present in 10.8 of 12 views on average, the same as large ones.
- **Their confidence is low in every view:** mean 0.23 when present, versus 0.45 for 2–4 kpx.
- **Only 2 candidates** were dropped while being confident in the views where they appeared.
- **Alternative fused scores:**

| Score | In-sample PQ | Split-half PQ |
|---|---|---|
| Baseline: summed confidence / 12 | 0.4765 | 0.4752 |
| Mean confidence when present | 0.4761 | 0.4740 |
| Max confidence | 0.4674 | 0.4658 |
| sqrt(sum/12 × mean) | 0.4761 | 0.4748 |
| Per-model normalisation | 0.4766 | 0.4752 |
| Small filaments boosted ×1.3 | 0.4705 | 0.4674 |

The small-filament problem is model uncertainty, not fusion.

Near the threshold, candidates are real only **27–40%** of the time, which is right at the break-even (~35%) of §1. That is why every threshold or boosting change is flat or negative.

## 6. Other systematic patterns (a5)

**Contrast is the strongest driver of recall:**

| Contrast | < 0.05 | 0.05–0.10 | 0.10–0.15 | > 0.15 |
|---|---|---|---|---|
| Recall | 0.17 | 0.53 | 0.72 | 0.82 |

**Size matters less:**

| Area | < 500 px | 2–4 kpx |
|---|---|---|
| Recall | 0.52 | 0.75 |

Small *and* faint is worst: 0.31 recall at < 500 px with contrast 0.05–0.10.

**Disk position:** recall falls towards the limb.

| r/R | < 0.5 | 0.7–0.85 |
|---|---|---|
| Recall | 0.69 | 0.60 |

**Observatory PQ:**

| Site | B | C | M | U | L | T |
|---|---|---|---|---|---|---|
| PQ | 0.44 | 0.47 | 0.46 | 0.48 | 0.49 | 0.50 |

All within the noise of 13–25 images each.

**Annotator is the largest unexplained factor.** PQ against individual record-id prefixes ranges from **0.37 to 0.55** (median 0.48, 18 prefixes with ≥ 4 records).

| Prefix | PQ | Why |
|---|---|---|
| 010401 | 0.37 | Over-predicts relative to this annotator: precision 0.55 |
| 030101 | 0.41 | Draws many small filaments (10.4 per record): recall 0.45 |

A test set weighted towards annotators like these would score ~0.06 lower. This is a plausible explanation for the LB gap, but it cannot be verified because test annotator identities are hidden.

## 7. Assessment: what is recoverable and roughly what it is worth

Oracle ceilings use the labels. "Realistic" assumes a recovered filament is matched at our current SQ (0.69).

| Loss bucket | Size | Ceiling ΔPQ | Recoverable? | Rough realistic ΔPQ |
|---|---|---|---|---|
| **Missed filaments on single-annotator images (s/k = 1/1)** | 210 FN, 144 never detected; mostly small, low-contrast | +0.136 oracle; +0.086 at model SQ if *all* found | **Main lever.** Needs better detection of faint/small filaments with confidence above the ~35% break-even. Lower thresholds do not work (candidates are at break-even) | +0.01 to +0.03 if a better detector recovers a quarter to a third of them at break-even precision |
| Missed 2/2, 2/3, 3/3 filaments | 32 clusters fully missed | part of the row above | Same lever; high value per recovery | ≤ +0.01 |
| Missed 1/2 filaments | 46 | +0.020 oracle, +0.008 realistic | Marginal | ~+0.003 |
| **Missed 1/3 filaments** | 62 | +0.014 oracle, **−0.001 realistic** | **Not worth recovering**: break-even at best | 0 |
| Shape-disagreement FN | 48 | — | **Not recoverable**: fixing one annotator breaks another | 0 |
| Partial-support FP | 122 | — | **Not a loss**: these predictions are net-positive | 0 |
| FP from predictions hitting no record (phantoms + near misses + fragments) | 169 FP / 121 predictions | **+0.037** (oracle pruning) | Partly. Needs a better keep/discard signal than confidence. Our LightGBM scorer and the refiner's keep head both failed, so confidence already carries most of the available signal | +0.005 to +0.01 |
| Near misses (IoU 0.3–0.5), FN and FP sides | ~65 FN + 67 FP | ~+0.03 if all converted (each ≈ +0.0005) | Hard. Mask agreement is already above human level; the leave-one-out consensus misses 23–29% of the time | ≤ +0.005 |
| Merged / split predictions | 35 + 8 FN | ~+0.01 | Splitting merged neighbours is plausible in post-processing (cheap to test) | ~+0.003 |
| Mask quality (SQ) | — | ~0 | Already above the human ceiling | 0 |
| Local → LB gap | ~0.06 | — | Not about k mix (≤ 0.007). Most consistent with the test set's annotator mix; not controllable | 0 |

## Bottom line

About **0.04 of PQ (oracle pruning) plus 0.09 (finding every worthwhile missed filament)** separates us from a label-aware oracle.

Of that:
- **Most of the recoverable part is recall of small, low-contrast filaments on single-annotator images.** This needs a better detector, specifically one whose confidence on faint filaments rises above ~35% precision. Post-processing or score re-weighting cannot do it; the candidates near the threshold are already at break-even.
- **Mask shape, annotator-count handling and fusion are not the bottleneck.**
- A realistic target from better small-filament detection plus modest FP pruning is **+0.02 to +0.04 local PQ** (≈ 0.50–0.52). It would likely remain ~0.06 lower on the public LB, because of who annotated the test set.

---

# Update (2026-10-16): cheap checks and the 3-channel input

## Cheap checks (cached predictions and labels only)

**Annotator identity is not predictable from a test file name** (`b1_annotator_identity.md`).
- All records of an image share one batch, but batches mix freely across dates and sites. On the 50 days with ≥ 2 images, only 6% have all their images in one batch.
- Leave-one-out accuracy:

| Target | Majority class | Best predictor |
|---|---|---|
| group | 0.31 | ≤ 0.32 |
| batch | 0.11 | ≤ 0.10 |
| first annotator | 0.10 | ≤ 0.08 |
| "has a hard annotator" | 0.54 | 0.58 |
| k | 0.58 | ≤ 0.58 |

- Predictors tried: year, observatory, year + observatory, hour, nearest image in time.

**Per-annotator post-processing** (`b2_per_annotator_pp.md`).
- Annotators were split at the median per-prefix PQ (101 hard records / 69 easy).
- Both subsets have the **same optimum**: conf 0.275, `union_cc`, erode 1, min area 150.

| Subset | PQ (in-sample) |
|---|---|
| Hard annotators | 0.445 |
| Easy annotators | 0.518 |

- The split is defined from the same errors, so the gap between subsets is optimistic.
- The conf difference is 0.00 < 0.05, so no per-subset CSVs were written.
- Written as requested: `submission_ens_b_a_c_conf035.csv` (conf 0.35, rest unchanged; 1,025 filaments over 176 images).

**Near-threshold band [0.15, 0.45)** (`b3_band_ceiling.md`): 681 fused candidates, 290 real (42.6%).
- "Real" means mean matched IoU across annotators > PQ/2.
- Share real by band: 0.33 at 0.15–0.25, 0.34 at 0.25–0.275, **0.54 at 0.275–0.35**, 0.57 at 0.35–0.45.
- Oracle ceilings (labels used):

| Perfect re-ranker scope | PQ | ΔPQ |
|---|---|---|
| Whole band | **0.519** | **+0.043** |
| Pure FP removal above 0.275 | | +0.023 |
| Pure recall below 0.275 | | +0.016 |
| All candidates ≥ 0.05 | 0.550 | |

- A re-ranker that captures a quarter to a third of the band ceiling is worth ~+0.01.

## Input representation change

**`src/channels.py`** (deterministic; precomputed as lossless PNGs by `src/build_c3.py`; off-disk = 0 in all channels):

| Channel | Content |
|---|---|
| ch0 | raw intensity |
| ch1 | raw ÷ azimuthal median per 2 px radius bin, encoded (ratio − 0.5)·255 |
| ch2 | ch1 minus its σ = 24 px on-disk background, encoded 128 + 400·Δ |

- **Disk:** the limb is located as the steepest radial drop along 360 rays, then fitted with a least-squares circle (2 outlier-rejection passes).

**Sanity checks** (`c1_channel_check.md`, panels in `channels/`):
- The fitted limb is 72–83 px inside the threshold circle (the halo bias is removed).
- Only **36 of 17.4 M** annotated pixels (one image) fall outside the fitted disk.
- **Limb darkening:** raw intensity falls to 0.64 × centre at r/R 0.95–1.0. The flattened channel stays at 1.000–1.006 across all radii.
- **Visual:** limb-region and faint filaments are clearly more visible in ch1/ch2. One active-region miss is barely visible in any channel (looks like label ambiguity).

**Training:** YOLO26m @ 2048, same recipe, split and labels as `y26m_2048` (40 epochs, ~9.3 min/epoch).
- Best mask mAP50 **0.672** (epoch 34), against 0.665 for grayscale.
- Then 4-view TTA, tuning, split-half check (`c2_eval_c3.md`).

| Model | PQ in-sample | PQ split-half (held-out) | TP | FP | FN |
|---|---|---|---|---|---|
| 2048 grayscale (`y26m_2048`) | 0.468 | 0.465 ± 0.010 | 804 | 298 | 454 |
| **2048 3-channel (`y26m_2048_c3`)** | **0.464** | **0.462 ± 0.013** | 814 | 331 | 444 |
| Ensemble b+a+c (baseline) | **0.4765** | **0.4752 ± 0.012** | 822 | 291 | 436 |
| Ensemble c3+a+c (c3 replaces b) | 0.472 | 0.470 ± 0.012 | 818 | 302 | 440 |
| Ensemble b+a+c+c3 (c3 added, 16 views) | 0.475 | 0.4746 ± 0.011 | 820 | 298 | 438 |

**Recall by GT contrast** (single models, then ensembles):

| Contrast | n GT | 2048 gray | 2048 c3 | Ensemble baseline | Ensemble +c3 |
|---|---|---|---|---|---|
| < 0.05 | 46 | 0.174 | 0.174 | 0.174 | 0.152 |
| 0.05–0.10 | 449 | 0.501 | **0.512** | 0.535 | 0.530 |
| 0.10–0.15 | 506 | 0.698 | **0.721** | 0.715 | 0.717 |
| 0.15–0.20 | 222 | 0.842 | 0.820 | 0.824 | 0.824 |

**Recall by r/R:**

| r/R | n GT | 2048 gray | 2048 c3 | Ensemble baseline | Ensemble +c3 |
|---|---|---|---|---|---|
| < 0.5 | 597 | 0.672 | 0.678 | 0.692 | 0.688 |
| 0.5–0.7 | 483 | 0.621 | 0.619 | 0.627 | 0.627 |
| 0.7–0.85 | 175 | 0.583 | **0.623** | 0.600 | 0.600 |

**Reading:**
- The new input moves recall where expected: +0.04 near the limb and +0.01 to +0.02 at contrast 0.05–0.15. That is +10 TP net for the single model.
- But it also produces **+33 FP**, so single-model PQ is −0.004 (in-sample) and −0.003 (held-out). Both differences are well inside the split-half SD (~0.01).
- The lowest-contrast bin (< 0.05, 46 GT) does not move at all. Those filaments are close to invisible in any channel; they reflect annotator judgement, not pixels the model fails to read.
- In the ensemble, c3 adds nothing: 0.4746 held-out vs 0.4752.
- **Conclusion:** the limb-darkening / local-contrast representation is not where the remaining PQ is. The baseline ensemble stays the best entry. `submission_y26m_2048_c3.csv` exists if wanted for a probe, but is not recommended.

## What I would do next (plans only; not started)

The two items below are sketched, as requested, and not built. Given the c3 result, I would run the re-ranker first (cheaper, measurable ceiling); the crop model second, only if the re-ranker shows a held-out gain.

**1. Semantic re-ranker for the band [0.15, 0.45)** (ceiling +0.043, realistic ~+0.01). Plan:
- **Model:** U-Net (ResNet-34, torchvision weights, as in `src/refiner.py`) on native-resolution 1024 crops of the training split.
- **Target:** per-annotator records as separate samples (soft by construction), BCE + Dice.
- **Inference:** sliding window with overlap, producing a full-disk probability map.
- **Features per fused candidate:** mean / max / quantiles of the probability inside the candidate mask, and the fraction of the mask above 0.5.
- **Re-rank** only band candidates: keep if a logistic blend of (ensemble score, mean semantic probability) passes a threshold. Fit the blend on half the validation images and evaluate on the other half (and swap). Candidates outside the band keep the current rule.
- **Go/no-go:** a held-out gain > +0.005 (about half the split-half SD) before any test submission.
- **Cost:** ~3 h training + ~20 min inference.
- **Caveat:** the LightGBM scorer (§5.5 of `EXPERIMENTS.md`) failed with hand features, so the semantic probability has to carry new information; the held-out check decides.

**2. Crop-trained detector with low-contrast oversampling.** Plan:
- **Data:** a dataset of 1024 px crops at imgsz 2048 (2× zoom), with 4 crops per record:
  - 2 random;
  - 2 centred on GT filaments, with sampling weight ∝ 1 / contrast for the lowest-contrast tercile;
  - crops must contain ≥ 1 GT or be kept as negatives at a 1:4 ratio.
- **Training:** same YOLO26m recipe, ~30 epochs.
- **Inference:** a 3×3 tiled pass at 2× zoom (as in `src/tiled_infer.py`), fused with full-image predictions using the recovery rule from `EXPERIMENTS.md` §5.8.
- **Evidence:** the tiled test with an untrained-for-zoom model recovered 50 FN but at ~30% precision; a zoom-trained model should raise that above the ~35% break-even.
- **Go/no-go:** held-out recovery precision ≥ 0.4 on added detections.
- **Cost:** ~6–7 h training.

---

# Update (2026-10-16, later): decision on the 3-channel input, and the re-ranker plan

## Branch taken

- **Held-out ensemble PQ with c3 added: 0.4746 (split-half)**, against 0.4752 for the baseline. This is below the 0.485 bar, so we take **branch 2**.
- No further channel recipes.
- c3 is **not** kept as an ensemble member, because it adds no held-out PQ.
- Note: c3 and c2 had already finished before these instructions arrived. The numbers are the ones in the previous update.

## Post-hoc input checks (`c3_input_checks.md`)

**Test set vs disk fit:** 0 of 2.40 M predicted filament pixels (baseline ensemble, 180 test images) fall outside the fitted disk. Zeroing off-disk is safe on test as well as train (36 of 17.4 M GT pixels).

**Per-channel distributions** (on-disk, 120 train images, uint8):

| Channel | Mean | Within-image SD | SD of image means |
|---|---|---|---|
| ch0 raw | 122.8 | 21.7 | 4.4 |
| ch1 flattened | 126.9 | 27.3 | 0.9 |
| ch2 contrast | 126.8 | 30.0 | 0.9 |

Channel 2 is centred and scaled comparably to channels 0 and 1, so **channel scale is not the reason c3 underperformed.** ch1/ch2 also remove most between-image brightness variation (SD of image means 0.9 vs 4.4).

**σ = 24 px background vs large filaments:** decoded ch2 contrast inside GT masks, relative to 1–2 kpx filaments:

| Area | < 500 | 2–4 k | 4–8 k | > 8 k |
|---|---|---|---|---|
| Relative ch2 contrast | 121% | 91% | 83% | 63% |

Large filaments are attenuated but not flattened out. ch0/ch1 carry them fully (flattened ratio 0.89–0.91 at every size).

## Why c3 did not help

**Training curve:** c3 trained normally and tracked the grayscale run closely. Best mask mAP50 was 0.672 (epoch 34) vs 0.665, with similar epoch-to-epoch noise. Training was not the problem.

**Recall moved where intended:**
- +0.04 at r/R 0.7–0.85.
- +0.01 to +0.02 at contrast 0.05–0.15.

**The added detections are near-threshold quality:**
- +10 TP but +33 FP for the single model.
- In the ensemble they duplicate what the grayscale members already find.

**The lowest-contrast filaments are untouched:** recall at contrast < 0.05 stays 0.17. The annotations there are not supported by visible signal in any representation.

**Reading:** the YOLO backbone already normalises limb darkening and local contrast internally well enough. Making them explicit changes which marginal candidates appear, not how separable real and spurious candidates are.

## Semantic re-ranker: plan (being built now)

**Sizing:**
- Oracle ceiling for the band [0.15, 0.45) is **+0.043**; realistic target **~+0.01**.
- So the effort is one model, one feature, one 2-parameter rule. No feature engineering, no stacking, no second model.

**Model:** U-Net with a ResNet-34 encoder (ImageNet weights; class reused from `src/refiner.py`).
- Input: grayscale replicated to 3 channels, as in the baseline.
- Native-resolution 768 px crops: 70% centred on a GT filament of the record (±200 px jitter), 30% at random on the disk.
- Target: the union mask of **one annotator record**. Each record is a separate sample, so the pixel probability estimates the share of annotators who would draw that pixel.
- Loss BCE + soft Dice; dihedral and brightness augmentation; ~12 epochs.

**Data hygiene:**
- Trained on the **training split only** (601 images / 984 records; same file-grouped split, seed 42).
- Validation maps are therefore out-of-sample: the re-ranker is tuned and scored only on validation candidates whose labels the semantic model never saw.
- Test maps come from the same model.

**Inference:** 1024 px tiles at stride 512 with cosine blending, producing a full-disk probability map.

**One agreement feature:** `agree` = mean probability inside each fused ensemble candidate's mask (12-view baseline fusion, before union/erosion).

**Keep rule** (2 parameters):
- Candidates with conf ≥ 0.45: keep.
- conf < 0.15: drop.
- Inside the band: keep iff conf + w·(agree − 0.5) ≥ t.
- w = 0, t = 0.275 is exactly the baseline. The unchanged post-processing (`union_cc`, erode 1, min area 150) follows.

**Tuning: split-half only.** (w, t) is chosen on one half of the validation images and scored on the other, both directions, 5 random halvings. The baseline (w = 0, best t on the same half) is scored identically.

**Decision rule:**
- The headline is held-out ensemble PQ vs 0.4765 / 0.4752.
- Any **in-sample gain under +0.01 is noise.**
- A submission CSV is written only if the held-out PQ beats the baseline's held-out PQ.

**Cost:** ~1 h training + ~20 min inference (one GPU job), CPU tuning.

## Semantic re-ranker: result (`d1_rerank.md`)

**Model:** U-Net (ResNet-34) trained on the training split only (601 images / 984 records), 12 epochs. Final loss 0.380 (from 1.317).
- Validation agreement maps are out-of-sample.
- Feature: mean probability inside each fused candidate's mask.

**Separability inside the band [0.15, 0.45)** (681 candidates, 42.6% real):

| Score | AUC for "real" |
|---|---|
| Ensemble confidence | **0.630** |
| Semantic agreement | **0.569** |
| conf + 0.5·(agree − 0.5) | 0.624 |

- Agreement is higher for real candidates (mean 0.78 vs 0.70), but it ranks them **worse** than the ensemble's own confidence.
- It adds no information on top of confidence.

**Keep rule:**

| Rule | PQ in-sample | PQ held-out (split-half) |
|---|---|---|
| Baseline (w = 0) | 0.4765 (t 0.275) | 0.4752 ± 0.012 |
| Re-ranked: conf + w·(agree − 0.5) ≥ t | 0.4765 (best is w = 0) | 0.4745 ± 0.011 |

- Held-out delta: **−0.0007** (SD 0.0021). 0 of 10 halves positive; 9 of 10 selected w = 0.
- In-sample, the grid's best is w = 0. Every w > 0 lowers in-sample PQ monotonically.
- Verdict: **no gain.** No submission was written.

**Why:**
- In the band, the ensemble confidence is already the best available ranking signal.
- A second model trained on the same labels learns the same notion of "filament-like". Inside the band it disagrees with the ensemble mostly where annotators themselves disagree, and that is not predictable from pixels.
- This matches the LightGBM keep-scorer (−0.013) and the refiner keep head, now with an independently trained out-of-sample model.

## Overall assessment after this round

**Held-out ensemble PQ is unchanged at 0.4752 (in-sample 0.4765).** All attempts this round were at or below it on held-out halves:
- 3-channel input: 0.4746 added / 0.470 replacing.
- Per-annotator post-processing: same optimum.
- Semantic re-ranker: 0.4745.

**The quantified headroom has proven hard to reach:**
- **Band re-ranking ceiling (+0.043):** the realistic share is ≈ 0 with any signal we can produce. Two independent re-rankers and a learned scorer found nothing beyond confidence.
- **Recall of low-contrast filaments:** the input change moved recall where expected (limb +0.04, contrast 0.05–0.15 +0.01 to +0.02) but bought as many FPs as TPs. The < 0.05 contrast bin did not move.

**What I would do next, and what I would not:**
1. **Not** the 1024-px crop detector as originally scoped. Its premise is that low-contrast filaments are under-detected for lack of scale. The c3 result shows the added candidates at that contrast arrive at roughly break-even precision. The earlier zoomed-tile test (§5.8 of `EXPERIMENTS.md`) showed the same (~30% precision). I would only run it as a cheap 10-epoch pilot, with the go/no-go set at **≥ 40% held-out precision** on added detections.
2. **Bank the current pipeline.**
   - Retrain the three ensemble members on all 707 images with their original recipes (~16 h GPU, one job at a time).
   - Re-use the validation-tuned post-processing (conf 0.275, `union_cc`, erode 1, min area 150).
   - This is the one remaining change with a reliable expected sign: 15% more data, typically +0.005 to +0.01.
3. **Shift remaining effort to the deliverables** that are 30% of the judging (report, reproducible repo, the organizers' IoU/Dice/one-to-many plots). The quantitative part is close to what the labels allow from this data.

**Submission files from this round:**
- `submission_ens_b_a_c_conf035.csv`: requested LB probe; conf 0.35; in-sample 0.461 locally.
- `submission_y26m_2048_c3.csv`: single c3 model; 0.464 in-sample / 0.462 held-out; not recommended.
- No configuration beat the baseline held-out, so no other CSVs were written.

---

# Update (2026-10-16, evening): final queue, figures, report draft, repo audit

**conf035 LB score:** not available. `submission_ens_b_a_c_conf035.csv` was written but not submitted; it needs a go-ahead. The full-data step reuses the validation-tuned conf 0.275 unchanged either way.

**Single-model LB check:** `submission_y26m_2048.csv` (local 0.468 in-sample / 0.465 held-out) scored **LB 0.39**, against 0.40 for the ensemble. Local PQ again predicts the LB ordering.

**GPU queue (one job at a time, `src/run_final_queue.sh`, systemd `filament-rfdetr` → `filament-finalq`):**
1. **RF-DETR-Seg Large**, 40 epochs, tiled 2×2 × 1100 px @ 768, original split (running).
   - Then 4-view flip TTA tiled inference (`src/rfdetr_infer.py`), and evaluation alone and as a 4th member with the same fusion, post-processing and split-half (`src/analysis/e1_eval_rfdetr.py`).
   - Kept only if the ensemble gains ≥ +0.005 held-out.
2. **Full-data retrain** (707 images) of the three YOLO26m members, plus RF-DETR if kept.
   - Original recipes; last-epoch weights, since validation is now in training.
   - Then test inference and `submission_ens_full.csv`, with the validation-tuned post-processing carried over unchanged.
   - Sanity check vs the current ensemble (`e2_full_ensemble.py`): masks per image, area distribution, IoU agreement; flags if agreement < 0.70, mean |Δ masks/image| > 1.5, or median area ratio outside [0.8, 1.25].
3. **Crop-detector pilot**, 10 epochs: 1024 px crops of the 3-channel PNGs at imgsz 2048, with low-contrast oversampling (`src/build_crops.py`).
   - Then tiled validation inference and the add-only gate: ≥ 40% of added held-out detections real, at ≥ 20 added (`e3_crop_pilot.py`).

**Figures** (`work/report_figs/`, `src/report/figures.py`; current best ensemble, validation):
- **Organizer-required:** IoU and Dice distributions (mean IoU of overlapping pairs 0.610); many-to-one / one-to-many. 962 GT touched by exactly one prediction, 262 by none, 33 by two; 33 predictions touch two GT.
- **Ours:**
  - PQ vs threshold with split-half range (peak 0.4765 at 0.275).
  - Recall by contrast / r/R / area.
  - PQ by observatory (0.441–0.497).
  - Qualitative panels (found / missed with no prediction near it / 3-annotator disagreement: 10 / 16 / 7 filaments drawn).

**Report draft:** `report/REPORT_DRAFT.md`, in the requested section order. Pending numbers are marked ⏳.

**Repo audit:**
- `Makefile`: one target per stage (data, prepare, train, infer, tune, submit, full, analysis, figures).
- `src/ensemble.py`: rebuilds the ensemble cache byte-identically to the one behind 0.4765.
- `envs/`: pinned requirements for the main environment, `.venv-gbm` and `.venv-rfdetr` (with the torch-reuse setup).
- Superseded runner scripts moved to `src/archive/runners/`; nothing deleted. The SAM weights moved to `work/weights/`.
- Top-level `README.md` mapped to the report sections.
