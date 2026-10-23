## Baseline

PQ 0.4765; TP 822, FP 291, FN 436, SQ 0.6873, D 1185.5

## Theory: is a prediction for a filament drawn by s of k annotators worth making? (q = model SQ)

| k | s | frac | dS | dD | net_per_pred | dPQ_x1000 | worth_it |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1/1 | 0.6873 | 0.5 | 0.449 | 0.3787 | yes |
| 2 | 1 | 1/2 | 0.6873 | 1 | 0.2107 | 0.1778 | yes |
| 2 | 2 | 2/2 | 1.375 | 1 | 0.898 | 0.7575 | yes |
| 3 | 1 | 1/3 | 0.6873 | 1.5 | -0.02754 | -0.02323 | no |
| 3 | 2 | 2/3 | 1.375 | 1.5 | 0.6597 | 0.5565 | yes |
| 3 | 3 | 3/3 | 2.062 | 1.5 | 1.347 | 1.136 | yes |

## GT instances by (k, s): recall

| k | s | GT | hits | FN | recall | median_area |
|---|---|---|---|---|---|---|
| 1 | 1 | 556 | 346 | 210 | 0.6223 | 1094 |
| 2 | 1 | 65 | 19 | 46 | 0.2923 | 927 |
| 2 | 2 | 162 | 129 | 33 | 0.7963 | 1436 |
| 3 | 1 | 77 | 15 | 62 | 0.1948 | 884 |
| 3 | 2 | 80 | 47 | 33 | 0.5875 | 1028 |
| 3 | 3 | 318 | 266 | 52 | 0.8365 | 1638 |

## FN taxonomy (all)

| cat | FN |
|---|---|
| FN: never detected | 264 |
| FN: near miss 0.3-0.5 | 65 |
| FN: filament hit for another annotator (shape disagreement) | 48 |
| FN: inside a pred covering >=2 GT | 35 |
| FN: weak overlap <0.3 | 16 |
| FN: split over >=2 preds | 8 |

## FN taxonomy by s/k

| cat | 1/1 | 1/2 | 1/3 | 2/2 | 2/3 | 3/3 |
|---|---|---|---|---|---|---|
| FN: filament hit for another annotator (shape disagreement) | 0 | 0 | 0 | 11 | 3 | 34 |
| FN: inside a pred covering >=2 GT | 18 | 4 | 4 | 3 | 6 | 0 |
| FN: near miss 0.3-0.5 | 39 | 3 | 8 | 4 | 5 | 6 |
| FN: never detected | 144 | 37 | 41 | 12 | 18 | 12 |
| FN: split over >=2 preds | 3 | 1 | 2 | 2 | 0 | 0 |
| FN: weak overlap <0.3 | 6 | 1 | 7 | 1 | 1 | 0 |

## Missed clusters (no annotator's version hit) by (k, s)

| k | s | missed_clusters |
|---|---|---|
| 1 | 1 | 210 |
| 2 | 1 | 46 |
| 2 | 2 | 11 |
| 3 | 1 | 62 |
| 3 | 2 | 15 |
| 3 | 3 | 6 |

## Final predictions by (k, records hit): value = sum IoU - PQ*k/2

| k | n_hit | preds | mean_sum_iou | FP_generated | total_value | dPQ_if_removed |
|---|---|---|---|---|---|---|
| 1 | 0 | 92 | 0 | 92 | -21.92 | 0.01849 |
| 1 | 1 | 346 | 0.694 | 0 | 157.7 | -0.133 |
| 2 | 0 | 10 | 0 | 20 | -4.765 | 0.00402 |
| 2 | 1 | 30 | 0.6566 | 30 | 5.403 | -0.004557 |
| 2 | 2 | 59 | 1.386 | 0 | 53.69 | -0.04529 |
| 3 | 0 | 19 | 0 | 57 | -13.58 | 0.01146 |
| 3 | 1 | 25 | 0.6867 | 50 | -0.7019 | 0.000592 |
| 3 | 2 | 42 | 1.328 | 42 | 25.75 | -0.02172 |
| 3 | 3 | 73 | 2.06 | 0 | 98.18 | -0.08282 |

## Where the 291 FP come from

| fp_kind | preds | FP |
|---|---|---|
| pred hits nobody: near miss (0.3-0.5) to some GT | 50 | 67 |
| pred hits nobody: no annotator drew anything there | 42 | 64 |
| pred hits nobody: weak overlap / fragment | 29 | 38 |
| pred hits some records, FP in the others (partial support) | 575 | 122 |

## Oracle experiments (real scorer; upper bounds, NOT achievable as-is)

| experiment | PQ | dPQ | TP | FP | FN | SQ | realistic_PQ_at_model_SQ |
|---|---|---|---|---|---|---|---|
| baseline | 0.4765 | 0 | 822 | 291 | 436 | 0.6873 |  |
| oracle prune: drop every pred with negative marginal value | 0.5151 | 0.03852 | 807 | 92 | 451 | 0.6883 |  |
| oracle prune: drop preds that hit no record | 0.5131 | 0.03657 | 822 | 122 | 436 | 0.6873 |  |
| oracle add missed clusters, s/k = 1 (227 clusters) | 0.6121 | 0.1356 | 1067 | 296 | 191 | 0.7518 | 0.5622 |
| oracle add missed clusters, s/k >= 2/3 (242 clusters) | 0.6191 | 0.1425 | 1096 | 312 | 162 | 0.753 | 0.5682 |
| oracle add missed clusters, s/k >= 1/2 (288 clusters) | 0.6318 | 0.1553 | 1142 | 358 | 116 | 0.7629 | 0.5721 |
| oracle add missed clusters, all missed clusters (350 clusters) | 0.634 | 0.1575 | 1204 | 482 | 54 | 0.7751 | 0.5649 |
| oracle add missed clusters, only s/k = 1/3 (62 clusters) | 0.4904 | 0.01383 | 884 | 415 | 374 | 0.7092 | 0.4752 |
| oracle add missed clusters, only s/k = 1/2 (46 clusters) | 0.4961 | 0.01955 | 868 | 337 | 390 | 0.7038 | 0.4844 |
