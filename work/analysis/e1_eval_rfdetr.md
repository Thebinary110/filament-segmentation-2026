## PQ (in-sample tuned; split-half = threshold chosen on held-out halves)

| model | PQ_in_sample | PQ_split_half | split_half_sd | TP | FP | FN | SQ | best |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-Seg L alone | 0.434 | 0.4315 | 0.01433 | 750 | 331 | 508 | 0.6767 | conf 0.35 paint grow 0 min 300 |
| ensemble b+a+c (baseline) | 0.4765 | 0.4752 | 0.01154 | 822 | 291 | 436 | 0.6873 | conf 0.275 union_cc grow -1 min 150 |
| ensemble b+a+c+r (RF-DETR added) | 0.4684 | 0.4644 | 0.01275 | 798 | 298 | 460 | 0.6909 | conf 0.3 union_cc grow -1 min 150 |
| ensemble r+a+c (RF-DETR replaces 2048) | 0.4684 | 0.4678 | 0.01179 | 813 | 314 | 445 | 0.687 | conf 0.3 union_cc grow -1 min 150 |

## Decision

- held-out delta of adding RF-DETR: -0.0107 (keep needs >= +0.005) -> DROP
