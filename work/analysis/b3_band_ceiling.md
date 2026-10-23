## Band summary

Band [0.15, 0.45): 681 candidates, 290 real (42.6%). Of these, 289 are currently kept (>= 0.275), 161 of them real; 392 are currently dropped, 129 of them real.

## Fused candidates by confidence band (real = mean matched IoU across annotators > PQ/2)

| band | candidates | real | share_real |
|---|---|---|---|
| <0.05 | 1380 | 271 | 0.1964 |
| 0.05-0.15 | 688 | 206 | 0.2994 |
| 0.15-0.25 | 334 | 109 | 0.3263 |
| 0.25-0.275 | 58 | 20 | 0.3448 |
| 0.275-0.35 | 153 | 83 | 0.5425 |
| 0.35-0.45 | 136 | 78 | 0.5735 |
| >=0.45 | 492 | 423 | 0.8598 |

## Oracle re-ranking ceilings (labels used; upper bounds)

| experiment | PQ | dPQ | TP | FP | FN |
|---|---|---|---|---|---|
| baseline (conf >= 0.275) | 0.4765 | 3.245e-05 | 822 | 291 | 436 |
| perfect re-ranker of [0.15, 0.45): keep >= 0.45 + real band candidates | 0.5192 | 0.04267 | 888 | 206 | 370 |
| perfect re-ranker of [0.15, 0.45) applied only above 0.275 (pure FP removal) | 0.4993 | 0.02276 | 811 | 172 | 447 |
| perfect re-ranker of [0.15, 0.275) only (pure recall add) | 0.4923 | 0.01577 | 881 | 314 | 377 |
| perfect re-ranker of all candidates >= 0.05 | 0.5496 | 0.07306 | 969 | 187 | 289 |
