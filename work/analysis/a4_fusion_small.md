## Real filaments (fused instance hits >=1 record): views present and fused score by size

| area_bin | true_instances | mean_views | mean_models | mean_conf_present | mean_fused_score | share_dropped_at_0_275 |
|---|---|---|---|---|---|---|
| <500 | 69 | 10.84 | 2.971 | 0.2319 | 0.2199 | 0.6377 |
| 500-1k | 255 | 10.71 | 2.91 | 0.3325 | 0.3198 | 0.4745 |
| 1k-2k | 356 | 10.57 | 2.871 | 0.3859 | 0.3721 | 0.4213 |
| 2k-4k | 228 | 10.76 | 2.912 | 0.4454 | 0.4326 | 0.3684 |
| >4k | 186 | 9.624 | 2.79 | 0.3887 | 0.354 | 0.5 |

## All fused candidates (max conf >= 0.1): share real and mean metric value, by size x views present

| area_bin | views_bin | instances | share_true | mean_value | mean_conf_present |
|---|---|---|---|---|---|
| <500 | 1-3 | 6 | 0.1667 | -0.267 | 0.0907 |
| <500 | 4-6 | 18 | 0.1667 | -0.2421 | 0.09057 |
| <500 | 7-9 | 34 | 0.2647 | -0.2416 | 0.1082 |
| <500 | 10-12 | 143 | 0.3916 | -0.0969 | 0.2036 |
| 500-1k | 1-3 | 27 | 0.1481 | -0.2043 | 0.102 |
| 500-1k | 4-6 | 67 | 0.2836 | -0.1883 | 0.09992 |
| 500-1k | 7-9 | 78 | 0.3333 | -0.1322 | 0.1097 |
| 500-1k | 10-12 | 413 | 0.4988 | 0.0462 | 0.2972 |
| 1k-2k | 1-3 | 58 | 0.3621 | -0.06763 | 0.1204 |
| 1k-2k | 4-6 | 64 | 0.3125 | -0.1404 | 0.09646 |
| 1k-2k | 7-9 | 87 | 0.3793 | -0.09662 | 0.1126 |
| 1k-2k | 10-12 | 447 | 0.6309 | 0.1861 | 0.3781 |
| 2k-4k | 1-3 | 17 | 0.4118 | -0.06977 | 0.1419 |
| 2k-4k | 4-6 | 48 | 0.3542 | -0.1434 | 0.1041 |
| 2k-4k | 7-9 | 57 | 0.2982 | -0.1978 | 0.1205 |
| 2k-4k | 10-12 | 301 | 0.6213 | 0.2269 | 0.4297 |
| >4k | 1-3 | 40 | 0.375 | -0.06477 | 0.1177 |
| >4k | 4-6 | 55 | 0.4182 | 0.04136 | 0.137 |
| >4k | 7-9 | 63 | 0.4603 | 0.05208 | 0.1773 |
| >4k | 10-12 | 194 | 0.6134 | 0.2049 | 0.4192 |

## Candidates dropped by the 0.275 threshold: are they confident where present?

| confident_when_present | area_bin | dropped_instances | of_which_true | share_true | mean_views | mean_value |
|---|---|---|---|---|---|---|
| False | <500 | 163 | 44 | 0.2699 | 9.773 | -0.2088 |
| False | 500-1k | 412 | 121 | 0.2937 | 9.206 | -0.1579 |
| False | 1k-2k | 396 | 150 | 0.3788 | 8.343 | -0.08706 |
| False | 2k-4k | 236 | 84 | 0.3559 | 8.631 | -0.1116 |
| False | >4k | 227 | 91 | 0.4009 | 7.374 | -0.03612 |
| True | >4k | 2 | 2 | 1 | 4 | 0.9281 |

## Alternative fused scores (union_cc, erode 1, min area 150; threshold tuned; in-sample vs split-half)

| score | best_thr | PQ_in_sample | PQ_split_half | split_half_sd | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| baseline: sum conf / 12 | 0.275 | 0.4765 | 0.4752 | 0.01154 | 822 | 291 | 436 |
| mean conf when present | 0.275 | 0.4761 | 0.474 | 0.0125 | 824 | 298 | 434 |
| max conf | 0.4 | 0.4674 | 0.4658 | 0.01321 | 801 | 288 | 457 |
| sqrt(sum/12 * mean) | 0.275 | 0.4761 | 0.4748 | 0.012 | 822 | 294 | 436 |
| sum / 12, small (<1k px) boosted x1.3 | 0.3 | 0.4705 | 0.4674 | 0.01104 | 815 | 311 | 443 |
| sum / (4 x models present) | 0.275 | 0.4766 | 0.4752 | 0.01153 | 822 | 291 | 436 |
