## PQ: single models and ensembles (best = in-sample tuned; split-half = threshold chosen on held-out halves)

| model | PQ_in_sample | PQ_split_half | split_half_sd | TP | FP | FN | SQ | best |
|---|---|---|---|---|---|---|---|---|
| 2048 gray (y26m_2048) | 0.4677 | 0.4652 | 0.01046 | 804 | 298 | 454 | 0.6864 | conf 0.275 paint grow -1 min 0 |
| 2048 3-channel (y26m_2048_c3) | 0.4639 | 0.4619 | 0.01264 | 814 | 331 | 444 | 0.6847 | conf 0.275 union_cc grow -1 min 150 |
| ensemble b+a+c (baseline) | 0.4765 | 0.4752 | 0.01154 | 822 | 291 | 436 | 0.6873 | conf 0.275 union_cc grow -1 min 150 |
| ensemble c3+a+c (c3 replaces b) | 0.4719 | 0.4696 | 0.01236 | 818 | 302 | 440 | 0.6859 | conf 0.275 union_cc grow -1 min 150 |
| ensemble b+a+c+c3 (c3 added) | 0.475 | 0.4746 | 0.01062 | 820 | 298 | 438 | 0.6881 | conf 0.275 union_cc grow -1 min 150 |

## Recall by GT contrast

| contrast | GT | 2048 gray (y26m_2048) | 2048 3-channel (y26m_2048_c3) | ensemble b+a+c (baseline) | ensemble c3+a+c (c3 replaces b) | ensemble b+a+c+c3 (c3 added) |
|---|---|---|---|---|---|---|
| <0.05 | 46 | 0.1739 | 0.1739 | 0.1739 | 0.1522 | 0.1522 |
| 0.05-0.10 | 449 | 0.5011 | 0.5122 | 0.5345 | 0.5367 | 0.5301 |
| 0.10-0.15 | 506 | 0.6976 | 0.7213 | 0.7154 | 0.7115 | 0.7174 |
| 0.15-0.20 | 222 | 0.8423 | 0.8198 | 0.8243 | 0.8153 | 0.8243 |
| >0.20 | 35 | 0.8857 | 0.8286 | 0.8286 | 0.8286 | 0.8286 |

## Recall by GT distance from disk centre (r/R)

| r_rel | GT | 2048 gray (y26m_2048) | 2048 3-channel (y26m_2048_c3) | ensemble b+a+c (baseline) | ensemble c3+a+c (c3 replaces b) | ensemble b+a+c+c3 (c3 added) |
|---|---|---|---|---|---|---|
| <0.5 | 597 | 0.6717 | 0.6784 | 0.6918 | 0.6834 | 0.6884 |
| 0.5-0.7 | 483 | 0.6211 | 0.619 | 0.6273 | 0.6273 | 0.6273 |
| 0.7-0.85 | 175 | 0.5829 | 0.6229 | 0.6 | 0.6057 | 0.6 |
| >0.85 | 3 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |

## Recall by GT area

| area | GT | 2048 gray (y26m_2048) | 2048 3-channel (y26m_2048_c3) | ensemble b+a+c (baseline) | ensemble c3+a+c (c3 replaces b) | ensemble b+a+c+c3 (c3 added) |
|---|---|---|---|---|---|---|
| <500 | 181 | 0.5359 | 0.5304 | 0.5193 | 0.5083 | 0.5249 |
| 500-1k | 328 | 0.5915 | 0.5884 | 0.5945 | 0.5945 | 0.5915 |
| 1k-2k | 346 | 0.6618 | 0.6647 | 0.6908 | 0.6763 | 0.6763 |
| 2k-4k | 247 | 0.7247 | 0.7571 | 0.753 | 0.7692 | 0.7692 |
| >4k | 156 | 0.6731 | 0.6923 | 0.6923 | 0.6859 | 0.6859 |
