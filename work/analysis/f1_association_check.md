## Overlap-only association (drop if > 10% of the candidate lies inside an existing mask)

| conf_threshold | candidates | dropped_by_overlap | dropped_but_new_true_match | dropped_duplicates_of_existing | kept_as_new | kept_new_true | kept_adjacent_no_match | kept_isolated_no_match |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 872 | 703 | 33 | 622 | 169 | 45 | 1 | 123 |
| 0.4 | 665 | 600 | 23 | 547 | 65 | 14 | 0 | 51 |
| 0.5 | 497 | 473 | 15 | 436 | 24 | 6 | 0 | 18 |

## Wrongly discarded candidates (new true matches) by containment

| cont_bin | n | median_area | median_iou_existing |
|---|---|---|---|
| 0.1-0.3 | 1 | 1146 | 0.03596 |
| 0.5-0.8 | 2 | 1418 | 0.5247 |
| >0.8 | 30 | 1362 | 0.4003 |

## How often annotated filaments touch each other (same annotator record)

| GT_instances | touching_another_GT_of_same_record_within_2px | share |
|---|---|---|
| 1258 | 12 | 0.009539 |
