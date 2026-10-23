## Annotator vs annotator (ordered pairs; annotations only, no model)

| split | k | images | pairs | PQ | SQ | RQ | TP | FP | FN | iou_p25 | iou_median | iou_p75 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 2 | 129 | 258 | 0.3621 | 0.6386 | 0.567 | 998 | 762 | 762 | 0.5728 | 0.6366 | 0.6966 |
| train | 3 | 127 | 762 | 0.3318 | 0.6326 | 0.5245 | 2656 | 2408 | 2408 | 0.5721 | 0.6264 | 0.6889 |
| val | 2 | 16 | 32 | 0.3397 | 0.6426 | 0.5286 | 120 | 107 | 107 | 0.5945 | 0.639 | 0.6948 |
| val | 3 | 24 | 144 | 0.3421 | 0.6274 | 0.5453 | 518 | 432 | 432 | 0.5634 | 0.6271 | 0.6882 |

## Our model (baseline ensemble) on validation, by number of annotators k

| k | images | PQ | SQ | RQ | TP | FP | FN | iou_p25 | iou_median | iou_p75 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 66 | 0.4832 | 0.694 | 0.6962 | 346 | 92 | 210 | 0.6241 | 0.7109 | 0.7591 |
| 2 | 16 | 0.4776 | 0.6858 | 0.6965 | 148 | 50 | 79 | 0.6194 | 0.6835 | 0.7507 |
| 3 | 24 | 0.4691 | 0.6808 | 0.6891 | 328 | 149 | 147 | 0.6098 | 0.6919 | 0.7505 |

## Our SQ on matched filaments, by (k, s) on multi-annotator val images

| k | s | matches | our_SQ |
|---|---|---|---|
| 2 | 1 | 19 | 0.6341 |
| 2 | 2 | 129 | 0.6934 |
| 3 | 1 | 15 | 0.7165 |
| 3 | 2 | 47 | 0.6681 |
| 3 | 3 | 266 | 0.681 |

## Leave-one-out: majority of the OTHER annotators' masks vs the held-out annotator, same filament cluster (val)

| cluster_support | n | mean_IoU | share_IoU_gt_0_5 |
|---|---|---|---|
| 2 | 242 | 0.5796 | 0.7686 |
| 3 | 318 | 0.5689 | 0.7107 |
