## Test set: baseline-ensemble predicted area outside the fitted disk

| test_images | predictions | pred_pixels | outside_pixels | share | images_with_any_outside | max_outside_in_one_image |
|---|---|---|---|---|---|---|
| 180 | 1173 | 2402357 | 0 | 0 | 0 | 0 |

## Test images with the most predicted pixels outside the disk

| stem | predictions | pred_pixels | outside_pixels |
|---|---|---|---|
| 20110120105534Ch | 2 | 2474 | 0 |
| 20160327133714Th | 3 | 8504 | 0 |
| 20160405014154Uh | 4 | 5877 | 0 |
| 20160505025554Uh | 4 | 3819 | 0 |
| 20160513133714Th | 6 | 7344 | 0 |

## Per-channel distribution of the PNGs on train (on-disk pixels, 120 images; uint8 units)

| channel | mean_of_means | sd_of_means_across_images | mean_within_image_std |
|---|---|---|---|
| ch0 raw | 122.8 | 4.408 | 21.68 |
| ch1 flattened | 126.9 | 0.9048 | 27.32 |
| ch2 local contrast | 126.8 | 0.8507 | 29.96 |

## Channel-2 local contrast inside GT masks by area (train, 250 images; negative = darker than background)

| area_bin | GT | ch2_inside_mean | flattened_ratio_inside | raw_ring_contrast | ch2_relative_to_1k_2k |
|---|---|---|---|---|---|
| <500 | 465 | -0.1207 | 0.8863 | 0.1307 | 1.211 |
| 500-1k | 771 | -0.1075 | 0.8905 | 0.1229 | 1.079 |
| 1k-2k | 785 | -0.09967 | 0.8916 | 0.1207 | 1 |
| 2k-4k | 498 | -0.09114 | 0.8911 | 0.1177 | 0.9145 |
| 4k-8k | 273 | -0.08241 | 0.8934 | 0.1165 | 0.8268 |
| >8k | 98 | -0.06245 | 0.907 | 0.09967 | 0.6266 |
