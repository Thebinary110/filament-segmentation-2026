## Sample submission / other metadata in the competition data

files with 'sample' in the name: none; test folder holds only the 180 JPEGs (no labels, no record ids).

## Annotators per image (train+val, 707 images)

| k | images |
|---|---|
| 1 | 411 |
| 2 | 145 |
| 3 | 151 |

## k by annotation group (first 2 digits of record id)

| group | 1 | 2 | 3 |
|---|---|---|---|
| 01 | 36 | 78 | 108 |
| 02 | 39 | 43 | 0 |
| 03 | 101 | 5 | 0 |
| 04 | 132 | 12 | 0 |
| 05 | 37 | 7 | 43 |
| 06 | 66 | 0 | 0 |

## Annotation batches (first 4 digits of record id)

| batch | images | k_values | years | sites |
|---|---|---|---|---|
| 0101 | 73 | 1,2,3 | 2011-2021 | BCLMTU |
| 0102 | 51 | 1,2,3 | 2011-2022 | BCLMTU |
| 0103 | 19 | 1,2 | 2011-2021 | BCLMTU |
| 0104 | 79 | 1,2,3 | 2011-2022 | BCLMTU |
| 0201 | 38 | 1,2 | 2011-2022 | BCLMTU |
| 0202 | 1 | 1 | 2011-2011 | U |
| 0204 | 43 | 1,2 | 2011-2022 | BCLMTU |
| 0301 | 48 | 1 | 2011-2021 | BCLMTU |
| 0303 | 43 | 1 | 2011-2019 | BCLMTU |
| 0304 | 15 | 1,2 | 2011-2021 | BCLTU |
| 0401 | 39 | 1 | 2011-2022 | BCLMTU |
| 0402 | 20 | 1 | 2011-2021 | CLMTU |
| 0403 | 41 | 1,2 | 2011-2022 | BCLMTU |
| 0404 | 44 | 1,2 | 2011-2022 | BCLMTU |
| 0501 | 48 | 1,2,3 | 2011-2022 | BCLMTU |
| 0502 | 23 | 1,2 | 2011-2017 | BCLMTU |
| 0503 | 16 | 1,2 | 2011-2019 | BCLMTU |
| 0601 | 23 | 1 | 2011-2022 | BCLMTU |
| 0604 | 43 | 1 | 2011-2021 | BCLMTU |

## k by year

| year | 1 | 2 | 3 |
|---|---|---|---|
| 2011 | 46 | 16 | 24 |
| 2012 | 47 | 21 | 24 |
| 2013 | 64 | 19 | 18 |
| 2014 | 60 | 25 | 17 |
| 2015 | 52 | 14 | 20 |
| 2016 | 49 | 15 | 16 |
| 2017 | 39 | 17 | 15 |
| 2018 | 15 | 6 | 5 |
| 2019 | 7 | 4 | 3 |
| 2020 | 6 | 1 | 2 |
| 2021 | 18 | 6 | 4 |
| 2022 | 8 | 1 | 3 |

## k by observatory

| site | 1 | 2 | 3 |
|---|---|---|---|
| B | 64 | 28 | 26 |
| C | 78 | 24 | 23 |
| L | 71 | 29 | 27 |
| M | 72 | 26 | 27 |
| T | 65 | 20 | 25 |
| U | 61 | 18 | 23 |

## Leave-one-out accuracy of predicting k from test-time information

| predictor | loo_accuracy |
|---|---|
| always majority class (k=1) | 0.5813 |
| year | 0.5813 |
| site | 0.5813 |
| year + site | 0.5601 |
| year + month | 0.4908 |
| nearest train image in time | 0.4017 |

## Test vs train: observatory shares and years

| set | images | site_B | site_C | site_L | site_M | site_T | site_U | year_min | year_max | median_year |
|---|---|---|---|---|---|---|---|---|---|---|
| train+val | 707 | 0.1669 | 0.1768 | 0.1796 | 0.1768 | 0.1556 | 0.1443 | 2011 | 2022 | 2014 |
| test | 180 | 0.1833 | 0.1833 | 0.1889 | 0.1444 | 0.15 | 0.15 | 2011 | 2022 | 2014 |

## Test images: k of the nearest train image in time

| k_of_nearest_train_image | test_images |
|---|---|
| 1 | 100 |
| 2 | 40 |
| 3 | 40 |

## Test images: time gap to nearest train image

| index | days to nearest train image |
|---|---|
| min | 0.0002315 |
| 25% | 0.6632 |
| 50% | 1.528 |
| 75% | 3 |
| max | 54.17 |

## Pooled PQ of the baseline under different k mixes (re-weighting validation per-k sums)

| k_mix | PQ |
|---|---|
| validation (66/16/24 images) | 0.4765 |
| all k=1 | 0.4832 |
| all k=3 | 0.4691 |
| train+val proportions | 0.4766 |
| test, k of nearest train image | 0.4764 |
