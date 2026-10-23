## Leave-one-out accuracy of predicting annotation identity from test-visible information

| target | classes | majority | year | site | year_site | hour | nearest_in_time | nearest_within_1day | images_with_neighbour_1day |
|---|---|---|---|---|---|---|---|---|---|
| group | 6 | 0.314 | 0.2956 | 0.3182 | 0.1994 | 0.2786 | 0.2164 | 0.2267 | 247 |
| batch | 19 | 0.1117 | 0.07921 | 0.08345 | 0.08345 | 0.1047 | 0.06789 | 0.05668 | 247 |
| first_annotator | 32 | 0.1033 | 0.06931 | 0.08204 | 0.06365 | 0.08487 | 0.06789 | 0.05668 | 247 |
| has_hard_annotator | 2 | 0.5431 | 0.4965 | 0.5431 | 0.5827 | 0.5163 | 0.4668 | 0.4251 | 247 |
| k | 3 | 0.5813 | 0.5813 | 0.5813 | 0.5601 | 0.5743 | 0.4017 | 0.3441 | 247 |

## Batches: time and observatory coverage

| batch | images | first | last | distinct_days | sites |
|---|---|---|---|---|---|
| 0101 | 73 | 2011 | 2021 | 72 | BCLMTU |
| 0102 | 51 | 2011 | 2022 | 51 | BCLMTU |
| 0103 | 19 | 2011 | 2021 | 19 | BCLMTU |
| 0104 | 79 | 2011 | 2022 | 79 | BCLMTU |
| 0201 | 38 | 2011 | 2022 | 38 | BCLMTU |
| 0202 | 1 | 2011 | 2011 | 1 | U |
| 0204 | 43 | 2011 | 2022 | 43 | BCLMTU |
| 0301 | 48 | 2011 | 2021 | 48 | BCLMTU |
| 0303 | 43 | 2011 | 2019 | 43 | BCLMTU |
| 0304 | 15 | 2011 | 2021 | 15 | BCLTU |
| 0401 | 39 | 2011 | 2022 | 38 | BCLMTU |
| 0402 | 20 | 2011 | 2021 | 20 | CLMTU |
| 0403 | 41 | 2011 | 2022 | 40 | BCLMTU |
| 0404 | 44 | 2011 | 2022 | 44 | BCLMTU |
| 0501 | 48 | 2011 | 2022 | 48 | BCLMTU |
| 0502 | 23 | 2011 | 2017 | 23 | BCLMTU |
| 0503 | 16 | 2011 | 2019 | 16 | BCLMTU |
| 0601 | 23 | 2011 | 2022 | 23 | BCLMTU |
| 0604 | 43 | 2011 | 2021 | 43 | BCLMTU |

## Do images from the same day share a batch?

| days_with_2plus_images | share_single_batch | mean_batches_per_day |
|---|---|---|
| 50 | 0.06 | 1.96 |

## Share of images with at least one 'hard' annotator, by year

| year | False | True |
|---|---|---|
| 2011 | 0.4419 | 0.5581 |
| 2012 | 0.5652 | 0.4348 |
| 2013 | 0.495 | 0.505 |
| 2014 | 0.5882 | 0.4118 |
| 2015 | 0.5233 | 0.4767 |
| 2016 | 0.475 | 0.525 |
| 2017 | 0.6479 | 0.3521 |
| 2018 | 0.6154 | 0.3846 |
| 2019 | 0.6429 | 0.3571 |
| 2020 | 0.3333 | 0.6667 |
| 2021 | 0.6786 | 0.3214 |
| 2022 | 0.6667 | 0.3333 |
