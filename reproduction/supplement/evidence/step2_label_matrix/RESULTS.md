# Step-2 Label-Matrix Audit

Validation passed. The matrix contains only binary observed labels (0/1) and
floating-point NaN missing values, and the assay-ID count matches the column count.

## Inputs

| Input | Absolute path | SHA-256 |
|---|---|---|
| Labels | `/mnt/data/chembl_supervised_paperlike/labels.npy` | `f631972cd9915bfc72e75f68bfd6b633cce99fc7951a712dccf5e6e54718405f` |
| Assay IDs | `/mnt/data/chembl_supervised_paperlike/assay_ids.txt` | `ed79496bd3f3a7b1c31a00c3e9db321f41768ea0826937c9859054b0d7742ae8` |

## Matrix and label totals

| Quantity | Value |
|---|---:|
| Shape | 511,898 molecules x 642 assays |
| Cells | 328,638,516 |
| Observed labels | 4,536,721 |
| Positive labels | 1,097,029 |
| Negative labels | 3,439,692 |
| Missing labels | 324,101,795 |
| Label density | 0.01380459 |
| Zero-label molecule rows | 0 |
| Constant-label assays | 182 |
| Observed labels from constant assays | 199,382 (4.3948%) |
| Assays with zero observed labels | 0 |

## Observed labels per assay

| Statistic | Value |
|---|---:|
| N | 642 |
| Min | 500 |
| 5th percentile | 538.05 |
| 25th percentile | 662 |
| Median | 915.5 |
| 75th percentile | 4245.75 |
| 95th percentile | 38727.3 |
| Max | 86035 |
| Mean | 7066.54 |
| SD (ddof=0) | 13267.3 |

## Positive prevalence per nonempty assay

Prevalence is `positive_count / observed_count`; assays with no observed labels
are excluded from this distribution.

| Statistic | Value |
|---|---:|
| N | 642 |
| Min | 0 |
| 5th percentile | 0.0334618 |
| 25th percentile | 0.203897 |
| Median | 0.500775 |
| 75th percentile | 1 |
| 95th percentile | 1 |
| Max | 1 |
| Mean | 0.566352 |
| SD (ddof=0) | 0.386118 |

## Observed labels per molecule

| Statistic | Value |
|---|---:|
| N | 511,898 |
| Min | 1 |
| 5th percentile | 1 |
| 25th percentile | 2 |
| Median | 3 |
| 75th percentile | 7 |
| 95th percentile | 55 |
| Max | 204 |
| Mean | 8.86255 |
| SD (ddof=0) | 15.2688 |

Full assay-level results are in `per_assay_summary.csv`. The exact molecule-level
count distribution is in `per_molecule_label_count_histogram.csv`; no molecular
identifiers or label values are copied into either output.
