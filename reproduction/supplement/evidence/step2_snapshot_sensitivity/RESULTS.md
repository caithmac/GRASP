# Step-2 snapshot sensitivity (descriptive)

This artifact describes test-metric sensitivity across eight already evaluated checkpoints. It is **not** a checkpoint-selection analysis, and no per-task test-selected best checkpoint is used or recommended as a model-selection result.

## Provenance and validation

- Input: `1.5b_p3_step2_matched_tdc_results/results_1.5b_p3_s2_matched/tdc_all_snapshots.csv`
- Input SHA256: `603ee64b4c22b964b457ac8611d268185d61b54b3053f534057e183d91a010eb`
- Analysis script SHA256: `8ed8b57207283f060ae2f2ff5fa37b582008850ddef9d95d6e7d3e3d8f643a9e`
- Validated design: exactly 8 unique checkpoints × 23 tasks; every unique task-step cell reports `n_seeds = 3`.
- Orientation: MAE is lower-is-better; every other reported metric is higher-is-better.
- Declared reference checkpoint: step 50000.
- The rationale for the original checkpoint choice is not inferred from these test results or from this retrospective analysis.

## Rank summary

| Step | Mean rank | Median rank | Task wins* |
|---:|---:|---:|---:|
| 10000 | 4.6522 | 4.0000 | 2 |
| 20000 | 4.4348 | 5.0000 | 4 |
| 30000 | 4.2609 | 4.0000 | 2 |
| 40000 | 4.1739 | 3.0000 | 3 |
| 50000 | 4.0870 | 4.0000 | 4 |
| 60000 | 5.0435 | 5.0000 | 2 |
| 70000 | 4.4348 | 5.0000 | 5 |
| 80000 | 4.9130 | 5.0000 | 1 |

*A task win is the best oriented reported mean among the eight snapshots; exact ties count as wins for every tied step. Ranks use average ranks for exact ties.

## Pairwise directions versus the declared reference

| Step vs 50000 | Better | Worse | Ties | Exact two-sided sign-test p** |
|---:|---:|---:|---:|---:|
| 10000 | 12 | 11 | 0 | 1 |
| 20000 | 10 | 13 | 0 | 0.67763948 |
| 30000 | 10 | 13 | 0 | 0.67763948 |
| 40000 | 11 | 12 | 0 | 1 |
| 60000 | 9 | 14 | 0 | 0.40487289 |
| 70000 | 10 | 13 | 0 | 0.67763948 |
| 80000 | 9 | 14 | 0 | 0.40487289 |

**P-values are exact two-sided sign tests over non-tied tasks. They are descriptive and uncorrected for the seven comparisons; they must not be read as confirmatory evidence or as a checkpoint-selection criterion.

Machine-readable per-task ranks and orientations are stored in `summary.json`; aggregate tables are in `summary_by_step.csv` and `pairwise_vs_reference.csv`.
