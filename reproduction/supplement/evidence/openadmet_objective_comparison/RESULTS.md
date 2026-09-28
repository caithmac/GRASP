# PASS primary-protocol MLM versus RTD objective rerun

All 46 objective/endpoint results, 276 split/method cells, 12 launch manifests, 12 shard markers, raw predictions, canonical split identities, and distinct checkpoint tensor provenance passed the fail-closed audit (independently_recomputed).

Positive MLM minus RTD MAE favors RTD. The interaction is the full-finetuning objective delta minus the frozen objective delta.

| Contrast | Mean | 95% endpoint bootstrap CI | Positive / negative / tie | Exact Wilcoxon Holm-3 p | Exact sign Holm-3 p |
|---|---:|---:|---:|---:|---:|
| frozen objective delta | +0.012186 | [+0.000157, +0.024798] | 16 / 7 / 0 | 0.096881866 | 0.1862793 |
| full objective delta | -0.000211 | [-0.012247, +0.010908] | 14 / 9 / 0 | 0.75398469 | 0.40487289 |
| full minus frozen interaction | -0.012397 | [-0.023406, -0.000734] | 6 / 17 / 0 | 0.083022594 | 0.10406899 |

Inference uses 23 endpoint-level paired effects. Bootstrap intervals use 100,000 deterministic endpoint resamples with seed 20260917. Exact Wilcoxon and exact sign-test p-values are adjusted in separate Holm families of three contrasts.
