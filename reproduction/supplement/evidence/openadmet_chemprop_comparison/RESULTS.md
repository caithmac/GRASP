# Same-split OpenADMET Chemprop comparison

**Validation: PASS.** All 69 Chemprop cells match ChemRasayan fixed full fine-tuning on prepared-data, inner-train, validation, and outer-test membership hashes, held-out row counts, and held-out targets.

Each endpoint first averages the same three split MAEs; inference and bootstrap resampling use the 23 paired endpoint means, not the 69 split cells.

| Baseline | Type | ChemRasayan | Baseline | Difference (95% endpoint bootstrap CI) | ChemR. wins | Wilcoxon raw / Holm-6 | Sign raw / Holm-6 |
|---|---|---:|---:|---:|---:|---:|---:|
| `dummy_median` | classical | 0.3742 | 0.5919 | -0.2177 [-0.2845, -0.1612] | 23/23 | 2.38419e-07 / 1.43051e-06 | 2.38419e-07 / 1.43051e-06 |
| `ecfp4_rf` | classical | 0.3742 | 0.4525 | -0.0783 [-0.1114, -0.0490] | 20/23 | 1.0252e-05 / 4.1008e-05 | 0.000488281 / 0.000976562 |
| `ecfp4_lgbm` | classical | 0.3742 | 0.4075 | -0.0334 [-0.0587, -0.0116] | 16/23 | 0.00603271 / 0.00603271 | 0.0931396 / 0.0931396 |
| `rdkit_desc_rf` | classical | 0.3742 | 0.4306 | -0.0564 [-0.0813, -0.0348] | 21/23 | 2.09808e-05 / 6.29425e-05 | 6.60419e-05 / 0.000264168 |
| `rdkit_desc_lgbm` | classical | 0.3742 | 0.4119 | -0.0377 [-0.0590, -0.0192] | 21/23 | 0.000407934 / 0.000815868 | 6.60419e-05 / 0.000264168 |
| `chemprop_dmpnn` | neural | 0.3742 | 0.4395 | -0.0654 [-0.1008, -0.0375] | 22/23 | 5.96046e-06 / 2.98023e-05 | 5.72205e-06 / 2.86102e-05 |

The Chemprop bootstrap interval uses 100,000 deterministic paired endpoint resamples (seed 20260917). Classical intervals are retained from the audited five-contrast analysis; all raw exact p-values were recomputed from endpoint differences before the six-contrast Holm adjustment.

Unrounded certified Chemprop difference and 95% interval: -0.06536292689838681 [-0.10075148102884383, -0.03745899405327064].

Point-estimate wins are descriptive endpoint directions, not per-endpoint tests.
