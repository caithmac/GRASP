# Same-split OpenADMET classical comparison

**Validation: PASS.** ChemRasayan `full_mix` and every classical method use the same 23 endpoints and the same three reconstructed cluster splits. Data and train/validation/test membership hashes and test sizes agree for every paired cell.

The analysis unit is the endpoint: each score below first averages the three split MAEs, then compares the 23 paired endpoint means. The three splits are **not** treated as independent observations. Difference = ChemRasayan MAE - classical MAE, so negative is better.

| Classical method | ChemRasayan | Classical | Mean difference (95% endpoint bootstrap CI) | Point wins | Exact Wilcoxon p / Holm | Exact sign p / Holm |
|---|---:|---:|---:|---:|---:|---:|
| `dummy_median` | 0.3742 | 0.5919 | -0.2177 [-0.2845, -0.1612] | 23/23 | 2.38419e-07 / 1.19209e-06 | 2.38419e-07 / 1.19209e-06 |
| `ecfp4_rf` | 0.3742 | 0.4525 | -0.0783 [-0.1114, -0.0490] | 20/23 | 1.0252e-05 / 4.1008e-05 | 0.000488281 / 0.000976562 |
| `ecfp4_lgbm` | 0.3742 | 0.4075 | -0.0334 [-0.0587, -0.0116] | 16/23 | 0.00603271 / 0.00603271 | 0.0931396 / 0.0931396 |
| `rdkit_desc_rf` | 0.3742 | 0.4306 | -0.0564 [-0.0813, -0.0348] | 21/23 | 2.09808e-05 / 6.29425e-05 | 6.60419e-05 / 0.000264168 |
| `rdkit_desc_lgbm` | 0.3742 | 0.4119 | -0.0377 [-0.0590, -0.0192] | 21/23 | 0.000407934 / 0.000815868 | 6.60419e-05 / 0.000264168 |

## Interpretation guardrails

- Point-estimate wins only count the direction of the 23 endpoint means; they are not themselves inferential claims.
- The Wilcoxon test evaluates paired signed ranks; the sign test evaluates directions only. Both are exact two-sided tests. Holm adjustment is applied across the five classical contrasts separately for each test family.
- Bootstrap intervals are deterministic percentile intervals from 100,000 paired endpoint resamples (base seed 20260917). They quantify endpoint heterogeneity, not training-seed uncertainty.
- This is a same-reconstructed-split comparison. It does not make cross-paper claims about unavailable author-defined splits.

## Alignment audit

- 345 audited classical cells; audit status PASS.
- 69 endpoint-split pairs aligned by cryptographic data and membership hashes.
- 345 retrieved classical prediction files additionally had their MAE recomputed and target arrays matched directly (69/69 unique endpoint-split pairs covered).
- 0 raw prediction files were absent and 0 present file(s) failed their recorded hash; these partial-retrieval files were excluded. The comparison uses the complete, hash-verified, PASS-audited `cells.csv`.

Machine-readable outputs: `summary.json`, `comparisons.csv`, and `endpoint_paired_differences.csv`.
