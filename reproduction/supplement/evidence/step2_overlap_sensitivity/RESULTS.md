# Step-2 overlap-filtered OpenADMET sensitivity

This is a post-hoc test-row exclusion analysis; models were not retrained.

| Exclusion | Original mean endpoint MAE | Unexposed-only mean endpoint MAE | Delta | Endpoints with exposed test rows | 95% bootstrap CI for delta |
|---|---:|---:|---:|---:|---:|
| Standardized exact | 0.374183 | 0.372414 | -0.001770 | 13/23 | [-0.006159, +0.001914] |
| Connectivity InChIKey | 0.374183 | 0.372099 | -0.002085 | 14/23 | [-0.006463, +0.001636] |
| ECFP4 Tanimoto >= 0.90 | 0.374183 | 0.372429 | -0.001754 | 17/23 | [-0.006351, +0.002487] |
| ECFP4 Tanimoto >= 0.70 | 0.374183 | 0.369296 | -0.004887 | 22/23 | [-0.013136, +0.001652] |

| Exclusion | Unexposed ChemRasayan - ECFP4-LightGBM MAE | 95% bootstrap CI | ChemRasayan wins |
|---|---:|---:|---:|
| Standardized exact | -0.033287 | [-0.059680, -0.011210] | 16/23 |
| Connectivity InChIKey | -0.033561 | [-0.059981, -0.011335] | 17/23 |
| ECFP4 Tanimoto >= 0.90 | -0.032987 | [-0.059566, -0.010513] | 17/23 |
| ECFP4 Tanimoto >= 0.70 | -0.035462 | [-0.062202, -0.013176] | 17/23 |

Boundary: this removes test rows by the stated Step-2 structure or ECFP4-neighbour rule. It does not address scaffold exposure, endpoint-assay semantic equivalence, or Step-1 ZINC exposure.
