# OpenADMET applicability-domain audit

Status: **PASS**

Across 23 endpoints and 69 endpoint--split cells, the mean endpoint-level Spearman association between nearest-training ECFP4 Tanimoto similarity and absolute error is -0.1093 (endpoint-bootstrap 95% interval [-0.1506, -0.0666]).
At the exploratory 0.5 similarity threshold, 56 cells contain at least five rows on each side. Their endpoint-averaged relative low-minus-high-similarity MAE gap is +0.2364 (95% interval [+0.0658, +0.4107]).

This is an exploratory same-endpoint, same-split diagnostic. It does not establish calibrated uncertainty, a causal effect of domain shift, or the model's relationship to the Step-1 pretraining domain.

Every saved target vector and train/validation/test SMILES hash was checked before analysis.
