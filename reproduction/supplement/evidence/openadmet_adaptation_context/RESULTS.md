# OpenADMET adaptation-context audit

Status: **PASS**

All 207 retained split--method cells were matched to the canonical 23-endpoint row table and passed identity, partition-size, SMILES-hash, prediction-length, and ordered-target checks.
Validation selection chose full fine-tuning 39 times, LoRA 18 times, and frozen mixing 12 times across 69 endpoint--split decisions.

Using endpoint means as the inferential unit, training size was associated with full-minus-frozen validation MAE at Spearman rho=-0.587 (95% bootstrap interval [-0.814, -0.211]; Holm-4 permutation p=0.0151).
Size-adjusted Bemis--Murcko richness was associated with full-minus-LoRA validation MAE at rho=+0.447 (95% interval [+0.084, +0.685]; Holm-4 permutation p=0.1024).

These post hoc associations are exploratory. They do not establish a causal adaptation rule, and they do not generalize beyond the three tested methods.
