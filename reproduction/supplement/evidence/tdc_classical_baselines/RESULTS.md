# Same-split TDC classical baselines

The fail-closed finalizer accepted all 23 endpoint files and the locked
PyTDC 0.4.1 split/label reference. Every endpoint records zero canonical-
SMILES and Murcko-scaffold overlap across train, validation, and test
partitions. Hyperparameters were selected on validation data; the held-out
test set was evaluated only after selection. Each selected ExtraTrees model
was repeated with seeds 0, 1, and 2 on one fixed scaffold partition.

- RDKit 2D descriptors were favorable to ECFP4 on 21/23 endpoints; ECFP4
  was favorable on 2/23 (exact two-sided sign-test raw p = 0.0000660;
  Holm-adjusted p = 0.0001981).
- Against the final ChemRasayan comparator, RDKit-descriptor ExtraTrees was
  favorable on 15/23 endpoints and ChemRasayan on 8/23 (raw p = 0.2100;
  Holm-adjusted p = 0.4201).
- Against the final ChemRasayan comparator, ECFP4 ExtraTrees was favorable
  on 8/23 endpoints, ChemRasayan on 14/23, with one tie (raw p = 0.2863;
  Holm-adjusted p = 0.4201).

The Holm family contains these three classical-control direction tests and is
separate from the three historical design-comparison tests.

The endpoint metrics mix MAE, AUROC, AUPRC, and Spearman correlation, so they
are not numerically averaged. Direction counts compare endpoint means and do
not establish per-endpoint significance from three algorithmic seeds.
