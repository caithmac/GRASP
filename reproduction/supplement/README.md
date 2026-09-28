# ChemRasayan anonymous reproducibility supplement

This archive is the submission-safe code and provenance supplement for the final ChemRasayan system described in the manuscript. It contains the exact scientific implementation used for the 93.5M-parameter, 12-layer, 768-dimensional discriminator; the 3-layer, 256-dimensional generator; the exposure-correcting Step-1 continuation; the 642-assay sparse Step-2 adaptation; current TDC and OpenADMET evaluation code; and the audit scripts used in the revision.

The archive intentionally excludes checkpoints, raw ZINC/ChEMBL/OpenADMET/TDC data, prediction-scale result directories, credentials, private host names, private user identifiers, and storage addresses. Cluster-specific values in staged copies were replaced with documented placeholders. Those substitutions change deployment wiring only, not model, data-processing, optimization, split, metric, or statistical logic.

## Directory map

- `src/pretraining/`: MolE/ChemRasayan package code, Step-2 trainer, encoder loader, exposure planner, and the exact ZINC/ChEMBL construction programs.
- `configs/`: resolved Step-1 Hydra configurations and the 211-entry radius-0 atom-environment vocabulary.
- `launch/`: restart-safe Step-1/Step-2/TDC pipeline plus current OpenADMET and audit launchers.
- `manifests/`: anonymized Kubernetes job specifications. Replace the clearly named storage, queue, and NFS placeholders before use.
- `src/evaluation/`: current OpenADMET and TDC evaluators, classical baselines, locked-split reference builder, and summarizers.
- `src/analysis/`: label-matrix, snapshot, leakage, overlap-sensitivity, objective, and atom-order audits.
- `provenance/`: retained training plans, checkpoint hashes, and Step-2 training metadata. Paths inside these files are generic container paths.
- `evidence/`: compact derived tables only; no molecule-level proprietary or raw benchmark data.
- `third_party/DeBERTa/`: the exact bundled DeBERTa Python dependency used by the model, with its own license.
- `MANIFEST.sha256`: SHA-256 of every release file except the manifest itself.
- `build_reproducibility_release.py`: deterministic archive builder with normalized ZIP metadata and two-build byte comparison.
- `CHECKPOINT_AND_DATA_HASHES.md`: hashes of excluded model/data artifacts and explicit provenance limits.

## Exact final training route

The final checkpoint is not the separately scaffolded 24-layer “large” model. It is the 12-layer/768-dimensional RTD-25% model in `configs/model/pretrain_rtd_25pct_phase4.yaml`, initialized from the retained 3M-phase source checkpoint and continued for an exposure-correcting 3,133,753 updates. The retained plan records 1,538,240,768 molecular presentations in total. Step 2 then uses `src/pretraining/supervised_pretrain_ddp.py` for 80,000 updates over the 511,898-by-642 sparse ChEMBL matrix; the 50,000-update encoder is the fixed principal downstream checkpoint.

The authoritative end-to-end launcher is:

```bash
bash launch/run_1.5b_phase4_to_tdc.sh
```

It expects the source checkpoint, ZINC parquet shards, ChEMBL matrix, TDC cache, and writable result/checkpoint volumes at the generic container paths used in the script. The archive does not distribute those large or licensed artifacts. `manifests/phase4_1.5b_to_tdc_job.yml` is a cluster template for the same pipeline.

The earlier Step-1 phases are retained as `launch/run_pretrain_rtd_1.5b.sh` and `launch/run_pretrain_rtd_1.5b_resume.sh` because the final continuation depends on their source checkpoint. Their retained comments accurately state that the intermediate resume was weights-only and restarted the optimizer schedule. The final continuation restores full optimizer/scheduler state within that continuation phase.

## Data construction

ZINC20 streaming construction:

```bash
python src/pretraining/download_zinc20_1.5b.py \
  --output_dir /mnt/data/zinc20_1.5b \
  --shard_size 1000000 \
  --val_size 100000
```

ChEMBL 36 sparse-label construction:

```bash
CHEMBL_VERSION=36 MIN_ASSAY_MOLS=500 \
CHEMBL_DB_PATH=/mnt/data/chembl/chembl_36_sqlite/chembl_36.db \
OUTPUT_DIR=/mnt/data/chembl_supervised_paperlike \
TDC_DATA_DIR=/mnt/tdc_data \
python src/pretraining/build_chembl_activity.py
```

The historical builder removed a reconstructed PyTDC test union, but the revision audit found that its default 70/10/20 split did not match the evaluated 70/15/15 split and that not every endpoint loaded. Therefore, the builder is released exactly, but its exclusion step must not be represented as proof of complete benchmark decontamination. See `src/analysis/audit_tdc_step2_exclusions.py`.

## Evaluation entry points

TDC:

- `src/evaluation/tdc/finetune_benchmarks.py`: retained three-seed historical/final evaluator.
- `src/evaluation/tdc/finetune_tdc_mole_protocol.py`: fixed-protocol MolE/TDC evaluator.
- `src/evaluation/tdc/summarize_tdc_run.py` and `summarize_tdc_snapshots.py`: completeness and aggregation.
- `src/evaluation/tdc/tdc_classical_baselines.py`: validation-selected ECFP4 and RDKit-descriptor ExtraTrees baselines on the fixed historical scaffold split.
- `src/evaluation/tdc/finalize_tdc_classical_baselines.py`: fail-closed result validator and endpoint-level comparison summarizer.
- `src/evaluation/tdc/build_tdc_classical_split_reference.py`: reconstructs the locked split/label reference from authorized raw snapshots and retained logs.
- `launch/run_tdc_classical_baselines.sh` and `manifests/tdc_classical_baselines_job.yml`: anonymous, configurable launch templates for the classical panel.

OpenADMET:

- `src/evaluation/openadmet/finetune_moljepa_benchmarks.py`: ChemRasayan evaluator on reconstructed OpenADMET splits.
- `openadmet_classical_baselines.py`: same-split dummy, ECFP4, and RDKit-descriptor baselines.
- `openadmet_chemprop_baseline.py`: same-split Chemprop baseline.
- `openadmet_step2_leakage_audit.py`: exact/connectivity/scaffold/similarity exposure audit.
- `src/analysis/analyze_openadmet_chemprop_comparison.py`: fail-closed 69-cell split/target/prediction audit and exact six-contrast endpoint-level inference.
- `src/analysis/analyze_step2_overlap_sensitivity.py`: post-hoc OpenADMET sensitivity analysis after exact and nearest-neighbor Step-2 exposure exclusions.
- `src/analysis/certify_openadmet_objective_rerun.py`: fail-closed validator for the provenance-locked MLM-S2 versus RTD-25%-S2 primary-protocol rerun.
- `src/evaluation/openadmet/finetune_moljepa_atom_order_certifiable_v2.py`: receipt-bound adapter that records ordered test-row and atom-serialization hashes around the locked OpenADMET evaluator.
- `src/analysis/certify_atom_order_robustness.py` and `certify_atom_order_robustness_v2.py`: independent fail-closed reconstruction and publication checks for the atom-order audit artifact contract.
- `launch/run_final_phase4_atom_order_audit_v2.sh` and `manifests/final_phase4_atom_order_job_v2.template.yml`: anonymized, indexed-job deployment sources for that audit. The launcher requires a separately finalized deployment manifest via `PRELAUNCH_MANIFEST`; the manifest itself is run-specific and is not distributed here.
- Matching restart-safe launchers and job manifests are in `launch/` and `manifests/`.

The compact overlap-sensitivity outputs are under `evidence/step2_overlap_sensitivity/`. They contain endpoint- and split-level aggregate metrics, source hashes, and exclusion counts, but no molecule strings or raw benchmark rows. The certified same-split Chemprop contrast and six-comparison multiplicity family are under `evidence/openadmet_chemprop_comparison/`. The certified 23-endpoint objective effects, anonymous PASS summary, and results narrative are under `evidence/openadmet_objective_comparison/`. Atom-order v2 source and deployment templates are available as listed above; no atom-order v2 result artifact is included unless and until the complete output passes the separate certification and anonymization gates. Internal summaries containing local or cluster paths are excluded. The fail-closed 23-endpoint TDC classical summary and its interpretation are under `evidence/tdc_classical_baselines/`; per-task payloads and raw `.tab` snapshots remain excluded.

## Verification

From the archive root:

```bash
sha256sum -c MANIFEST.sha256
python -m compileall -q src
```

From the parent repository, rebuild the archive deterministically with:

```bash
python paper_iclr/reproducibility_release/build_reproducibility_release.py
```

The builder regenerates `RELEASE_INVENTORY.csv` first, regenerates `MANIFEST.sha256` second, writes members in sorted order with fixed ZIP metadata, rejects unsafe or duplicate paths and forbidden data/checkpoint payloads, and requires two byte-identical archive builds.

The SHA-256 manifest covers the redacted release bytes. Original run hashes in `provenance/` and `CHECKPOINT_AND_DATA_HASHES.md` refer to the unredacted files/artifacts present during the experiments, so hashes of the staged launchers and manifests are expected to differ.

## Scope and known limits

- No checkpoint or raw dataset is embedded; the listed hashes permit identity checks when separately authorized artifacts are obtained.
- The TDC split reference redacts a source-repository commit identifier for anonymous review. `ANONYMIZATION.md` records the canonical content hash of the original unredacted reference so authorized holders can audit identity without publishing the identifier.
- A single aggregate bytewise hash for all 1,539 ZINC training shards was not retained. The available row/shard counts and checkpoint lineage are reported rather than inventing one.
- Raw split membership and original cluster-assignment lists are not recoverable from retained OpenADMET artifacts. Available membership hashes and held-out predictions are distributed in the separate OpenADMET result supplement.
- The Kubernetes manifests are executable templates after replacing `NFS_SERVER_PLACEHOLDER`, `/EXTERNAL_NFS_PATH`, `WORKSPACE_PVC`, and `REVIEW_QUEUE` with site-specific values.
