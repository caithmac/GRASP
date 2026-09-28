# Environment and dependency notes

## Recorded training container

The retained Kubernetes jobs used `nvcr.io/nvidia/pytorch:23.10-py3`. The final Step-1 continuation and Step-2 training used four GPUs. The manuscript reports mixed precision, per-device batch 32, gradient accumulation 2 for the final Step-1 continuation, and gradient accumulation 4 for Step 2.

The cluster image was used as a base rather than captured by immutable OCI digest. Consequently, the image tag is exact to the retained manifests, but a registry digest cannot be certified from the retained records.

## Python dependencies

`environments/environment.yml` and `environments/requirements_local.txt` document the project environment. The launchers additionally install or require:

- PyTorch and torchvision supplied by the container (constrained to the installed versions during package installation)
- PyTorch Lightning, torchmetrics, Hydra/OmegaConf
- PyTorch Geometric and its compiled extensions
- RDKit, NumPy, SciPy, pandas, pyarrow, scikit-learn
- PyTDC for TDC reconstruction
- LightGBM for the same-split classical baseline
- Chemprop 2.3.0, RDKit 2026.03.6, cuik-molmaker-pin 2026.03.6, scikit-learn 1.7.2, and pandas 2.3.3 for the current Chemprop job
- `third_party/DeBERTa`, the bundled source dependency used for shortest-path relative attention

Network-resolved installations can drift. For archival reruns, build a wheelhouse and record hashes of every wheel. The launchers retain the package versions that were explicitly pinned, but the historical main training run did not preserve a complete lock file for every transitive dependency.

## Hardware and distributed execution

- Final Step-1 continuation: four GPUs; DDP; local batch 32; gradient accumulation 2; global batch 256.
- Step 2: four GPUs; DDP; local batch 32; gradient accumulation 4; global batch 512.
- Current evaluation manifests express their own CPU/GPU requirements and task-array concurrency.

Hardware product selectors and site-specific storage bindings are deployment concerns and were anonymized in the submission copy. The model and optimizer configuration remains unchanged.

## Anonymized infrastructure placeholders

- `/workspace/chemrasayan`: repository root mounted in the job.
- `NFS_SERVER_PLACEHOLDER`: site NFS server address.
- `/EXTERNAL_NFS_PATH`: site NFS export path.
- `WORKSPACE_PVC`: writable persistent-volume claim.
- `REVIEW_QUEUE`: batch queue name.
- `anonymous-review`: non-identifying job/user label.

No credential is required in the code. Optional Weights & Biases logging is enabled only when a runtime environment supplies its own credential; no credential value is included here.

