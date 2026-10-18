# Final-layer versus mixed-layer readout comparison

[Open the notebook in Colab](https://colab.research.google.com/github/caithmac/GRASP/blob/main/notebooks/04_readout_ablation.ipynb).

Select a GPU runtime and run the cells in order. The default uses the public GRASP Step2 checkpoint and performs 24 fits: six endpoint datasets, four arms and one training seed. Set `MODEL_KEY="A1"` to run a separate experiment with the pinned public A1 checkpoint. No EIDF allocation is used.

| Adaptation | Final-layer atoms | Mixed-layer atoms |
|---|---|---|
| Frozen encoder | Layer 12 | Learned mixture of layers 4/8/10/12 |
| Full encoder fine-tuning | Layer 12 | Learned mixture of layers 4/8/10/12 |

All four arms use the same atom-attention pooling and LayerNorm/MLP prediction head. The frozen arms still train pooling and head parameters. They are **not strict linear probes**. The final-layer full arm updates the whole encoder; it is not last-transformer-block-only training. The optional LoRA method in the general library is not part of this four-arm notebook.

## Dataset selection

The candidate pool is the project's 23 public OpenADMET endpoints. The notebook uses `random.Random(20261018).sample(sorted(pool), 6)`, without inspecting new results:

1. ASAP SARS-CoV-2 potency
2. Biogen rat liver microsomal clearance
3. ASAP LogD
4. ExpansionRx Caco-2 permeability
5. ExpansionRx solubility
6. ExpansionRx LogD

The complete pool, selection seed, immutable source URLs and source SHA256 checksums are exported. Raw data are downloaded from the original public providers when the user runs the notebook; they are not bundled in this repository. Source dataset terms apply separately from the model/code license.

Each dataset gets one new scaffold-grouped train/validation/test split with target proportions 80/10/10. These are deliberately disclosed new splits, not the paper's or the challenges' test memberships. All four arms receive the same saved CSVs. Connectivity/scaffold groups remain disjoint, including duplicates and stereoisomers. All acyclic molecules form one group. Known eligibility exclusions are logged before splitting, and unexpected featurizer errors stop the run.

## Training and interpretation

- Default: 20 epochs, patience 5, batch size 8, training seed 0.
- Frozen/full learning rates are fixed separately at `1e-3`/`1e-5`, shared across the two readouts within each adaptation setting. This is not exhaustive hyperparameter optimization.
- Head and pooling initialization are checked for equality across arms. Frozen encoders have no trainable parameters.
- Target scaling is fitted on training labels only. Early stopping uses validation MAE. Held-out predictions are made after selecting the checkpoint.
- `final_vs_mixed.csv` reports final MAE minus mixed MAE: negative favors the final layer. Compare adaptation methods in the per-dataset summary. Different property scales should not be pooled into one raw-error claim.
- Add training seeds before viewing test results if repeated fitting is wanted. Repeated seeds share a test cohort; they are not independent datasets or pretrained seeds.

`USE_DRIVE=True` preserves completed cells across Colab disconnects. Changing a scientific setting requires a new `RUN_NAME`. Completed fits resume only when model, protocol and split identities match; an interrupted individual fit restarts. `KEEP_PREDICTORS=False` keeps scores, predictions and configuration while discarding that completed fit's roughly 0.37 GB weight file. The final ZIP always excludes weights and raw source downloads.

## Library support

Python callers can select the readout while retaining the existing four-layer default:

```python
from grasp import GRASPPredictor

predictor = GRASPPredictor.from_encoder(
    encoder, task="regression", method="frozen", mix_layers=(12,)
)
```

`grasp.train.fit(..., mix_layers=(12,))` uses the same selection. The setting is saved with the predictor and restored on reload. Old predictor configurations without `mix_layers` continue to use layers 4/8/10/12.

For local validation with already downloaded model files:

```bash
python -m pip install -e '.[notebooks,test]'
python -m pytest -q tests
python scripts/verify_readout_notebook.py --model-dir /path/to/GRASP --work-dir /path/to/new_validation_directory
```

The validator audits the six full public datasets and executes four tiny CPU fits on a disposable subset. It skips the installation cell and uses local checkpoint transport. This is an execution check, not a full benchmark or a Colab GPU measurement.
