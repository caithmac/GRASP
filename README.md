# GRASP

**Graph Representation Learning with Assay Supervision for Molecular Properties**

GRASP is a 93.5M-parameter molecular graph Transformer. It was trained first on 1.54 billion ZINC20 molecule presentations with replaced-token detection, then adapted on sparse measurements from 642 ChEMBL assays. The fixed public encoder is the **50,000-step Step 2 snapshot** used for the paper's downstream evaluations.

This repository provides molecular embeddings and a complete path to fine-tune, save, reload, and run a property predictor. The pretrained encoder alone does **not** predict a property value. The reported OpenADMET results use a learned layer mixture, atom-attention pooler, and a task-specific head trained on each endpoint; calling `encode()` returns the final-layer CLS embedding instead.

## Install

Use Python 3.10–3.13 and a PyTorch installation suitable for your CPU or CUDA system. Then:

```bash
git clone https://github.com/caithmac/GRASP.git
cd GRASP
pip install -e .
```

The core package includes the adapted MolE encoder and the DeBERTa attention implementation. See [license and attribution](#license-and-attribution) below. Model weights are downloaded from [caithmac/GRASP on Hugging Face](https://huggingface.co/caithmac/GRASP) when the default model ID is used.

**Try in Colab:** [extract embeddings](https://colab.research.google.com/github/caithmac/GRASP/blob/main/notebooks/01_embeddings.ipynb) · [fine-tune and predict](https://colab.research.google.com/github/caithmac/GRASP/blob/main/notebooks/02_finetune.ipynb). Run the cells from top to bottom; the first code cell installs this repository in Colab. For fine-tuning, select a GPU under **Runtime → Change runtime type** when one is available. The example runs on CPU too, more slowly. Colab runtimes are temporary, so download trained predictor files you want to keep.

## Extract representations

```python
from grasp import GRASPEncoder

encoder = GRASPEncoder.from_pretrained("caithmac/GRASP")
vectors = encoder.encode(["CCO", "c1ccccc1"])
print(vectors.shape)  # (2, 768)
```

For a CSV with a `smiles` column:

```bash
python -m grasp embed --model caithmac/GRASP --input-csv molecules.csv --output-csv embeddings.csv
```

The API reports the row of an invalid SMILES and rejects molecules exceeding 511 atoms (the encoder's 512 positions include CLS). It does not silently drop or truncate molecules. The pretraining corpus was limited to 96 heavy atoms; larger downstream molecules can be processed within the positional capacity, but may be outside the familiar training distribution.

## Fine-tune on a property

Prepare `train.csv` and `valid.csv`, each with `smiles,target` columns and no overlapping SMILES. For binary classification, targets must be `0` or `1`. The command saves a complete predictor directory, including its encoder, readout, vocabulary, task settings, and target scaling.

```bash
python -m grasp finetune \
  --model caithmac/GRASP \
  --train-csv train.csv --valid-csv valid.csv \
  --task regression --method full --output-dir runs/my_property

python -m grasp predict \
  --model runs/my_property \
  --input-csv molecules.csv --output-csv predictions.csv
```

`--method` can be `full` (default), `lora`, or `frozen`. These use the paper's layers 4/8/10/12, learned atom-attention pooling, and 512-unit endpoint head. The default learning rates are `1e-5`, `5e-5`, and `1e-3`, respectively. Regression predictions are returned in the training target's original units; binary predictions are probabilities for class 1. Use a validation set separated by chemical structure when assessing transfer. The [two notebooks](notebooks/) show embedding extraction and a short illustrative fine-tuning run.

## Model identity and paper reproduction

The source 50k encoder state dict has SHA-256 `7940ca66fc8f6785d6ae2e63b3965f69425e872cd28f4a975493fb1972742ee1`. `scripts/prepare_model.py` checks this identity before converting to `model.safetensors`, then verifies every tensor exactly. The release package retains the 211-token radius-0 vocabulary and the final encoder architecture explicitly.

The [reproduction supplement](reproduction/README.md) contains the sanitized training, evaluation, audit, and provenance material. It is separate from the short user workflow. No raw ZINC, ChEMBL, OpenADMET, or TDC datasets are included.

**Paper:** Srivastava et al., *GRASP: Graph Representation Learning with Assay Supervision for Molecular Properties*. The arXiv identifier will be added after posting. The model card describes the evaluation protocol and limitations; the notebook's illustrative run does not reproduce paper scores.

## License and attribution

The adapted MolE-derived code and model release retain the upstream [CC BY-NC 4.0 terms](LICENSE). The bundled DeBERTa code retains its [MIT license](vendor/DEBERTA_LICENSE). GRASP builds on Recursion Pharmaceuticals' MolE and the DeBERTa implementation. See [NOTICE](NOTICE) for source attribution. Respect the terms of any datasets you use for fine-tuning.
