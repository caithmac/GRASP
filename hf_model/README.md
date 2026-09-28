---
license: cc-by-nc-4.0
library_name: pytorch
tags:
  - chemistry
  - molecular-representation-learning
  - graph-transformer
  - replaced-token-detection
  - assay-supervision
  - grasp
---

# GRASP

**Graph Representation Learning with Assay Supervision for Molecular Properties** is a 93.5M-parameter molecular graph Transformer. It was trained using replaced-token detection on 1.54B ZINC20 molecule presentations, then sparse supervision over 642 ChEMBL 36 assays. This model repository contains the fixed **50,000-step Step 2 encoder** used for downstream adaptation in the paper.

## Use

Install the [GRASP code repository](https://github.com/caithmac/GRASP) in Python 3.10–3.13, then:

```python
from grasp import GRASPEncoder

model = GRASPEncoder.from_pretrained("caithmac/GRASP")
embeddings = model.encode(["CCO", "c1ccccc1"])
print(embeddings.shape)  # (2, 768)
```

This returns the final encoder-layer CLS state. The encoder does not output experimental property predictions by itself. The paper's primary OpenADMET protocol learns a layer mixture, atom-attention pooler, and endpoint head; the repository provides full, LoRA, and frozen fine-tuning modes for new labeled datasets.

For a browser-based start, open the shared [embedding notebook in Colab](https://colab.research.google.com/drive/1i-NLImnIBcnv-qn5sun0J0Xkuge8rGhG?usp=sharing) or [fine-tuning notebook](https://colab.research.google.com/drive/1BcYhx7Jp_WPJFgvqRw_S53W3dge2CWfQ?usp=sharing). The [source notebooks](https://github.com/caithmac/GRASP/tree/main/notebooks) are versioned with the code repository.

## Model details and evaluation

The encoder has 12 layers, hidden size 768, 12 heads, shortest-path relative attention, and 211 atom-environment tokens including special tokens. The original fixed checkpoint SHA-256 is `7940ca66fc8f6785d6ae2e63b3965f69425e872cd28f4a975493fb1972742ee1`. `model.safetensors` is a tensor-identical conversion of that checkpoint; the repository's `SHA256SUMS` records the converted file identity.

In the arXiv paper, endpoint-specific full fine-tuning of GRASP reaches mean MAE 0.374 across 23 reconstructed cluster-held-out OpenADMET endpoints. This figure is **not** a zero-shot score for these encoder weights. The paper also reports a separate 22-task TDC ADMET evaluation. See the paper and the [reproduction supplement](https://github.com/caithmac/GRASP/tree/main/reproduction) for splits, training settings, baselines, and limitations.

## Intended use and limitations

GRASP is intended for research on molecular representations and downstream property models. The encoder was trained on 2D molecular graphs; no 3D geometry or experimental uncertainty is provided. Its radius-0 atom tokens and graph-distance inputs do not encode tetrahedral or E/Z stereochemistry, so the released checkpoint should not be used to distinguish stereoisomers. Molecular predictions after downstream fitting can fail outside the training chemical domain and should be checked experimentally. Atom ordering can affect outputs because the encoder has learned absolute position embeddings; any ordering-related difference is not evidence of stereo sensitivity. The pretraining inputs were limited to 96 heavy atoms; molecules up to 511 atoms can be processed by the code, but larger molecules may be well outside the training distribution. Do not use generated predictions as clinical or toxicological determinations.

## Training data and license

Structural pretraining used ZINC20 molecule presentations. Sparse assay adaptation used ChEMBL 36 molecules and 642 selected binding or functional assays. Raw datasets are not redistributed here. The model and MolE-derived implementation retain CC BY-NC 4.0 terms; the bundled DeBERTa code has a separate MIT license. The code repository includes attribution and exact provenance records.

## Citation

Satya Pratik Srivastava, Rohan Gorantla, Sharath Krishna Chundru, Harshit Singh, Antonia S. J. S. Mey, and Rajeev Kumar Singh. *GRASP: Graph Representation Learning with Assay Supervision for Molecular Properties*. arXiv preprint (identifier pending).
