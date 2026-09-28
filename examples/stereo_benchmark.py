"""Exploratory stereo-aware prediction check on public OdorNet odor labels.

Run: python -m examples.stereo_benchmark
This is not a GRASP paper benchmark or an experimental assay benchmark.
"""

from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grasp import GRASPEncoder


SOURCE = "88d146360dfc548462a5ebd6a081ea0590f2819e"
BASE = f"https://raw.githubusercontent.com/NKU-DOIE/OdorNet/{SOURCE}/data/processed/"
TARGET = "green&herbal"
RDLogger.DisableLog("rdApp.warning")  # GRASP's retained tokenizer uses RDKit's legacy Morgan API.


def load_data():
    frames = []
    for name in ("dataset_train_aligned.csv", "dataset_val_aligned.csv"):
        response = requests.get(BASE + name, timeout=60)
        response.raise_for_status()
        frames.append(pd.read_csv(BytesIO(response.content)))

    # Keep specified stereo and one row per isomer. Connectivity groups stay intact in CV.
    rows = {}
    conflicts = set()
    for smiles, label in pd.concat(frames, ignore_index=True)[["SMILES", TARGET]].itertuples(index=False):
        if pd.isna(label):
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() > 96:  # GRASP's pretraining size range
            continue
        isomer = Chem.MolToSmiles(mol, isomericSmiles=True)
        connectivity = Chem.MolToSmiles(mol, isomericSmiles=False)
        if isomer == connectivity:
            continue
        label = int(label)
        if isomer in rows and rows[isomer][1] != label:
            conflicts.add(isomer)
        else:
            rows[isomer] = (connectivity, label)
    for isomer in conflicts:
        rows.pop(isomer, None)

    smiles = sorted(rows)
    groups = np.array([rows[s][0] for s in smiles])
    labels = np.array([rows[s][1] for s in smiles])
    return smiles, groups, labels


def fingerprint_matrix(smiles, include_chirality):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=include_chirality
    )
    result = np.empty((len(smiles), 2048), dtype=np.float32)
    for i, value in enumerate(smiles):
        DataStructs.ConvertToNumpyArray(
            generator.GetFingerprint(Chem.MolFromSmiles(value)), result[i]
        )
    return result


def pair_accuracy(groups, labels, scores):
    by_group = defaultdict(list)
    for i, group in enumerate(groups):
        by_group[group].append(i)
    accuracies = []
    for indices in by_group.values():
        positives = [i for i in indices if labels[i] == 1]
        negatives = [i for i in indices if labels[i] == 0]
        if positives and negatives:
            differences = [scores[p] - scores[n] for p in positives for n in negatives]
            accuracies.append(np.mean([0.5 if abs(d) < 1e-5 else float(d > 0) for d in differences]))
    return float(np.mean(accuracies)), len(accuracies)


def main():
    smiles, groups, labels = load_data()
    print(f"OdorNet {SOURCE[:8]}, target: {TARGET}")
    print(f"{len(smiles)} stereo-specified molecules; {len(set(groups))} connectivity groups")
    source = "hf_model" if Path("hf_model/model.safetensors").is_file() else "caithmac/GRASP"
    encoder = GRASPEncoder.from_pretrained(source, device="cpu")
    grasp_parts = []
    for start in range(0, len(smiles), 128):
        end = min(start + 128, len(smiles))
        grasp_parts.append(encoder.encode(smiles[start:end], batch_size=32))
        print(f"Encoded {end}/{len(smiles)} molecules", flush=True)
    grasp = np.concatenate(grasp_parts)
    achiral = fingerprint_matrix(smiles, include_chirality=False)
    chiral = fingerprint_matrix(smiles, include_chirality=True)

    # Fails if the chosen representation stops distinguishing a known stereo pair.
    example = ["F[C@H](Cl)Br", "F[C@@H](Cl)Br"]
    example_fps = fingerprint_matrix(example, True)
    assert not np.array_equal(example_fps[0], example_fps[1])

    features = {
        "GRASP": grasp,
        "GRASP + achiral Morgan": np.concatenate([grasp, achiral], axis=1),
        "GRASP + chiral Morgan": np.concatenate([grasp, chiral], axis=1),
    }
    folds = list(GroupKFold(n_splits=5).split(grasp, labels, groups))
    for train, test in folds:
        assert set(groups[train]).isdisjoint(groups[test])

    results = {}
    for name, x in features.items():
        scores = np.empty(len(labels))
        for train, test in folds:
            predictor = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=1000),
            )
            predictor.fit(x[train], labels[train])
            scores[test] = predictor.predict_proba(x[test])[:, 1]
        pair_score, n_groups = pair_accuracy(groups, labels, scores)
        auroc = float(roc_auc_score(labels, scores))
        results[name] = (auroc, pair_score)
        print(f"{name:24s} AUROC={auroc:.3f}  "
              f"discordant-pair accuracy={pair_score:.3f} ({n_groups} groups)")
    return results


if __name__ == "__main__":
    main()
