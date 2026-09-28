import pandas as pd
import pytest

from grasp.encoder import validate_smiles
from grasp.train import fit, read_labeled_csv


def test_invalid_smiles_reports_row():
    with pytest.raises(ValueError, match="row 1"):
        validate_smiles(["CCO", "not_smiles"])


def test_over_capacity_is_explicit():
    with pytest.raises(ValueError, match="511"):
        validate_smiles(["C" * 512])


def test_binary_labels_must_be_zero_or_one(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"smiles": ["CCO"], "target": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="0 or 1"):
        read_labeled_csv(path, "smiles", "target", "binary")


def test_equivalent_smiles_cannot_cross_validation(tmp_path):
    train = tmp_path / "train.csv"
    valid = tmp_path / "valid.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN"], "target": [0.0, 1.0]}).to_csv(train, index=False)
    pd.DataFrame({"smiles": ["OCC"], "target": [0.5]}).to_csv(valid, index=False)
    with pytest.raises(ValueError, match="overlap"):
        fit(model_source="unused", train_csv=train, valid_csv=valid, output_dir=tmp_path / "out",
            task="regression")
