"""Generate a tiny illustrative RDKit-computed property dataset."""
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen

SMILES = [
    "CCO", "CCCO", "CCCCO", "CCN", "CCCN", "CC(=O)O", "CCC(=O)O",
    "COC", "CCOC", "CCCOC", "CC(=O)N", "CCC(=O)N", "c1ccccc1",
    "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1", "c1ccncc1", "c1ccoc1",
    "CCCl", "CCCCl", "CCBr", "CC(C)O", "CC(C)C", "CC(C)(C)O",
    "CC(C)N", "CC(C)C(=O)O", "CCOCC", "CCCOCC", "CC(C)CO", "CC(C)CC",
]


def main() -> None:
    output = Path("examples/data")
    output.mkdir(parents=True, exist_ok=True)
    records = [{"smiles": smiles, "target": Crippen.MolLogP(Chem.MolFromSmiles(smiles))}
               for smiles in SMILES]
    frame = pd.DataFrame(records)
    frame.iloc[[i for i in range(len(frame)) if i % 5 != 0]].to_csv(output / "demo_train.csv", index=False)
    frame.iloc[[i for i in range(len(frame)) if i % 5 == 0]].to_csv(output / "demo_valid.csv", index=False)
    binary = frame.copy()
    binary["target"] = (binary["target"] > float(frame["target"].median())).astype(int)
    binary.iloc[[i for i in range(len(binary)) if i % 5 != 0]].to_csv(output / "demo_binary_train.csv", index=False)
    binary.iloc[[i for i in range(len(binary)) if i % 5 == 0]].to_csv(output / "demo_binary_valid.csv", index=False)
    print("Wrote illustrative RDKit logP training and validation CSVs")


if __name__ == "__main__":
    main()
