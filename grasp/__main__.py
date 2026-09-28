"""Command-line access to GRASP embeddings, fine-tuning, and predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .encoder import GRASPEncoder
from .predictor import GRASPPredictor
from .train import fit


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m grasp")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("finetune", help="Train a property head and adapt GRASP")
    train.add_argument("--model", default="caithmac/GRASP")
    train.add_argument("--train-csv", required=True)
    train.add_argument("--valid-csv", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--task", choices=("regression", "binary"), required=True)
    train.add_argument("--method", choices=("full", "lora", "frozen"), default="full")
    train.add_argument("--smiles-column", default="smiles")
    train.add_argument("--target-column", default="target")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--patience", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device")
    for name in ("predict", "embed"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--input-csv", required=True)
        command.add_argument("--output-csv", required=True)
        command.add_argument("--smiles-column", default="smiles")
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--device")
    args = parser.parse_args()
    if args.command == "finetune":
        print(fit(model_source=args.model, train_csv=args.train_csv, valid_csv=args.valid_csv,
                  output_dir=args.output_dir, task=args.task, method=args.method,
                  smiles_column=args.smiles_column, target_column=args.target_column,
                  epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                  learning_rate=args.learning_rate, seed=args.seed, device=args.device))
        return
    frame = pd.read_csv(args.input_csv)
    if args.smiles_column not in frame:
        raise ValueError(f"Missing SMILES column: {args.smiles_column}")
    smiles = frame[args.smiles_column].tolist()
    if args.command == "predict":
        model = GRASPPredictor.from_pretrained(args.model, device=args.device)
        frame["prediction"] = model.predict(smiles, batch_size=args.batch_size)
    else:
        model = GRASPEncoder.from_pretrained(args.model, device=args.device)
        embeddings = model.encode(smiles, batch_size=args.batch_size)
        for index in range(embeddings.shape[1]):
            frame[f"embedding_{index}"] = embeddings[:, index]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
