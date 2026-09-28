"""Calculate an RTD continuation in molecule presentations, not nominal steps."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parquet_rows(data_dir: Path) -> tuple[int, int]:
    import pyarrow.parquet as pq

    shards = sorted(data_dir.glob("train_shard_*.parquet"))
    if not shards:
        raise SystemExit(f"ERROR: no train_shard_*.parquet under {data_dir}")
    return sum(pq.ParquetFile(path).metadata.num_rows for path in shards), len(shards)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prior-examples", type=int, default=736_000_000)
    parser.add_argument("--effective-batch", type=int, default=256)
    parser.add_argument("--target-passes", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_rows, shard_count = parquet_rows(args.data_dir)
    target_examples = math.ceil(train_rows * args.target_passes)
    remaining = max(0, target_examples - args.prior_examples)
    additional_steps = math.ceil(remaining / args.effective_batch)
    final_examples = args.prior_examples + additional_steps * args.effective_batch
    plan = {
        "data_dir": str(args.data_dir),
        "train_parquet_shards": shard_count,
        "train_rows": train_rows,
        "prior_examples": args.prior_examples,
        "prior_exposure_passes": args.prior_examples / train_rows,
        "target_passes": args.target_passes,
        "target_examples": target_examples,
        "effective_batch": args.effective_batch,
        "additional_optimizer_steps": additional_steps,
        "additional_examples": additional_steps * args.effective_batch,
        "final_examples": final_examples,
        "final_exposure_passes": final_examples / train_rows,
        "note": (
            "Exposure passes count presentations. Earlier interrupted runs restarted "
            "shuffle order, so this is not a claim of exactly one unique-data epoch."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
