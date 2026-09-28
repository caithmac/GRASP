"""
Download ZINC20 1.54B SMILES from HuggingFace, write parquet shards.
Streaming — never loads full dataset into memory.
Usage: python download_zinc20_1.5b.py --output_dir /mnt/data/zinc20_1.5b
"""
import argparse
import os
import sys
import time
from datasets import load_dataset

# pandas imported after arg check to fail fast if datasets missing

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
parser.add_argument("--shard_size", type=int, default=1_000_000)
parser.add_argument("--val_size", type=int, default=100_000)
parser.add_argument("--max_mols", type=int, default=None, help="Cap for testing")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print(f"Loading haydn-jones/ZINC20 (streaming)...", flush=True)
ds = load_dataset("haydn-jones/ZINC20", split="train", streaming=True)

# Peek to find SMILES column
first = next(iter(ds.take(1)))
smiles_col = [c for c in first.keys() if "SMILES" in c.upper() and "SELFIES" not in c.upper()]
if not smiles_col:
    print(f"ERROR: could not find SMILES column. Available: {list(first.keys())}", flush=True)
    sys.exit(1)
smiles_col = smiles_col[0]
print(f"SMILES column: '{smiles_col}'", flush=True)

import pandas as pd

val_buf = []
train_buf = []
shard_idx = 0
total = 0
t0 = time.time()
last_report = 0

for row in ds:
    smi = row.get(smiles_col)
    if smi and isinstance(smi, str) and len(smi) > 0:
        entry = {"smiles": smi}
        if total < args.val_size:
            val_buf.append(entry)
        else:
            train_buf.append(entry)
        total += 1

    # Write val.parquet first
    if len(val_buf) >= args.val_size and not os.path.exists(os.path.join(args.output_dir, "val.parquet")):
        df_val = pd.DataFrame(val_buf)
        df_val.to_parquet(os.path.join(args.output_dir, "val.parquet"), index=False)
        print(f"  Wrote val.parquet ({len(val_buf):,} mols)", flush=True)
        val_buf = []  # free memory

    # Write train shard when buffer full
    if len(train_buf) >= args.shard_size:
        df_train = pd.DataFrame(train_buf)
        path = os.path.join(args.output_dir, f"train_shard_{shard_idx:05d}.parquet")
        df_train.to_parquet(path, index=False)
        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        print(f"  [{elapsed/3600:.1f}h] Wrote {path} ({len(train_buf):,} mols, {total:,.0f} total, {rate:,.0f} mol/s)", flush=True)
        shard_idx += 1
        train_buf = []
        last_report = total

    if args.max_mols and total >= args.max_mols:
        print(f"Reached --max_mols={args.max_mols}, stopping.", flush=True)
        break

# Final train shard
if train_buf:
    df_train = pd.DataFrame(train_buf)
    path = os.path.join(args.output_dir, f"train_shard_{shard_idx:05d}.parquet")
    df_train.to_parquet(path, index=False)
    print(f"  Wrote {path} ({len(train_buf):,} mols)", flush=True)
    shard_idx += 1

# Write val.parquet if not written yet (small test run)
if val_buf and not os.path.exists(os.path.join(args.output_dir, "val.parquet")):
    df_val = pd.DataFrame(val_buf)
    df_val.to_parquet(os.path.join(args.output_dir, "val.parquet"), index=False)
    print(f"  Wrote val.parquet ({len(val_buf):,} mols)", flush=True)

elapsed = time.time() - t0
print(f"DONE. {total:,} molecules in {shard_idx} train shards + val.parquet. {elapsed/3600:.1f} hours.", flush=True)
