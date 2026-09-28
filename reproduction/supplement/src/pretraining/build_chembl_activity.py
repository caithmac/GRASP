"""Build ChEMBL supervised pretraining dataset following Mayr et al. protocol.

Steps:
  1. Download ChEMBL SQLite (if not cached)
  2. Query binding/functional assays with potency measurements
  3. Binarize via Mayr 4-step protocol
  4. Filter: keep assays with >= MIN_ASSAY_MOLS measurements
  5. Remove TDC ADMET test-set molecules (canonicalized SMILES match)
  6. Save: smiles.txt, labels.npy (float16, NaN=missing), assay_ids.txt

Env vars:
  CHEMBL_DB_PATH   /mnt/data/chembl/chembl_34_sqlite/chembl_34.db
  OUTPUT_DIR       /mnt/data/chembl_supervised
  TDC_DATA_DIR     /mnt/tdc_data
  MIN_ASSAY_MOLS   100
  CHEMBL_VERSION   34
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles

CHEMBL_DB_PATH  = os.environ.get("CHEMBL_DB_PATH",  "/mnt/data/chembl/chembl_36_sqlite/chembl_36.db")
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR",       "/mnt/data/chembl_supervised")
TDC_DATA_DIR    = os.environ.get("TDC_DATA_DIR",     "/mnt/tdc_data")
MIN_ASSAY_MOLS  = int(os.environ.get("MIN_ASSAY_MOLS", "5000"))
CHEMBL_VERSION  = os.environ.get("CHEMBL_VERSION",   "36")

VALID_RELATIONS = {"<", "<=", ">", ">=", "=", "~"}
ACTIVE_COMMENTS   = {"active"}
INACTIVE_COMMENTS = {"inactive", "not active", "not_active", "not-active"}

STANDARD_TYPES = {
    "IC50", "Ki", "EC50", "Kd", "AC50", "GI50", "CC50",
    "MIC", "LC50", "Potency", "ED50", "LD50",
}


def find_db_in_tar(tar_path: Path) -> str:
    """Return the path of the .db file inside the tarball."""
    with tarfile.open(tar_path, "r:gz") as t:
        members = t.getnames()
    db_members = [m for m in members if m.endswith(".db")]
    if not db_members:
        raise FileNotFoundError(f"No .db file found in {tar_path}. Contents: {members[:10]}")
    # pick the largest by name length heuristic (main DB vs auxiliary)
    return sorted(db_members, key=len)[-1]


def download_chembl(db_path: str, version: str) -> str:
    """Download + extract ChEMBL SQLite. Returns actual path to .db file."""
    db_path = Path(db_path)
    extract_root = db_path.parent.parent  # /mnt/data/chembl/
    extract_root.mkdir(parents=True, exist_ok=True)

    tar_path = extract_root / f"chembl_{version}_sqlite.tar.gz"

    # Download if needed
    if not tar_path.exists() or tar_path.stat().st_size < 1_000_000:
        tar_url = (
            f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
            f"chembl_{version}/chembl_{version}_sqlite.tar.gz"
        )
        print(f"  Downloading {tar_url} ...")
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(tar_path), tar_url], check=True)
    else:
        print(f"  Tarball already at {tar_path} ({tar_path.stat().st_size/1e9:.1f} GB)")

    # Inspect tarball to find actual .db path
    print(f"  Inspecting tarball structure...")
    db_member = find_db_in_tar(tar_path)
    actual_db_path = extract_root / db_member
    print(f"  DB in tarball: {db_member}")
    print(f"  Expected path after extract: {actual_db_path}")

    if actual_db_path.exists() and actual_db_path.stat().st_size > 1_000_000_000:
        print(f"  Already extracted ({actual_db_path.stat().st_size/1e9:.1f} GB), skipping")
        return str(actual_db_path)

    print(f"  Extracting to {extract_root} ...")
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(path=str(extract_root))

    if not actual_db_path.exists():
        raise FileNotFoundError(f"DB not found at {actual_db_path} after extraction")
    print(f"  Extracted. DB at {actual_db_path} ({actual_db_path.stat().st_size/1e9:.1f} GB)")
    return str(actual_db_path)


def query_chembl(db_path: str) -> pd.DataFrame:
    """Pull binding/functional activity measurements with precomputed pChEMBL."""
    print("  Querying ChEMBL (may take several minutes)...")
    con = sqlite3.connect(db_path)
    sql = """
        SELECT
            cs.canonical_smiles,
            act.molregno,
            act.assay_id,
            act.pchembl_value,
            act.activity_comment,
            act.potential_duplicate
        FROM activities act
        JOIN assays a               ON act.assay_id  = a.assay_id
        JOIN compound_structures cs ON act.molregno  = cs.molregno
        WHERE a.assay_type IN ('B', 'F')
          AND cs.canonical_smiles IS NOT NULL
          AND (act.potential_duplicate IS NULL OR act.potential_duplicate = 0)
          AND (
               act.data_validity_comment IS NULL
               OR act.data_validity_comment = 'Manually validated'
          )
    """
    df = pd.read_sql_query(sql, con)
    con.close()
    print(f"  Raw rows: {len(df):,}")
    return df


def binarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mayr et al. binarization using ChEMBL's precomputed pChEMBL_value.

    Priority:
      1. activity_comment (Active/Inactive strings) — no numeric value needed
      2. pchembl_value >= 5.5 → active; <= 4.5 → inactive; else discard
    Step 4: remove contradictions within same (smiles, assay_id).
    """
    df = df.copy()
    df["label"] = np.nan

    # ── Step 1: activity_comment ───────────────────────────────────────────
    comment = df["activity_comment"].fillna("").str.lower().str.strip()
    df.loc[comment.isin(ACTIVE_COMMENTS),   "label"] = 1.0
    df.loc[comment.isin(INACTIVE_COMMENTS), "label"] = 0.0

    # ── Step 2+3: pChEMBL_value for rows without comment label ────────────
    no_comment_label = df["label"].isna()
    pval = df["pchembl_value"]
    df.loc[no_comment_label & (pval >= 5.5), "label"] = 1.0
    df.loc[no_comment_label & (pval <= 4.5), "label"] = 0.0
    # weak actives (4.5 < pval < 5.5) → remain NaN → discarded below

    # ── Keep only definite labels ──────────────────────────────────────────
    df = df[df["label"].notna()][["canonical_smiles", "assay_id", "label"]].copy()
    df["label"] = df["label"].astype(np.int8)
    print(f"  After binarization: {len(df):,} labeled measurements")

    # ── Step 4: remove contradictions ─────────────────────────────────────
    key = ["canonical_smiles", "assay_id"]
    nunique = df.groupby(key)["label"].nunique()
    consistent_keys = nunique[nunique == 1].reset_index()[key]
    df = df.merge(consistent_keys, on=key).drop_duplicates(key)
    print(f"  After removing contradictions: {len(df):,} unique (smiles, assay) pairs")
    return df


def canonicalize_smiles(s: str) -> str | None:
    mol = MolFromSmiles(s)
    if mol is None:
        return None
    return MolToSmiles(mol)


def get_tdc_test_smiles() -> set[str]:
    """Return canonical SMILES of all TDC ADMET benchmark test-set molecules."""
    print("  Collecting TDC ADMET test-set SMILES...")
    os.makedirs(TDC_DATA_DIR, exist_ok=True)
    test_smiles: set[str] = set()

    # TDC ADMET group datasets
    admet_datasets = [
        ("ADME", "BBB_Martins"),
        ("ADME", "Caco2_Wang"),
        ("ADME", "HIA_Hou"),
        ("ADME", "Pgp_Broccatelli"),
        ("ADME", "Bioavailability_Ma"),
        ("ADME", "Lipophilicity_AstraZeneca"),
        ("ADME", "Solubility_AqSolDB"),
        ("ADME", "CYP2C19_Veith"),
        ("ADME", "CYP2D6_Veith"),
        ("ADME", "CYP3A4_Veith"),
        ("ADME", "CYP1A2_Veith"),
        ("ADME", "CYP2C9_Veith"),
        ("ADME", "CYP2C9_Substrate_CarbonMangels"),
        ("ADME", "CYP2D6_Substrate_CarbonMangels"),
        ("ADME", "CYP3A4_Substrate_CarbonMangels"),
        ("ADME", "Half_Life_Obach"),
        ("ADME", "Clearance_Hepatocyte_AZ"),
        ("ADME", "Clearance_Microsome_AZ"),
        ("Tox",  "hERG"),
        ("Tox",  "hERG_Karim"),
        ("Tox",  "AMES"),
        ("Tox",  "DILI"),
        ("Tox",  "Skin_Reaction"),
        ("Tox",  "Carcinogens_Lagunin"),
        ("Tox",  "ClinTox"),
        ("Tox",  "Tox21"),
        ("HTS",  "SARSCoV2_Vitro_Touret"),
        ("HTS",  "SARSCoV2_3CLPro_Diamond"),
    ]

    for module_name, dataset_name in admet_datasets:
        try:
            if module_name == "ADME":
                from tdc.single_pred import ADME
                task = ADME(name=dataset_name, path=TDC_DATA_DIR)
            elif module_name == "Tox":
                from tdc.single_pred import Tox
                task = Tox(name=dataset_name, path=TDC_DATA_DIR)
            elif module_name == "HTS":
                from tdc.single_pred import HTS
                task = HTS(name=dataset_name, path=TDC_DATA_DIR)
            else:
                continue
            split = task.get_split(method="scaffold", seed=42)
            for s in split["test"]["Drug"]:
                c = canonicalize_smiles(s)
                if c:
                    test_smiles.add(c)
        except Exception as e:
            print(f"    Warning: could not load {dataset_name}: {e}")

    # Also add BACE explicitly
    try:
        from tdc.single_pred import ADME as ADME2
        task = ADME2(name="BACE", path=TDC_DATA_DIR)
        split = task.get_split(method="scaffold", seed=42)
        for s in split["test"]["Drug"]:
            c = canonicalize_smiles(s)
            if c:
                test_smiles.add(c)
    except Exception as e:
        print(f"    Warning BACE: {e}")

    print(f"  TDC test molecules: {len(test_smiles):,}")
    return test_smiles


def build_label_matrix(df: pd.DataFrame, tdc_smiles: set[str]):
    """Build (smiles list, float16 label matrix, assay_id list)."""

    # Canonicalize ChEMBL SMILES
    print("  Canonicalizing SMILES...")
    df["canon"] = df["canonical_smiles"].apply(canonicalize_smiles)
    df = df[df["canon"].notna()].copy()

    # Remove TDC test molecules
    before = df["canon"].nunique()
    df = df[~df["canon"].isin(tdc_smiles)]
    after  = df["canon"].nunique()
    print(f"  Removed TDC test mols: {before - after:,} ({before:,} → {after:,} unique molecules)")

    # Filter assays: keep only assays with >= MIN_ASSAY_MOLS unique molecules
    assay_counts = df.groupby("assay_id")["canon"].nunique()
    valid_assays = assay_counts[assay_counts >= MIN_ASSAY_MOLS].index
    df = df[df["assay_id"].isin(valid_assays)]
    print(f"  Assays with >= {MIN_ASSAY_MOLS} mols: {len(valid_assays):,}")

    # Re-filter molecules: must still appear in at least 1 assay
    valid_mols = df["canon"].unique()
    print(f"  Unique molecules remaining: {len(valid_mols):,}")

    # Build pivot: rows = molecules, cols = assays
    assay_ids = sorted(df["assay_id"].unique())
    assay_to_idx = {a: i for i, a in enumerate(assay_ids)}
    mol_to_idx   = {m: i for i, m in enumerate(valid_mols)}

    n_mols   = len(valid_mols)
    n_assays = len(assay_ids)
    print(f"  Building label matrix: {n_mols:,} × {n_assays:,} ...")

    labels = np.full((n_mols, n_assays), np.nan, dtype=np.float16)
    for row in df.itertuples(index=False):
        mi = mol_to_idx[row.canon]
        ai = assay_to_idx[row.assay_id]
        labels[mi, ai] = row.label

    return list(valid_mols), labels, assay_ids


def main():
    print("=== Build ChEMBL supervised pretraining dataset ===")
    print(f"  ChEMBL DB: {CHEMBL_DB_PATH}")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"  Min assay mols: {MIN_ASSAY_MOLS}")

    # 1. Download
    print("\n[1] Download ChEMBL SQLite")
    actual_db_path = download_chembl(CHEMBL_DB_PATH, CHEMBL_VERSION)

    # 2. Query
    print("\n[2] Query ChEMBL")
    raw = query_chembl(actual_db_path)

    # 3. Binarize (Mayr 4-step)
    print("\n[3] Binarize (Mayr et al.)")
    labeled = binarize(raw)
    del raw

    # 4. TDC test set removal
    print("\n[4] TDC ADMET test-set removal")
    tdc_smiles = get_tdc_test_smiles()

    # 5. Build matrix
    print("\n[5] Build label matrix")
    smiles_list, label_matrix, assay_ids = build_label_matrix(labeled, tdc_smiles)

    # 6. Save
    print("\n[6] Saving outputs")
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    smiles_path  = out / "smiles.txt"
    labels_path  = out / "labels.npy"
    assays_path  = out / "assay_ids.txt"

    smiles_path.write_text("\n".join(smiles_list))
    np.save(str(labels_path), label_matrix)
    assays_path.write_text("\n".join(str(a) for a in assay_ids))

    print(f"  smiles.txt  → {len(smiles_list):,} molecules")
    print(f"  labels.npy  → shape {label_matrix.shape}, dtype {label_matrix.dtype}")
    print(f"  assay_ids   → {len(assay_ids):,} assays")
    print(f"  Sparsity: {np.isnan(label_matrix).mean()*100:.1f}% missing")
    print("\nDone.")


if __name__ == "__main__":
    main()
