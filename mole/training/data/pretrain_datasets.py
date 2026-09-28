import os
import glob
import hashlib
import logging
import pickle
import random
from typing import List, Optional, Union

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse

RDLogger.DisableLog("rdApp.*")

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mole.training.data.datasets import MolDataset
from mole.training.data.utils import open_dictionary

logger = logging.getLogger(__name__)


class MolPreTrainDataset(MolDataset):
    """MolDataset with random atom-token masking for RTD pre-training."""

    def __init__(self, smiles, dictionary_inp, mask_prob=0.15, radius_inp=0, useFeatures_inp=False):
        super().__init__(
            smiles=smiles,
            dictionary_inp=dictionary_inp,
            radius_inp=radius_inp,
            useFeatures_inp=useFeatures_inp,
            cls_token=True,
            labels=None,
        )
        self.mask_prob = mask_prob
        self.mask_token_id = dictionary_inp["MASK"]

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        tokens = data.x.clone()
        L = len(tokens)

        # All positions except CLS (index 0) are masking candidates
        mask_candidates = torch.ones(L, dtype=torch.bool)
        mask_candidates[0] = False

        mask_pos = mask_candidates & (torch.rand(L) < self.mask_prob)

        original_ids = tokens.clone()
        mlm_labels = torch.full((L,), -100, dtype=torch.long)
        mlm_labels[mask_pos] = tokens[mask_pos]

        tokens[mask_pos] = self.mask_token_id
        data.x = tokens
        data.original_ids = original_ids
        data.mlm_labels = mlm_labels
        return data


class TokenizedMolPreTrainDataset(IterableDataset):
    """
    Streaming dataset that reads from pre-tokenized .pt shards.
    Automatically handles DDP and multi-worker splitting.
    """

    def __init__(
        self,
        shard_paths: List[str],
        mask_token_id: int,
        mask_prob: float = 0.15,
        shuffle: bool = True,
    ):
        super().__init__()
        self.shard_paths = sorted(shard_paths)
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.shuffle = shuffle

    def _get_worker_shards(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0

        total_workers = num_workers * world_size
        global_worker_id = rank * num_workers + worker_id

        # Split shards among all workers across all nodes
        my_shards = [
            s for i, s in enumerate(self.shard_paths)
            if i % total_workers == global_worker_id
        ]
        return my_shards

    def __iter__(self):
        shards = self._get_worker_shards()
        if self.shuffle:
            random.shuffle(shards)

        for path in shards:
            # .pt files are a sequence of pickle chunks (each a list of dicts).
            # Read one chunk at a time to keep peak RAM ~163MB/worker vs ~8GB.
            with open(path, "rb") as f:
                while True:
                    try:
                        chunk = pickle.load(f)
                    except EOFError:
                        break
                    if self.shuffle:
                        random.shuffle(chunk)
                    for item in chunk:
                        tokens = torch.from_numpy(item["tokens"]).long()
                        L = len(tokens)

                        mask_candidates = torch.ones(L, dtype=torch.bool)
                        mask_candidates[0] = False  # skip CLS
                        mask_pos = mask_candidates & (torch.rand(L) < self.mask_prob)

                        original_ids = tokens.clone()
                        mlm_labels = torch.full((L,), -100, dtype=torch.long)
                        mlm_labels[mask_pos] = tokens[mask_pos]
                        tokens[mask_pos] = self.mask_token_id

                        yield Data(
                            x=tokens,
                            edge_index=torch.stack([
                                torch.from_numpy(item["row"]).long(),
                                torch.from_numpy(item["col"]).long()
                            ]),
                            edge_attr=torch.from_numpy(item["data"]).long(),
                            original_ids=original_ids,
                            mlm_labels=mlm_labels,
                        )


class TokenizedMLMDataset(IterableDataset):
    """
    Reads MLM-specific .pt shards produced by build_mlm_data.py.
    Each item has radius-0 input tokens + radius-2 target tokens.
    Applies masking to radius-0 inputs; builds r2_mlm_labels (radius-2 IDs at
    masked positions, -100 elsewhere) for cross-entropy against 141k-class head.
    """

    def __init__(
        self,
        shard_paths: List[str],
        mask_token_id: int,
        mask_prob: float = 0.15,
        shuffle: bool = True,
    ):
        super().__init__()
        self.shard_paths = sorted(shard_paths)
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.shuffle = shuffle

    def _get_worker_shards(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size, rank = 1, 0
        total_workers = num_workers * world_size
        global_worker_id = rank * num_workers + worker_id
        return [s for i, s in enumerate(self.shard_paths) if i % total_workers == global_worker_id]

    def __iter__(self):
        shards = self._get_worker_shards()
        if self.shuffle:
            random.shuffle(shards)
        for path in shards:
            with open(path, "rb") as f:
                while True:
                    try:
                        chunk = pickle.load(f)
                    except EOFError:
                        break
                    if self.shuffle:
                        random.shuffle(chunk)
                    for item in chunk:
                        tokens = torch.from_numpy(item["tokens"]).long()
                        r2_tokens = torch.from_numpy(item["r2_tokens"]).long()
                        L = len(tokens)

                        mask_candidates = torch.ones(L, dtype=torch.bool)
                        mask_candidates[0] = False  # skip CLS
                        mask_pos = mask_candidates & (torch.rand(L) < self.mask_prob)

                        # r2_mlm_labels: radius-2 target at masked positions, -100 elsewhere
                        r2_mlm_labels = torch.full((L,), -100, dtype=torch.long)
                        r2_mlm_labels[mask_pos] = r2_tokens[mask_pos]

                        tokens[mask_pos] = self.mask_token_id

                        yield Data(
                            x=tokens,
                            r2_mlm_labels=r2_mlm_labels,
                            edge_index=torch.stack([
                                torch.from_numpy(item["row"]).long(),
                                torch.from_numpy(item["col"]).long()
                            ]),
                            edge_attr=torch.from_numpy(item["data"]).long(),
                        )


class StreamingParquetDataset(IterableDataset):
    """
    Streaming dataset that reads raw SMILES parquet shards and tokenizes on-the-fly.
    Avoids pre-tokenization storage (~5GB/shard). Handles DDP + multi-worker splitting.
    """

    def __init__(
        self,
        shard_paths: List[str],
        dictionary: dict,
        mask_token_id: int,
        mask_prob: float = 0.15,
        radius: int = 0,
        use_features: bool = False,
        max_atoms: int = 128,
        shuffle: bool = True,
        deterministic_mask_seed: Optional[int] = None,
        replicate_across_ranks: bool = False,
    ):
        super().__init__()
        self.shard_paths = sorted(shard_paths)
        self.dictionary = dictionary
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.radius = radius
        self.use_features = use_features
        self.max_atoms = max_atoms
        self.shuffle = shuffle
        self.deterministic_mask_seed = deterministic_mask_seed
        self.replicate_across_ranks = replicate_across_ranks

    def _get_worker_shards(self):
        if self.replicate_across_ranks:
            return list(self.shard_paths)
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size, rank = 1, 0
        total_workers = num_workers * world_size
        global_worker_id = rank * num_workers + worker_id
        return [s for i, s in enumerate(self.shard_paths) if i % total_workers == global_worker_id]

    def _smiles_to_data(self, smi: str):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumHeavyAtoms() > self.max_atoms:
                return None

            # Atom environment tokens
            info = {}
            atomenv = {}
            disconnected = [i for i, a in enumerate(mol.GetAtoms()) if a.GetDegree() == 0]
            AllChem.GetMorganFingerprint(
                mol, self.radius, bitInfo=info,
                includeRedundantEnvironments=True, useFeatures=self.use_features,
            )
            for k, v in info.items():
                for e in v:
                    if e[1] == self.radius or e[0] in disconnected:
                        atomenv[e[0]] = self.dictionary.get(k, self.dictionary["UNK"])
            atomenv = dict(sorted(atomenv.items()))
            tokens_list = list(atomenv.values()) or [self.dictionary["UNK"]] * mol.GetNumAtoms()

            # Distance matrix → sparse COO with CLS row/col
            dist_mat = Chem.GetDistanceMatrix(mol)
            dist_mat[dist_mat == 1.0e08] = -1
            dist_mat = sparse.coo_matrix(dist_mat + 1)
            dist_mat = sparse.vstack([np.zeros(dist_mat.shape[0])[None, :], dist_mat])
            dist_mat = sparse.hstack([np.zeros(dist_mat.shape[0])[:, None], dist_mat])

            tokens = np.insert(tokens_list, 0, self.dictionary["CLS"])
            tokens = torch.tensor(tokens, dtype=torch.long)
            L = len(tokens)

            mask_candidates = torch.ones(L, dtype=torch.bool)
            mask_candidates[0] = False
            if self.deterministic_mask_seed is None:
                mask_draw = torch.rand(L)
            else:
                digest = hashlib.blake2b(
                    smi.encode("utf-8"), digest_size=8, person=b"mole-rtd"
                ).digest()
                seed = int.from_bytes(digest, "little") ^ self.deterministic_mask_seed
                generator = torch.Generator().manual_seed(seed & ((1 << 63) - 1))
                mask_draw = torch.rand(L, generator=generator)
            mask_pos = mask_candidates & (mask_draw < self.mask_prob)
            original_ids = tokens.clone()
            mlm_labels = torch.full((L,), -100, dtype=torch.long)
            mlm_labels[mask_pos] = tokens[mask_pos]
            tokens[mask_pos] = self.mask_token_id

            return Data(
                x=tokens,
                edge_index=torch.tensor(np.array([dist_mat.row, dist_mat.col]), dtype=torch.long),
                edge_attr=torch.tensor(dist_mat.data, dtype=torch.long),
                original_ids=original_ids,
                mlm_labels=mlm_labels,
            )
        except Exception:
            return None

    def __iter__(self):
        import pandas as pd
        shards = self._get_worker_shards()
        if self.shuffle:
            random.shuffle(shards)
        for path in shards:
            # Historical ZINC shards in this project use both `smiles` and
            # `SMILES`. Resolve the physical column case-insensitively before
            # projecting it; pyarrow column projection itself is case-sensitive.
            import pyarrow.parquet as pq

            columns = pq.ParquetFile(path).schema.names
            smiles_col = next(
                (name for name in columns if name.lower() == "smiles"), None
            )
            if smiles_col is None:
                raise ValueError(
                    f"No SMILES column in {path}; available columns: {columns}"
                )
            df = pd.read_parquet(path, columns=[smiles_col])
            smiles_list = df[smiles_col].tolist()
            if self.shuffle:
                random.shuffle(smiles_list)
            for smi in smiles_list:
                data = self._smiles_to_data(smi)
                if data is not None:
                    yield data


class MolPreTrainDataModule(LightningDataModule):
    """
    DataModule for RTD pre-training.
    Supports:
    - Single CSV/Parquet file (legacy)
    - Directory of SMILES Parquet shards
    - Directory of pre-tokenized .pt shards (highest performance)
    """

    def __init__(
        self,
        data,
        vocabulary_inp,
        validation_data=None,
        MASK_token=None,
        UNK_token=None,
        CLS_token=None,
        radius_inp=0,
        useFeatures_inp=False,
        mask_prob=0.15,
        val_fraction=0.01,
        batch_size=64,
        num_workers=4,
        val_num_workers=0,
        max_atoms=128,
        **kwargs,
    ):
        super().__init__()
        self.data = data
        self.validation_data = validation_data
        self.radius_inp = radius_inp
        self.useFeatures_inp = useFeatures_inp
        self.mask_prob = mask_prob
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers
        self.prepare_data_per_node = False
        self.max_atoms = max_atoms

        self.dictionary_inp = open_dictionary(
            vocabulary_inp,
            mask_token=MASK_token,
            unk_token=UNK_token,
            cls_token=CLS_token,
        )

    def _get_shard_paths(self, path, extension=".pt", exclude=None):
        if os.path.isdir(path):
            paths = sorted(glob.glob(os.path.join(path, f"*{extension}")))
            if exclude:
                paths = [p for p in paths if os.path.basename(p) not in exclude]
            return paths
        return [path]

    def _load_smiles(self, path):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.parquet")))
            # Exclude val.parquet to ensure it's used only for validation
            files = [f for f in files if not f.endswith("val.parquet")]
            dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["smiles"])
        else:
            loader = getattr(pd, "read_" + path.split(".")[-1])
            df = loader(path)

        s = df.smiles.astype("string[pyarrow]")
        if self.max_atoms is not None:
            heavy = s.str.replace(r"[^A-Za-z]", "", regex=True).str.len()
            s = s[heavy <= self.max_atoms].reset_index(drop=True)
        return s

    def setup(self, stage):
        if stage == "fit":
            # Detect if input is pre-tokenized
            pt_shards = self._get_shard_paths(self.data, ".pt", exclude={"val_tokenized.pt"})
            if pt_shards:
                logger.info(f"Detected {len(pt_shards)} pre-tokenized shards.")
                self.train_dataset = TokenizedMolPreTrainDataset(
                    shard_paths=pt_shards,
                    mask_token_id=self.dictionary_inp["MASK"],
                    mask_prob=self.mask_prob,
                    shuffle=True,
                )
                
                # Validation: try to find val_tokenized.pt or fallback to loading val.parquet
                val_pt = os.path.join(self.data, "val_tokenized.pt")
                if os.path.exists(val_pt):
                    logger.info("Using pre-tokenized validation set.")
                    # We can use TokenizedMolPreTrainDataset for val too, 
                    # but if it's small, loading fully is fine.
                    # For simplicity, let's use the IterableDataset with shuffle=False.
                    self.val_dataset = TokenizedMolPreTrainDataset(
                        shard_paths=[val_pt],
                        mask_token_id=self.dictionary_inp["MASK"],
                        mask_prob=self.mask_prob,
                        shuffle=False,
                    )
                elif self.validation_data:
                    smiles_val = self._load_smiles(self.validation_data)
                    self.val_dataset = MolPreTrainDataset(
                        smiles_val, self.dictionary_inp,
                        mask_prob=self.mask_prob,
                        radius_inp=self.radius_inp,
                        useFeatures_inp=self.useFeatures_inp,
                    )
                else:
                    logger.warning("No validation data found for tokenized mode.")
                    self.val_dataset = None
            else:
                # Parquet streaming mode — tokenize on-the-fly, no storage overhead
                parquet_shards = self._get_shard_paths(self.data, ".parquet", exclude={"val.parquet"})
                if parquet_shards:
                    logger.info(f"No .pt shards found. Streaming {len(parquet_shards)} parquet shards on-the-fly.")
                    self.train_dataset = StreamingParquetDataset(
                        shard_paths=parquet_shards,
                        dictionary=self.dictionary_inp,
                        mask_token_id=self.dictionary_inp["MASK"],
                        mask_prob=self.mask_prob,
                        radius=self.radius_inp,
                        use_features=self.useFeatures_inp,
                        max_atoms=self.max_atoms,
                        shuffle=True,
                    )
                    val_parquet = os.path.join(self.data, "val.parquet")
                    if os.path.exists(val_parquet):
                        self.val_dataset = StreamingParquetDataset(
                            shard_paths=[val_parquet],
                            dictionary=self.dictionary_inp,
                            mask_token_id=self.dictionary_inp["MASK"],
                            mask_prob=self.mask_prob,
                            radius=self.radius_inp,
                            use_features=self.useFeatures_inp,
                            max_atoms=self.max_atoms,
                            shuffle=False,
                            deterministic_mask_seed=42,
                            replicate_across_ranks=True,
                        )
                    elif self.validation_data:
                        smiles_val = self._load_smiles(self.validation_data)
                        self.val_dataset = MolPreTrainDataset(
                            smiles_val, self.dictionary_inp,
                            mask_prob=self.mask_prob,
                            radius_inp=self.radius_inp,
                            useFeatures_inp=self.useFeatures_inp,
                        )
                    else:
                        self.val_dataset = None
                    return

                # SMILES mode (Legacy)
                smiles = self._load_smiles(self.data)

                if self.validation_data is not None:
                    smiles_val = self._load_smiles(self.validation_data)
                    smiles_train = smiles
                else:
                    n_val = max(1, int(len(smiles) * self.val_fraction))
                    smiles_val = smiles.iloc[:n_val]
                    smiles_train = smiles.iloc[n_val:]

                self.train_dataset = MolPreTrainDataset(
                    smiles_train, self.dictionary_inp,
                    mask_prob=self.mask_prob,
                    radius_inp=self.radius_inp,
                    useFeatures_inp=self.useFeatures_inp,
                )
                self.val_dataset = MolPreTrainDataset(
                    smiles_val, self.dictionary_inp,
                    mask_prob=self.mask_prob,
                    radius_inp=self.radius_inp,
                    useFeatures_inp=self.useFeatures_inp,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=not isinstance(self.train_dataset, IterableDataset),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.val_num_workers > 0,
            prefetch_factor=4 if self.val_num_workers > 0 else None,
        )


class StreamingMLMParquetDataset(IterableDataset):
    """
    On-the-fly MLM dataset: reads SMILES parquet shards, computes radius-0 input
    tokens + radius-2 target tokens per atom, applies masking.
    Avoids pre-tokenization storage (728 GB for ZINC 415M at 1.7 GB/shard).
    Handles DDP + multi-worker shard splitting identically to StreamingParquetDataset.
    """

    def __init__(
        self,
        shard_paths: List[str],
        r0_vocab: dict,
        r2_vocab: dict,
        mask_token_id: int,
        mask_prob: float = 0.15,
        max_atoms: int = 96,
        shuffle: bool = True,
    ):
        super().__init__()
        self.shard_paths = sorted(shard_paths)
        self.r0_vocab = r0_vocab
        self.r2_vocab = r2_vocab
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.max_atoms = max_atoms
        self.shuffle = shuffle

    def _get_worker_shards(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size, rank = 1, 0
        total_workers = num_workers * world_size
        global_worker_id = rank * num_workers + worker_id
        return [s for i, s in enumerate(self.shard_paths) if i % total_workers == global_worker_id]

    def _smiles_to_data(self, smi: str):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumHeavyAtoms() > self.max_atoms:
                return None

            disconnected = [i for i, a in enumerate(mol.GetAtoms()) if a.GetDegree() == 0]
            r0_env, r2_env = {}, {}
            for radius, target in ((0, r0_env), (2, r2_env)):
                info = {}
                AllChem.GetMorganFingerprint(
                    mol, radius, bitInfo=info,
                    includeRedundantEnvironments=True, useFeatures=False,
                )
                for hash_id, hits in info.items():
                    for atom_idx, hit_radius in hits:
                        if hit_radius == radius or atom_idx in disconnected:
                            target[atom_idx] = hash_id

            unk_r0 = self.r0_vocab.get("UNK", 2)
            unk_r2 = self.r2_vocab.get("UNK", 2)
            cls_r0 = self.r0_vocab.get("CLS", self.r0_vocab.get("UNK", 2))
            cls_r2 = self.r2_vocab.get("CLS", 1)

            n = mol.GetNumAtoms()
            r0_toks = [self.r0_vocab.get(r0_env.get(i), unk_r0) for i in range(n)]
            r2_toks = [self.r2_vocab.get(r2_env.get(i), unk_r2) for i in range(n)]

            tokens   = torch.tensor([cls_r0] + r0_toks, dtype=torch.long)
            r2_full  = torch.tensor([cls_r2] + r2_toks, dtype=torch.long)
            L = len(tokens)

            mask_pos = torch.zeros(L, dtype=torch.bool)
            mask_pos[1:] = torch.rand(L - 1) < self.mask_prob  # skip CLS

            r2_mlm_labels = torch.full((L,), -100, dtype=torch.long)
            r2_mlm_labels[mask_pos] = r2_full[mask_pos]
            tokens[mask_pos] = self.mask_token_id

            dist_mat = Chem.GetDistanceMatrix(mol)
            dist_mat[dist_mat == 1e8] = -1
            dist_mat = sparse.coo_matrix(dist_mat + 1)
            # prepend CLS row/col
            dist_mat = sparse.vstack([np.zeros(dist_mat.shape[1])[None, :], dist_mat.toarray()])
            dist_mat = sparse.coo_matrix(
                sparse.hstack([np.zeros(dist_mat.shape[0])[:, None], dist_mat])
            )

            return Data(
                x=tokens,
                r2_mlm_labels=r2_mlm_labels,
                edge_index=torch.tensor(np.array([dist_mat.row, dist_mat.col]), dtype=torch.long),
                edge_attr=torch.tensor(dist_mat.data, dtype=torch.long),
            )
        except Exception:
            return None

    def __iter__(self):
        shards = self._get_worker_shards()
        if self.shuffle:
            random.shuffle(shards)
        for path in shards:
            df = pd.read_parquet(path, columns=["smiles"])
            smiles_list = df["smiles"].tolist()
            if self.shuffle:
                random.shuffle(smiles_list)
            for smi in smiles_list:
                data = self._smiles_to_data(smi)
                if data is not None:
                    yield data


class MolMLMDataModule(LightningDataModule):
    """DataModule for MolE-MLM pre-training (radius-2 target).

    Preferred: pre-tokenized ``_mlm.pt`` shards under ``data/``.
    Fallback: on-the-fly tokenization from ``parquet_data/`` parquets using
    ``r2_vocab_path`` — avoids 728 GB pre-tokenization storage for ZINC 415M.
    """

    def __init__(
        self,
        data: str,
        vocabulary_inp: str,
        mask_prob: float = 0.15,
        batch_size: int = 128,
        num_workers: int = 2,
        parquet_data: Optional[str] = None,
        r2_vocab_path: Optional[str] = None,
        max_atoms: int = 96,
        **kwargs,
    ):
        super().__init__()
        self.data = data
        self.parquet_data = parquet_data
        self.r2_vocab_path = r2_vocab_path
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_atoms = max_atoms
        self.prepare_data_per_node = False
        self.dictionary_inp = open_dictionary(vocabulary_inp)

    def _get_shard_paths(self, path, suffix="_mlm.pt", exclude=None):
        if os.path.isdir(path):
            paths = sorted(glob.glob(os.path.join(path, f"*{suffix}")))
            if exclude:
                paths = [p for p in paths if os.path.basename(p) not in exclude]
            return paths
        return [path]

    def setup(self, stage):
        if stage != "fit":
            return

        # --- Pre-tokenized shards (fast path) ---
        train_shards = self._get_shard_paths(self.data, suffix="_mlm.pt", exclude={"val_mlm.pt"})
        if train_shards:
            logger.info(f"MLM DataModule: {len(train_shards)} pre-tokenized shards from {self.data}")
            self.train_dataset = TokenizedMLMDataset(
                shard_paths=train_shards,
                mask_token_id=self.dictionary_inp["MASK"],
                mask_prob=self.mask_prob,
                shuffle=True,
            )
            val_pt = os.path.join(self.data, "val_mlm.pt")
            self.val_dataset = TokenizedMLMDataset(
                shard_paths=[val_pt], mask_token_id=self.dictionary_inp["MASK"],
                mask_prob=self.mask_prob, shuffle=False,
            ) if os.path.exists(val_pt) else None
            return

        # --- On-the-fly parquet fallback ---
        parquet_dir = self.parquet_data or self.data
        parquet_shards = sorted(glob.glob(os.path.join(parquet_dir, "train_shard_*.parquet")))
        if not parquet_shards:
            raise RuntimeError(
                f"No _mlm.pt shards in {self.data} and no parquet shards in {parquet_dir}. "
                "Set parquet_data= or ensure pre-tokenized shards exist."
            )
        if not self.r2_vocab_path or not os.path.exists(self.r2_vocab_path):
            raise RuntimeError(
                f"r2_vocab_path={self.r2_vocab_path!r} not found. "
                "Run build_mlm_data.py --mode vocab first."
            )
        with open(self.r2_vocab_path, "rb") as f:
            r2_vocab = pickle.load(f)
        logger.info(
            f"MLM DataModule (on-the-fly): {len(parquet_shards)} parquet shards, "
            f"r2_vocab={len(r2_vocab)} tokens"
        )
        dataset_kwargs = dict(
            r0_vocab=self.dictionary_inp,
            r2_vocab=r2_vocab,
            mask_token_id=self.dictionary_inp["MASK"],
            mask_prob=self.mask_prob,
            max_atoms=self.max_atoms,
        )
        self.train_dataset = StreamingMLMParquetDataset(
            shard_paths=parquet_shards, shuffle=True, **dataset_kwargs
        )
        val_parquet = os.path.join(parquet_dir, "val.parquet")
        self.val_dataset = StreamingMLMParquetDataset(
            shard_paths=[val_parquet], shuffle=False, **dataset_kwargs
        ) if os.path.exists(val_parquet) else None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
