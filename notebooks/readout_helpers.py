"""Shared protocol for the six-dataset final-layer versus mixed-layer Colab."""
import gc
import hashlib
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

PROTOCOL_VERSION = "grasp-readout-2x2-colab-20261018-v1"
ARMS = {
    "frozen_final": {"method": "frozen", "mix_layers": [12]},
    "frozen_mixed": {"method": "frozen", "mix_layers": [4, 8, 10, 12]},
    "full_final": {"method": "full", "mix_layers": [12]},
    "full_mixed": {"method": "full", "mix_layers": [4, 8, 10, 12]},
}

SOURCES = {'asap_admet': {'filename': 'ASAP_ADMET.csv',
                'revision': '060aee45a3cf2ab030b325e16c301a06d2d19d71',
                'sha256': '3388480e9a312c17c9b621352f9414b0456ac19fc44604570e82a112e2200603',
                'url': 'https://huggingface.co/datasets/openadmet/ASAP_Polaris_OpenADMET_challenge/resolve/060aee45a3cf2ab030b325e16c301a06d2d19d71/ADMET.csv'},
 'asap_potency': {'filename': 'ASAP_Potency.csv',
                  'revision': '060aee45a3cf2ab030b325e16c301a06d2d19d71',
                  'sha256': '2c7e877684e01ba079552aff302e74cda89023e80c966c995748336d3cf1e5ed',
                  'url': 'https://huggingface.co/datasets/openadmet/ASAP_Polaris_OpenADMET_challenge/resolve/060aee45a3cf2ab030b325e16c301a06d2d19d71/Potency.csv'},
 'biogen': {'filename': 'ADME_public_set_3521.csv',
            'revision': 'b00df003de117ce9e5b381afd886095c5f2af2d5',
            'sha256': '2cfabc2667740c224487876c33b23124159ef43294e0f9e4d926cb6276c95a3b',
            'url': 'https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/b00df003de117ce9e5b381afd886095c5f2af2d5/ADME_public_set_3521.csv'},
 'expansion': {'filename': 'expansion_data_train.csv',
               'revision': '6b898ccc43d10d25b230fb09e22a6e30c30022b5',
               'sha256': 'c5214ad8c8a4d7d4d09082fcce24fe16c97a980a1e0873b91b8c8e474b79f6e4',
               'url': 'https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data/resolve/6b898ccc43d10d25b230fb09e22a6e30c30022b5/expansion_data_train.csv'},
 'pxr': {'filename': 'PXR_train.csv',
         'revision': 'c320ab14df0ed66a485b01839e9db8874f81509b',
         'sha256': 'efa9096110c9123386141cb505507d10f3de845c764d07e9c7375f644c82757c',
         'url': 'https://huggingface.co/datasets/openadmet/pxr-challenge-train-test/resolve/c320ab14df0ed66a485b01839e9db8874f81509b/pxr-challenge_TRAIN.csv'}}

DATASETS = {'asap_hlm': {'family': 'ASAP',
              'name': 'HLM',
              'smiles_column': 'CXSMILES',
              'source': 'asap_admet',
              'source_filter_column': 'Set',
              'source_filter_value': 'Train',
              'target_column': 'HLM',
              'transform': 'positive_log10'},
 'asap_ksol': {'family': 'ASAP',
               'name': 'KSOL',
               'smiles_column': 'CXSMILES',
               'source': 'asap_admet',
               'source_filter_column': 'Set',
               'source_filter_value': 'Train',
               'target_column': 'KSOL',
               'transform': 'positive_log10'},
 'asap_logd': {'family': 'ASAP',
               'name': 'LogD',
               'smiles_column': 'CXSMILES',
               'source': 'asap_admet',
               'source_filter_column': 'Set',
               'source_filter_value': 'Train',
               'target_column': 'LogD',
               'transform': 'identity'},
 'asap_mdr1': {'family': 'ASAP',
               'name': 'MDR1 Efflux',
               'smiles_column': 'CXSMILES',
               'source': 'asap_admet',
               'source_filter_column': 'Set',
               'source_filter_value': 'Train',
               'target_column': 'MDR1-MDCKII',
               'transform': 'positive_log10'},
 'asap_mers': {'family': 'ASAP',
               'name': 'MERS-CoV-2 Potency',
               'smiles_column': 'CXSMILES',
               'source': 'asap_potency',
               'source_filter_column': 'Set',
               'source_filter_value': 'Train',
               'target_column': 'pIC50 (MERS-CoV Mpro)',
               'transform': 'identity'},
 'asap_mlm': {'family': 'ASAP',
              'name': 'MLM',
              'smiles_column': 'CXSMILES',
              'source': 'asap_admet',
              'source_filter_column': 'Set',
              'source_filter_value': 'Train',
              'target_column': 'MLM',
              'transform': 'positive_log10'},
 'asap_sars': {'family': 'ASAP',
               'name': 'SARS-CoV-2 Potency',
               'smiles_column': 'CXSMILES',
               'source': 'asap_potency',
               'source_filter_column': 'Set',
               'source_filter_value': 'Train',
               'target_column': 'pIC50 (SARS-CoV-2 Mpro)',
               'transform': 'identity'},
 'biogen_hlm': {'family': 'Biogen ADME',
                'name': 'HLM CLint',
                'smiles_column': 'SMILES',
                'source': 'biogen',
                'source_filter_column': None,
                'source_filter_value': None,
                'target_column': 'LOG HLM_CLint (mL/min/kg)',
                'transform': 'identity'},
 'biogen_hppb': {'family': 'Biogen ADME',
                 'name': 'HPPB',
                 'smiles_column': 'SMILES',
                 'source': 'biogen',
                 'source_filter_column': None,
                 'source_filter_value': None,
                 'target_column': 'LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)',
                 'transform': 'identity'},
 'biogen_mdr1': {'family': 'Biogen ADME',
                 'name': 'MDR1 Efflux',
                 'smiles_column': 'SMILES',
                 'source': 'biogen',
                 'source_filter_column': None,
                 'source_filter_value': None,
                 'target_column': 'LOG MDR1-MDCK ER (B-A/A-B)',
                 'transform': 'identity'},
 'biogen_rlm': {'family': 'Biogen ADME',
                'name': 'RLM CLint',
                'smiles_column': 'SMILES',
                'source': 'biogen',
                'source_filter_column': None,
                'source_filter_value': None,
                'target_column': 'LOG RLM_CLint (mL/min/kg)',
                'transform': 'identity'},
 'biogen_rppb': {'family': 'Biogen ADME',
                 'name': 'RPPB',
                 'smiles_column': 'SMILES',
                 'source': 'biogen',
                 'source_filter_column': None,
                 'source_filter_value': None,
                 'target_column': 'LOG PLASMA PROTEIN BINDING (RAT) (% unbound)',
                 'transform': 'identity'},
 'biogen_solubility': {'family': 'Biogen ADME',
                       'name': 'Solubility',
                       'smiles_column': 'SMILES',
                       'source': 'biogen',
                       'source_filter_column': None,
                       'source_filter_value': None,
                       'target_column': 'LOG SOLUBILITY PH 6.8 (ug/mL)',
                       'transform': 'identity'},
 'expansion_caco2_efflux': {'family': 'ExpansionRx',
                            'name': 'Caco-2 Efflux',
                            'smiles_column': 'SMILES',
                            'source': 'expansion',
                            'source_filter_column': None,
                            'source_filter_value': None,
                            'target_column': 'Caco-2 Permeability Efflux',
                            'transform': 'positive_log10'},
 'expansion_caco2_pappa': {'family': 'ExpansionRx',
                           'name': 'Caco-2 Permeability',
                           'smiles_column': 'SMILES',
                           'source': 'expansion',
                           'source_filter_column': None,
                           'source_filter_value': None,
                           'target_column': 'Caco-2 Permeability Papp A>B',
                           'transform': 'positive_log10'},
 'expansion_hlm': {'family': 'ExpansionRx',
                   'name': 'HLM CLint',
                   'smiles_column': 'SMILES',
                   'source': 'expansion',
                   'source_filter_column': None,
                   'source_filter_value': None,
                   'target_column': 'HLM CLint',
                   'transform': 'positive_log10'},
 'expansion_ksol': {'family': 'ExpansionRx',
                    'name': 'KSOL',
                    'smiles_column': 'SMILES',
                    'source': 'expansion',
                    'source_filter_column': None,
                    'source_filter_value': None,
                    'target_column': 'KSOL',
                    'transform': 'positive_log10'},
 'expansion_logd': {'family': 'ExpansionRx',
                    'name': 'LogD',
                    'smiles_column': 'SMILES',
                    'source': 'expansion',
                    'source_filter_column': None,
                    'source_filter_value': None,
                    'target_column': 'LogD',
                    'transform': 'identity'},
 'expansion_mbpb': {'family': 'ExpansionRx',
                    'name': 'MBPB',
                    'smiles_column': 'SMILES',
                    'source': 'expansion',
                    'source_filter_column': None,
                    'source_filter_value': None,
                    'target_column': 'MBPB',
                    'transform': 'positive_log10'},
 'expansion_mgmb': {'family': 'ExpansionRx',
                    'name': 'MGMB',
                    'smiles_column': 'SMILES',
                    'source': 'expansion',
                    'source_filter_column': None,
                    'source_filter_value': None,
                    'target_column': 'MGMB',
                    'transform': 'positive_log10'},
 'expansion_mlm': {'family': 'ExpansionRx',
                   'name': 'MLM CLint',
                   'smiles_column': 'SMILES',
                   'source': 'expansion',
                   'source_filter_column': None,
                   'source_filter_value': None,
                   'target_column': 'MLM CLint',
                   'transform': 'positive_log10'},
 'expansion_mppb': {'family': 'ExpansionRx',
                    'name': 'MPPB',
                    'smiles_column': 'SMILES',
                    'source': 'expansion',
                    'source_filter_column': None,
                    'source_filter_value': None,
                    'target_column': 'MPPB',
                    'transform': 'positive_log10'},
 'pxr': {'family': 'PXR',
         'name': 'PXR Activity',
         'smiles_column': 'SMILES',
         'source': 'pxr',
         'source_filter_column': None,
         'source_filter_value': None,
         'target_column': 'pEC50',
         'transform': 'identity'}}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding='utf-8')
    temporary.replace(path)

def scaffold_split(frame, seed):
    groups = [group.index.to_list() for _, group in frame.groupby("scaffold", sort=True)]
    np.random.default_rng(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    partitions = {"train": [], "valid": [], "test": []}
    for group in groups:
        if len(partitions["train"]) + len(group) <= 0.8 * len(frame):
            partitions["train"].extend(group)
        elif len(partitions["valid"]) + len(group) <= 0.1 * len(frame):
            partitions["valid"].extend(group)
        else:
            partitions["test"].extend(group)
    result = {key: frame.loc[sorted(rows)].reset_index(drop=True) for key, rows in partitions.items()}
    assert sum(map(len, result.values())) == len(frame)
    assert all(len(part) for part in result.values()), "Too few scaffold groups to create three nonempty splits."
    for column in ("source_row", "connectivity", "scaffold"):
        sets = [set(part[column]) for part in result.values()]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), f"Split leakage: {column}"
    return result

def save_splits(frame, task, directory, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parts = scaffold_split(frame, seed)
    if task == "binary":
        assert all(set(part.target) == {0, 1} for part in parts.values()), "A binary partition lacks one class; do not silently resplit."
    csv_bytes = {split: part.to_csv(index=False).encode("utf-8") for split, part in parts.items()}
    for split, data in csv_bytes.items():
        path = directory / f"{split}.csv"
        assert not path.is_file() or path.read_bytes() == data, "Existing splits differ: choose a new RUN_ROOT."
    for split, data in csv_bytes.items():
        (directory / f"{split}.csv").write_bytes(data)
    manifest = dict(split_seed=seed, policy="Shared scaffold-grouped 80/10/10 target proportions; all acyclic molecules form one group.",
                    counts={key: len(part) for key, part in parts.items()},
                    sha256={key: sha256(directory / f"{key}.csv") for key in parts},
                    disjoint_source_rows=True, disjoint_connectivity=True, disjoint_scaffolds=True)
    write_json(directory / "split_manifest.json", manifest)
    return manifest

def enable_matched_initialization():
    import grasp.encoder as encoder_module
    import grasp.predictor as predictor_module
    from mole.training.models.mole import AtomEnvEmbeddings
    class MatchedEmbeddings(AtomEnvEmbeddings):
        def __init__(self, config):
            # Loading a different encoder architecture must not shift task-head or loader RNG.
            with torch.random.fork_rng(devices=[]):
                super().__init__(config)
    encoder_module.AtomEnvEmbeddings = predictor_module.AtomEnvEmbeddings = MatchedEmbeddings

def score_predictions(target, prediction, task):
    target, prediction = np.asarray(target), np.asarray(prediction)
    assert target.shape == prediction.shape and np.isfinite(prediction).all()
    if task == "regression":
        return dict(mae=float(mean_absolute_error(target, prediction)), rmse=float(np.sqrt(mean_squared_error(target, prediction))),
                    r2=float(r2_score(target, prediction)))
    assert set(target) == {0, 1} and ((prediction >= 0) & (prediction <= 1)).all()
    return dict(auroc=float(roc_auc_score(target, prediction)), auprc=float(average_precision_score(target, prediction)))


def selected_datasets(seed=20261018):
    return random.Random(seed).sample(sorted(DATASETS), 6)


def prepare_task(name, cache_dir, output_dir, vocabulary):
    from mole.training.data.datasets import MolDataset
    spec = DATASETS[name]
    source = SOURCES[spec['source']]
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = cache_dir/source['filename']
    if not raw.exists():
        response = requests.get(source['url'], timeout=(15, 120))
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != source['sha256']:
            raise ValueError(f'Source checksum mismatch for {name}')
        temporary = raw.with_suffix(raw.suffix+'.tmp')
        temporary.write_bytes(response.content)
        temporary.replace(raw)
    if sha256(raw) != source['sha256']:
        raise ValueError(f'Cached source checksum mismatch: {raw}')
    frame = pd.read_csv(raw)
    required = [spec['smiles_column'], spec['target_column']]
    if spec['source_filter_column']:
        required.append(spec['source_filter_column'])
    if set(required).difference(frame.columns):
        raise ValueError(f'Source schema changed for {name}')
    accepted, audit = [], []
    for source_row, row in frame.iterrows():
        reason = ''
        smiles = str(row[spec['smiles_column']])
        value = pd.to_numeric(row[spec['target_column']], errors='coerce')
        if spec['source_filter_column'] and str(row[spec['source_filter_column']]) != spec['source_filter_value']:
            reason = 'outside_declared_source_training_subset'
        elif not np.isfinite(value):
            reason = 'missing_or_nonnumeric_target'
        elif spec['transform'] == 'positive_log10' and value <= 0:
            reason = 'nonpositive_target_before_log10'
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or not mol.GetNumAtoms():
                reason = 'unparsable_or_empty_smiles'
            elif mol.GetNumAtoms() > 511:
                reason = 'above_encoder_atom_limit'
            else:
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                try:
                    MolDataset(pd.Series([canonical]), vocabulary, cls_token=True)[0]
                except Exception as exc:
                    raise RuntimeError(f'Unexpected featurizer failure at {name} source row {source_row}') from exc
                target = float(np.log10(value+.001)) if spec['transform'] == 'positive_log10' else float(value)
                accepted.append(dict(source_row=int(source_row), smiles=canonical, target=target,
                                     connectivity=Chem.MolToSmiles(mol, isomericSmiles=False),
                                     scaffold=Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol),
                                                              isomericSmiles=False) or '<acyclic>'))
        audit.append(dict(source_row=int(source_row), original_smiles=smiles, excluded=bool(reason), reason=reason))
    result = pd.DataFrame(accepted)
    if len(result) < 30:
        raise ValueError(f'Too few eligible molecules for {name}')
    pd.DataFrame(audit).to_csv(output_dir/'data_audit.csv', index=False)
    result.to_csv(output_dir/'eligible_rows.csv', index=False)
    write_json(output_dir/'data_manifest.json', dict(dataset=name, spec=spec, source=source,
               source_rows=len(frame), eligible_rows=len(result), excluded_rows=len(frame)-len(result),
               policy='Eligibility before new scaffold split; no salt, charge or tautomer repair. Log targets use log10(x+0.001) after excluding nonpositive values.'))
    return result


def readout_fingerprint(model_dir, seed, arm):
    from grasp import GRASPEncoder, GRASPPredictor
    torch.manual_seed(seed)
    encoder = GRASPEncoder.from_pretrained(model_dir, device='cpu')
    predictor = GRASPPredictor.from_encoder(encoder, task='regression', **ARMS[arm])
    digest = hashlib.sha256()
    for key, value in sorted(predictor.model.state_dict().items()):
        if key.startswith(('head.', 'pooler.')):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    encoder_trainable = sum(p.numel() for p in predictor.model.encoder.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    result = dict(head_and_pool_sha256=digest.hexdigest(), encoder_trainable=encoder_trainable,
                  total_trainable=total_trainable,
                  post_initialization_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    if ARMS[arm]['method'] == 'frozen':
        assert encoder_trainable == 0
    else:
        assert encoder_trainable > 0
    del predictor, encoder
    gc.collect()
    return result


def validate_cached_cell(directory, plan):
    directory = Path(directory)
    plan_file = directory/'run_plan.json'
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError('Settings, readout, model or splits changed: choose a new RUN_ROOT.')
    complete = directory/'FIT_COMPLETE.json'
    if not complete.exists():
        return None
    if not plan_file.exists():
        raise ValueError('Completed cell has no run plan.')
    result = json.loads(complete.read_text())
    if sha256(directory/'test_predictions.csv') != result['prediction_sha256']:
        raise ValueError('Saved test predictions changed; refusing to reuse cached metrics.')
    return result


def train_cell(model_dir, split_dir, directory, *, arm, seed, epochs, patience, batch_size,
               learning_rate, device, identity, keep_predictors=False):
    from grasp import GRASPPredictor
    from grasp.train import fit
    split_dir, directory = Path(split_dir), Path(directory)
    hashes = {k:sha256(split_dir/f'{k}.csv') for k in ('train', 'valid', 'test')}
    if hashes != json.loads((split_dir/'split_manifest.json').read_text())['sha256']:
        raise ValueError('Shared split files changed.')
    precision = 'bf16' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    plan = dict(protocol=PROTOCOL_VERSION, arm=arm, adaptation=ARMS[arm], seed=seed,
                epochs=epochs, patience=patience, batch_size=batch_size, learning_rate=learning_rate,
                device=device, precision=precision, identity=identity, split_sha256=hashes)
    cached = validate_cached_cell(directory, plan)
    if cached is not None:
        if keep_predictors and not (directory/'predictor.safetensors').exists():
            raise ValueError('This cell was saved without weights; use a new RUN_ROOT to retain predictors.')
        print(f'SKIP completed {directory.name}', flush=True)
        return cached
    write_json(directory/'run_plan.json', plan)
    started = time.monotonic()
    outcome = fit(model_source=model_dir, train_csv=split_dir/'train.csv', valid_csv=split_dir/'valid.csv',
                  output_dir=directory, task='regression', seed=seed, epochs=epochs, patience=patience,
                  batch_size=batch_size, learning_rate=learning_rate, device=device, **ARMS[arm])
    predictor = GRASPPredictor.from_pretrained(directory, device=device)
    if predictor.config['mix_layers'] != ARMS[arm]['mix_layers']:
        raise RuntimeError('Saved predictor lost its selected readout.')
    test = pd.read_csv(split_dir/'test.csv')
    prediction = predictor.predict(test.smiles.tolist(), batch_size=batch_size)
    metrics = score_predictions(test.target.to_numpy(), prediction, 'regression')
    if not all(np.isfinite(v) for v in metrics.values()):
        raise FloatingPointError('Non-finite test metric')
    exported = test[['source_row', 'smiles', 'target']].copy()
    exported['prediction'] = prediction
    temporary = directory/'test_predictions.csv.tmp'
    exported.to_csv(temporary, index=False)
    temporary.replace(directory/'test_predictions.csv')
    result = dict(arm=arm, seed=seed, test_rows=len(test), seconds=time.monotonic()-started,
                  best_epoch=outcome['best_epoch'], validation_mae=outcome['validation_score'],
                  encoder_trainable=sum(p.numel() for p in predictor.model.encoder.parameters() if p.requires_grad),
                  total_trainable=sum(p.numel() for p in predictor.model.parameters() if p.requires_grad),
                  precision=predictor.config['precision'], metrics=metrics,
                  prediction_sha256=sha256(directory/'test_predictions.csv'))
    write_json(directory/'FIT_COMPLETE.json', result)
    del predictor
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    if not keep_predictors:
        # Only this completed cell's freshly generated weight file is discarded.
        (directory/'predictor.safetensors').unlink()
    return result


def comparison_tables(results):
    rows = pd.DataFrame(results)
    if rows.duplicated(['dataset', 'seed', 'arm']).any():
        raise ValueError('Duplicate experiment cells')
    expected = set(ARMS)
    pairs = []
    for (dataset, seed), group in rows.groupby(['dataset', 'seed']):
        if set(group.arm) != expected:
            raise ValueError(f'Missing arm for {dataset} seed {seed}')
        values = group.set_index('arm').mae
        for method in ('frozen', 'full'):
            final, mixed = float(values[f'{method}_final']), float(values[f'{method}_mixed'])
            pairs.append(dict(dataset=dataset, seed=int(seed), adaptation=method,
                              final_mae=final, mixed_mae=mixed, final_minus_mixed=final-mixed))
    summary = rows.groupby(['dataset', 'arm']).mae.agg(['mean', 'std', 'count']).reset_index()
    return pd.DataFrame(pairs), summary
