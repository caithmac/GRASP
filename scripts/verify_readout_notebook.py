"""Audit all six real data splits and execute tiny real-checkpoint CPU fits.

Dependencies must already be installed. Only installation/model transport and
the disposable smoke-test settings are replaced; the checked notebook sources
execute in order. This does not claim a Colab GPU or full benchmark run.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import nbformat
import pandas as pd
import torch
from rdkit import RDLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    torch.set_num_threads(2)
    RDLogger.DisableLog('rdApp.warning')
    path = root/'notebooks/04_readout_ablation.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert all(not c.get('outputs') and c.get('execution_count') is None for c in notebook.cells)
    helper = (root/'notebooks/readout_helpers.py').read_text(encoding='utf-8')
    assert notebook.metadata.protocol_helper_sha256 == hashlib.sha256(helper.encode()).hexdigest()
    assert args.work_dir.resolve() != args.model_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    scope = {'__name__':'__main__', 'display':lambda value:print(str(value), flush=True)}
    report = {'datasets':{}, 'scope':'Six full public-dataset split audits, four tiny real-checkpoint CPU fits; no full benchmark or Colab GPU execution.'}
    train_source = None
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        source = cell.source
        compile(source, cell.id, 'exec')
        tag = cell.metadata.tags[0]
        if tag == 'install':
            import ast
            tree = ast.parse(source)
            code_revision = next(ast.literal_eval(n.value) for n in tree.body
                                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='CODE_REVISION' for t in n.targets))
            import importlib.metadata, os, subprocess
            scope.update(CODE_REVISION=code_revision, sys=sys, os=os,
                         subprocess=subprocess, importlib=__import__('importlib'))
            continue
        if tag == 'model':
            original = 'MODEL_DIR = Path(snapshot_download(MODEL_SPEC["repo"], revision=MODEL_SPEC["revision"],\n    allow_patterns=["model.safetensors", "config.json", "vocab.json"]))'
            assert original in source
            source = source.replace(original, 'MODEL_DIR = LOCAL_MODEL_DIR')
            scope['LOCAL_MODEL_DIR'] = args.model_dir.resolve()
        if tag == 'train':
            train_source = source
        exec(compile(source, cell.id, 'exec'), scope)
        if tag == 'settings':
            scope.update(RUN_ROOT=args.work_dir.resolve(), REQUIRE_GPU=False, DEVICE='cpu',
                         EPOCHS=1, PATIENCE=1, BATCH_SIZE=4, TRAINING_SEEDS=[0], KEEP_PREDICTORS=False)
        if tag == 'prepare':
            for row in scope['overview']:
                dataset = row['dataset']
                directory = args.work_dir/'splits'/dataset
                parts = {k:pd.read_csv(directory/f'{k}.csv') for k in ('train','valid','test')}
                for col in ('source_row','connectivity','scaffold'):
                    sets = [set(p[col]) for p in parts.values()]
                    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
                report['datasets'][dataset] = row
            # Change only these disposable fit inputs after the complete split audit.
            chosen = 'asap_logd'
            directory = args.work_dir/'splits'/chosen
            manifest = json.loads((directory/'split_manifest.json').read_text())
            for split, count in (('train',8),('valid',4),('test',4)):
                frame = pd.read_csv(directory/f'{split}.csv').head(count)
                frame.to_csv(directory/f'{split}.csv', index=False)
                manifest['sha256'][split] = scope['sha256'](directory/f'{split}.csv')
                manifest['counts'][split] = len(frame)
            manifest['smoke_only_reduced_after_full_audit'] = True
            scope['write_json'](directory/'split_manifest.json', manifest)
            scope['DATASET_NAMES'] = [chosen]
    assert len(scope['results']) == 4 and len(scope['paired']) == 2
    before = json.loads(json.dumps(scope['results']))
    exec(compile(train_source, 'resume-train-cell', 'exec'), scope)
    assert scope['results'] == before
    assert not list((args.work_dir/'runs').rglob('predictor.safetensors'))
    with zipfile.ZipFile(args.work_dir/'grasp_readout_results.zip') as bundle:
        assert not any(n.endswith('.safetensors') or n.startswith('raw/') for n in bundle.namelist())
        assert 'final_vs_mixed.csv' in bundle.namelist() and 'run_manifest.json' in bundle.namelist()
    assert len(report['datasets']) == 6
    report.update(status='passed', matched_head_initialization=True, notebook_workflow_executed=True,
                  installation_cell_not_executed=True, model_transport_replaced_with_local_weights=True,
                  completed_fit_resume=True, compact_export=True,
                  pinned_model_sha256=scope['IDENTITY']['model_files_sha256']['model.safetensors'])
    (args.work_dir/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('READOUT_NOTEBOOK_VERIFIED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
