"""Build the output-free, self-contained Colab from the checked protocol helper."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.code_revision):
        raise ValueError('Use an immutable 40-character GRASP code commit')
    root = Path(__file__).resolve().parents[1]
    helper = (root/'notebooks/readout_helpers.py').read_text(encoding='utf-8')
    helper_hash = hashlib.sha256(helper.encode('utf-8')).hexdigest()
    cells = []

    def md(text):
        cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))

    def code(text, tag):
        cells.append(dict(cell_type='code', metadata={'tags':[tag]}, source=text.splitlines(True),
                          execution_count=None, outputs=[]))

    md('''# GRASP: final-layer versus mixed-layer readouts on six random datasets

This notebook compares **four methods on the same checkpoint and the same saved splits**:

| Encoder adaptation | Final layer 12 | Learned mixture of layers 4/8/10/12 |
|---|---|---|
| Frozen encoder | `frozen_final` | `frozen_mixed` |
| Full encoder fine-tuning | `full_final` | `full_mixed` |

Every method uses the same atom-attention pooling and nonlinear prediction head. Frozen means the encoder stays fixed while pooling/head parameters train; **this is not a strict linear probe**. Final-layer readout does not mean training only the last transformer block. LoRA is outside this four-arm comparison.

Six endpoint datasets were sampled without replacement from the existing **23-endpoint public OpenADMET pool**, using Python `random.Random(20261018).sample(sorted(pool), 6)`: **ASAP SARS-CoV-2 potency, Biogen rat liver microsomal clearance, ASAP LogD, ExpansionRx Caco-2 permeability, ExpansionRx solubility and ExpansionRx LogD**. The complete pool and selection seed are exported.

Select **Runtime → Change runtime type → GPU**, then run cells in order. The default is the released 93.5M-parameter GRASP checkpoint, one training seed and 24 fits (6 datasets × 4 arms). A1 is selectable. Increase seeds before examining test results for a larger experiment.
''')
    md('''## 1. Install

Use a fresh runtime and run this cell before importing scientific packages. The existing PyTorch/CUDA installation is retained. If an already imported package changes, restart the session and rerun from here.
''')
    code('''import os, sys, subprocess, importlib.metadata
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
CODE_REVISION = "REVISION"
if not (3, 10) <= sys.version_info[:2] < (3, 14):
    raise RuntimeError("Use Python 3.10–3.13 for this pinned GRASP release")
watched = {"numpy":"numpy", "pandas":"pandas", "scipy":"scipy", "rdkit":"rdkit", "sklearn":"scikit-learn"}
loaded = {name:importlib.metadata.version(package) for name,package in watched.items() if name in sys.modules}
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    f"git+https://github.com/caithmac/GRASP.git@{CODE_REVISION}",
    "rdkit==2025.3.6", "scipy==1.15.3", "scikit-learn==1.7.2"], check=True)
changed = [name for name,version in loaded.items() if importlib.metadata.version(watched[name]) != version]
if changed:
    raise RuntimeError(f"Restart the session: already imported packages changed: {changed}")
print("Installation ready")
'''.replace('"REVISION"', json.dumps(args.code_revision)), 'install')
    md('''## 2. Shared protocol

The helper below is embedded so this notebook is self-contained. It verifies public source checksums, audits exclusions before splitting, saves scaffold-disjoint partitions, checks identical head initialization and resumes completed experiment cells only when their configurations match.
''')
    code('PROTOCOL_HELPER_SHA256 = '+repr(helper_hash)+'\n'+helper, 'protocol')
    md('''## 3. Settings — choose before viewing test results

`GRASP` and `A1` are immutable public Step2 checkpoints. This is a within-checkpoint readout experiment; changing checkpoints makes a separate experiment. These released checkpoints also differ in training history, so cross-checkpoint results do not isolate absolute positions.

Use `USE_DRIVE=True` to preserve results across Colab disconnects. Completed fits resume; an interrupted individual fit restarts. Without Drive, files disappear when the runtime is discarded. Keep the same `RUN_NAME` only for identical settings. `KEEP_PREDICTORS=False` saves scores and predictions but discards each completed fit's approximately 0.37 GB weight file.

Learning rates are fixed separately for frozen/full adaptation and shared between final/mixed readouts. There is no test-driven method selection or exhaustive hyperparameter search. Epoch and seed defaults are a screening budget, not paper-score reproduction.
''')
    code('''from pathlib import Path
MODEL_KEY = "GRASP"                 # "GRASP" or "A1"
DATASET_SELECTION_SEED = 20261018
SPLIT_SEED = 20261018
TRAINING_SEEDS = [0]                 # use [0, 1, 2] for repeats on the same split
EPOCHS, PATIENCE, BATCH_SIZE = 20, 5, 8
LEARNING_RATES = {"frozen":1e-3, "full":1e-5}
USE_DRIVE = False
KEEP_PREDICTORS = False
REQUIRE_GPU = True
RUN_NAME = "grasp_readout_six_v1"
MODELS = {
    "GRASP":{"repo":"caithmac/GRASP", "revision":"4804c0a558307c944da5e8de1d182b4b4c994dae"},
    "A1":{"repo":"caithmac/GRASP-A1", "revision":"d8f631e096f1d2dfd281962a21389142cde437ee"},
}
DATASET_NAMES = selected_datasets(DATASET_SELECTION_SEED)
assert len(DATASET_NAMES) == len(set(DATASET_NAMES)) == 6
assert TRAINING_SEEDS and len(TRAINING_SEEDS) == len(set(TRAINING_SEEDS))
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
RUN_ROOT = (Path("/content/drive/MyDrive/GRASP") if USE_DRIVE else Path("/content"))/RUN_NAME
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Selected endpoint datasets:")
display(pd.DataFrame([{"dataset":name, "family":DATASETS[name]["family"], "endpoint":DATASETS[name]["name"]} for name in DATASET_NAMES]))
print("Planned fits:", len(DATASET_NAMES)*len(TRAINING_SEEDS)*len(ARMS))
''', 'settings')
    md('''## 4. Load the pinned checkpoint and lock this run

The model download is about 0.37 GB. Both checkpoint repositories are public; no token is embedded or required. Encoder construction is isolated from the task-head RNG stream. All four arms start with identical pooling/head weights for each training seed.
''')
    code('''from huggingface_hub import snapshot_download
from grasp.encoder import read_vocab
if REQUIRE_GPU and DEVICE != "cuda":
    raise RuntimeError("Select a GPU runtime, or deliberately set REQUIRE_GPU=False for a slow CPU run")
MODEL_SPEC = MODELS[MODEL_KEY]
MODEL_DIR = Path(snapshot_download(MODEL_SPEC["repo"], revision=MODEL_SPEC["revision"],
    allow_patterns=["model.safetensors", "config.json", "vocab.json"]))
VOCABULARY = read_vocab(MODEL_DIR/"vocab.json")
IDENTITY = {"code_revision":CODE_REVISION, "protocol_helper_sha256":PROTOCOL_HELPER_SHA256,
            "checkpoint":MODEL_SPEC, "model_files_sha256":{name:sha256(MODEL_DIR/name)
            for name in ("model.safetensors", "config.json", "vocab.json")}}
run_plan = {"protocol":PROTOCOL_VERSION, "identity":IDENTITY, "candidate_pool":sorted(DATASETS),
            "selected_datasets":DATASET_NAMES, "dataset_selection_seed":DATASET_SELECTION_SEED,
            "split_seed":SPLIT_SEED, "training_seeds":TRAINING_SEEDS, "arms":ARMS,
            "epochs":EPOCHS, "patience":PATIENCE, "batch_size":BATCH_SIZE, "learning_rates":LEARNING_RATES}
RUN_ROOT.mkdir(parents=True, exist_ok=True)
if (RUN_ROOT/"run_manifest.json").exists():
    if json.loads((RUN_ROOT/"run_manifest.json").read_text()) != run_plan:
        raise ValueError("Run settings changed: choose a new RUN_NAME")
write_json(RUN_ROOT/"run_manifest.json", run_plan)
write_json(RUN_ROOT/"environment.json", {"python":sys.version, "device":DEVICE,
    "gpu":torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    "versions":{package:importlib.metadata.version(package) for package in
    ("torch", "torch-geometric", "rdkit", "numpy", "pandas", "scipy", "scikit-learn", "safetensors", "huggingface-hub")}})
enable_matched_initialization()
print("Checkpoint and protocol locked:", RUN_ROOT)
''', 'model')
    md('''## 5. Prepare one shared split per dataset

All four methods receive identical **new scaffold-grouped train/validation/test splits**, targeting 80/10/10 proportions. These are not the existing paper or challenge test memberships. Duplicates and stereoisomers with the same connectivity stay together; all acyclic molecules form one group. No salts, charges or tautomers are repaired. The exclusion log and target transformations are saved for every dataset. No training fit receives test labels.
''')
    code('''overview = []
for dataset in DATASET_NAMES:
    directory = RUN_ROOT/"splits"/dataset
    frame = prepare_task(dataset, RUN_ROOT/"raw", directory, VOCABULARY)
    manifest = save_splits(frame, "regression", directory, SPLIT_SEED)
    overview.append({"dataset":dataset, "eligible_rows":len(frame), **manifest["counts"]})
display(pd.DataFrame(overview))
''', 'prepare')
    md('''## 6. Verify the comparison before training

The check below compares the actual pooling/head tensors across all four arms. Frozen encoders must have zero trainable parameters; full fine-tuning must update encoder parameters. The final-only arm has a fixed single-layer weight of one, while the mixed arm learns depth weights.
''')
    code('''head_checks = {}
for seed in TRAINING_SEEDS:
    checks = {arm:readout_fingerprint(MODEL_DIR, seed, arm) for arm in ARMS}
    assert len({x["head_and_pool_sha256"] for x in checks.values()}) == 1
    assert len({x["post_initialization_rng_sha256"] for x in checks.values()}) == 1
    head_checks[str(seed)] = checks
write_json(RUN_ROOT/"initialization_checks.json", head_checks)
display(pd.DataFrame(head_checks[str(TRAINING_SEEDS[0])]).T)
''', 'heads')
    md('''## 7. Train and evaluate

Each fit selects its checkpoint on **validation MAE**. Only afterward does it predict the held-out test set. Scores, predictions and completion markers are saved after every fit; rerunning this cell skips matching completed cells. If a GPU runs out of memory, choose a smaller batch size and a new `RUN_NAME` so all four arms use the same changed setting.
''')
    code('''results = []
for dataset in DATASET_NAMES:
    for seed in TRAINING_SEEDS:
        for arm, settings in ARMS.items():
            print(f"\\n{dataset} | seed {seed} | {arm}", flush=True)
            result = train_cell(MODEL_DIR, RUN_ROOT/"splits"/dataset,
                RUN_ROOT/"runs"/dataset/f"seed_{seed}"/arm,
                arm=arm, seed=seed, epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATES[settings["method"]], device=DEVICE,
                identity=IDENTITY, keep_predictors=KEEP_PREDICTORS)
            results.append({"dataset":dataset, **{k:v for k,v in result.items() if k != "metrics"}, **result["metrics"]})
            pd.DataFrame(results).to_csv(RUN_ROOT/"results.csv", index=False)
paired, summary = comparison_tables(results)
paired.to_csv(RUN_ROOT/"final_vs_mixed.csv", index=False)
summary.to_csv(RUN_ROOT/"summary.csv", index=False)
display(summary.pivot(index="dataset", columns="arm", values="mean"))
display(paired)
''', 'train')
    md('''## 8. Export results

MAE is lower-is-better. In `final_vs_mixed.csv`, a negative `final_minus_mixed` favors the final layer. Compare frozen versus full within the same readout in `summary.csv`. Do not average raw errors from different property scales into a single performance claim. Multiple seeds share the same test molecules; these are not independent test cohorts. One training seed is an exploratory result.

The ZIP contains results, predictions, saved splits, source/selection manifests and initialization checks. Raw source downloads and trained weight files are excluded. Downloading the ZIP does not preserve any trained predictors you elected to keep; copy those separately.
''')
    code('''import zipfile
archive = RUN_ROOT/"grasp_readout_results.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(RUN_ROOT.rglob("*")):
        if path.is_file() and path.suffix in (".csv", ".json") and "raw" not in path.relative_to(RUN_ROOT).parts:
            bundle.write(path, path.relative_to(RUN_ROOT).as_posix())
print("Results archive:", archive)
try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(str(archive))
''', 'export')
    md('''## Sources and scope

Pinned public sources are listed with URLs and SHA256 hashes in `SOURCES` above and in each exported data manifest: [OpenADMET ExpansionRx](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data), [ASAP/Polaris OpenADMET](https://huggingface.co/datasets/openadmet/ASAP_Polaris_OpenADMET_challenge), [Biogen Computational ADME](https://github.com/molecularinformatics/Computational-ADME) and [PXR](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test). Dataset terms apply to their respective sources. GRASP code/weights are CC BY-NC 4.0, with separate MIT attribution for DeBERTa; see the repository/model cards.

This is a downstream readout comparison using new disclosed splits, not reproduction of published scores, proof of no pretraining overlap, or a test of completed A2/A3 checkpoints. No EIDF resources are used. Test results must not drive retuning on the same held-out cohort.
''')
    for index, cell in enumerate(cells):
        cell['id'] = f'readout-{index:02d}'
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['id'], 'exec')
    notebook = dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(provenance=[], toc_visible=True),
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python'), protocol_helper_sha256=helper_hash), nbformat=4, nbformat_minor=5)
    path = root/'notebooks/04_readout_ablation.ipynb'
    path.write_text(json.dumps(notebook, indent=1)+'\n', encoding='utf-8')
    print('BUILT', path, len(cells), 'cells')


if __name__ == '__main__':
    main()
