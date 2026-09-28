"""Write the two minimal public example notebooks without saved outputs."""
import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


def write(name, cells):
    for index, cell in enumerate(cells):
        cell["id"] = f"{name.split('.')[0]}-{index}"
    payload = {"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    Path("notebooks", name).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main():
    setup = """import os
import subprocess
import sys
from pathlib import Path

try:
    import google.colab
except ImportError:
    pass
else:
    repo = Path('/content/GRASP')
    if not repo.is_dir():
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/caithmac/GRASP.git', str(repo)], check=True)
    else:
        subprocess.run(['git', '-C', str(repo), 'pull', '--ff-only'], check=True)
    os.chdir(repo)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=True)
"""
    write("01_embeddings.ipynb", [
        markdown("# GRASP molecular embeddings\n\nIn Colab, run all cells in order; the first code cell installs GRASP. Locally, run from the repository root after `pip install -e .`. The pretrained encoder returns representations, not property predictions. This notebook uses local release weights when present; otherwise it downloads `caithmac/GRASP`.\n"),
        code(setup),
        code("from pathlib import Path\nfrom grasp import GRASPEncoder\n\nmodel_source = 'hf_model' if Path('hf_model/model.safetensors').is_file() else 'caithmac/GRASP'\nencoder = GRASPEncoder.from_pretrained(model_source)\nsmiles = ['CCO', 'CC(=O)O', 'c1ccccc1']\nembeddings = encoder.encode(smiles)\nprint(embeddings.shape)  # (3, 768)\n"),
        code("import numpy as np\n\nunit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)\nprint(np.round(unit @ unit.T, 3))  # illustrative cosine similarities\n"),
    ])
    write("02_finetune.ipynb", [
        markdown("# Fine-tune, save, reload, predict\n\nIn Colab, select a GPU under **Runtime → Change runtime type** if available, then run all cells in order; the first code cell installs GRASP. Locally, run from the repository root after `pip install -e .`. This *illustrative* dataset uses RDKit-computed logP for 30 simple SMILES. It is not an experimental benchmark and does not reproduce paper scores. For real work, replace the CSVs with independent training and structure-separated validation data. Download the saved predictor directory before your Colab runtime ends if you want to keep it.\n"),
        code(setup),
        code("from pathlib import Path\nimport torch\nfrom scripts.make_demo_data import main as make_demo_data\nfrom grasp.train import fit\nfrom grasp import GRASPPredictor\n\nmake_demo_data()\nmodel_source = 'hf_model' if Path('hf_model/model.safetensors').is_file() else 'caithmac/GRASP'\n"),
        code("result = fit(model_source=model_source,\n             train_csv='examples/data/demo_train.csv',\n             valid_csv='examples/data/demo_valid.csv',\n             output_dir='runs/notebook_demo',\n             task='regression', method='full',\n             epochs=1, patience=1, batch_size=24,\n             device='cuda' if torch.cuda.is_available() else 'cpu')\nprint(result)\n"),
        code("predictor = GRASPPredictor.from_pretrained('runs/notebook_demo')\nnew_smiles = ['CCO', 'CCOC', 'c1ccccc1']\nprint(list(zip(new_smiles, predictor.predict(new_smiles))))\n"),
    ])


if __name__ == "__main__":
    main()
