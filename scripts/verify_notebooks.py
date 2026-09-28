"""Execute notebook code cells in order from the repository root."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    for name in ("01_embeddings.ipynb", "02_finetune.ipynb"):
        path = Path("notebooks") / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        scope = {"__name__": "__main__"}
        for index, cell in enumerate(payload["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"{name}:cell{index}", "exec"), scope)
        print(f"PASS: {name}")


if __name__ == "__main__":
    main()
