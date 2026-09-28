"""Fail on private paths, credentials, or raw datasets in tracked release files."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

FORBIDDEN_EXTENSIONS = {".ckpt", ".pt", ".parquet", ".sqlite", ".db", ".npy", ".npz"}
SENSITIVE = [
    re.compile(r"[A-Za-z]:\\Users\\sps26", re.IGNORECASE),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?:api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{12,}['\"]", re.IGNORECASE),
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    failures = []
    for name in filter(None, paths):
        path = root / name
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            failures.append(f"raw/checkpoint file: {name}")
            continue
        if path.stat().st_size > 25_000_000:
            failures.append(f"oversized tracked file: {name}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(content) for pattern in SENSITIVE):
            failures.append(f"sensitive pattern: {name}")
    if failures:
        raise SystemExit("Release audit failed:\n" + "\n".join(failures))
    print(f"PASS: scanned {len(paths) - 1} tracked release files")


if __name__ == "__main__":
    main()
