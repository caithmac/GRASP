"""Publish the checked model payload after Hugging Face authentication."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import HfApi


def verify_payload(path: Path) -> None:
    for line in (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        digest = hashlib.sha256()
        with (path / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Payload hash mismatch: {name}")
    for name in ("README.md", "LICENSE", "NOTICE"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="caithmac/GRASP")
    parser.add_argument("--payload", type=Path, default=Path("hf_model"))
    args = parser.parse_args()
    verify_payload(args.payload)
    api = HfApi()
    identity = api.whoami()
    owner = args.repo_id.split("/", 1)[0]
    if identity.get("name") != owner and owner not in {org["name"] for org in identity.get("orgs", [])}:
        raise PermissionError(f"Authenticated HF account cannot publish under {owner}")
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(args.payload), repo_id=args.repo_id,
                      repo_type="model", commit_message="Release fixed GRASP Step-2 encoder")
    print(f"Published https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
