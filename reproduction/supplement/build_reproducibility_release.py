#!/usr/bin/env python3
"""Build and verify the deterministic anonymous reproducibility archive."""

from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


SOURCE = Path(__file__).resolve().parent
OUTPUT = SOURCE.parent / "output" / "ChemRasayan_anonymous_reproducibility_release.zip"
ARCHIVE_ROOT = "reproducibility_release"
INVENTORY = SOURCE / "RELEASE_INVENTORY.csv"
MANIFEST = SOURCE / "MANIFEST.sha256"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)
FORBIDDEN_SUFFIXES = {".ckpt", ".db", ".npy", ".parquet", ".pt", ".pyc", ".sqlite"}
ALLOWED_PICKLES = {
    "configs/tokenizer/vocabulary_207atomenvs_radius0_ZINC_guacamole.pkl",
    "src/pretraining/mole/training/data/vocabularies/"
    "vocabulary_207atomenvs_radius0_ZINC_guacamole.pkl",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_relative(path: Path) -> str:
    relative = path.relative_to(SOURCE).as_posix()
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"unsafe release path: {relative!r}")
    if "\\" in relative or relative.startswith("/"):
        raise ValueError(f"non-canonical release path: {relative!r}")
    return relative


def release_files(*, omit: set[str] | None = None) -> list[tuple[str, Path]]:
    omit = omit or set()
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for path in SOURCE.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symbolic links are not permitted: {path}")
        if not path.is_file():
            continue
        relative = safe_relative(path)
        if relative in omit:
            continue
        if "__pycache__" in PurePosixPath(relative).parts:
            raise ValueError(f"Python cache is not permitted: {relative}")
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            raise ValueError(f"forbidden release extension: {relative}")
        if suffix == ".pkl" and relative not in ALLOWED_PICKLES:
            raise ValueError(f"unexpected pickle payload: {relative}")
        folded = relative.casefold()
        if folded in seen:
            raise ValueError(f"duplicate case-insensitive release path: {relative}")
        seen.add(folded)
        found.append((relative, path))
    return sorted(found)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def regenerate_metadata() -> None:
    inventory_files = release_files(omit={INVENTORY.name, MANIFEST.name})
    inventory_lines = ["path,bytes,sha256"]
    for relative, path in inventory_files:
        data = path.read_bytes()
        quoted_relative = relative.replace('"', '""')
        inventory_lines.append(f'"{quoted_relative}",{len(data)},{sha256(data)}')
    atomic_write(INVENTORY, ("\n".join(inventory_lines) + "\n").encode("utf-8"))

    manifest_files = release_files(omit={MANIFEST.name})
    lines = [f"{sha256(path.read_bytes())}  {relative}" for relative, path in manifest_files]
    atomic_write(MANIFEST, ("\n".join(lines) + "\n").encode("ascii"))


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_once(path: Path, members: list[tuple[str, Path]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, source in members:
            archive_name = f"{ARCHIVE_ROOT}/{relative}"
            parsed = PurePosixPath(archive_name)
            if parsed.is_absolute() or ".." in parsed.parts:
                raise ValueError(f"unsafe archive member: {archive_name}")
            archive.writestr(zip_info(archive_name), source.read_bytes())


def verify_archive(path: Path, members: list[tuple[str, Path]]) -> None:
    expected_names = [f"{ARCHIVE_ROOT}/{relative}" for relative, _ in members]
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != expected_names or names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("archive names are missing, duplicated, or unsorted")
        for info, (_, source) in zip(infos, members):
            parsed = PurePosixPath(info.filename)
            if parsed.is_absolute() or ".." in parsed.parts or "\\" in info.filename:
                raise ValueError(f"unsafe archive member: {info.filename}")
            if info.date_time != FIXED_TIME or info.create_system != 3:
                raise ValueError(f"non-deterministic ZIP metadata: {info.filename}")
            if info.external_attr != 0o100644 << 16 or info.compress_type != zipfile.ZIP_DEFLATED:
                raise ValueError(f"unexpected ZIP permissions/compression: {info.filename}")
            if archive.read(info) != source.read_bytes():
                raise ValueError(f"archive payload mismatch: {info.filename}")


def build(output: Path = OUTPUT) -> str:
    regenerate_metadata()
    members = release_files()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        for _ in range(2):
            with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
                temporary = Path(handle.name)
            temporary_paths.append(temporary)
            build_once(temporary, members)
            verify_archive(temporary, members)
        first = temporary_paths[0].read_bytes()
        second = temporary_paths[1].read_bytes()
        if first != second:
            raise RuntimeError("two deterministic ZIP builds differ byte-for-byte")
        digest = sha256(first)
        os.replace(temporary_paths[0], output)
        temporary_paths.pop(0)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    print(
        f"built {output} with {len(members)} sorted members "
        f"({output.stat().st_size} bytes; sha256={digest})"
    )
    return digest


if __name__ == "__main__":
    build()
