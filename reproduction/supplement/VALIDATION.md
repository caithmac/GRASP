# Release validation

The staged release passed the following checks before archive creation:

- Python bytecode compilation for all files under `src/` and `third_party/`.
- Bash syntax checking for every `.sh` launcher.
- JSON parsing for every `.json` file.
- YAML parsing for every `.yml` and `.yaml` file.
- No checkpoint or raw-data extensions (`.pt`, `.ckpt`, `.npy`, `.parquet`, `.sqlite`, `.db`) are present. The two byte-identical `.pkl` files are copies of the 1,647-byte atom-token vocabulary identified in the hash record.
- No Python cache or compiled-bytecode directory is included.
- Load-bearing training, evaluation, classical-baseline, overlap-sensitivity, and audit files were compared byte-for-byte against the originals after applying the documented infrastructure and anonymous-provenance substitutions; all scientific-logic parity checks passed.
- The atom-order v2 runner and both certifiers are byte-identical to their source copies. The staged v2 launcher and indexed-job template differ only in release-tree locations and documented anonymous infrastructure placeholders; no v2 result file or run-specific deployment manifest is staged.
- The primary-protocol objective evidence reports `PASS`, exactly 46 endpoint/model results, 276 split/method cells, 12 launch manifests, and 12 shard markers. Its public summary, 23-row endpoint CSV, and narrative match the remotely certified bytes; private internal summaries, raw predictions, job metadata, and storage paths are excluded.
- A case-insensitive scan found no private user identifier, Windows home path, private NFS path, private IP address, project-author email/name, AWS key pattern, OpenAI-style secret, GitHub personal-access-token pattern, or bearer credential value.
- A second scan checked credential assignments (`api key`, `password`, `client secret`, `access token`) for non-placeholder values. No credential value was found. Environment-variable names that allow users to provide their own optional credentials are not credentials and are retained.
- The final ZIP builder rejected unsafe, absolute, traversal, duplicate, case-colliding, cache, checkpoint, and raw-data paths before packaging.
- The final ZIP was produced twice from a sorted file list with fixed timestamps, permissions, compression, and member ordering; both builds were byte-identical and had the same SHA-256.

`MANIFEST.sha256` validates release contents. `RELEASE_INVENTORY.csv` gives file size and SHA-256 for all payload files present before the inventory and manifest were generated.
