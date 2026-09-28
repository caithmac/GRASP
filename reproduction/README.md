# GRASP paper reproduction

`supplement/` is the retained, sanitized scientific code and provenance bundle associated with the paper. It includes the staged pretraining/evaluation sources, configurations, launch templates, audits, and compact derived evidence. Its `MANIFEST.sha256` covers the files inside that bundle and should be checked from that directory.

The bundle preserves historical source names and anonymous submission records so its hashes and scientific lineage remain auditable. For routine embeddings or property fine-tuning, use the root README and the `grasp` package; the cluster launch templates in the supplement require site-specific infrastructure and separately licensed datasets.

The copied source bundle had an older inventory and manifest that did not cover later documentation and evidence additions. The release staging regenerated both metadata files over the complete 216-file supplement; all current entries verify byte-for-byte. No scientific source or result file was changed for this refresh.
