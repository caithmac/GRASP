# Anonymization record

The staging process copied source files into this directory and replaced only deployment-specific identifiers:

| Original category | Release placeholder |
|---|---|
| Private repository mount | `/workspace/chemrasayan` |
| Private NFS server | `NFS_SERVER_PLACEHOLDER` |
| Private NFS export | `/EXTERNAL_NFS_PATH` |
| Private persistent-volume claim | `WORKSPACE_PVC` |
| Private queue | `REVIEW_QUEUE` |
| Private cluster user/job prefix | `anonymous-review` |
| Source-repository commit identifier in the TDC split reference | `REDACTED_FOR_ANONYMOUS_REVIEW` |

No model dimensions, learning rates, schedules, masks, seeds, dataset filters, split rules, metrics, statistical tests, or checkpoint hashes were changed. Because launchers and manifests were redacted, their release hashes differ from the original-run hashes recorded in provenance files.

The unredacted `tdc_classical_split_reference.json` had canonical JSON SHA-256 `a66d3a13182b11e5ccbc3a8de84ba802ba13b59e912441b69ede11b4fed1309e` before the source-repository commit identifier was replaced. This digest is retained only as an audit identifier; the identifying commit value is not distributed. The redaction changes provenance metadata only, not any task, split, membership, label, protocol, or historical-log hash.

The validation scan rejects the private username, Windows home path, private NFS path, private IP address, and common credential-value formats. Scientific uses of the word “token” are retained because atom tokens and tokenization are part of the method; no authentication token value is present.
