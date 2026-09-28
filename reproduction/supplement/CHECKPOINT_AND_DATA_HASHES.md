# Excluded checkpoint and data identities

All hashes are SHA-256. The underlying checkpoints and raw datasets are intentionally not embedded in this anonymous supplement.

## Final ChemRasayan lineage

| Artifact | SHA-256 | Note |
|---|---|---|
| Step-1 3M-phase source checkpoint | `0b6f0067113853fad79ece1bbfc75e9d391081ff30fa9ca1ffb777c840de2b15` | Source for exposure-correcting continuation |
| Final Step-1 Phase-4 weights | `4ffcef04907e1d9ef039e4657eccf5df2bde276046a51b88acda9b71c1f8e48d` | 1.538B-presentation final Step-1 checkpoint |
| Step-2 10k encoder | `910331069cde7c9d4bc0cbd3af3a6acbc7dd186cbed4e888a00fdb0c9ab994c6` | Snapshot sensitivity grid |
| Step-2 20k encoder | `f5a68297af1654c98ad6a2587ff40c201121d571001943cee96e3bf24e2538fb` | Snapshot sensitivity grid |
| Step-2 30k encoder | `4343ae5b8d2909a45d3fc4e4fa9c487901b5eb334e18b0dedd8f0c0e33099596` | Snapshot sensitivity grid |
| Step-2 40k encoder | `4ef3f9e59084f58ff2c4da3517c78fbcc9d001b0c844a7603817554d1f057d21` | Snapshot sensitivity grid |
| Step-2 50k encoder | `7940ca66fc8f6785d6ae2e63b3965f69425e872cd28f4a975493fb1972742ee1` | Fixed principal downstream checkpoint |
| Step-2 60k encoder | `7ecf17381a4a330860dcdaf59407899ece461f7174799dff3828dd34ea2620ba` | Snapshot sensitivity grid |
| Step-2 70k encoder | `a81bd0f297ae593bf42afa8072f0bd454fe7d41e7a5144620fe88c9b2cde0438` | Snapshot sensitivity grid |
| Step-2 80k encoder | `405531c0955d90007f43a91f18d4ae2bd109d526fe73d214b21051d5c399dfc8` | Final Step-2 training snapshot |
| Radius-0 vocabulary | `660efd9732c9cbf62987b5e71fd8ae8908a3d70e8d1d7890cd30747cc72a689e` | Included at `configs/tokenizer/` |

## Step-2 matrix identities

| Artifact | SHA-256 | Shape/role |
|---|---|---|
| `labels.npy` | `f631972cd9915bfc72e75f68bfd6b633cce99fc7951a712dccf5e6e54718405f` | float16, 511,898 molecules by 642 assays |
| `assay_ids.txt` | `ed79496bd3f3a7b1c31a00c3e9db321f41768ea0826937c9859054b0d7742ae8` | 642 ordered assay identifiers |
| Complete 8-snapshot by 23-task by 3-seed TDC input table | `603ee64b4c22b964b457ac8611d268185d61b54b3053f534057e183d91a010eb` | Input to snapshot-sensitivity analysis |

The ZINC training corpus consisted of 1,539 training parquet shards containing 1,538,240,669 rows plus a fixed validation parquet. A single aggregate bytewise hash over all shards was not retained; this supplement records that limitation instead of substituting a newly reconstructed dataset hash.

## Controlled-comparison checkpoints

| Artifact | SHA-256 | Scope |
|---|---|---|
| MLM Step-2 comparison checkpoint | `1160ecaea098f0500d5b6138203f7bd7c5852ef8b34d3d5cc908aeed57c04b86` | Current provenance-locked OpenADMET objective rerun |
| RTD Step-2 comparison checkpoint | `daa0d70f4a7ee44807192ccfe8735696cf26e3b6d10dc0a74bc877e586b34dba` | Current provenance-locked OpenADMET objective rerun |

These controlled-comparison identities are not the final ChemRasayan Step-2 checkpoint unless explicitly stated above.

