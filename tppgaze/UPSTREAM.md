# Origin of this folder

This folder is the **original TPP-Gaze implementation**, unmodified.

| | |
|---|---|
| Source | https://github.com/phuselab/tppgaze |
| Commit | `59cf572ba679b8e3351da29dbbbd2ff66bcfa552` (2025-02-20) |
| Licence | MIT ([`LICENSE`](LICENSE)) |
| Paper | D'Amelio et al., "TPP-Gaze: Modelling Gaze Dynamics in Space and Time with Neural Temporal Point Processes", WACV 2025 ([arXiv:2410.23409](https://arxiv.org/abs/2410.23409)) |

**Unmodified.** Every upstream file distributed here is byte-identical to that commit. `tests/test_repository_hygiene.py` checks this against `tests/data/tppgaze_upstream_59cf572.sha256`. This note is the only added file.

**Not distributed: `data/1009.jpg`.** This is the sample photograph used by the upstream `demo.py` and README example. The image-specific redistribution rights of the photo are not stated, so it is left out of this repository.
- **Effect.** No ASD-ScanPath code uses it.
- **To run the upstream demo:** download it from the upstream repository (`data/1009.jpg` at commit `59cf572`) into `tppgaze/data/`. Git ignores it there, and the test still checks it against the upstream hash if it is present.

**Where our changes are.** All ASD-ScanPath changes live outside this folder, as wrappers (`training/conditioned.py`, `training/model_utils.py`, `training/generate_cond.py`); see [`../docs/architecture.md`](../docs/architecture.md).

**Pretrained weights.** `data/model_transformer.pth` is not included; download it with `download_models.sh` or from the upstream release v1.0.0.

**Released-code quirk.** The released transformer attends across the batch dimension (`batch_first=False` with batch-first tensors). It is kept as released; see [`../docs/architecture.md`](../docs/architecture.md).
