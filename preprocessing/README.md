# Preprocessing

| Script | Input | Output (Git-ignored) |
|---|---|---|
| `preprocess_dataset.py` | Saliency4ASD (`TrainingData/TrainingData/`) | `processed_data/asd_scanpaths.json`, `td_scanpaths.json`, report and statistics |
| `validate_dataset.py` | `processed_data/` | Console checks: every image present in both groups, value ranges, etc. |
| `run_sensitivity.py` | Saliency4ASD raw files | `docs/audits/preprocessing_sensitivity.txt` (retention by group for minimum lengths 4, 3 and 2) |
| `preprocess_gaze4asd_v2.py` | Gaze4ASD (`external_data/Gaze4ASD/`) | `external_data/Gaze4ASD_processed_v2/`: three versions (fit-to-height with clipping as primary; stretch with clipping; fit-to-height with removal) and validation reports |

**Run from the repository root.** Raw data are only read: the Saliency4ASD script compares SHA-256 hashes of the raw files before and after the run.

**Documentation.** The rules, the coordinate mapping and its evidence are in [`../docs/preprocessing.md`](../docs/preprocessing.md).

**Superseded.** The native-size Gaze4ASD script (`preprocess_gaze4asd.py`) is superseded and kept in [`../archive/code/`](../archive/README.md).
