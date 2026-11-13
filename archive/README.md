# Archive (superseded material, kept for provenance)

> **Do not cite or build on anything in this folder.** It is kept so that the history of the project can be traced. Current code is in `training/`, `preprocessing/` and `scripts/`; current results are in `results/`.

## `code/`

| File | Why it is superseded |
|---|---|
| `preprocess_gaze4asd.py` | Mapped Gaze4ASD gaze assuming native-size, centred stimuli. The coordinate audit found this contradicted by the data (median *r* 0.23 with the authors' saliency maps, against 0.9999 for fit-to-height). Replaced by `preprocessing/preprocess_gaze4asd_v2.py`. |
| `eval_external.py`, `stats_external.py` | Evaluation and statistics on the native-mapping data. Replaced by `training/eval_external_v2.py` and `stats_external_v2.py`. |
| `build_final_package.py` | Built the first results package from the native-mapping evaluation |
| `early_gaze4asd_audits/` | First exploratory inspections of the Gaze4ASD release and early off-image policy audits. They were done before the corrected mapping, and replaced by the validation reports of `preprocess_gaze4asd_v2.py`. |

**Not maintained.** These scripts are not maintained and may need old paths (for example `external_data/Gaze4ASD_processed/`).

## `configs/`

| File | Note |
|---|---|
| `phase2_lr1e-3_no_saved_results.yaml` | An earlier Phase 2 configuration (learning rate 10⁻³, base trained apart from the backbone). No outputs or metrics from it could be found, it is not reported, and nothing indicates that the test split was used with it. It was the Phase 2 trainer's default until the 2026-11 cleanup. |

## `results/`

| Folder | Why it is superseded |
|---|---|
| `external_gaze4asd_native_mapping/` | Gaze4ASD results under the contradicted native-size mapping. See its `SUPERSEDED.txt`. A figure that showed stimulus photographs was removed. |
| `final_results_2026-11-03/` | Findings package written before the research-completion analyses. It reports the in-domain gain without a confidence interval (later found inconclusive), and the generated shares from one run (60/50/29%), which E1 superseded. See its `SUPERSEDED.txt`. Two figures that showed stimulus photographs were removed. |
| `in_domain_behaviour_tau2_initial/` | First in-domain behavioural analysis (`training/analyze_phase2.py`, τ = 2, no seed-only baseline). E1 supersedes it for every conclusion. |

**Paths inside these files** refer to the repository layout before 2026-11 (for example `results/external_gaze4asd_v2/`, now `results/external_evaluation/`); see [`../docs/reproducibility.md`](../docs/reproducibility.md).
