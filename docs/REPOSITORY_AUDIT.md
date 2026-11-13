# Repository Audit: 2026-11-13 cleanup and handover

**Scope.** The whole local project and the public repository (`github.com/Anish-R-18/ASD-ScanPath`, `main` at `bca2af0`) were audited and reorganised for handover.
- **What was not changed.** No model was retrained, and no checkpoint, dataset, experiment output or manuscript file was modified.
- **Git history.** Rewritten **locally** on 2026-11-13 to remove restricted material from every past commit; see [`GIT_HISTORY_CLEANUP.md`](GIT_HISTORY_CLEANUP.md). It is not pushed.
- **Commits.** No new commits were made. The reorganisation is staged on top of the rewritten history (`main` = `3707633`, formerly `bca2af0`; two rewrite passes).

## 1. Files removed from the tracked tree

| File | Reason | Where it is now |
|---|---|---|
| `results/final_results/figures/fig4a_representative_scanpaths_saliency4asd.png` | Shows Saliency4ASD stimulus photographs, **including two children** | `local_only/removed_from_repository/` (Git-ignored) |
| `results/final_results/figures/fig4b_representative_scanpaths_gaze4asd.png` | Shows Gaze4ASD stimulus photographs (identifiable faces) | same |
| `results/external_gaze4asd/fig_representative_scanpaths.png` | Byte-identical to `fig4b` | same |
| `docs/TPP-Gaze_..._Processes (2).pdf` | Publisher PDF of the TPP-Gaze paper; it is freely available, so it is linked instead ([arXiv:2410.23409](https://arxiv.org/abs/2410.23409)) | same |
| `docs/UsageAgreement.txt` | File distributed with the Saliency4ASD data; its key clauses are quoted in `docs/dataset_documentation.md` | same |
| `configs/config.yaml` | Byte-identical duplicate of `tppgaze/data/config.yaml`, which the code actually uses | Deleted (the original remains) |
| `tppgaze/data/1009.jpg` | Upstream TPP-Gaze sample photograph. Its image-specific redistribution rights are unclear. It was removed from the tree and from history (second history pass), and is Git-ignored. Not used by our code. | Kept locally only; available upstream |
| `test/README.md` | Merged into `inference/README.md` | Deleted |
| `inference.py` (root) | Moved to `inference/inference.py`. Git shows it as delete + add because of the size of the uncommitted changes. | `inference/inference.py` |

**Also removed (local, untracked):**
- every `__pycache__/` folder;
- the empty `analysis/` and `research_completion/` folders;
- a stale PowerPoint lock file (`presentation/~$ASD_ScanPath_Research_Presentation.pptx`).

**Redacted, not removed.** These files had record-level dataset content cut out, and the full versions are kept in `local_only/removed_from_repository/`:
- **`docs/audits/duration_audit.txt`:** two "Top 10" tables listing individual fixations (file, segment, duration, screen position).
- **`docs/audits/dataset_audit.txt`:** per-segment tables and row counts for six sample files.
- **`results/in_domain/saliency4asd_preprocessing_report.txt`:** its section 7, which lists raw example scanpaths. It is redacted on every copy by `scripts/collect_results.py`.

## 2. Files moved

| From | To |
|---|---|
| `training/config.yaml` | `configs/phase1_baseline.yaml` |
| `training/config_phase2_embonly.yaml` | `configs/phase2_embedding_only.yaml` |
| `inference.py` | `inference/inference.py` |
| `demo/` (untracked) | `inference/demo/` (`local_only/` stays Git-ignored) |
| `test/input/`, `test/outputs/` (ignored) | `inference/inputs/`, `inference/outputs/` (ignored) |
| `analysis/run_sensitivity.py` | `preprocessing/run_sensitivity.py` |
| `docs/{dataset_audit,duration_audit,preprocessing_sensitivity,tpp_compatibility}.txt` | `docs/audits/` |
| `docs/figures/make_architecture_diagram.py` | `scripts/make_architecture_diagram.py` (still writes to `docs/figures/`) |
| `results/external_gaze4asd_v2/` | `results/external_evaluation/` |
| `research_completion/scripts/*.py`, `external_reference/` (untracked) | `scripts/research_completion/` |
| `research_completion/*.md` (untracked) | `docs/research_audit/` (`MANUSCRIPT_CHANGELOG.md` moved to `paper/`, which is Git-ignored) |
| `research_completion/outputs/` (untracked) | `training/runs/research_completion/` (Git-ignored) |
| `presentation/*` (17-slide deck, untracked) | `presentation/supporting_material/research_presentation_17_slide/` |
| `presentation/meeting_walkthrough/` (untracked) | `presentation/supporting_material/meeting_walkthrough/` |

**Folder name kept.** `presentation/supervisor_8_slide/` kept its name. It was going to be renamed `supervisor_presentation/`, but the restricted deck inside it was open in PowerPoint; the existing name is descriptive, so it stays.

## 3. Files archived (in `archive/`, with explanations in [`../archive/README.md`](../archive/README.md))

| Archived | From |
|---|---|
| `code/preprocess_gaze4asd.py`, `code/eval_external.py`, `code/stats_external.py`, `code/build_final_package.py` | `preprocessing/`, `training/`. All are based on the superseded native-size Gaze4ASD mapping. |
| `code/early_gaze4asd_audits/` (6 scripts) | `analysis/` |
| `configs/phase2_lr1e-3_no_saved_results.yaml` | `training/config_phase2.yaml` |
| `results/external_gaze4asd_native_mapping/` | `results/external_gaze4asd/` (superseded mapping) |
| `results/final_results_2026-11-03/` | `results/final_results/` (written before the research-completion analyses) |
| `results/in_domain_behaviour_tau2_initial/` | New copy of the first in-domain behavioural analysis (τ = 2), superseded by E1 |

## 4. Files added

| Added | Purpose |
|---|---|
| `README.md` (rewritten) | Entry point: objective, implementation, architecture, phases, data, installation, every workflow, verified results, limitations, reproducibility, references |
| `docs/`: `research_overview`, `architecture`, `dataset_documentation`, `preprocessing`, `training_guide`, `evaluation_guide`, `results_interpretation`, `limitations`, `reproducibility`, `handover_guide`, `README`, this file | Research and handover documentation |
| `docs/research_audit/` (6 audit documents + index) | Research-completion audit, previously untracked |
| `results/final/KEY_RESULTS.md`, `results/README.md` | Verified headline numbers with sources; results index |
| `results/in_domain/`, `behavioural_analysis/`, `sensitivity_analysis/`, `temperature_analysis/`, `figures/` | Curated group-level summaries with `PROVENANCE.csv` (37 files) |
| `scripts/collect_results.py` | Rebuilds those folders from local outputs. It replaces machine paths with `<project_root>`, removes record-level examples, and verifies with `--check`. |
| `scripts/research_completion/` | E1–E3, A3–A6 and the consistency-check scripts, previously untracked |
| `inference/demo/` | Reproducible demo: blank-canvas results and the synthetic scene, previously untracked |
| `presentation/` | 8-slide supervisor deck (shareable version), meeting walkthrough (without the local photo), 17-slide deck with Q&A and checklist |
| `tests/` (6 test modules, helper, upstream hash manifest) | Lightweight checks that need no data |
| `environment.yml` | Conda environment |
| `.gitattributes` | Keeps the upstream hash manifest byte-stable across platforms |
| `.gitignore` (rewritten) | Blocks datasets, checkpoints, raw outputs, local-only and restricted material, and the unpublished manuscript. Data files (`*.csv`, `*.json`) are ignored except in curated folders. |
| `README.md` in `configs/`, `training/`, `preprocessing/`, `scripts/`, `inference/`, `presentation/`, `tests/`, `archive/` | Per-folder guides |
| `tppgaze/UPSTREAM.md` | Origin of the vendored TPP-Gaze code. It is the only file added to that folder. |

## 5. Important corrections

1. **In-domain result.** The old README said Phase 2 "gains 0.0073 … no confidence interval was computed", and `results/external_gaze4asd_v2/README.md` said "Phase 2 is better by 0.0073". Both now report the E3 interval [−0.0012, +0.0163] and call the gain **inconclusive**. The external package builder was updated accordingly.
2. **Generated contrast shares.** The old README used 60%, 50% and 29% (one GPU temperature run). It now uses the E1 values (69%, 64% and 32%, with a seed-only baseline), as the manuscript does, and gives 60/50/29% as the independent estimate. The fixation counts now come from E1 (10.19 / 9.60) instead of the temperature run (10.24 / 9.62).
3. **Phase 2 trainer default.** `train_tpp_cond.py` defaulted to `config_phase2.yaml`, the unreported learning-rate-10⁻³ configuration. Running it without `--config` would have trained the wrong setup. It now defaults to `configs/phase2_embedding_only.yaml`. `train_tpp.py` now defaults to `configs/phase1_baseline.yaml`.
4. **Phase 2 smoke test.** The check required non-zero gradients in trainable base parameters, which cannot exist in embedding-only mode, so `smoke` always reported FAIL for the reported configuration. In embedding-only mode it now requires the base to be fully frozen. Both modes were re-tested and pass.
5. **Absolute paths.**
   - All ten research-completion scripts hard-coded the original workstation's absolute project path; they now use the repository root.
   - The presentation build scripts were made location-independent.
   - Curated summaries contained the training-server and local home paths; these are now written as `<project_root>`.
6. **Inference outputs.** The default output folder moved from `test/outputs/` to `inference/outputs/`, which is Git-ignored.
7. **Citation and dataset setup.**
   - The placeholder BibTeX entry (`journal={TBD}`, "et al.") was removed. The README now cites the original model and datasets with DOIs, and states that a project citation is pending authorship.
   - The vague dataset-placement instruction was replaced with the exact layout the code expects.
8. **Dependencies.** `pandas` was moved to optional, because only archived scripts use it.
9. **Consistency check.** `z_consistency_check.py` now finds the manuscript changelog in `paper/`. When the changelog is not available, it checks the values only instead of failing.

## 6. Tests and checks performed

| Check | Result |
|---|---|
| `python -m unittest discover -s tests -t .` | **33 tests, all pass** (Python 3.12.4, PyTorch 2.3.1 CPU, with checkpoints present) |
| Same suite on a fresh export of the staged files, in a new Git repository, with no data, checkpoints or local outputs | 30 pass; the 3 checkpoint tests skip as designed |
| Upstream TPP-Gaze check on a Windows-style (CRLF) and a Unix-style (LF) checkout | Passes on both |
| `tppgaze/` against upstream `phuselab/tppgaze` at `59cf572` (fresh clone, `diff -r`) | Identical |
| `inference/demo/run_demo.py --check` from the new location | 35 of 35 scanpaths regenerated identically; E1 cross-check within 10⁻⁵ |
| `inference/inference.py --condition both --compare-phase1` on the synthetic scene | Runs in about 15 s on CPU and writes to `inference/outputs/` |
| `train_tpp.py smoke` and `train_tpp_cond.py smoke`, with the outputs redirected to a scratch folder | PASS (Phase 2 also with the archived configuration); `training/runs/` verified unchanged by hash |
| `scripts/research_completion/z_consistency_check.py` | 44 cited numbers recomputed, 0 problems |
| `e2_*.py`, `e3_*.py` | Resolve their new paths and refuse to overwrite the existing outputs, as designed |
| `training/build_results_v2.py` (new output path) | All 10 figures, 16 data tables and `temperature_sensitivity.txt` byte-identical to the previously tracked versions. Only `README.md`, `final_external_comparison.txt` and `corrected_mapping_methodology.txt` (intended corrections) and `provenance.csv` changed. |
| `scripts/collect_results.py --check` | 37 of 37 files match their sources |
| Both presentation `slide_content.py` scripts | Run from the new locations and produce identical `SLIDE_CONTENT.md` |
| Pre/post hash manifest of all 677 non-dataset files | No content lost. The 70 files without an exact match are the intended path edits, regenerated caches, the deleted lock file and the merged README. |
| Publication scan of everything Git would publish | No datasets, checkpoints, raw outputs, stimulus photographs, files over 10 MB, absolute machine paths, or participant- or record-level tables |

## 7. Remaining issues

**Git history.** The following were committed in `d48d165` (2026-10-04) and carried through all 7 commits:
- the three stimulus figures (`fig4a`, which shows children; `fig4b`; and `fig_representative_scanpaths`);
- `docs/UsageAgreement.txt`;
- the TPP-Gaze paper PDF;
- the unredacted `docs/dataset_audit.txt` and `docs/duration_audit.txt`.

**What has been done.** The local history was rewritten to remove the figures, the agreement and the PDF, and to redact the two audits in every commit. Verification found none of this material reachable from `main` or the staged tree ([`GIT_HISTORY_CLEANUP.md`](GIT_HISTORY_CLEANUP.md)).

**What remains.**
- **Still public.** The **public GitHub history still contains the material** until the owner approves a force-push.
- **Earlier copies.** Copies made before then (clones, forks, caches, archives) cannot be recalled.

**Staged diff after the rewrite.** The restricted files no longer exist in HEAD, so the staged diff now shows 243 changed files: 163 added, including `docs/GIT_HISTORY_CLEANUP.md`; 67 moved; 11 modified; and 2 deleted (`inference.py`, moved with changes, and `test/README.md`, merged). The staged tree has 279 files: the 278-file handover plus that document.

**Decisions pending (owner):**
- the licence for the project's own code (none has been added; `tppgaze/` stays MIT);
- `CITATION.cff` (needs the manuscript authorship);
- publishing the manuscript (`paper/` is Git-ignored);
- releasing the trained checkpoints;
- force-pushing the rewritten history.

**Research questions that remain open** (see [`limitations.md`](limitations.md)):
- Saliency4ASD vs Gaze4ASD data terms;
- Gaze4ASD mapping confirmation;
- the TD count (132 vs 133);
- participant age;
- the earlier Phase 2 configuration.

**Older documents.** The manuscript and the presentation decks quote pre-reorganisation paths; the mapping is in [`reproducibility.md`](reproducibility.md). They were intentionally not edited.

**Local-only material** (never commit): `local_only/`, `presentation/supervisor_8_slide/local_only_restricted/`, `presentation/supporting_material/meeting_walkthrough/01_example_image/`, `inference/demo/local_only/`, `inference/inputs/`, `outputs/`, `external_data/Eye-Tracking Dataset/` (unused).
