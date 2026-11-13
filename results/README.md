# Results

Curated, group-level result summaries. Start with [`final/KEY_RESULTS.md`](final/KEY_RESULTS.md).

## What is here

- **Contents:** statistics, summaries and plots only.
- **Not here:** datasets, processed scanpaths, per-trial or per-participant records, generated scanpaths, checkpoints, or images of dataset stimuli.
- **Where it comes from:** each folder is copied from local, Git-ignored experiment outputs (`training/runs/`) by a script.
- **Provenance:** each folder's `PROVENANCE.csv` (or `tables/provenance.csv`) gives the source file and its SHA-256 for every copied file.
- **Machine paths:** absolute paths inside some summaries are written as `<project_root>`.

| Folder | Question it answers | Built by |
|---|---|---|
| [`final/`](final/KEY_RESULTS.md) | Headline numbers, verdicts and strength of evidence | Written from the files below |
| [`in_domain/`](in_domain/) | Phase 1 and Phase 2 on the Saliency4ASD held-out test set, training summaries, the image split, preprocessing statistics, and the in-domain confidence interval (E3) | `scripts/collect_results.py` |
| [`external_evaluation/`](external_evaluation/README.md) | Gaze4ASD: Phase 1 vs Phase 2 likelihood; real ASD vs TD effects; generated behaviour at τ = 1 vs τ = 2; the coordinate mapping | `training/build_results_v2.py` |
| [`behavioural_analysis/`](behavioural_analysis/) | E1: how the label changes generated behaviour, compared with real data and a seed-only baseline; MultiMatch validation (A5, E1d); τ = 2 reference (E1b) | `scripts/collect_results.py` |
| [`sensitivity_analysis/`](sensitivity_analysis/) | Robustness to coordinate mapping and off-image policy (E2, A3, A4, A6); CPU vs GPU agreement (E1c) | `scripts/collect_results.py` |
| [`temperature_analysis/`](temperature_analysis/) | Selection of the sampling temperature on Saliency4ASD **validation** images | `scripts/collect_results.py` |
| [`figures/`](figures/) | The six manuscript figures (none shows a stimulus photograph) | `scripts/collect_results.py` |

The experiment codes (E1–E3, A3–A6) are defined in [`docs/research_audit/EXPERIMENT_PLAN.md`](../docs/research_audit/EXPERIMENT_PLAN.md). Their full write-ups are in [`docs/research_audit/ADDITIONAL_RESULTS.md`](../docs/research_audit/ADDITIONAL_RESULTS.md).

## Which results are authoritative

1. **Final values** come from the final manuscript and the research-completion audit; [`final/KEY_RESULTS.md`](final/KEY_RESULTS.md) lists them.
   - The research-completion runs (E1–E3) **add to** the original evaluations; they do not replace them silently.
   - **Generated contrast shares.** E1 (CPU, with the seed-only baseline) gives 69%, 64% and 32%. The earlier temperature run (GPU) gives 60%, 50% and 29%. The difference is Monte Carlo noise (checked in E1c), and both are reported.
   - **External likelihood.** The primary values (Table 4) come from the original GPU evaluation. The three-version comparison (E2) is a CPU re-scoring, which agrees within 0.0002 nats.
2. **Superseded results** are in [`../archive/results/`](../archive/README.md) and must not be cited:
   - **The old native-size Gaze4ASD mapping.** It was contradicted by the data, and every value based on it is superseded.
   - **The 2026-11-03 findings package.** It predates the in-domain confidence interval and the E1 experiment, so it still described the in-domain gain as an improvement.
   - **The first in-domain behavioural analysis (τ = 2).** E1 replaces it for every conclusion.

## How to rebuild

```bash
python training/build_results_v2.py      # results/external_evaluation/ (deletes and rewrites that folder)
python scripts/collect_results.py         # all other folders
python scripts/collect_results.py --check # verify results/ against the local outputs
```

Both scripts need the local outputs in `training/runs/`, which are not distributed. See [`docs/reproducibility.md`](../docs/reproducibility.md).
