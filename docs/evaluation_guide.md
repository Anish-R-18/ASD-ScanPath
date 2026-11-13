# Evaluation Guide

**Two separate questions.** Likelihood and generated behaviour answer different questions, so they are evaluated separately.

**Before running anything.**
- **Working directory.** Run every command from the repository root.
- **What is needed.** All steps need the checkpoints, the processed data and the images. Raw outputs go to `training/runs/`, which Git ignores.
- **Existing outputs are protected.** Most scripts **refuse to overwrite** an existing output folder; move or rename it to re-run.
- **Scripts that do overwrite.** `stats_external_v2.py`, `gen_replicates_v2.py`, `analyze_phase2.py`, the `analyze` steps and the two result builders overwrite their outputs.

## 1. Likelihood

**What is compared.** Both models score the same real scanpaths in the same batches. The gain is Phase 1 NLL minus Phase 2 NLL, in nats per fixation; positive favours Phase 2.

| Step | Command | Output | Notes |
|---|---|---|---|
| In-domain test NLL | (written by the training runs) | `training/runs/*/test_eval*.json` | 45 test images, 903 scanpaths |
| In-domain uncertainty (E3) | `python scripts/research_completion/e3_indomain_gain_ci.py` | `training/runs/research_completion/e3_indomain_gain_ci_v1/` | CPU. Image-cluster bootstrap and sign-flip test, because Saliency4ASD has no participant IDs. A gate first reproduces the saved GPU values (within 7 × 10⁻⁵). |
| External evaluation | `python training/eval_external_v2.py` | `training/runs/external_gaze4asd_v2/` | Three corrected Gaze4ASD versions, plus the old native-size data as a reference. Also generates scanpaths at τ = 2 (20 per image and condition). |
| External statistics and figures | `python training/stats_external_v2.py` | same folder, `statistics/` | Real ASD vs TD per participant (Hedges' g); generated contrasts per image (Cohen's d_z) |
| Phase 1 vs Phase 2 on Gaze4ASD | `python training/compare_phase1_phase2.py run`, then `verify`, then `analyze` | `.../phase1_vs_phase2/` | Participant, image and crossed bootstraps. `verify` runs label and loss checks. |
| Mapping and policy sensitivity (E2) | `python scripts/research_completion/e2_mapping_policy_nll.py` | `training/runs/research_completion/e2_mapping_policy_nll_v1/` | CPU re-scoring under all three versions, gated against the saved GPU results |

## 2. Generated behaviour

**Measures.** All are computed in pixels, normalised by the image diagonal, by `training/eval_external_v2.py::metric_row`:
- dispersion;
- saccade amplitude;
- revisit rate;
- direction concentration;
- fixation count, rate and duration;
- total fixation time;
- horizontal saccade fraction.

**Primary measures.** Dispersion, saccade amplitude and revisit rate were fixed before E1. However, they were chosen because the earlier temperature run reproduced them, so the choice is not fully independent.

**How real and generated data are compared.**
- **Real ASD vs TD contrasts** are computed per participant (Gaze4ASD) or per image.
- **Generated contrasts** are computed per image and averaged over seeds.

| Step | Command | Output |
|---|---|---|
| τ = 2 generation replicates (Monte Carlo stability) | `python training/gen_replicates_v2.py` | `training/runs/external_gaze4asd_v2/comparison/` |
| **E1** (main behavioural experiment) | `python scripts/research_completion/e1_phase_tau1.py run <shard> 5` for shards 0–4, then `merge`, then `analyze` | `training/runs/research_completion/e1_phase_tau1_v2/` |
| E1b: τ = 2 reference, re-scored like E1 | `python scripts/research_completion/e1b_tau2_reference.py` | `.../e1b_tau2_reference_v1/` |
| E1c: CPU vs GPU agreement | `python scripts/research_completion/e1c_cpu_gpu_check.py` | `.../e1c_cpu_gpu_check_v2/` |
| Real-effect robustness (A3), Policy B accounting (A4), image-centred check (A6) | `python scripts/research_completion/a346_sensitivity_and_stats.py` | `.../a346_sensitivity_stats_v1/` |
| Earlier in-domain behavioural analysis (τ = 2; superseded by E1) | `python training/analyze_phase2.py` | `training/runs/phase2_embedding_only/analysis/` |

**E1 design.**
- **Images.** 30 Gaze4ASD images and 45 Saliency4ASD test images.
- **Sampling.** 20 scanpaths per image, label and seed, over 10 seeds, at τ = 1.
- **Conditions.** Phase 2 with the ASD label, Phase 2 with the TD label, and Phase 1 on the same two seed streams.
- **Label effect.** (P2 ASD − P2 TD) − (P1 stream A − P1 stream B). The Phase 1 difference is what random seeds alone produce.
- **Realism.** Each label is also compared with its own real group.
- **Run time.** E1 runs on CPU as five parallel shards (two threads each), about 3.3–3.8 hours per shard.

**Seeds** follow a fixed rule recorded in [`../results/behavioural_analysis/e1_environment.json`](../results/behavioural_analysis/e1_environment.json).

## 3. MultiMatch

**Our implementation.** `training/metrics.py::multimatch` aggregates with the mean, where the reference toolbox uses the median, and it allows 2-fixation scanpaths.

**Validation against the reference implementation.** These scripts need the source of `multimatch-gaze` 0.1.3; pass the folder that contains the `multimatch_gaze` package:
```bash
python scripts/research_completion/a5_multimatch_validation.py <path-to-multimatch_gaze-src>   # A5: implementation check
python scripts/research_completion/e1d_multimatch_reference.py <path-to-multimatch_gaze-src>   # E1d: E1 re-scored with the reference
```
- **What A5 shows.** With median aggregation, our implementation reproduces the MATLAB toolbox test case to 3 × 10⁻⁶.
- **What E1d shows.** All E1 conditioning conclusions are unchanged under the reference implementation.
- **Use relative comparisons only.** MultiMatch values are for relative comparisons within this implementation, not for comparison with other papers.
- **Details.** See [`research_audit/MULTIMATCH_VALIDATION.md`](research_audit/MULTIMATCH_VALIDATION.md).

## 4. Sampling temperature

| Step | Command | Data |
|---|---|---|
| Selection | `python training/temperature_sensitivity.py run`, then `windowed`, then `analyze` | Saliency4ASD **validation** images only |
| External check | `python training/eval_gaze4asd_tau1.py run`, then `analyze` | Gaze4ASD; four prespecified (not preregistered) tests |

**How temperature works.** It raises each spatial σ to the power τ; durations are not tempered. NLL is always computed without temperature.

## 5. Statistics

**Resampling unit.**
- In-domain: images.
- Gaze4ASD: participants, images, or both (crossed).

**Effect sizes.**
- Real group differences: Hedges' g, per participant.
- Generated contrasts: Cohen's d_z, paired by image. Under common random numbers it is not on the same scale as g.

**Tests and corrections.**
- **Tests.** Sign-flip and permutation tests use 10,000 resamples, so the smallest attainable p is about 10⁻⁴.
- **Corrections.** Holm correction is applied within families, not across them.

**Exploratory status.** All analyses are exploratory; none was preregistered.

Full statistical audit: [`research_audit/STATISTICAL_VALIDATION.md`](research_audit/STATISTICAL_VALIDATION.md).

## 6. Rebuilding the curated summaries

```bash
python training/build_results_v2.py        # results/external_evaluation/
python scripts/collect_results.py          # results/in_domain, behavioural_analysis, sensitivity_analysis, temperature_analysis, figures
python scripts/collect_results.py --check
PYTHONIOENCODING=utf-8 python scripts/research_completion/z_consistency_check.py   # recompute 44 cited numbers
```

On Windows, `PYTHONIOENCODING=utf-8` (in PowerShell: `$env:PYTHONIOENCODING="utf-8"`) lets the console print the "−" characters in the reports.
