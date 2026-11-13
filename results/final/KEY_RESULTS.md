# Key Results

This page lists the verified headline numbers of the project, with the file each one comes from. The values match the final manuscript and the research-completion audit ([`docs/research_audit/`](../../docs/research_audit/RESEARCH_COMPLETION_AUDIT.md)); `tests/test_results_consistency.py` checks them against the result files.

**Conventions**
- NLL is the negative log-likelihood in nats per fixation, so lower is better.
- A gain is Phase 1 NLL minus Phase 2 NLL, so a positive gain favours Phase 2.
- Intervals are 95% bootstrap intervals.
- All analyses are exploratory; none was preregistered.
- No clinical or diagnostic validity is claimed.

## 1. Likelihood

| Comparison | Result | Verdict | Source |
|---|---|---|---|
| Released TPP-Gaze vs Phase 1, Saliency4ASD test (45 images, 903 scanpaths) | −0.1665 → −0.4957 | **Phase 1 improves clearly** | [`in_domain/phase1_test_eval.json`](../in_domain/phase1_test_eval.json) |
| Phase 1 vs Phase 2, Saliency4ASD test | −0.4957 vs −0.5030; gain +0.0073, image CI [−0.0012, +0.0163]; 27 of 45 images favour Phase 2; sign-flip p = 0.11 | **Small and inconclusive** | [`in_domain/e3_indomain_gain_ci.json`](../in_domain/e3_indomain_gain_ci.json) |
| Phase 1 vs Phase 2, Gaze4ASD (primary: fit-to-height, clipping) | −0.5568 vs −0.5396; gain −0.0172, participant CI [−0.0224, −0.0122], image CI [−0.0324, −0.0018] | **Phase 2 worse** | [`external_evaluation/tables/external_nll_phase1_vs_phase2.csv`](../external_evaluation/tables/external_nll_phase1_vs_phase2.csv) |
| Gaze4ASD by group (primary) | TD −0.0229, CI [−0.0290, −0.0171]; ASD +0.0104, CI [+0.0027, +0.0168] by participant and [−0.0018, +0.0240] by image | TD worse; the ASD gain clears zero only by participant | same |
| Gaze4ASD, stretch mapping | gain −0.0274, participant CI [−0.0322, −0.0226] | Phase 2 worse | [`sensitivity_analysis/e2_mapping_policy_nll_results.json`](../sensitivity_analysis/e2_mapping_policy_nll_results.json) |
| Gaze4ASD, fit-to-height with off-image fixations removed (Policy B) | gain −0.0087, participant CI [−0.0130, −0.0042] | Phase 2 worse | same |

**Robustness of the likelihood results**
- Phase 1 fits Gaze4ASD better in all three preprocessing versions.
- The small ASD gain on Gaze4ASD is fragile:
  - it falls to +0.0027 under the stretch mapping, with an interval that includes zero;
  - it is not ASD-specific, because the ASD embedding also beats Phase 1 on TD trials (post hoc).

## 2. Real ASD vs TD differences on Gaze4ASD (participant level, 33 ASD vs 133 TD)

| Measure | Hedges' g [95% CI] | Direction |
|---|---|---|
| Revisit rate | −1.04 [−1.44, −0.68] | ASD lower |
| Spatial dispersion | +0.80 [+0.41, +1.24] | ASD higher |
| Total fixation time | −0.71 [−1.11, −0.35] | ASD lower |
| Saccade amplitude | +0.65 [+0.22, +1.11] | ASD higher |
| Direction concentration | +0.55 [+0.11, +1.01] | ASD higher |

The other measures do not differ significantly: fixation count, mean fixation duration, fixation rate and horizontal saccade fraction.
- **Robustness.** All five effects hold under the stretch mapping and under Policy B.
- **Off-image gaze.** ASD children looked off the image more often: 7.35% of fixations against 2.72% for TD, about 2.7×.

Source: [`external_evaluation/tables/real_asd_vs_td_participant_level.csv`](../external_evaluation/tables/real_asd_vs_td_participant_level.csv) and [`sensitivity_analysis/a346_real_effects_policy_accounting_results.json`](../sensitivity_analysis/a346_real_effects_policy_accounting_results.json).

## 3. Generated behaviour (experiment E1: Gaze4ASD, 30 images, 10 seeds, 20 scanpaths per seed, τ = 1)

**How the label effect is measured.** It is the Phase 2 ASD−TD contrast minus the difference between two Phase 1 runs on the same seed streams, which shows what the seeds alone produce.

| Finding | Result |
|---|---|
| The label changes generated behaviour | 8 of 9 measures move beyond the seed-only baseline (Holm p ≤ 0.001). The shifts are small: 1.1% and 0.8% of the image diagonal for dispersion and saccade amplitude. |
| Spatial contrasts follow the real direction, at reduced size | Generated ASD−TD contrast as a share of the real one: dispersion 69% (0.0117 vs 0.0169), saccade amplitude 64% (0.0096 vs 0.0151), revisit rate 32% (−0.0261 vs −0.0805). An independent run gave 60%, 50% and 29%, so treat the shares as approximate. |
| Several contrasts have the wrong direction | Total fixation time, direction concentration and fixation count (the real count difference itself is not significant) |
| Neither group becomes more realistic | No spatial measure moves significantly closer to its own real group, and ASD revisit gets worse (−11.8%). The TD condition contracts in dispersion from 0.112 to 0.101, past real TD (0.108); ASD stays at 0.113. |
| Fixation over-production | 10.19 (ASD label) and 9.60 (TD label) fixations per scanpath, against 6.25 and 6.41 real. The label's own shift in count (+0.66) is about a fifth of this shared excess. |

Source: [`behavioural_analysis/e1_results.json`](../behavioural_analysis/e1_results.json) and [`behavioural_analysis/e1_analysis_report.txt`](../behavioural_analysis/e1_analysis_report.txt).

## 4. Sampling temperature

- **Effect on behaviour.** Temperature changes generated behaviour but never the likelihood: NLL is computed without temperature.
- **How it works.** It raises each spatial mixture σ to the power τ. Every σ in this model is below 1, so τ = 2 narrows the spatial spread.
- **Selection.** τ = 1 was chosen on Saliency4ASD validation images, then checked once on Gaze4ASD with paired seeds.
- **Error compared with τ = 2:**

| Measure | ASD | TD |
|---|---|---|
| Saccade amplitude | 52.5% lower | 49.9% lower |
| Revisit rate | 66.1% lower | 56.5% lower |

- **Contrast sizes.** At τ = 2 the same contrasts keep their directions but shrink to 46%, 33% and 15% of the real size.

Source: [`external_evaluation/temperature_sensitivity.txt`](../external_evaluation/temperature_sensitivity.txt) and [`temperature_analysis/`](../temperature_analysis/).

## 5. Strength of evidence (from the research-completion audit)

| Strength | Findings |
|---|---|
| Strong | No external likelihood gain overall, and TD is worse. The real spatial ASD−TD differences. The label changes generated behaviour. Spatial contrast directions are reproduced at reduced size; count, total-time and direction-concentration contrasts are wrong. τ = 1 is more realistic than τ = 2 for the ASD measures and revisit rate. Fixation over-production. |
| Weak or preprocessing-dependent | The external ASD gain. The exact size of the generated shares. |
| Inconclusive | The in-domain Phase 2 gain, and why the external TD fit gets worse. |
| Not testable with the released data | Age effects, participant-level in-domain uncertainty, clinical relevance |
