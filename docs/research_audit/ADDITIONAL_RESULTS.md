# Additional Results: Research Completion

Each experiment's design is in the experiment plan, which was written before running. These analyses were not preregistered, and all tests are exploratory.

Conventions:
- NLL is in nats per fixation, lower is better.
- **Gain > 0 favours Phase 2.**
- Intervals are percentile 95% bootstrap with 10,000 resamples.
- *g* is Hedges' *g*, ASD minus TD.

Outputs are in `training/runs/research_completion/<experiment>_v*/`; the scripts are in `scripts/research_completion/`. No checkpoint, dataset or earlier result was changed, and hashes were checked before and after every model run.

## E1. Phase 1 vs Phase 2 generated behaviour at τ=1 (common random numbers)

Outputs:
- `training/runs/research_completion/e1_phase_tau1_v2/`: `e1_results.json`, `e1_analysis_report.txt`, `summaries.npz`, `multimatch_per_image.csv`, `run_checks.json`, `environment.json`, and 80 generated files (4 sets × 10 replicates × 2 datasets).
- τ=2 reference: `training/runs/research_completion/e1b_tau2_reference_v1/`.
- CPU vs GPU check: `training/runs/research_completion/e1c_cpu_gpu_check_v2/`. v1 is superseded because it lacked the contrast test.

### Completion and integrity

- **Shards.** All 5 shards finished, exit code 0. Each covers exactly its 2 replicates, with no missing values in its own replicates and none outside them. Real-data arrays are identical across shards.
- **Merge.** The merge found replicates 1–10 each exactly once.
- **Generated files.**
  - 40 per dataset, each with full image coverage: 30 Gaze4ASD and 45 Saliency4ASD images.
  - 598–600 Gaze4ASD scanpaths and 900 Saliency4ASD scanpaths per set and replicate. A few empty samples were dropped, as in all earlier runs.
  - MultiMatch has 2,400 rows, none missing.
- **Cache gate.** In every shard, cached vs uncached sampling differed by at most 4.8e-7, with identical masks.
- **Checkpoints.** Hashes were identical before and after every shard: Phase 1 `b6134130…b0bb`, Phase 2 `35319673…a9a4`.
- **Environment.** Python 3.12.4, torch 2.3.1+cpu, numpy 1.26.4, on CPU. Details are in `environment.json`.

### CPU vs saved GPU τ=1 Phase 2 (same seeds, different random streams)

- **Group means.** All 18 group-by-metric means agree within 1.6% (image-paired sign-flip, smallest p = 0.061 for TD dispersion, unadjusted).
- **Densities.** CPU and GPU densities agree to within 2e-4 nats per fixation (E2), so there is no plausible platform effect on sampling.
- **Contrasts.**
  - The ASD−TD contrasts differ by Monte Carlo amounts. CPU vs GPU: dispersion +0.0117 vs +0.0101 (p = 0.10), amplitude +0.0096 vs +0.0075 (p = 0.050), revisit −0.0261 vs −0.0237 (p = 0.40).
  - With nine contrasts compared, none survives Holm correction.
  - Single-seed contrasts vary widely: amplitude ranges from +0.0059 to +0.0126 across CPU seeds.
- **Consequence.** The "share of real contrast" figures (paper Table 5: 60%, 50%, 29%) carry several percentage points of Monte Carlo uncertainty; E1 gives 69%, 64% and 32%. E1's own comparisons are internally consistent, because all four sets were sampled on the same platform with the same seeds.

### Conditioning effect: does the Phase 2 ASD−TD contrast exceed the Phase 1 seed-only contrast?

The conditioning effect is the Phase 2 contrast (ASD-conditioned − TD-conditioned) minus the Phase 1 seed-null contrast (seed g=0 − g=1), paired by image (n = 30 or 45). *d_z* is the paired standardized effect over images. Under common random numbers it is large even for small absolute shifts, so it measures detectability, not practical importance. The primary family (3 measures × 2 datasets) is Holm-corrected; the other measures are corrected within each dataset.

**Gaze4ASD (external, 30 images)**

| Measure | Real ASD−TD (image-level) | Phase 2 | Phase 1 null | Conditioning effect [95% CI] | *d_z* | Holm p | Direction vs real | Share of real |
|---|---|---|---|---|---|---|---|---|
| **Dispersion** (primary) | +0.0169 | +0.0117 | +0.0009 | +0.0107 [+0.0098, +0.0116] | 4.3 | 0.0006 | Correct | 69% |
| **Saccade amplitude** (primary) | +0.0151 | +0.0096 | +0.0018 | +0.0078 [+0.0070, +0.0087] | 3.1 | 0.0006 | Correct | 64% |
| **Revisit rate** (primary) | −0.0805 | −0.0261 | −0.0034 | −0.0227 [−0.0275, −0.0181] | −1.7 | 0.0006 | Correct | 32% |
| Direction concentration | +0.0270 | −0.0052 | +0.0033 | −0.0086 [−0.0123, −0.0049] | −0.8 | 0.0009 | **Wrong** | — |
| Total fixation time (s) | −0.0822 | +0.0210 | −0.0051 | +0.0261 [+0.0200, +0.0320] | 1.5 | 0.0009 | **Wrong** | — |
| Fixation count | −0.159 (n.s.) | +0.593 | −0.069 | +0.662 [+0.617, +0.707] | 5.1 | 0.0009 | **Wrong** | — |
| Fixation rate | +0.053 (n.s.) | +0.190 | −0.019 | +0.209 [+0.195, +0.222] | 5.4 | 0.0009 | Same sign, 3.6× real | — |
| Fixation duration (s) | −0.0072 (n.s.) | −0.0213 | +0.0018 | −0.0230 [−0.0248, −0.0212] | −4.5 | 0.0009 | Same sign, 2.9× real | — |
| Horizontal fraction | +0.0010 (n.s.) | +0.0004 | −0.0005 | +0.0009 [−0.0050, +0.0066] | 0.05 | 0.77 | — | — |

The conditioning effect has the same sign in all 10 seeds for seven measures. The exceptions are direction concentration (9 of 10) and horizontal fraction (6 of 10, no effect).

**Saliency4ASD held-out test (in-domain, 45 images)**

| Measure | Real | Phase 2 | Conditioning effect [95% CI] | Holm p | Direction | Share |
|---|---|---|---|---|---|---|
| **Dispersion** | +0.0058 [−0.0001, +0.0118] | +0.0103 | +0.0098 [+0.0092, +0.0105] | 0.0006 | Correct (real CI touches 0) | 178% |
| **Saccade amplitude** | +0.0111 | +0.0060 | +0.0060 [+0.0053, +0.0068] | 0.0006 | Correct | 54% |
| **Revisit rate** | −0.0638 | −0.0144 | −0.0141 [−0.0180, −0.0102] | 0.0006 | Correct | 23% |
| Fixation duration | −0.0413 | −0.0218 | −0.0205 | 0.0009 | Correct | 53% |
| Fixation rate | +0.474 | +0.225 | +0.211 | 0.0009 | Correct | 48% |
| Horizontal fraction | −0.0528 | −0.0087 | −0.0094 | 0.0009 | Correct | 16% |
| Fixation count | −0.904 | +0.719 | +0.678 | 0.0009 | **Wrong** | — |
| Total fixation time | −0.430 | +0.022 | +0.024 | 0.0009 | **Wrong** | — |
| Direction concentration | +0.0295 (CI touches 0) | −0.0116 | −0.0105 | 0.0009 | **Wrong** | — |

**Interpretation**

- **The embedding changes generated behaviour systematically.** Conditioning shifts 8 of 9 measures well beyond seed noise, on both datasets.
- **Spatial measures follow real group differences.** On all three primary spatial measures the shift is in the real direction on both datasets. It is under-sized (23–69% of real), except in-domain dispersion, where it overshoots a real contrast whose interval touches zero.
- **Temporal measures and direction concentration do not.** The ASD condition produces **more** fixations and more total time than the TD condition, the opposite of the real data. Direction concentration also goes the wrong way on both datasets. On Gaze4ASD, mean duration and rate shifts are about 3× the (non-significant) real differences.
- **Statistical vs practical.**
  - The effects are statistically unambiguous, with *d_z* of 1–5 under common random numbers.
  - In absolute terms they are small next to the model's overall miscalibration. The dispersion and amplitude shifts are 0.8–1.1% of the image diagonal. The revisit-rate shift is 2.3 percentage points. The fixation-count shift (+0.66) is about a fifth of the over-production that both conditions share: 9.6–10.2 generated vs 6.2–6.4 real fixations.
- **Off-image gaze.**
  - Generated scanpaths almost never leave the image: 0.42% (ASD condition), 0.25% (TD condition) and 0.43–0.44% (Phase 1) of fixations, against 7.35% and 2.72% for real ASD and TD.
  - The conditioned contrast (+0.17 percentage points) has the real sign, but it is a tiny fraction of the real +4.6 points.

### Match to real: does Phase 2 match each real group better than Phase 1?

Per-image absolute error against the matching real group: P1_s0 vs real ASD and P1_s1 vs real TD, compared with P2_ASD and P2_TD. A positive gain favours Phase 2. Holm correction covers the 18 tests per dataset.

**Gaze4ASD**

| Measure | ASD: error P1 → P2 | ASD gain (Holm p) | TD: error P1 → P2 | TD gain (Holm p) |
|---|---|---|---|---|
| Saccade amplitude | 0.0174 → 0.0180 | −3.2% (0.79) | 0.0123 → 0.0134 | −9.1% (1.0) |
| Dispersion | 0.0159 → 0.0158 | +0.7% (1.0) | 0.0139 → 0.0148 | −6.5% (1.0) |
| Revisit rate | 0.0541 → 0.0605 | **−11.8% (0.014)** | 0.0586 → 0.0510 | +13.0% (1.0) |
| Direction concentration | 0.0945 → 0.0975 | −3.2% (0.091) | 0.0709 → 0.0653 | **+7.9% (0.004)** |
| Fixation duration | 0.0201 → 0.0267 | **−32.5% (0.002)** | 0.0273 → 0.0196 | **+28.2% (0.012)** |
| Fixation count | 3.63 → 3.94 | **−8.5% (0.002)** | 3.54 → 3.19 | **+10.0% (0.002)** |
| Fixation rate | 0.363 → 0.465 | **−28.1% (0.002)** | 0.434 → 0.336 | **+22.6% (0.002)** |
| Total fixation time | 0.794 → 0.801 | −0.8% (0.085) | 0.717 → 0.697 | **+2.8% (0.002)** |
| Horizontal fraction | 0.0712 → 0.0715 | −0.4% (1.0) | 0.0740 → 0.0737 | +0.4% (1.0) |

**Saliency4ASD test.** The pattern is the same.
- **TD** improves on count (+11.3%), total time (+2.8%) and, marginally, dispersion (+19.0%, Holm 0.060) and duration (+10.0%, Holm 0.059).
- **ASD** gets worse on count (−9.8%), total time (−1.3%) and direction concentration (−6.4%), all Holm 0.002.
- No spatial measure improves for ASD.

**Interpretation.**
- Conditioning does **not** bring either group's absolute **spatial** behaviour closer to the real group on Gaze4ASD. No spatial measure improves significantly for either group, and ASD revisit rate gets worse.
- It **does** improve TD **temporal** behaviour: fewer, longer fixations, closer to real TD. The ASD condition moves temporal behaviour away from real ASD.
- So the correct-direction spatial **contrast** arises because the two conditions move apart. Neither moves towards its own real group.
- This is consistent with the τ=2 finding that "contrast gains come mainly from moving TD away from real TD". At τ=1 the TD spatial losses are smaller and no longer significant.

### Agreement with the τ=2 comparison

The τ=2 reference was recomputed from the saved τ=2 scanpaths with the same clipped metric code (E1b).

| Aspect | τ=2 | τ=1 (E1) | Agreement |
|---|---|---|---|
| Phase 2 contrast direction, all 9 measures | Spatial three correct; count, total time and direction wrong | Same | **Same direction everywhere** (horizontal fraction ≈ 0 at both) |
| Share of real: dispersion, amplitude, revisit | 46%, 33%, 15% | 69%, 64%, 32% | τ=1 larger, as expected from wider sampling |
| TD spatial match (amplitude, dispersion, revisit) | Phase 2 **worse**: −25.5%, −32.5%, −21.4%, all Holm ≤ 0.003 | −9.1%, −6.5%, +13.0%, all n.s. | **Differs:** the τ=2 TD spatial penalty largely disappears at τ=1 |
| TD temporal match (count, rate, duration, total) | Better: +7.7%, +15.8%, +18.1%, +2.6%; Holm 0.04, 0.18, 0.83, 0.026 | Better: +10.0%, +22.6%, +28.2%, +2.8%; all Holm ≤ 0.012 | Same direction, clearer at τ=1 |
| ASD temporal match (count, rate, duration) | Worse: −9.3%, −31.0%, −30.7% | Worse: −8.5%, −28.1%, −32.5% | Same |
| ASD revisit match | Worse (−6.7%, Holm 0.040) | Worse (−11.8%, Holm 0.014) | Same |

The τ=2 Phase 1 was a single independent-seed set, so τ=2 Holm p-values come from a noisier design. The direction agreement is the robust part of the comparison.

The paper's τ=2 statement ("better on one measure, TD total time, after Holm across 22 tests") used unclipped coordinates and a 22-test family. Recomputed with clipping and E1's 18-test family, TD fixation count also passes (Holm 0.040). This is a difference in analysis choices, not a contradiction: the paper's family is the one that was specified.

### MultiMatch: is each real group closer to its own condition than the other, beyond the seed null?

This uses our own implementation, mean aggregation. Re-scored on exactly the same pairs with the reference implementation (E1d; MultiMatch validation §4b), all eight significant conclusions are unchanged. The only sign change is ASD direction, which is not significant in either implementation.

| Real group | Shape | Length | Direction | Position | Duration |
|---|---|---|---|---|---|
| ASD: conditioning effect (Holm p) | −0.0016 (0.001) | −0.0021 (0.001) | +0.0011 (0.41) | **−0.0062 (0.001)** | +0.0056 (0.001) |
| TD: conditioning effect (Holm p) | +0.0019 (0.001) | +0.0026 (0.001) | −0.0014 (0.41) | **+0.0074 (0.001)** | −0.0050 (0.001) |

- **Spatial dimensions.** Real ASD scanpaths are *less* similar to ASD-conditioned samples than to TD-conditioned samples on shape, length and position. Real TD scanpaths are more similar to TD-conditioned samples.
- **What this means.** The ASD condition's spatial shift (wider, longer saccades) makes samples less similar, pair by pair, to real ASD scanpaths, even though the group *means* move in the real direction.
- **Duration** shows the opposite pattern for both groups.
- **Size.** All effects are ≤ 0.0074 on a 0–1 scale, which is practically negligible.

## E2. Phase 1 vs Phase 2 NLL on Gaze4ASD under the alternative mapping and policy

Output: `training/runs/research_completion/e2_mapping_policy_nll_v1/e2_results.json`.

**Gate.** Aggregate CPU NLL was within 1.2e-4 to 1.8e-4 per fixation of the saved GPU values in all three versions. The primary overall gain was reproduced: −0.0172 vs −0.0172 saved. Per-trial values differ slightly by platform (median 0.005, unbiased). See the plan for the gate change.

| Version | Group | Joint gain | CI, participants | CI, images | Duration gain | Position gain |
|---|---|---|---|---|---|---|
| **Fit-height A (primary)** | Overall | **−0.0172** | [−0.0225, −0.0122] | [−0.0323, −0.0024] | −0.0052 | −0.0120 |
| | ASD | **+0.0104** | [+0.0028, +0.0169] | [−0.0019, +0.0244] | +0.0029 | +0.0075 |
| | TD | **−0.0230** | [−0.0291, −0.0171] | [−0.0404, −0.0057] | −0.0069 | −0.0161 |
| Stretch A | Overall | −0.0274 | [−0.0322, −0.0226] | [−0.0456, −0.0079] | −0.0053 | −0.0221 |
| | ASD | +0.0027 | [−0.0041, +0.0087] | [−0.0091, +0.0160] | +0.0029 | −0.0003 |
| | TD | −0.0336 | [−0.0394, −0.0282] | [−0.0547, −0.0118] | −0.0070 | −0.0266 |
| Fit-height B | Overall | −0.0087 | [−0.0130, −0.0042] | [−0.0243, +0.0069] | −0.0055 | −0.0032 |
| | ASD | +0.0152 | [+0.0077, +0.0215] | [+0.0030, +0.0294] | +0.0029 | +0.0122 |
| | TD | −0.0134 | [−0.0183, −0.0082] | [−0.0311, +0.0042] | −0.0071 | −0.0062 |

**Interpretation**

- **Robust to mapping and policy.**
  - Phase 2 is worse overall and for TD, with every participant interval excluding zero.
  - The **duration** component is identical across versions: ASD +0.0029 and TD −0.0070, with all intervals excluding zero. This is expected, because durations do not depend on the coordinate mapping.
- **Depends on mapping and policy:** the ASD *position* gain.
  - It is +0.0075 under fit-height A, about 0 under stretch (−0.0003), and +0.0122 under Policy B.
  - The ASD joint gain clears zero by participants under fit-height A and B, but not under stretch.
  - Stretch is the less supported mapping (saliency-map r = 0.91 vs 0.9999), but it is not excluded, so the ASD positional benefit is not mapping-independent.
- **Policy B** changes the data: it removes more ASD fixations (A4), so its NLL values describe a different, filtered population. It is reported as a sensitivity analysis, not as an alternative primary result.
- **Conclusion for the paper.** "Phase 2 helps external ASD and hurts external TD" holds for durations under every version. For positions, it holds only under fit-to-height.

### Robustness to preprocessing (E2 with A3 and A4)

| Question | Answer | Evidence |
|---|---|---|
| Does the overall external NLL conclusion change? | **No.** Phase 1 is better overall in every version (−0.0172, −0.0274, −0.0087), and all participant intervals exclude 0. The image interval includes 0 only under Policy B. | E2 |
| Do the group-specific conclusions change? | **TD: no.** Phase 2 is worse in every version. **ASD: partly.** The duration gain is identical everywhere; the position gain disappears under stretch, so the ASD joint gain is not robust to the mapping. | E2 |
| Do the real behavioural contrasts change? | **No in direction and significance** for the five significant effects. Under Policy B, the count and total-time effects grow, because the policy removes ASD data. | A3, A4 |
| Do the generated-vs-real comparisons change? | The generated scanpaths do not depend on the mapping; the real reference does. The τ=1 share of the real contrast is stable for amplitude (48–53%) and revisit (28–29%). For dispersion it varies (60–81%). The τ=1 benefit holds in 4/4 prespecified tests under Policy B and 3/4 under stretch. | A3 |
| Should the primary mapping change? | **No.** Fit-to-height has the strongest support: the authors' saliency maps give median r = 0.9999, against 0.91 for stretch. Stretch is weaker but not excluded by gaze alone, so it remains a legitimate sensitivity analysis, and the mapping-dependence of the ASD position gain is a real caveat. Policy B is not group-neutral. Fit-to-height with Policy A stays primary. | `mapping_methodology.txt`; A4 |

## E3. In-domain gain with image-level uncertainty (Saliency4ASD test)

Output: `training/runs/research_completion/e3_indomain_gain_ci_v1/e3_results.json`.

**Gate.** Re-scored aggregates were Phase 1 −0.49561 (saved −0.49567) and Phase 2 −0.50295 (saved −0.50302), log-likelihood per fixation. The maximum difference was 6.6e-5.

| Group | Component | Gain | 95% CI, 45 images | Images favouring Phase 2 | Sign-flip p |
|---|---|---|---|---|---|
| Overall | Joint | **+0.0073** | **[−0.0012, +0.0163]** | 27/45 | 0.110 |
| Overall | Duration | +0.0019 | [+0.0002, +0.0035] | 28/45 | 0.029 |
| Overall | Position | +0.0055 | [−0.0029, +0.0139] | 26/45 | 0.216 |
| ASD | Joint | +0.0086 | [+0.0011, +0.0162] | 28/45 | 0.025 |
| ASD | Position | +0.0082 | [+0.0013, +0.0152] | — | 0.028 |
| TD | Joint | +0.0064 | [−0.0058, +0.0189] | 25/45 | 0.328 |
| TD | Duration | +0.0029 | [+0.0005, +0.0054] | — | 0.023 |

**The in-domain gain is not statistically distinguishable from zero across images.**
- The direction favours Phase 2.
- The ASD subset and the duration component clear zero, unadjusted.
- These sub-results are exploratory and would not survive a correction across the seven rows shown.

Images are the only available clusters, because Saliency4ASD has no participant IDs. A participant-level interval would probably be wider, not narrower.

## A3. Behavioural results under the alternative mapping and policy

Output: `training/runs/research_completion/a346_sensitivity_stats_v1/a346_results.json`.

**Real ASD vs TD participant-level effects (Hedges' g; Holm p in parentheses).**

| Measure | Fit-height A | Stretch A | Fit-height B |
|---|---|---|---|
| Revisit rate | −1.04 (.001) | −1.03 (.001) | −1.11 (.001) |
| Spatial dispersion | +0.80 (.001) | +0.79 (.002) | +0.72 (.003) |
| Total fixation time | −0.71 (.002) | −0.71 (.002) | −1.08 (.001)† |
| Saccade amplitude | +0.65 (.010) | +0.66 (.007) | +0.75 (.001) |
| Direction concentration | +0.55 (.025) | +0.55 (.032) | +0.63 (.006) |
| Fixation count | −0.21 (1.0) | −0.21 (1.0) | −0.46 (.081)† |
| Duration, rate, horizontal fraction | \|g\| ≤ 0.16, all p = 1 | same | same |

The paper's Holm p-values come from its own seed (amplitude .0105); this run used a new seed, so p-values differ by Monte-Carlo noise.

† Under Policy B the larger effects partly reflect the policy itself. It removes whole off-image trials and fixations more often for ASD (A4), which lowers ASD fixation count and total time.

**All five significant real effects are robust in direction and significance across all three versions.**

**τ=1 vs τ=2 realism: error reduction at τ=1 (95% CI of the paired reduction).**

| Group, measure | Fit-height A | Stretch A | Fit-height B |
|---|---|---|---|
| ASD amplitude | 52.5% [.017, .022] | 53.0% [.008, .019] | 53.0% [.016, .022] |
| ASD revisit | 66.1% | 66.2% | 66.7% |
| ASD dispersion | 47.8% | 40.7% | 37.6% |
| TD amplitude | 49.9% [.009, .018] | **32.9% [−.0002, .012]** | 47.7% |
| TD revisit | 56.5% | 52.4% | 53.5% |
| TD dispersion | 34.1% [.004, .012] | **19.0% [−.002, .008]** | **24.2% [−.0002, .010]** |
| Generated share of real ASD–TD contrast, dispersion (τ=1 / τ=2) | 0.60 / 0.46 | 0.64 / 0.50 | 0.81 / 0.62 |
| Same, amplitude | 0.50 / 0.33 | 0.53 / 0.35 | 0.48 / 0.31 |
| Same, revisit | 0.29 / 0.15 | 0.29 / 0.15 | 0.28 / 0.15 |

- **Robust:** τ=1 is more realistic for all three ASD measures and for TD revisit rate, under every version.
  - Of the paper's four prespecified tests (amplitude and revisit rate, ASD and TD; Table 6), all four keep intervals excluding zero under Policy B, and three of four under stretch.
  - Dispersion was not among the prespecified tests.
- **Depends on the version:** TD dispersion (CI includes zero under stretch and Policy B) and TD amplitude (CI touches zero under stretch).
- The paper's τ=1 benefit should be described as robust for ASD and for revisit rate, and weaker for TD spatial measures.

## A4. What the out-of-image policy removes

| | ASD | TD |
|---|---|---|
| Trials removed by Policy B | 26 of 771 (3.37%) | 49 of 3,605 (1.36%) |
| Fixations removed | 8.07% | 3.03% |
| Fixations at the image edge under Policy A (clipped) | 7.35% | 2.72% |
| Participants losing at least one trial | 11 of 33 | 33 of 133 |
| Most trials lost by one participant | 6 | 7 |

**Participant-level off-image fraction.**
- ASD 0.073 vs TD 0.027, a difference of +0.046, 95% CI [+0.020, +0.074], permutation p ≈ 0.0001.
- No participant is removed entirely: 33 and 133 remain.

**Interpretation.**
- ASD children look outside the image area about 2.7 times as often as TD children. This is itself a group difference.
- Policy B removes it, and Policy A (clipping) keeps it as edge fixations.
- Neither policy is neutral, and no new filtering rule is proposed.
- The paper should:
  - report this difference as a finding;
  - keep Policy A as primary, because it retains all trials and was fixed before modelling;
  - show Policy B only as a sensitivity analysis.

## A6. Design balance and image-centred participant effects

- **Trials per participant:** ASD 11–30 (median 25), TD 19–30 (median 27).
- **Participants per image:** ASD 19–32, TD 103–130.
- **Image-centred effects** (each image's mean subtracted before averaging per participant): revisit −1.03, dispersion +0.79, total time −0.70, amplitude +0.66, direction concentration +0.56. The Holm conclusions are identical to the uncentred analysis (see the statistical validation).
- **Conclusion:** differences in which images each participant saw do not explain the real group effects.
