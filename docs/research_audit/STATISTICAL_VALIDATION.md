# Statistical Validation

This reviews the statistical methods behind the paper and adds four targeted checks:
- **E3**, uncertainty for the in-domain gain (new evaluation);
- **A6**, an image-centred robustness test (re-analysis only);
- **E1**, the statistical design of the τ=1 comparison;
- **E1c**, CPU vs GPU sampling agreement.

All tests are exploratory, and none were preregistered.

## 1. Units of analysis and independence

| Analysis | Unit | Repeated observations treated as independent? | Assessment |
|---|---|---|---|
| Real ASD vs TD (Gaze4ASD) | One mean per participant (33 vs 133) | No; trials are averaged within participant | Appropriate |
| External NLL, Phase 1 vs 2 | Participant and image clusters, bootstrapped separately and together | No; fixation events are never resampled individually | Appropriate |
| In-domain NLL (Saliency4ASD) | Previously none; **now image clusters (E3)** | Previously no interval at all | **Corrected by E3** |
| Generated vs real, generated contrasts | One value per image, pooled over 10 seeds (n = 30 or 45) | No | Appropriate. Real per-image means come from the same participants across images, so image-level intervals ignore participant crossing. They are therefore somewhat optimistic for real-data contrasts. |
| Temperature comparisons | Images, with seeds paired across temperatures | No | Appropriate. Seeds are paired, not independent units. |
| MultiMatch | Pairs averaged within image (at most 30), then images | No | Appropriate for comparisons. Absolute values depend on the implementation (see MultiMatch validation). |
| E1 conditioning effect (new) | Image; Phase 2 contrast minus Phase 1 seed-null contrast, pooled over 10 seeds; same seeds for both models | No | Appropriate. Common random numbers remove most sampling noise, so tiny absolute effects become highly significant (see Section 3). |

## 2. Methods checked in code

| Method | What the code does | Assessment |
|---|---|---|
| Hedges' *g* | Pooled SD with small-sample correction J = 1 − 3/(4(n₁+n₂) − 9) | Correct (Hedges 1981); recomputed independently in an earlier audit |
| Cohen's *d_z* | Mean of paired differences divided by their SD | Correct. Not comparable with *g*; noted in the manuscript. |
| Permutation tests | Two-sided, mean difference, 10,000 relabellings; p = (k+1)/(N+1), floor 0.0001 | Correct. Group SDs are similar (ratios 0.91–1.38), so the equal-variance exchangeability concern is small. |
| Bootstrap intervals | Percentile, 10,000 resamples. Participant bootstrap stratified by group; groups resampled separately for *g* intervals | Correct. Intervals for differences of absolute values (gap measures) can be off-centre near zero; noted. |
| Sign-flip tests | Paired image-level or participant-level | Correct under symmetric differences |
| Holm | Step-down, within each family | Correct. No correction across families. |
| Correlations (per-image *r*) | Point estimates only | **Need intervals or more cautious wording.** With 30 images, *r* = 0.08–0.39 is consistent with zero. |

## 3. Targeted robustness checks

**A6. Image composition.** Participants viewed different subsets of images: ASD 11–30 trials (median 25), TD 19–30 (median 27). Image-centred participant means subtract each image's overall mean first.

| Measure | Original *g* | Image-centred *g* | Holm p (image-centred) |
|---|---|---|---|
| Revisit rate | −1.04 | −1.03 | 0.0010 |
| Spatial dispersion | +0.80 | +0.79 | 0.0018 |
| Total fixation time | −0.71 | −0.70 | 0.0024 |
| Saccade amplitude | +0.65 | +0.66 | 0.0084 |
| Direction concentration | +0.55 | +0.56 | 0.0264 |
| Other five measures | \|g\| ≤ 0.21 | \|g\| ≤ 0.21 | 1.0000 |

All real effects are unchanged in direction, size and significance, so image composition does not explain them. The rerun's p-values differ slightly from the paper's because a new random seed was used; the paper's values remain the reported ones.

**E3. In-domain gain.** The 903 Saliency4ASD test scanpaths were re-scored on CPU. The gate reproduced the saved NLL within 6.6e-5. A paired image-level cluster bootstrap and sign-flip test over 45 images:

| Group | Component | Gain (Phase 2 better) | 95% CI, images | Images favouring Phase 2 | Sign-flip p |
|---|---|---|---|---|---|
| Overall | Joint | +0.0073 | [−0.0012, +0.0163] | 27 / 45 | 0.110 |
| Overall | Duration | +0.0019 | [+0.0002, +0.0035] | 28 / 45 | 0.029 |
| Overall | Position | +0.0055 | [−0.0029, +0.0139] | 26 / 45 | 0.216 |
| ASD | Joint | +0.0086 | [+0.0011, +0.0162] | 28 / 45 | 0.025 |
| TD | Joint | +0.0064 | [−0.0058, +0.0189] | 25 / 45 | 0.328 |

**The overall in-domain gain is not distinguishable from zero at the image level.** The manuscript's "small in-domain improvement" must be qualified: the point estimate favours Phase 2, but the 95% interval includes zero. Only the ASD subset and the duration component clear zero, and these are unadjusted, exploratory subgroup results.

**E1. Statistical design of the τ=1 Phase 1 vs Phase 2 comparison.**
- **Null.** The Phase 1 contrast between two seeds (g=0 vs g=1) uses exactly the seeds of the two Phase 2 conditions. The conditioning effect is therefore the embedding's effect, net of the specific seed draws. Phase 1 seed-null contrasts are small, as expected: for example, dispersion +0.0009 and amplitude +0.0018 on Gaze4ASD.
- **Families.**
  - Primary family: 3 measures × 2 datasets, Holm.
  - Secondary families: the 9 contrasts within each dataset, the 18 match-to-real tests within each dataset, and the 10 MultiMatch tests.
  - Families were fixed in the experiment plan before the run.
- **Effect sizes.** *d_z* (mean / SD of per-image paired differences) is reported. Under common random numbers it measures *detectability*, not importance: dispersion *d_z* = 4.3 corresponds to a shift of 1.1% of the image diagonal. Practical size is reported as the share of the real contrast, and as absolute error relative to the shared miscalibration.
- **Monte Carlo uncertainty of descriptive shares.**
  - Single-seed Phase 2 contrasts vary widely; for example, the Gaze4ASD amplitude contrast ranges from +0.0059 to +0.0126 across CPU seeds.
  - The paper's GPU estimate and E1's CPU estimate of the same quantity differ by up to 0.0021 for amplitude (image-paired p = 0.050, Holm 0.45 over 9 contrasts; E1c).
  - Shares such as "60%" are therefore uncertain by several percentage points, and should be given as approximate, or with a seed-bootstrap interval.
- **Seed counts.** Per-seed sign agreement (for example 10/10 seeds) is a useful, distribution-free robustness check. It is reported alongside the image-level tests.

**E1c. CPU vs GPU sampling.**
- Two independent runs with the same seeds but different random streams agree in distribution: 18 group-by-measure means within 1.6%, smallest p = 0.061 unadjusted.
- Per-group NLL differs by at most 2e-4 nats per fixation (E2).
- No evidence of a platform effect was found.
- A residual risk remains: a platform effect smaller than about 2% of a group mean cannot be excluded with 10 seeds.

## 4. Conclusions that need more cautious wording

1. **In-domain gain.** Say "a small, statistically inconclusive in-domain gain (95% CI [−0.0012, +0.0163] across images)", not "improvement".
2. **External ASD gain** (+0.0104). It clears zero only by participants; by images and crossed, the intervals include zero. The paper already says so; keep it.
3. **Per-image correlations.** Report them as weak and uncertain, or add intervals.
4. **Real image-level contrasts** (manuscript Table 5). Their intervals treat images as independent, although the same participants contribute to many images. Use the participant-level results (Table 4) for inference.
5. **Policy B real effects.** Policy B removes more ASD data (see the additional results). Its larger total-time effect (*g* −1.08 vs −0.71) partly reflects that removal and should not be read as a stronger effect.
6. **MultiMatch.** Values come from a mean-based variant. See the MultiMatch validation for the one interpretation that changes.
7. **"Statistically significant" conditioning effects in E1.** Report them together with their absolute size. With common random numbers, Holm p ≤ 0.001 coexists with shifts of about 1% of the image diagonal. Do not describe them as large behavioural differences.
8. **Generated shares of real contrasts.** Give them as approximate (for example "about 50–65%"), not to the percentage point. They vary by several points between independent sampling runs.

## 5. What was deliberately not done

- No new hypothesis tests beyond those listed in the experiment plan, with one exception: the E1c CPU vs GPU tests. These were added after the merge, to investigate the agreement check (max |z| = 2.39). They are diagnostics, not findings.
- No search over metrics, thresholds or subsets.
- No mixed-effects models. With crossed participant × image structure and unbalanced groups, a mixed model would be the natural next step for a confirmatory analysis. It was not run here, to avoid adding another modelling choice without a preregistered plan.
