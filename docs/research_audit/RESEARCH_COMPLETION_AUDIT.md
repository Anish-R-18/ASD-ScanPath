# Research Completion Audit

**Scope:** everything the manuscript draft (`paper/`) claims, checked against local data, code and outputs, plus the analyses in the experiment plan. Detailed numbers are in the additional results, statistical validation and MultiMatch validation documents. Open items are in the unresolved issues document.

**Constraints:**
- No retraining, and no checkpoint, dataset or earlier output changed.
- The test split was used for evaluation only.
- The remote server was not touched.
- Nothing was committed or pushed.

**Classes:**
- **A:** resolvable with local data and code (resolved here)
- **B:** needed a new experiment or analysis (run here)
- **C:** needs external information or confirmation
- **D:** cannot be resolved with the released data

## 1. Issue register

| # | Issue | Class | Action | Status |
|---|---|---|---|---|
| 1 | Phase 1 vs Phase 2 behaviour compared only at τ=2, not at the selected τ=1 | B | E1: common random numbers at τ=1, Gaze4ASD and Saliency4ASD test; E1b τ=2 reference; E1c CPU vs GPU check | **Resolved.** Conditioning shifts behaviour systematically; spatial contrasts are correct in direction, temporal ones wrong; neither group's spatial match improves |
| 2 | No uncertainty for the in-domain Phase 2 gain | B | E3: re-scoring plus image cluster bootstrap | **Resolved: CI includes zero** |
| 3 | External Phase 1 vs 2 NLL only under the primary mapping | B | E2: stretch and Policy B | Resolved: overall and TD robust; ASD position depends on mapping |
| 4 | Real ASD vs TD effects under alternative mapping and policy | A | A3 re-analysis | Resolved: robust |
| 5 | τ=1 benefit under alternative mapping and policy | A | A3 re-analysis | Resolved: 4/4 prespecified tests under Policy B, 3/4 under stretch |
| 6 | What Policy B removes, by group | A | A4 accounting | Resolved: removes ASD data disproportionately |
| 7 | Image-composition confounding of real effects | A | A6 image-centred test | Resolved: no effect |
| 8 | MultiMatch implementation correctness | A | A5 line-by-line comparison, MATLAB test case, reference re-run; E1d reference re-run of E1 | Resolved: mean vs median discrepancy; one temperature interpretation flips; all E1 conclusions unchanged |
| 9 | Statistical units, intervals, multiplicity | A | Code audit | Resolved, with wording changes listed |
| 10 | CPU re-computation vs original GPU results | A | Gates in E2, E3 and E1c | Resolved. Aggregates within 2e-4 nats; sampling agrees in distribution; contrast differences within Monte Carlo noise |
| 11 | Age confounding | D | Raw files searched: no age field | Cannot adjust; disclose |
| 12 | Coordinate mapping confirmation | C | Evidence re-read (saliency-map r 0.9999 vs 0.91 stretch vs 0.23 native); dependence quantified by E2 and A3 | Needs the authors or their presentation code |
| 13 | 132 vs 133 TD | C | Raw files confirm 133 | Ask the authors, or check the main paper |
| 14 | Zhong et al. and Gaze4ASD full texts | C | Access attempted, blocked (no institutional access) | Supervisor or library |
| 15 | Saliency4ASD sequence-to-participant link | C | Counts recomputed | Ask the maintainers |
| 16 | Earlier Phase 2 configuration (learning rate 1e-3) without results | C | Searched locally: no outputs | Ask whoever ran it |
| 17 | Dataset terms: Saliency4ASD (CC BY 4.0 vs usage agreement), Gaze4ASD (README says MIT; stimulus source unstated) | C | Terms read; conflict documented; nothing redistributed | Supervisor, institution or maintainers |
| 18 | Stimulus photographs in public repository history | C | Not changed (no commits allowed) | Owner's decision |
| 19 | Cause of the worse external TD fit | D | Cannot be isolated (age, tracker, stimuli, site confounded) | Disclose |
| 20 | Intermediate temperatures | — | Deliberately not run (see the experiment plan) | Optional future validation-only sweep |
| 21 | Over-production of fixations; no stopping model | D (for this study) | Quantified in E1 | Model limitation; would need a new model component |

## 2. Answers to the research questions

**Q1. Does ASD/TD conditioning improve in-domain likelihood?**
- **Not demonstrably.** The point estimate favours Phase 2 by +0.0073 nats per fixation, but the 95% image-level interval is [−0.0012, +0.0163]: 27/45 images favour Phase 2, sign-flip p = 0.11 (E3).
- Unadjusted sub-results clear zero: the ASD subset (+0.0086, [+0.0011, +0.0162]) and the duration component (+0.0019, [+0.0002, +0.0035]). With seven related rows, none would survive correction.
- **Verdict: inconclusive.** "Modest improvement" must become "a small, statistically inconclusive gain".

**Q2. Does it improve external likelihood?**
- **No.** On Gaze4ASD, Phase 2 is worse than Phase 1 overall (−0.0172; participant CI [−0.0225, −0.0122]; image CI [−0.0323, −0.0024]) and for TD (−0.0230). This holds in all three preprocessing versions, by participants (E2).
- The ASD subset is slightly better (+0.0104): it clears zero by participants only, and depends on the mapping (position gain about 0 under stretch). Its duration part (+0.0029) is robust.
- The ASD gain is not ASD-specific: the ASD embedding also beats Phase 1 on TD trials, according to the earlier post hoc label-swap analysis.
- **Verdict: strong negative result.** Conditioning does not improve external likelihood.

**Q3. Does it reproduce meaningful ASD–TD behavioural differences?**
- **Partly, and only in the sense of group contrasts** (E1, τ=1, common random numbers).
- **Real effect?** Yes. The embedding systematically changes generated behaviour: 8 of 9 measures shift beyond the Phase 1 seed-null on both datasets, all Holm p ≤ 0.001.
- **Spatial measures.** For all three primary spatial measures, the shift has the real direction on both datasets: dispersion, saccade amplitude and revisit rate, at 69%, 64% and 32% of the real Gaze4ASD contrast.
- **Wrong direction.**
  - The ASD condition generates more fixations and more total time than the TD condition, the opposite of the real data.
  - Direction concentration is wrong on both datasets.
  - Mean duration and rate shift about 3× the non-significant real differences.
- **Per-group realism.**
  - Phase 2 does not bring either group's spatial behaviour closer to its real group: no spatial measure improves significantly, and ASD revisit worsens (−11.8%, Holm 0.014).
  - It improves TD temporal behaviour (count +10%, rate +23%, duration +28%) and worsens ASD temporal behaviour (−8% to −33%).
  - The correct contrasts therefore come from the two conditions moving apart, not from each moving towards its real group.
- **MultiMatch.** Real ASD scanpaths are slightly *less* similar to ASD-conditioned than to TD-conditioned samples on position (−0.0062), whereas real TD is more similar to TD-conditioned samples (+0.0074). Both effects are negligible in size.
- **Statistical vs practical.** The shifts are statistically unambiguous (*d_z* 1–5) but small: 0.8–1.1% of the image diagonal. They are dwarfed by the shared over-production of about 3.5 fixations per scanpath.
- **Verdict.** Conditioning reproduces the *direction* of the main real spatial contrasts at reduced size. It does not reproduce temporal or direction-concentration contrasts, and it does not make either group's samples more realistic. "Meaningful" applies only to the spatial contrast direction.

**Q4. Does the conclusion depend on sampling temperature?**
- **The qualitative conclusion does not; the sizes do.**
- At τ=2 and τ=1, the Phase 2 contrast has the same direction on all nine measures (E1b).
- τ=1 roughly doubles the amplitude and revisit shares and raises the dispersion share: amplitude 33 → 64%, revisit 15 → 32%, dispersion 46 → 69%. These are E1 CPU values; the GPU τ=1 values are 50%, 29% and 60%.
- One τ=2 finding does not hold at τ=1: the significant TD spatial-match penalty (−21% to −33%, Holm ≤ 0.003) shrinks to non-significant values (−9% to +13%). The TD temporal improvement becomes clearer.
- Separately, τ=1 is more realistic than τ=2 on all four prespecified tests (A3). NLL does not depend on temperature.
- τ was selected on validation data only, and that remains the correct procedure.

**Q5. Are results robust to alternative coordinate mappings and filtering policies?**
- **Robust:**
  - all five significant real effects;
  - Phase 1 better overall and for TD;
  - the duration components;
  - the τ=1 benefit for ASD measures and TD revisit.
- **Not robust:**
  - the external ASD position gain, about 0 under stretch;
  - the TD amplitude and dispersion τ benefits;
  - the size of count and total-time effects under Policy B, which removes off-image ASD data. ASD children have about 2.7× more off-image gaze, a group difference itself.
- **Generated scanpaths** do not depend on the mapping. The dispersion share varies (60–81%); amplitude and revisit shares are stable.
- **Primary choice.** Fit-to-height with Policy A stays primary: strongest evidence, all data kept, fixed before modelling.

**Q6. Is age a potential confound, and can it be adjusted for?**
- **It is a potential confound, and it cannot be adjusted for.**
- Neither release has participant-level age: Gaze4ASD gives only group means (ASD 5.04, TD 4.54 years, range 3–7), and Saliency4ASD gives a range (5–12, mean 8).
- **Within Gaze4ASD,** a 0.5-year mean difference of unknown spread could contribute to real group effects.
- **Across datasets,** age is confounded with tracker, display, stimuli and site.
- This is a disclosure item; only the dataset authors could resolve it.

**Q7. Is the MultiMatch implementation sufficiently validated?**
- **For relative comparisons within one implementation: yes, with one disclosed difference.**
  - It follows the published algorithm. The difference is that it aggregates with the mean where the reference toolbox uses the median, and it allows 2-fixation scanpaths.
  - With the median it reproduces the MATLAB toolbox test case to 3e-6.
- **Conclusions under the reference implementation.**
  - Temperature conclusions and the position own-vs-other signs are unchanged.
  - The ASD-direction "closer to real-vs-real" reading reverses.
  - TD duration own-vs-other loses its interval.
  - All E1 conditioning conclusions are unchanged on the same pairs (E1d).
- **For absolute values or comparison with other papers: not without switching to the reference.**

**Q8. Which findings are strong, weak, exploratory or inconclusive?**

| Strength | Finding |
|---|---|
| **Strong** (robust to resampling unit, preprocessing, platform) | Phase 2 does not improve overall external NLL, and TD external NLL is worse. Real ASD vs TD spatial differences (revisit, dispersion, amplitude, direction concentration) and total time. The embedding systematically changes generated behaviour. Spatial contrast directions are reproduced at reduced size; count, total-time and direction-concentration contrasts are wrong. τ=1 is more realistic than τ=2 for ASD measures and revisit. Fixation over-production. |
| **Moderate** | τ=1 TD amplitude benefit (3/3 versions, CI touching 0 under stretch). TD temporal match improves with conditioning (both datasets, both temperatures). ASD temporal match worsens. |
| **Weak or preprocessing-dependent** | External ASD joint gain (participant level only; mapping-dependent). TD dispersion τ benefit. Size of the generated share of real contrasts (Monte Carlo spread of several points; dispersion share 60–81% across versions). |
| **Inconclusive** | In-domain Phase 2 gain. Any statement about why the external TD fit worsens. |
| **Exploratory only** | Post hoc label-swap analysis. In-domain subgroup and component results. All MultiMatch condition-specificity results. Per-image correlations. Policy B and stretch values. |
| **Not testable with released data** | Age effects. Participant-level in-domain uncertainty. Clinical relevance (none claimed). |

## 3. Cross-document consistency check (done for this audit)

- **Mapping evidence.** A draft of the unresolved issues document gave r = 0.23 as the stretch value. 0.23 is native centred placement; stretch is 0.91. This is corrected in both documents, with the source file cited.
- **τ tests.** A draft answer said "5 of 6"; the prespecified family has four tests. Corrected.
- **Shares.** E1 (CPU) shares (69/64/32%) differ from the paper's GPU shares (60/50/29%) by Monte Carlo noise (E1c: no contrast difference survives Holm). Both are reported as such. The paper's values are not replaced silently.
- **Rounding.** NLL is reported to 4 decimals, *g* and *d_z* to 2, p-values to 2–4 significant digits. Each document states its conventions.
