# Research Q&A Preparation: ASD-ScanPath

This document has 43 likely questions, grouped by topic. Each has:
- **Say:** a short spoken answer (about 20–40 s).
- **If they push:** the deeper technical answer for a follow-up.
- **Source:** where the evidence is. "§" means a section of `paper/final/ASD_ScanPath_Final.md`.

## How to answer
1. **Number first, then meaning.** For example: "Plus 0.0073, but the interval includes zero, so it's inconclusive."
2. **Don't defend weak results.** If something is inconclusive or negative, say so plainly. That is the strength of this work.
3. **Say "we didn't test that" when it's true.** Then say what *would* test it. Never guess at a cause on the spot.
4. **Never move toward diagnosis.** The model describes group-level statistics, not people.

---

## A. Motivation and novelty

### Q1. Why study ASD gaze patterns at all?
**Say:**
> Eye-tracking studies report that autistic and typically developing children differ in where they look, although the differences depend on the stimulus. If a model can reproduce those differences, that tells us something about what the model has captured. Our interest is in modelling, not in diagnosis.

**If they push:** In our own analysis of Gaze4ASD (33 ASD, 133 TD children, one value per child), five measures differ significantly after Holm correction:

| Measure | Hedges' g |
|---|---|
| Revisit rate | −1.04 |
| Spatial dispersion | +0.80 |
| Total fixation time | −0.71 |
| Saccade amplitude | +0.65 |
| Direction concentration | +0.55 |

- Fixation duration, count, rate and horizontal fraction do not differ significantly.
- These five keep their direction and significance under both alternative preprocessing versions.

**Source:** §1, §2.3, §6.5, Table 6; refs [1, 2].

### Q2. Why a generative model instead of a classifier?
**Say:**
> A classifier only tells you whether groups can be separated. A generative model has to reproduce *how* gaze unfolds: where, in what order, for how long. So we can score it on real held-out data with likelihood, and compare its generated behaviour with real behaviour, measure by measure. We deliberately did not want to build a classifier.

**If they push:** We use the model in two modes (§3.1):
- *Likelihood evaluation* scores observed scanpaths, using the observed history and no temperature.
- *Behavioural generation* samples scanpaths autoregressively.

The second mode lets us separate two questions: whether group *contrasts* are right, and whether each group's *realism* improves.

**Source:** §3.1, §5.

### Q3. What is the actual research contribution?
**Say:**
> A controlled test of minimal ASD/TD conditioning in an existing gaze model. Phase 1 and Phase 2 have identical base weights, so every difference comes from 1,024 parameters. We tested likelihood in-domain and on an external dataset, compared generated behaviour with common random numbers, and showed that a correct group contrast can appear without either group becoming more realistic.

**If they push:** The four listed contributions (§1) are:
1. Minimal conditioning: a zero-initialized embedding added before both heads of a frozen model.
2. In-domain and external likelihood, with cluster-level uncertainty and sensitivity analyses for the coordinate mapping and off-image gaze.
3. A common-random-numbers behavioural comparison that separates the embedding's effect from seed noise, and contrasts from per-group realism.
4. A temperature analysis showing that temperature changes generated differences but not likelihood.

**Source:** §1.

### Q4. What is new compared with TPP-Gaze?
**Say:**
> Architecturally, only the 1,024-parameter group embedding. Everything else is TPP-Gaze's released code and pretrained weights, unmodified. What's new is the question and the evaluation. We do not claim to improve on TPP-Gaze, and on the external data we actually fit worse.

**If they push:** Table 1 separates the published method, the released code and our choices. Our own choices are:
- duration normalization computed from our training split;
- τ = 1, chosen on validation data (the released configuration uses τ = 2);
- a minimum of 3 fixations (the TPP-Gaze paper required 4);
- the zero-initialized 2 × 512 embedding.

**Source:** §3.2, Table 1, §4.3.

### Q5. How does this relate to SpFormer and Zhong et al.?
**Say:**
> SpFormer is a transformer scanpath model that includes fixation duration in its fixation features; it isn't about autism. Zhong et al. predict ASD scanpaths with a semantic-map approach. We could only read their abstract, so we make no claims about how their method compares. Our design is different: a minimal label on an existing joint model, with an unconditioned control and an external test. We don't claim to be better than either.

**If they push:**
- According to its abstract, Zhong et al. use a two-branch framework with global spatial descriptors and semantic maps, a dynamically updated difference-of-semantics map, n-reference transfer learning, and two ASD scanpath datasets built by the authors.
- We tried to get the full text twice, and it was not accessible. So we cannot say how it handles duration or external data.

**Source:** §2.1, §2.3, §2.4; `docs/research_audit/UNRESOLVED_ISSUES.md` §1.

### Q6. Isn't a mostly negative result a weak contribution?
**Say:**
> I'd argue it's a clear answer to a clear question. Because the base weights are identical, the worse external fit can only come from the embedding, which makes the negative result unambiguous. And the lesson that a correct group contrast doesn't mean per-group realism applies to any conditional gaze generator.

**If they push:** The audit classes these findings as **strong**, meaning robust to resampling unit, preprocessing and platform:
- no external likelihood improvement, and a worse TD fit;
- the embedding systematically changes generated behaviour;
- spatial contrast directions are reproduced at reduced size;
- count, total-time and direction-concentration contrasts are wrong;
- fixations are over-produced.

The in-domain gain is the only main result classed as inconclusive.

**Source:** `docs/research_audit/RESEARCH_COMPLETION_AUDIT.md`, Q8; §7.

---

## B. Architecture

### Q7. Why temporal point processes?
**Say:**
> Fixations happen at irregular times, and how long each one lasts matters. A temporal point process treats each fixation as an event in continuous time and predicts the time to the next event as a probability distribution, given everything so far. That matches gaze naturally, and it gives a proper likelihood for both *where* and *how long*.

**If they push:**
- It is a *marked* point process: the duration is the inter-event time, and the location is the mark.
- TPP-Gaze models the inter-event time directly with a log-normal mixture, the intensity-free approach of Shchur et al. [16], instead of an intensity function.
- The objective has no survival term for the end of the sequence. At sampling time there is no stopping model either, which is relevant to over-production (Q34).

**Source:** §2.2, §3.1, §3.7, Table 1.

### Q8. Why a Gaussian mixture for location?
**Say:**
> The next fixation could go to several places, like a face, a sign or an object. A mixture of 16 Gaussians can put probability on several targets at once, which a single Gaussian can't. This is TPP-Gaze's design, and we kept it unchanged.

**If they push:**
- The mixture has J = 16 components, diagonal covariance, tanh on the means (so they stay inside [−1, 1]) and softmax weights.
- All mixture weight has σ < 1 in normalized coordinates (median RMS σ = 0.131 on validation images). That is why τ = 2 narrows the components rather than widening them.

**Source:** §3.6, §3.8, §6.8, Table 1.

### Q9. Why a log-normal mixture for duration?
**Say:**
> Durations are always positive and skewed: mostly short, with some long ones. Modelling the log of the duration with a mixture handles both, and it's straightforward to evaluate and to sample from. Again, this is inherited from TPP-Gaze, following Shchur et al.

**If they push:**
- K = 4 components over log-durations.
- The log-durations are normalized by the training-split mean (−1.4366) and standard deviation (0.6936), and the log-scales are clamped to [−5, 3].
- Equation 3 in the paper gives the density.
- Durations are never tempered.

**Source:** §3.6, §4.4, Table 1.

### Q10. Why add only a 1,024-parameter embedding?
**Say:**
> Because our question was what a *minimal* group label can do. A tiny, zero-initialized addition to a frozen model means any difference is attributable to the label. It was a deliberate choice for a clean test, not a claim that this is the best way to condition.

**If they push:**
- The embedding is a 2 × 512 table, and the selected row is added to the 512-d context before both heads at every step.
- A constant added vector cannot easily express *image-specific* group differences. That is consistent with the roughly uniform shift we see across images (Q38), but we did not test it as a cause.
- Richer conditioning, evaluated against the same unconditioned control and per-group criteria, is future work.

**Source:** §3.5, §9.

### Q11. Why freeze the base model?
**Say:**
> So that the comparison is clean. The embedding starts at zero, so Phase 2 begins exactly equal to Phase 1, and we verified that all 1,290 base tensors are identical in both checkpoints. That means Phase 1 is exactly Phase 2 with a zero embedding, and any difference comes only from the 1,024 parameters.

**If they push:**
- In Phase 2 the base is in evaluation mode, so there is no dropout, and only the embedding is trained, at learning rate 10⁻⁴.
- Without freezing, a difference could come from refitting the base model, and we could not attribute it to the group label.

**Source:** §3.5, §4.4.

### Q12. Could full fine-tuning work better?
**Say:**
> Possibly, but we didn't test it, so I can't say. It would also change the question: if the whole network is retrained with the label, you can't separate the label's effect from refitting. If we tried it, we'd keep the same unconditioned control and check per-group realism, not just contrasts.

**If they push:**
- Our external result is a caution. Even 1,024 parameters improved validation NLL (−0.5845 to −0.5964) and still made the external fit worse. Fitting the training groups better does not guarantee transfer.
- Disclose if relevant: another Phase 2 configuration (learning rate 10⁻³) exists in the repository, but its results could not be found locally, so it is not reported.

**Source:** §4.4, §6.1, §6.3.

### Q13. Where exactly does the embedding enter, and why there?
**Say:**
> It's added to the 512-dimensional fused context, the joined image and history features, immediately before both prediction heads. That way one vector can shift both where the next fixation goes and how long it lasts, and it's applied at every generation step.

**If they push:**
- The equation is c̃ₙ = cₙ + e_ℓ.
- The embedding does not enter the image or history encoders, so it does not change how the history is encoded, only the input to the heads.

**Source:** §3.5, §3.8, Figure 1.

### Q14. What is the "batch-layout issue", and does it affect your results?
**Say:**
> The released transformer is configured for sequence-first input but receives batch-first tensors, so its attention mixes across the scanpaths in a batch instead of along each sequence. Temporal order comes from the LSTM. We kept it because the pretrained weights were learned with it, and we disclose it.

**If they push:**
- **Likelihood.** On Gaze4ASD, shuffling batch composition changed overall NLL by about 0.0001 or less.
- **Generation.** The 20 scanpaths sampled together share an image and label and can influence each other.
- **Demo.** The demo samples one scanpath at a time (n_sim = 1), so its samples come from a slightly different procedure than E1's batches of 20. The demo's `--check` reproduces two saved E1 calls at n_sim = 20 within 1e-5.

**Source:** §3.4, §8 (Implementation); `inference/demo/DEMO_GUIDE.md` §5.

---

## C. Datasets and preprocessing

### Q15. Why use Saliency4ASD and Gaze4ASD?
**Say:**
> Saliency4ASD is a public dataset with ASD and control children viewing 300 natural images. That's enough to fine-tune the model and still hold out test images. Gaze4ASD is independent, with a different site, eye tracker, age range and stimuli. That makes it a real external test, so we used it only for evaluation.

**If they push:**
- Gaze4ASD is not fully independent. Its 30 stimuli were selected for large ASD/TD differences and are mostly smiling faces.
- The supplement also describes fine-tuning a saliency model on Saliency4ASD, so the stimulus selection may depend on our training data.

**Source:** §4.1, §4.2, Table 2.

### Q16. Why are the participant counts inconsistent?
**Say:**
> The Gaze4ASD README says 132 TD children, but the release has 133 TD participant folders, and all of them contribute trials. We analysed the data as released and report the discrepancy; only the authors can resolve it. Saliency4ASD has a similar issue: the README says 14 children per group per image, but some images have up to 17 sequences.

**If they push:**
- The raw Saliency4ASD files hold 8–17 (ASD) and 9–15 (TD) sequences per image, separated by index resets.
- We treat each sequence as one scanpath, but a sequence may not be one participant.
- There are no participant IDs, so in-domain uncertainty can only be estimated across images.

**Source:** §4.1, §4.2, §8.

### Q17. Why is the coordinate mapping partly inferred?
**Say:**
> Gaze4ASD records gaze in screen pixels on a 2560 × 1440 display, but it doesn't record where each image was drawn on the screen. So we had to infer the placement. The evidence strongly supports fit-to-height, but without the original presentation code we can't confirm it.

**If they push:**
- **Fit-to-height:** scale = 1440/H, horizontally centred (Eqs. 7–9).
- **Evidence:**
  - The authors' own saliency maps match this geometry at median r = 0.9999 over 4,533 maps, against 0.91 for stretch and 0.23 for native-size placement.
  - 96.5% of fixations fall inside the image, against 81.4% for native placement.
  - Face-landmark scale estimates are closer to fit-to-height, though that estimator reads low.
- An earlier native-size analysis was contradicted by the data and is superseded; none of its values are used.

**Source:** §4.3; `UNRESOLVED_ISSUES.md` §2.

### Q18. Why fit-to-height rather than stretch?
**Say:**
> Because it matches the data best, and we fixed it as primary before evaluating any model. Gaze alone can't rule out stretch, so we kept stretch as a sensitivity analysis. The main conclusions hold under both.

**If they push:** Under stretch:

| Group | Gain |
|---|---|
| Overall | −0.0274 |
| TD | −0.0336 |
| ASD | +0.0027 (interval includes zero) |

- So "Phase 1 better overall" and "TD worse" survive under stretch.
- The ASD gain does not. Its position part goes from +0.0075 under fit-to-height to −0.0003 under stretch, so it is mapping-dependent.

**Source:** §4.3, §6.4, Table 5.

### Q19. What happens to out-of-image fixations?
**Say:**
> Our primary policy clips them to the image edge; the sensitivity policy removes them. Neither is neutral, because autistic children looked outside the image about 2.7 times as often as TD children. The main likelihood conclusions hold under both policies.

**If they push:**
- **How often.** Under clipping, 7.35% of ASD fixations and 2.72% of TD fixations lie outside the image. At participant level that is 0.073 vs 0.027, a difference of +0.046 (95% CI [+0.020, +0.074]).
- **What removal does.** It deletes 8.07% of ASD fixations and 3.03% of TD fixations, so it removes more ASD data. That is partly why the Policy B total-time effect looks larger.
- **Generated data.** The model rarely produces off-image gaze; only 0.25–0.44% of generated fixations needed clipping.

**Source:** §4.3, §6.4, §3.8.

### Q20. Could age differences confound the results?
**Say:**
> Yes, potentially, and we can't adjust for it. Within Gaze4ASD the group mean ages differ by about half a year (5.04 vs 4.54 years). Across datasets the ranges are 5–12 versus 3–7. Neither release has individual ages.

**If they push:**
- Real ASD–TD effects could partly reflect age.
- The cross-dataset shift is confounded with age, eye tracker, display, stimuli and site all at once. That's why we don't attribute the worse external TD fit to any single factor.
- Participant-level age would have to come from the dataset authors.

**Source:** §4.2, §7, §8; `UNRESOLVED_ISSUES.md` §3.

### Q21. Why a minimum of 3 fixations, when TPP-Gaze used 4?
**Say:**
> That was our own preprocessing choice, and the paper states that it differs from TPP-Gaze's threshold of 4. We didn't run a sensitivity analysis on it, so I can't say whether 4 would change anything.

**Source:** §4.3.

---

## D. Experiments and statistics

### Q22. Why is the in-domain result inconclusive?
**Say:**
> The gain, +0.0073 nats per fixation, is small compared with how much it varies between images. The 95% interval runs from −0.0012 to +0.0163, which includes zero. Only 27 of 45 images favour Phase 2, and the sign-flip test gives p = 0.11. With 45 test images, we can't distinguish it from no effect.

**If they push:**
- Validation NLL improved more (−0.5845 to −0.5964). But validation images were used to select the checkpoint, so that number is optimistic. The test set is the fair estimate.
- The ASD subset (+0.0086, [+0.0011, +0.0162]) and the duration component (+0.0019, [+0.0002, +0.0035]) clear zero only without correction. With seven related rows, none would survive correction.

**Source:** §6.1, §6.2, Table 3; `RESEARCH_COMPLETION_AUDIT.md` Q1.

### Q23. Why is the external likelihood worse?
**Say:**
> Honestly, we don't know why; we didn't test causes. What we do know is that the TD embedding fits *both* groups worse than having no embedding. The two datasets differ in age, tracker, display, stimuli and site all at once, so we can't attribute it to one factor.

**If they push:** A post hoc label-swap analysis scored each embedding on each group:

| | Real ASD trials | Real TD trials |
|---|---|---|
| ASD embedding vs Phase 1 | +0.0104 | +0.0189 |
| TD embedding vs Phase 1 | −0.0571 | −0.0229 |

- The correct label beats the wrong one by +0.0128 (participant CI [+0.0049, +0.0212]), almost entirely in position.
- The ASD gain is therefore not ASD-specific.
- Also, Gaze4ASD stimuli were selected for group differences and are mostly faces.

**Source:** §6.3, §7.

### Q24. What do the confidence intervals mean here?
**Say:**
> They're 95% bootstrap intervals. We resample whole images, or whole children, 10,000 times and recompute the gain each time. If the interval includes zero, the data are compatible with no effect. We resample clusters rather than fixations, because fixations from the same child or image are correlated.

**If they push:**
- The bootstraps are *paired*: both models are scored on the same resample.
- Externally we resample participants (stratified by group), images, or both crossed.
- In-domain there are no participant IDs, so we use images plus a sign-flip test.
- Intervals are unadjusted unless stated; Holm correction is applied within prespecified families.

**Source:** §5.1, §5.5.

### Q25. Why use identical random seeds?
**Say:**
> So that the only thing that changes between outputs is the model, not luck. The ASD label, the TD label and Phase 1 all draw from the same random-number streams. Phase 1 on two different seed streams tells us how big a difference randomness alone produces. We subtract that "seed-only null" to get the conditioning effect.

**If they push:**
- Effect = (P2 ASD − P2 TD) − (P1 stream A − P1 stream B).
- For dispersion: the Phase 2 contrast is 0.0117 and the null is 0.0009, giving an effect of 0.0107.
- Because the shared seeds remove most of the noise, d_z values are large (1–5) even for shifts of about 1% of the image diagonal. d_z measures how detectable an effect is, not how large it is.

**Source:** §5.3, §5.5, Table 7.

### Q26. What does the CPU/GPU reproducibility check establish?
**Say:**
> Training and the original evaluations ran on a GPU; the newer analyses ran on CPU. Before trusting any CPU result, we checked that it reproduced the saved GPU numbers: within 0.00007 nats per fixation in-domain and 0.0002 externally. So the platform doesn't change our likelihood conclusions.

**If they push:**
- **Per-trial values.** They differ slightly between platforms (median 0.005 nats per trial), without bias.
- **Generation.** We compared E1 (CPU, τ = 1) with the earlier GPU temperature run at τ = 1. All 18 group means agree within 1.6%, and no ASD−TD contrast differs after Holm correction.
- **What that implies for the shares.** Single-seed contrasts vary, so the shares carry several points of Monte Carlo uncertainty: 69/64/32% in E1 vs 60/50/29% in the GPU run.
- **What it does *not* establish.** It does not show that samples are bit-identical across platforms. Demo outputs are reproducible on the same CPU and torch build.

**Source:** §4.5, §5.3, §6.6; `ADDITIONAL_RESULTS.md` (E1c).

### Q27. Why are some findings called exploratory?
**Say:**
> Nothing was preregistered. The three primary E1 measures were chosen because an earlier run had reproduced them. The label-swap analysis was post hoc. And the τ = 2 results on Gaze4ASD had been seen before the temperature tests. So we report these as exploratory evidence, not confirmation.

**If they push:**
- Holm correction is applied within families, not across them.
- With 10,000 resamples, the smallest attainable p-value is about 0.0001.
- Per-image correlations between real and generated contrasts were not analysed with intervals and are not used.

**Source:** §5.3, §5.5, §8.

### Q28. How was MultiMatch validated?
**Say:**
> Our implementation follows the published algorithm but averages aligned differences with the mean, where the reference toolbox uses the median. Switched to the median, it reproduces the original MATLAB test case to within 3 millionths. We also re-scored all E1 comparisons with the reference Python implementation, and every conclusion held.

**If they push:**
- **Effect of using the mean.** On 2,992 Gaze4ASD pairs, the mean variant scores 0.011–0.034 lower on shape, direction and length.
- **E1 re-scoring.** 407 of 72,000 pairs were excluded by the reference's 3-fixation minimum. All 8 significant effects kept their sign and significance (for example, position +0.0084 and −0.0069 under the reference).
- **Changed reading.** One temperature reading (ASD direction "closer to real-vs-real") flipped under the reference, and was dropped.
- **Scope.** Absolute values are implementation-specific, so we use MultiMatch only for within-implementation comparisons.

**Source:** §5.4, §6.7; `MULTIMATCH_VALIDATION.md`.

### Q29. Your d_z values go up to 5. Aren't those huge effects?
**Say:**
> No. d_z is the paired difference divided by its standard deviation across images. With common random numbers the noise is tiny, so even small shifts give large d_z. It measures how *detectable* an effect is, not how big it is. That's why we also report absolute sizes (about 1% of the image diagonal) and shares of the real contrast.

**Source:** §5.5, §6.6; `STATISTICAL_VALIDATION.md` §4.

### Q30. Why τ = 1 when the official TPP-Gaze configuration uses τ = 2?
**Say:**
> We chose it on Saliency4ASD validation images. τ = 1 gave lower per-image error for saccade amplitude and revisit rate, and it's the model's own distribution, the same one the likelihood scores. Gaze4ASD was not used to choose it.

**If they push:** Validation error was lower at τ = 1 by:

| Measure | ASD | TD |
|---|---|---|
| Saccade amplitude | 35.6% | 15.6% |
| Revisit rate | 64.9% | 41.2% |

Only τ = 1 and τ = 2 were compared; no intermediate values were tested.

**Source:** §6.8, §8 (Model selection).

---

## E. Results and interpretation

### Q31. Why do some behavioural differences have the correct direction?
**Say:**
> The embedding moves the two labels apart along the spatial dimensions where real children differ. ASD-labelled samples become more spread out than TD-labelled ones, with larger saccades and fewer revisits. But that happens mostly because the TD condition contracts, not because the ASD condition moves. We haven't tested *why* the embedding learned this direction.

**If they push:**
- **Dispersion.** The ASD condition stays at 0.113 in both models. The TD condition moves from 0.112 to 0.101, past real TD at 0.108.
- **Saliency4ASD.** The primary contrasts again follow the real direction: amplitude 54% and revisit rate 23% of real. Dispersion overshoots a real contrast whose interval touches zero.

**Source:** §6.6, §6.7.

### Q32. Why do other measures move in the wrong direction?
**Say:**
> Most of them are about timing: fixation count and total fixation time. Direction concentration is wrong too. The sampler has no stopping or viewing-time model and fills the 3-second window, so it over-produces fixations for both labels, and the ASD label increases that. Real autistic children actually spent *less* time fixating the image. One possible contributor, which we haven't tested, is off-image gaze, which the model essentially can't produce.

**If they push:**

| Measure | Real ASD−TD | Phase 2 contrast | Notes |
|---|---|---|---|
| Fixation count | −0.159 (not significant) | +0.593 | Conditioning effect +0.662 |
| Total fixation time | −0.0822 s | +0.0210 s | |
| Direction concentration | +0.0270 | −0.0052 | |

Mean duration and fixation rate shift about three times their (non-significant) real differences.

**Source:** §6.6, Table 7, §7.

### Q33. Why doesn't individual-group realism improve?
**Say:**
> A contrast measures the *distance between* the groups; realism measures whether each group is *in the right place*. You can get the distance right by moving one group too far. That's what happens here: TD contracts past real TD, and ASD doesn't move. So the gap looks right, but neither group got closer to reality spatially.

**If they push:** Table 8 shows the change in per-image error against the matching real group:
- No spatial measure improves significantly for either group.
- ASD revisit rate gets worse: −11.8%, Holm p = 0.0140.
- Timing moves TD toward real TD (count +10.0%, rate +22.6%, duration +28.2%) and ASD away from real ASD (count −8.5%, rate −28.1%, duration −32.5%).
- MultiMatch agrees. Real ASD scanpaths are slightly *less* similar to ASD-conditioned samples in position (−0.0062); real TD scanpaths are more similar to TD-conditioned ones (+0.0074). All effects are below 0.01 on a 0–1 scale.

**Source:** §6.7, Table 8, §7.

### Q34. Why does the over-production of fixations matter?
**Say:**
> Because it's the biggest error in the generated behaviour. Both labels produce about 10 fixations per scanpath, against about 6 in real data. The conditioning shift in count is only +0.66, about a fifth of that shared excess. So any group comparison on timing is dominated by a calibration problem common to both labels.

**If they push:**
- Generated: 10.19 (ASD label) and 9.60 (TD label). Real: 6.25 and 6.41.
- The excess is largely because the sampler fills the 3 s window. Even inside the real viewing window, it remains +1.23 and +1.00 fixations (temperature run).
- Modelling stopping, viewing time and off-image gaze is the first item of future work.

**Source:** §6.7, §7, §9.

### Q35. Does temperature improve the model?
**Say:**
> No. Temperature only changes how we *sample*. The likelihood is computed without it, so the model's fit is the same at any τ. τ = 1 gave more realistic samples than τ = 2 on our four prespecified tests, but that's a sampling choice, not a better model.

**If they push:** On Gaze4ASD, error at τ = 1 was lower than at τ = 2 by:

| Measure | ASD | TD |
|---|---|---|
| Saccade amplitude | 52.5% | 49.9% |
| Revisit rate | 66.1% | 56.5% |

- Each was in 10 of 10 seeds, Holm p ≤ 0.0005. The tests are exploratory, because τ = 2 results had been seen.
- All four hold under Policy B; three of four hold under stretch.
- At τ = 2, every E1 contrast keeps its direction but shrinks, to 46%, 33% and 15%.

**Source:** §3.8, §6.8, Table 9.

### Q36. What exactly does "69%, 64% and 32%" mean?
**Say:**
> Take the real ASD-minus-TD difference in spatial dispersion. The model's generated ASD-minus-TD difference is about 69% as large, with the same sign. For saccade amplitude it's about 64%, and for revisit rate about 32%. So: right direction, smaller size. And they're approximate; another independent run gave 60, 50 and 29.

**If they push:**
- **How they are computed.** Phase 2 contrast ÷ real image-level contrast on Gaze4ASD: 0.0117/0.0169, 0.0096/0.0151 and −0.0261/−0.0805. The seed-only null is near zero (for example, 0.0009 for dispersion).
- **Seed spread.** Single-seed shares vary a lot: for amplitude, 0.39 to 0.83.
- **Absolute size.** The dispersion and amplitude shifts are 1.1% and 0.8% of the image diagonal.

**Source:** §6.6, Table 7, Figure 5.

### Q37. Can the model distinguish ASD from TD reliably?
**Say:**
> No, and we never claimed it could. We didn't build or evaluate a classifier. The in-domain likelihood gain is inconclusive, and externally the ASD embedding even fits TD children better than no embedding. This is not a diagnostic model.

**If they push:**
- The only related number is post hoc: the correct label beats the wrong label by +0.0128 nats per fixation on Gaze4ASD, almost all in position. That is tiny, and it was never tested as classification.
- In the demo, single samples vary more from seed to seed than between labels.

**Source:** §6.3, §8 (Clinical validity).

### Q38. Is the generated ASD−TD shift specific to each image?
**Say:**
> No, it looks roughly uniform. On all 45 Saliency4ASD test images, the model's generated ASD-minus-TD dispersion difference is positive, between +0.002 and +0.020. The real per-image difference ranges from −0.047 to +0.058, so on some images the model has the opposite sign; image 116 in the demo is one of them.

**If they push:**
- The real per-image differences are themselves noisy, with only 6–14 viewers per group.
- The paper doesn't use per-image correlations, because they were not analysed with intervals.

**Source:** `inference/demo/demo_results/manifest.json`, `inference/demo/DEMO_GUIDE.md` §4; §8 (Statistics).

### Q39. Why do the three demo panels look almost the same?
**Say:**
> Because that's what the design does. Same seed, same frozen base model; the only difference is 1,024 numbers. The group effect is real but small, around 1% of the image diagonal on average, and it shows up only over many samples and images, not in one picture.

**Source:** §6.6; `DEMO_GUIDE.md` §5.

---

## F. Limitations and future work

### Q40. Is this a diagnostic tool?
**Say:**
> No. It's a research model of group-level gaze statistics. It has no clinical or diagnostic validity, it was never evaluated as a classifier, and a generated scanpath does not describe any person.

**Source:** Abstract, §1, §8.

### Q41. What prevents clinical use?
**Say:**
> Almost everything about the study design. There's no clinical validation, and the model wasn't built or tested for classification; it only uses group labels. The external ASD sample is 33 children, age is a confound, the statistics are exploratory, and the data terms are still unresolved. Most importantly, conditioning didn't even improve the external fit. Clinical use would need a completely different kind of study.

**Source:** §8.

### Q42. What would be needed for stronger generalization?
**Say:**
> First, confirm the Gaze4ASD display mapping and participant count with the authors. Then get participant-level age, so it can be adjusted for. Beyond that, we'd need more ASD participants, independent data with preregistered tests, and a model that handles stopping and off-image gaze.

**Source:** §8, §9.

### Q43. How would you improve the model, and what is the next experiment?
**Say:**
> First, fix the shared calibration problem, because over-production is the largest error: model viewing time, stopping and off-image gaze. Then I'd rerun the same E1 comparison, with the same seed-only null and the same per-group realism tests, ideally preregistered. Only after that would I try richer conditioning, against the same unconditioned control.

**If they push:** That ordering is my proposal, based on the evidence. The manuscript's future-work list (§9) contains the same items:
- confirm the mapping and participant count;
- obtain age;
- model stopping and off-image gaze;
- compare richer conditioning against the same control and criteria;
- replicate with preregistered tests.

**Source:** §9.

---

## G. If you don't know the answer

> That's a good question, and I don't want to guess. We didn't test that directly. What we *can* say is [closest supported fact], and testing it properly would need [the experiment].

**Questions only you can answer.** Prepare your own answers to these before the talk:
- What part of the work did you do yourself?
- What was the hardest problem, and how did you solve it?
- What would you do differently if you started again?
