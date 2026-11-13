# Interpreting the Results

All numbers come from [`../results/final/KEY_RESULTS.md`](../results/final/KEY_RESULTS.md), which also gives the source file of each. This page explains what the results mean and, just as importantly, what they do not mean.

## 1. Fine-tuning helped; the label did not improve likelihood

- **Fine-tuning helped clearly.** Phase 1 moves the held-out NLL from −0.1665 (released TPP-Gaze) to −0.4957, so adapting TPP-Gaze to child gaze data is useful.
- **The in-domain gain from the label is inconclusive.**
  - The point estimate is +0.0073 nats per fixation, and the 95% image-level interval is [−0.0012, +0.0163].
  - 27 of 45 images favour Phase 2 (sign-flip p = 0.11).
  - Read it as "no demonstrated gain", not as a small improvement.
  - Some subsets clear zero without correction: the ASD subset (+0.0086) and the duration component (+0.0019). With seven related comparisons, none survives correction.
- **On Gaze4ASD, the label makes the fit worse.**
  - Phase 1 is better by 0.0172. The TD trials get worse (−0.0229).
  - This holds in all three preprocessing versions, so it does not depend on how we prepared the data.
  - **Small ASD gain (+0.0104).** It clears zero only when resampling participants, and it depends on the mapping (+0.0027 under stretch). It is also not ASD-specific: the ASD embedding beats Phase 1 on TD trials too (post hoc).
- **Why external fit gets worse is untested.** The two datasets differ in age (5–12 vs 3–7 years), eye tracker, display, stimuli and site all at once. The Gaze4ASD stimuli were also selected for group differences.

## 2. The label changes generated behaviour, but not realism

**How the label effect is measured.** E1 compares the Phase 2 ASD−TD contrast with the difference between two Phase 1 runs on the same seed streams. That Phase 1 difference is what random seeds alone produce.

**What E1 shows:**
- **The label has a real effect.** 8 of 9 measures move beyond the seed-only baseline. The shifts are statistically clear but physically small: about 1% of the image diagonal.
- **Some directions are right.** Dispersion, saccade amplitude and revisit rate move in the real ASD−TD direction, at about 69%, 64% and 32% of the real difference.
  - **The shares are approximate.** An independent run gave 60%, 50% and 29%; the difference is Monte Carlo noise.
  - **The effects are uniform.** The generated ASD−TD dispersion difference is positive on all 45 Saliency4ASD test images, even on images where the real difference is negative. The label acts as a roughly uniform shift, not as an image-specific effect.
- **Some directions are wrong.** These are temporal measures and direction concentration:
  - With the ASD label the model makes more fixations and more total fixation time than with the TD label. Real ASD children spent *less* time fixating the image, and made slightly (not significantly) fewer fixations.
  - Direction concentration is also reversed.
- **No group becomes more realistic.**
  - No spatial measure moves significantly closer to its own real group, and ASD revisit rate gets worse.
  - The correct spatial contrasts arise because the two conditions **move apart**, not because each moves towards its group. The TD condition contracts in dispersion from 0.112 to 0.101, overshooting real TD (0.108), while the ASD condition stays at 0.113.
  - MultiMatch shows the same picture: real ASD scanpaths are, if anything, slightly closer to TD-conditioned samples.
- **Over-production is the largest error.**
  - Both labels produce about 10 fixations per scanpath, against about 6 in the real data.
  - The cause is that the sampler has no stopping model and fills the 3 s window.
  - The label's own shift in fixation count (+0.66) is about a fifth of this shared excess.

**Lesson for evaluating conditional scanpath models.** A correct group contrast is weak evidence of group-specific realism, so report per-group realism next to contrasts.

## 3. Temperature changes behaviour, not likelihood

- **What it changes.** Temperature only rescales the spatial mixture components when sampling, and NLL is computed without it.
- **Why τ = 2 narrows.** Every σ in this model is below 1, so τ = 2 narrows the spread rather than adding randomness.
- **Selection of τ = 1.** It was selected on validation images. On Gaze4ASD it reduces saccade-amplitude and revisit-rate error by 50–66% compared with τ = 2.
- **Effect on contrasts.** The contrast directions are the same at τ = 2, but their sizes shrink (46%, 33% and 15%).
- **Practical rule.** Any report of generated behaviour must state the temperature, and generated behaviour is never evidence about likelihood.

## 4. Real ASD vs TD differences

- **What differs.** On Gaze4ASD, ASD children show, compared with TD children:
  - fewer revisits;
  - more dispersion;
  - less total fixation time;
  - larger saccades;
  - more concentrated saccade directions;
  - about 2.7× more off-image gaze.
- **Robustness.** These effects hold under the alternative mapping and off-image policy.
- **Not adjusted for age.** The group means differ by 0.5 years, and no participant-level age exists.
- **What these are.** They are descriptive group differences in one dataset, not markers of autism.

## 5. What the results do not show

- **No diagnostic or screening use.** No classifier was built or tested, and no clinical validation exists.
- **Not about individual children.** A generated scanpath is a sample from a group-level model.
- **No comparison with other models.** Nothing here shows that this model is better or worse than other ASD scanpath models; no such comparison was run.
- **No mechanism.** The generated differences do not establish any biological or clinical mechanism.

## 6. How strong each finding is

See Section 5 of [`../results/final/KEY_RESULTS.md`](../results/final/KEY_RESULTS.md), and Q8 in [`research_audit/RESEARCH_COMPLETION_AUDIT.md`](research_audit/RESEARCH_COMPLETION_AUDIT.md).
