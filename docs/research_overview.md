# Research Overview

## Objective

The project asks what happens when a neural temporal point process model of gaze is told a viewer's group.
- **The model:** TPP-Gaze.
- **The groups:** autistic children (ASD) and typically developing children (TD).
- **What "told" means:** the model receives a minimal ASD/TD label.

It does not try to build the best possible ASD scanpath model. Instead it is a controlled test of one small change.

## Background and gap

**Why model scanpaths.**
- **What summaries lose.** Aggregate gaze measures and static saliency maps lose the order and timing of fixations.
- **What scanpath models keep.** Scanpath models predict sequences.
- **What TPP-Gaze adds.** It predicts both *where* the next fixation lands and *how long* it lasts, as a marked temporal point process.

**What was missing.** TPP-Gaze has no group input.
- **Why not two models.** Training separate ASD and TD models would change every weight. A difference between such models could come from the group, or from training noise and different data sizes.
- **What we use instead.** A single shared model with a small label keeps everything else fixed.

## Research questions

| | Question | Answer (short) |
|---|---|---|
| Q1 | Does the label improve the likelihood of held-out Saliency4ASD scanpaths? | **Not demonstrably.** The gain is +0.0073 nats per fixation, with a 95% CI that includes zero. |
| Q2 | Does it improve the likelihood on Gaze4ASD, an external dataset? | **No.** Phase 1 is better by 0.0172, in all three preprocessing versions. |
| Q3 | Do generated ASD and TD scanpaths differ like real children do? | **Partly.** Spatial contrasts point the right way at reduced size; temporal and direction-concentration contrasts point the wrong way; neither group becomes more realistic. |
| Q4 | Does the conclusion depend on the sampling temperature? | **The direction does not; the sizes do.** Temperature never changes the likelihood. |
| Q5 | Are the results robust to the Gaze4ASD coordinate mapping and off-image policy? | **Mostly.** The weak external ASD gain is the exception. |
| Q6 | Is age a confound, and can it be adjusted for? | **Possibly, and no.** Participant-level age is not released. |

The full answers, with their evidence, are in [`research_audit/RESEARCH_COMPLETION_AUDIT.md`](research_audit/RESEARCH_COMPLETION_AUDIT.md) (Section 2).

## Approach

1. **Phase 1.** Fine-tune the released TPP-Gaze on Saliency4ASD (ASD and TD together, no label). This adapts the model to child gaze data and is the control.
2. **Phase 2.** Add a zero-initialised `nn.Embedding(2, 512)` to the fused context, freeze everything else, and train only the 1,024 embedding values.
   - At zero initialisation Phase 2 is numerically identical to Phase 1.
   - All 1,290 base tensors are identical in the two checkpoints, so every measured difference comes from the label.
3. **Compare Phase 1 and Phase 2** on the same data and the same random seeds:
   - **likelihood:** in-domain, and external under three preprocessing versions;
   - **generated behaviour:** ASD−TD contrasts against a seed-only baseline from Phase 1, and the realism of each group against its own real group.

## Our contribution compared with TPP-Gaze

**What is new:**
- a minimal ASD/TD conditioning of TPP-Gaze;
- an evaluation that keeps likelihood and generated behaviour separate;
- an unconditioned control on common random numbers;
- an external dataset with a corrected coordinate mapping and sensitivity versions;
- per-group realism checks alongside group contrasts.

**What we do not claim:**
- that this is better than other ASD scanpath models (no comparison was made);
- that this is the first work on ASD scanpath prediction;
- any diagnostic or clinical use.

**Related work.** Zhong et al. (*Pattern Recognition*, 2026) predict ASD scanpaths with semantic information. Only their abstract could be accessed, so no comparison is made.

## Main findings

See [`../results/final/KEY_RESULTS.md`](../results/final/KEY_RESULTS.md) and [`results_interpretation.md`](results_interpretation.md).

## Future directions

1. **Confirm the facts behind the external data.** Ask the Gaze4ASD authors for:
   - the display rectangle (to confirm the mapping);
   - the TD participant count (132 vs 133);
   - participant-level age.
2. **Model viewing time and stopping**, so the sampler does not fill the 3 s window. This would address fixation over-production, the largest error.
3. **Model off-image gaze**, which is a real group difference that the model cannot currently express.
4. **Compare richer conditioning against the same baseline.** For example, conditioning in other positions or with more parameters, judged against the same unconditioned control and per-group realism criteria.
5. **Replicate on independent data** with preregistered tests.
