# Experiment Plan: Research Completion, Version 1

Written before running anything. Every experiment answers a specific question from the audit. No model is retrained, no checkpoint is changed, no test or external data is used for model or temperature selection, and existing outputs are left intact. New outputs go to `training/runs/research_completion/<experiment>_v1/`.

Hardware: the local GPU (RTX 3050) cannot be used, because the installed PyTorch is CPU-only, and I chose not to change your environment. All new experiments run on the local CPU. The remote GPU server is not used, so no files are synchronized.

## E1. Phase 1 vs Phase 2 generated behaviour at τ=1

**Question.** At the selected temperature, does group conditioning produce ASD–TD behavioural differences beyond seed noise? Does it match each real group better than unconditioned Phase 1?

**Why it is needed.** The paper's Phase 1 vs Phase 2 behavioural comparison exists only at τ=2, and τ=2 artificially narrows spatial spread. The final generation setting is τ=1.

**Design: common random numbers.** For each image, seed and group-seed g ∈ {0, 1}, four sets of 20 scanpaths are generated:

| Set | Model | Condition | Seed |
|---|---|---|---|
| P2_ASD | Phase 2 | ASD | seed(g=0) |
| P2_TD | Phase 2 | TD | seed(g=1) |
| P1_s0 | Phase 1 (zero embedding) | none | seed(g=0) |
| P1_s1 | Phase 1 (zero embedding) | none | seed(g=1) |

- Seeds follow the existing rule `100000*(rep+1) + 1000*(image_rank+1) + g`, with 10 replicates.
- Both models have bit-identical base weights. A Phase 2 set and the Phase 1 set with the same seed therefore differ only by the embedding.
- P1_s0 − P1_s1 is a seed-only null contrast.

**Settings.** Existing `sample_conditioned`, τ=1, 3 s window, 20 scanpaths per call, batch composition unchanged. Phase 1 uses the same sampler with an in-memory zero embedding, as in the earlier comparison.

**Data.**
- Gaze4ASD primary (fit-to-height, Policy A): 30 images, external.
- Saliency4ASD held-out test: 45 images, in-domain. The test set is used only for evaluation, as in earlier analyses; nothing is selected on it.

**Metrics.** Same definitions as the paper (`metric_row`, clipped coordinates): fixation duration, count, rate, total time, saccade amplitude, horizontal fraction, dispersion, revisit rate, direction concentration.

Primary, fixed now: the ASD–TD contrasts for dispersion, saccade amplitude and revisit rate. These are the measures whose real direction the model reproduces. All other measures are secondary.

MultiMatch (own implementation, Gaze4ASD only): similarity to the own group vs the other group.

**Statistics.** Image-level, paired, pooled over seeds; 10,000 bootstrap and sign-flip resamples; Holm within each family; per-seed sign counts.

**Gates.**
- Cached image features must reproduce uncached sampling.
- CPU Phase 2 must agree in distribution with the saved GPU τ=1 run: same seeds, so seed-level metric means should be within seed noise. CPU and GPU random streams differ, so exact equality is not expected.
- Checkpoint hashes are checked before and after.

**Cost.** 4 sets × 75 images × 10 seeds = 3,000 sampler calls on CPU. Image features are computed once per image. The sampler code itself is unchanged, and caching is verified by a gate. Estimate: 1–2 hours.

**Leakage.** None. No parameter, temperature or model choice depends on these results.

**Execution change (v1 aborted, v2 run).** The design, seeds, metrics and analysis are unchanged; only the execution changed.
- **v1 stopped.** The first attempt (`training/runs/research_completion/e1_phase_tau1_v1/`) ran one process with all 16 CPU threads. It used about 13,900 CPU-seconds in 28 minutes without finishing one replicate, which points to thread oversubscription. It was stopped. Its folder holds only config files, a log and an empty `generated/` folder, and no results were read from it.
- **Timing check.** `scripts/research_completion/e1_timing_check.py` timed one real sampler call at 1, 2, 4, 8 and 16 threads: 16.5, 8.6, 7.6, 13.6 and 16.0 s. The output was identical at every thread count.
- **v2 sharded.** `training/runs/research_completion/e1_phase_tau1_v2/` splits the 10 replicates over 5 processes, 2 replicates each, with 2 threads per process. Seeds depend only on replicate, image and group, so sharding cannot change any sample. A `merge` step combines the shards and checks that every replicate appears exactly once.
- **Shared base.** To fit in memory, each process builds Phase 1 as a zero-embedding wrapper around the same base object as Phase 2. The base weights are first asserted bit-identical to a separately loaded Phase 1 checkpoint. One feature cache then serves both models.
- **Revised cost.** About 3,000 calls at about 9 s each, over 3 concurrent processes (memory limit), so roughly 2.5–3 hours.
- **Actual run.**
  - Shards 0–2 started at 11:52–11:53 and shard 3 at 11:56. Each first Gaze4ASD replicate took about 46 minutes, because of contention and memory paging (about 900 pages read in per second, 1.7 GB free).
  - Shard 4 therefore started at about 12:39, concurrently, rather than after the others.
  - Between launches, only `analyze()` and its print lines were edited, adding d_z and the τ=2 reference. The `run()` sampling code is identical for all shards.

## E1b. τ=2 reference for E1 (added while E1 was running, before any E1 result was seen)

**Question.** Do the E1 τ=1 findings agree with the earlier τ=2 Phase 1 vs Phase 2 comparison?

**Why it is needed.**
- The τ=2 comparison used unclipped coordinates, while E1 uses clipped ones.
- Its Phase 1 set used an independent seed (g=2), so it has no common-random-number null.

**Design.**
- Recompute the τ=2 metrics from the saved τ=2 scanpaths with E1's metric code, clipped, without sampling.
- **Gate:** the unclipped recomputation must reproduce the saved τ=2 summaries.
- **Compare:**
  - the Phase 2 ASD−TD contrast and its share of the real contrast;
  - the Phase 1 vs Phase 2 match-to-real error by group, with the same paired tests and Holm family (18 tests) as E1.

**Result of the gate.**
- Real-data metrics are identical.
- Generated metrics match within 1.1e-4, except threshold-based measures: horizontal fraction up to 0.0083 and revisit rate up to 0.0033 in single cells.
- The cause is that the saved scanpaths are rounded to 5 decimals, the same rounding effect documented for the τ=1 run.

**Cost.** Seconds. **Leakage:** none.

## E1c. CPU vs GPU agreement diagnostic (added after the E1 merge)

**Trigger.** The planned agreement check reported max |z| = 2.39 over 18 group-by-measure comparisons. The largest was TD dispersion, a primary measure.

**Design.**
- Pair the per-image means of the CPU (E1) and GPU (saved τ=1) Phase 2 runs by image, and run a sign-flip test over 30 images.
- Also test each ASD−TD contrast difference, with both the image-paired test and a seed-level z.

**Status.** Diagnostic only; no conclusion of the paper depends on it. v1 was superseded by v2, which added the contrast test. Output: `training/runs/research_completion/e1c_cpu_gpu_check_v2/`.

## E1d. E1 MultiMatch under the reference implementation (added after E1, as part of the A5 validation)

**Why it is needed.** E1's MultiMatch results use our mean-based implementation. A5 had validated the τ comparison, but not E1's comparison against the seed-null.

**Design.**
- Reconstruct exactly E1's scanpath pairs from the saved scanpaths, using E1's pairing seeds.
- **Gate:** our implementation on these pairs must reproduce E1's stored values (tolerance 1e-9).
- Score the same pairs with `multimatch-gaze` 0.1.3, and repeat E1's paired analysis and Holm family.

**Run record.** The first attempt reloaded the coordinates in float64, and the gate failed at 3.7e-7. E1 had computed them in float32, the sampler's output type. The script was corrected to use float32, and the gate passed with a difference of 0.0. No output was written by the failed attempt.

**Cost.** CPU only; no sampling or model evaluation. **Leakage:** none.

## E2. Phase 1 vs Phase 2 NLL under alternative mappings and policies

**Question.** Is "Phase 1 better overall externally" robust to the mapping (stretch) and the out-of-image policy (B)?

**Why it is needed.** Phase 1 vs Phase 2 was scored only on the primary dataset.

**Design.**
- Score both models on stretch Policy A and fit-to-height Policy B, using the existing processed files, in file order with batch size 128, matching the V2 evaluation.
- Re-score the primary version too, as a regression gate.
- Paired cluster bootstrap over participants (stratified by group) and over images, 10,000 resamples.

**Gate.** CPU Phase 2 per-trial log-likelihoods must match the saved GPU values for every version (tolerance 1e-3 per trial).

**Gate change (recorded before rerunning).** The per-trial gate failed on the primary version, so it was changed.
- **Diagnosis:** 3,838 of 4,376 trials differ by more than 1e-3 in position log-likelihood (median 0.0049, maximum 0.089), but the differences are unbiased, and the aggregate differs by only 1.7e-4 nats per event.
- **Interpretation:** this is a platform-level numeric difference between CPU and GPU, probably GPU TF32 or cuDNN convolution arithmetic. That cause is a hypothesis, not tested.
- **New gate:** aggregate NLL within 5e-4 per event for every version, and the saved primary Phase 1 vs Phase 2 gain (−0.0172) reproduced within 5e-4.
- **Consequence:** E2's per-trial values are CPU values. Within a version, both models are scored on the same platform, so the comparison is internally consistent.
- The first, failed attempt left only two config files, which were removed before the rerun.

**Cost.** About 13,000 trials × 2 models on CPU, plus 30 image features. Minutes.

**Leakage.** None. The mapping was fixed before any model evaluation, and these runs cannot change it.

## E3. Uncertainty for the in-domain Phase 2 gain (added during the audit, before running)

**Question.** Is the Saliency4ASD held-out gain (+0.0073 nats per fixation) distinguishable from zero, given image-level variation?

**Why it is needed.** The paper reports this gain without a confidence interval, because per-scanpath log-likelihoods were not saved.

**Design.**
- Re-score the 903 test scanpaths with Phase 1 and Phase 2 on CPU, using the original order and batch size 128.
- Gate: reproduce the saved aggregates, Phase 1 −0.4957 and Phase 2 −0.5030, within 1e-4.
- Paired image-level cluster bootstrap and sign-flip test over the 45 test images. Saliency4ASD has no participant IDs, so images are the only available cluster.

**Cost.** Minutes on CPU.

**Leakage.** None. This is evaluation only, and no choice depends on it.

## A3. Behavioural sensitivity to mapping and policy (no model run)

- Real ASD vs TD participant effects in all three versions, read from the existing statistics.
- τ=1 vs τ=2 prespecified realism tests, recomputed with real values from each version. Generated scanpaths do not depend on the mapping, so the saved ones are reused.

## A4. Out-of-image policy accounting (no model run)

From the processed datasets: trials, fixations and participants removed by group under Policy B, and how group balance changes.

## A5. MultiMatch validation (no model run)

1. Compare our code line by line with the published method [Jarodzka et al. 2010; Dewhurst et al. 2012] and the reference Python implementation, `multimatch-gaze` 0.1.3 (Wagner et al.).
2. Reproduce the reference's MATLAB test case (`segment_10_sub-19` vs `sub-01`, no simplification) with our code, the reference code, and a median variant of our code.
3. Compare our scores with the reference on real and generated Gaze4ASD pairs.
4. Re-run the paper's MultiMatch conclusions (τ=1 vs τ=2, own vs other condition) with the reference implementation.

## A6. Statistical robustness (no model run)

1. Audit independence, clustering, interval construction, multiple-comparison correction and effect sizes in the existing scripts.
2. One targeted robustness check for the real ASD vs TD effects. Participants viewed different subsets of images (12–30 each). An image-centred analysis subtracts each image's overall mean before averaging per participant, then repeats the participant-level test. This addresses image-composition confounding.
3. No other new tests.

## Deliberately not run

| Proposal | Reason |
|---|---|
| Retraining Phase 1 or Phase 2 | No identified question requires it |
| Age-adjusted analysis | No participant-level age exists in the released data (see the audit) |
| New mappings or filtering rules | Not justified by evidence; would add researcher degrees of freedom |
| Intermediate temperatures | Would be selection on external data if judged on Gaze4ASD; a validation-only sweep could be planned separately |
| Re-running τ=2 Phase 1 vs Phase 2 | Already exists |
