# Limitations and Unresolved Questions

## Limitations

| Area | Limitation |
|---|---|
| Conditioning | The in-domain gain is inconclusive (95% image-level interval includes zero). The external fit is worse overall, and the small external ASD gain depends on the coordinate mapping. |
| Generated behaviour | Fixation count and total fixation time are over-produced, and their group contrasts go in the wrong direction, as does direction concentration. No group's spatial realism improves. Generated contrast sizes carry Monte Carlo uncertainty of several percentage points. |
| Coordinate mapping | The Gaze4ASD fit-to-height mapping is strongly supported but not confirmed from the original presentation code. Position NLL, absolute spatial measures and MultiMatch depend on it. |
| Off-image gaze | It differs by group (ASD 7.35% vs TD 2.72% of fixations), so both clipping (Policy A) and removal (Policy B) bias group comparisons, in different ways. The model rarely generates off-image gaze. |
| Participant identity | Saliency4ASD has no participant IDs, and its sequences may not map one-to-one to participants. In-domain uncertainty can therefore be estimated only across images. |
| Participant count | Gaze4ASD's README reports 132 TD children, but the release contains 133 TD folders. |
| Age | Age is a potential confound within Gaze4ASD (the group means differ by 0.5 years) and across datasets (5–12 vs 3–7 years). Participant-level age is not available, so it **cannot be adjusted for**. |
| Independence and sample size | The Gaze4ASD stimuli were selected for group differences, may have been selected using Saliency4ASD, and are mostly faces. The external ASD group has 33 children, against 133 TD. |
| Cross-dataset shift | Age, eye tracker (Tobii T120 at 1280 × 1024 vs Tobii Pro Fusion at 2560 × 1440), viewing distance, stimuli, site and diagnostic procedure all differ at once. The reason the external TD fit gets worse cannot be attributed to any single factor. |
| Model selection | Validation images selected both the checkpoints and the temperature, and only τ = 1 and τ = 2 were compared. An earlier Phase 2 configuration (learning rate 10⁻³) has no saved results and is not reported. |
| Statistics | All analyses are exploratory and not preregistered. Holm correction applies within families, not across them. E1's primary measures were chosen on the basis of the earlier temperature run. |
| Implementation | The released transformer's batch-dimension attention was **kept for fidelity**, so scanpaths scored or generated together can influence each other (about 0.0001 nats on likelihood). MultiMatch uses mean aggregation, so its absolute values are implementation-specific. The research-completion analyses ran on CPU, and their per-trial values differ slightly from the GPU results (aggregates agree within 0.0002). Sampling with the same seed can differ across PyTorch versions and devices. |
| Literature access | The full texts of Zhong et al. (2026) and of the main Gaze4ASD paper were not accessible, so statements about them rely on the abstract, the README and the supplement. |
| Data terms | The Saliency4ASD licence record (CC BY 4.0) conflicts with its usage agreement, and the source of the Gaze4ASD stimuli is not stated. Both are pending clarification. |
| Clinical validity | None is established. This is not a diagnostic or screening tool. |

## Unresolved questions and who can resolve them

| Question | Who can resolve it |
|---|---|
| The display rectangle used for the Gaze4ASD stimuli (confirms or refutes fit-to-height) | Gaze4ASD authors |
| 132 vs 133 TD participants | Gaze4ASD authors, or the main paper |
| Participant-level age | Dataset authors (probably needs an ethics amendment) |
| Whether each Saliency4ASD sequence is one participant | Saliency4ASD maintainers |
| Which Saliency4ASD terms apply: research use, publishing derived statistics and figures, redistributing stimuli | Maintainers, supervisor or institution |
| Whether the earlier Phase 2 configuration (learning rate 10⁻³) was run, and on which split | Whoever ran it |
| Full texts of Zhong et al. (2026) and the Gaze4ASD paper | Supervisor or library (institutional access) |
| Why the TD embedding fits Gaze4ASD worse | Cannot be isolated with the released data |
| When to publish the locally rewritten history (force-push), which removes the older stimulus figures and record-level audits | Repository owner (see [`GIT_HISTORY_CLEANUP.md`](GIT_HISTORY_CLEANUP.md)) |

Details and the evidence for each item are in [`research_audit/UNRESOLVED_ISSUES.md`](research_audit/UNRESOLVED_ISSUES.md).
