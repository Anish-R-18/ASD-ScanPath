# Research-Completion Audit

**What these documents are.** They record the final phase of the research.
- **What was checked.** Every claim of the first manuscript draft was checked against the data, code and outputs.
- **What was added.** The analyses it was missing (E1–E3, A3–A6) were run.
- **What was settled.** Each research question was answered with a stated strength of evidence.
- **Ground rules.** No model was retrained, and no checkpoint or dataset was changed.

| Document | Contents |
|---|---|
| [`RESEARCH_COMPLETION_AUDIT.md`](RESEARCH_COMPLETION_AUDIT.md) | Issue register; answers to the research questions Q1–Q8; strength of each finding |
| [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) | Design of E1 (behaviour against a seed-only baseline), E1b–E1d, E2 (mapping/policy NLL), E3 (in-domain CI) and A3–A6 |
| [`ADDITIONAL_RESULTS.md`](ADDITIONAL_RESULTS.md) | Full results of these experiments |
| [`STATISTICAL_VALIDATION.md`](STATISTICAL_VALIDATION.md) | Units of analysis, intervals, multiplicity and wording corrections |
| [`MULTIMATCH_VALIDATION.md`](MULTIMATCH_VALIDATION.md) | Our MultiMatch implementation against the reference (`multimatch-gaze`) |
| [`UNRESOLVED_ISSUES.md`](UNRESOLVED_ISSUES.md) | What cannot be settled with the released data, and who could settle it |

**Where the files are now.**
- **Scripts.** In [`../../scripts/research_completion/`](../../scripts/README.md).
- **Raw outputs.** Local and Git-ignored, in `training/runs/research_completion/`.
- **Curated summaries.** In [`../../results/`](../../results/README.md).
- **How paths are written.** These documents now use the repository paths. The original documents were written in a `research_completion/` folder, and the mapping between old and new paths is in [`../reproducibility.md`](../reproducibility.md).
- **Changelog.** A changelog of the manuscript revision stays with the unpublished manuscript.
