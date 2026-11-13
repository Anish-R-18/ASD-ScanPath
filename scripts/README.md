# Scripts

## `research_completion/`: follow-up experiments and checks

These scripts produced the final research-completion results. They:
- read the checkpoints and data without changing them;
- run on CPU;
- write raw outputs to `training/runs/research_completion/<experiment>/`, which Git ignores;
- **refuse to overwrite** an existing output folder.

Designs are in [`../docs/research_audit/EXPERIMENT_PLAN.md`](../docs/research_audit/EXPERIMENT_PLAN.md), results in [`../docs/research_audit/ADDITIONAL_RESULTS.md`](../docs/research_audit/ADDITIONAL_RESULTS.md).

| Script | Experiment |
|---|---|
| `e3_indomain_gain_ci.py` | E3: image-level 95% CI and sign-flip test for the in-domain Phase 2 gain |
| `e2_mapping_policy_nll.py` | E2: Phase 1 vs Phase 2 NLL under the three Gaze4ASD versions |
| `e1_phase_tau1.py` | E1: label effect on generated behaviour at τ = 1, against a Phase 1 seed-only baseline. Commands: `run <shard> 5` (×5 in parallel), then `merge`, then `analyze`. |
| `e1_timing_check.py` | Checks that sampling does not depend on the CPU thread count (used to plan E1) |
| `e1b_tau2_reference.py` | E1b: the saved τ = 2 samples re-scored with E1's metric code |
| `e1c_cpu_gpu_check.py` | E1c: agreement of E1's CPU samples with the saved GPU τ = 1 run |
| `e1d_multimatch_reference.py <src>` | E1d: E1 MultiMatch re-scored with `multimatch-gaze` 0.1.3 |
| `a5_multimatch_validation.py <src>` | A5: our MultiMatch implementation against `multimatch-gaze` 0.1.3 and the MATLAB test case |
| `a346_sensitivity_and_stats.py` | A3: real effects and the τ benefit under each mapping and policy. A4: what Policy B removes, by group. A6: image-centred robustness. |
| `z_consistency_check.py` | Recomputes the 44 key numbers cited in the research-audit documents and checks that each appears in its document |

**MultiMatch scripts.** `<src>` is a folder that contains the `multimatch_gaze` package source (version 0.1.3).

**MultiMatch test data.** `external_reference/` holds the two test scanpaths from the `multimatch-gaze` repository, used by A5. They are distributed under its MIT licence (`LICENSE_multimatch_gaze.txt`).

## Other scripts

| Script | Purpose |
|---|---|
| `collect_results.py` | Copies the curated summaries from `training/runs/` into `results/`, with provenance (`--check` only verifies) |
| `make_architecture_diagram.py` | Draws `docs/figures/architecture_diagram.png` and `.svg` |
