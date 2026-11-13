# `training/` package

The code is imported as the `training` package (for example `from training.conditioned import ConditionedTPPGaze`). Run scripts from the repository root.

**Raw outputs** go to `training/runs/`, which Git ignores because it holds checkpoints and per-trial data. The curated summaries are in [`../results/`](../results/README.md).

## Model and data (imported by everything else)

| File | Role |
|---|---|
| `conditioned.py` | **Our addition.** `ConditionedTPPGaze` wraps the unmodified TPP-Gaze and adds a zero-initialised `nn.Embedding(2, 512)` (ASD = 0, TD = 1). |
| `model_utils.py` | Builds the official TPP-Gaze with our duration statistics, loads checkpoints, freezes and caches the backbone, and computes the joint NLL. The optional group vector is added to the fused context here. |
| `generate_cond.py` | Conditioned sampler; mirrors the official `TPPGaze.sample()` and adds the embedding at each step |
| `dataset.py` | Saliency4ASD scanpaths, images and batching |
| `splits.py` | Reproducible image-level 70/15/15 split |
| `external_gaze4asd_loader.py` | Gaze4ASD loader (stimuli by filename, strict validation) |
| `metrics.py` | Behavioural measures and our MultiMatch implementation |

## Training

| File | Role |
|---|---|
| `train_tpp.py` | Phase 1 trainer: `smoke`, `short`, `train` (default config `configs/phase1_baseline.yaml`) |
| `train_tpp_cond.py` | Phase 2 trainer: `smoke`, `short`, `train` (default config `configs/phase2_embedding_only.yaml`) |

## Evaluation and statistics

| File | Role |
|---|---|
| `eval_external_v2.py` | Gaze4ASD evaluation of the frozen Phase 2 model under three versions; also generates scanpaths at τ = 2 |
| `stats_external_v2.py` | Statistics and figures for the above |
| `gen_replicates_v2.py` | Monte Carlo replicates of the generated contrasts (τ = 2) |
| `compare_phase1_phase2.py` | Phase 1 vs Phase 2 on Gaze4ASD: `run`, then `verify`, then `analyze` |
| `temperature_sensitivity.py` | Temperature selection on Saliency4ASD **validation** images: `run`, then `windowed`, then `analyze` |
| `eval_gaze4asd_tau1.py` | Gaze4ASD behaviour at τ = 1 vs τ = 2: `run`, then `analyze` |
| `analyze_phase2.py` | First in-domain behavioural analysis (τ = 2). Its conclusions are superseded by E1; its outputs are in `archive/results/in_domain_behaviour_tau2_initial/`. |
| `build_results_v2.py` | Builds `results/external_evaluation/` from local outputs |

**The follow-up experiments** (E1–E3, A3–A6) are in [`../scripts/research_completion/`](../scripts/README.md).

**Superseded scripts** (old Gaze4ASD mapping) are in [`../archive/code/`](../archive/README.md).

**Commands, inputs and outputs** for all of the above: [`../docs/evaluation_guide.md`](../docs/evaluation_guide.md).
