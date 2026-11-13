# Reproducibility

## Environments

| Use | Environment |
|---|---|
| Training and the original evaluations | NVIDIA GPU, PyTorch 2.12.1 + CUDA 13.0 |
| Research-completion analyses (E1–E3, A3–A6), inference, demo and tests | Windows 11, Python 3.12.4, PyTorch 2.3.1 (CPU), NumPy 1.26.4, SciPy 1.13.0, OmegaConf 2.3.1 |

Install with `pip install -r requirements.txt` or `conda env create -f environment.yml`.

## What is fixed and recorded

**The image split.**
- **Where it is.** It is published in [`../results/in_domain/saliency4asd_image_split_seed0.json`](../results/in_domain/saliency4asd_image_split_seed0.json).
- **How it is made.** `training/splits.py` regenerates it exactly from seed 0, and `tests/test_data_logic.py` checks this.

**The checkpoints.**
- **Hashes.** Their SHA-256 hashes are listed in [`training_guide.md`](training_guide.md) and in `results/behavioural_analysis/e1_environment.json`.
- **Verified before and after every run.** The research-completion scripts and the demo check them before and after each run.

**Seeds.**
- **Fixed rules.** Every generation run uses a recorded seed rule, for example `100000·(replicate+1) + 1000·(image rank+1) + g` in E1.
- **Common random numbers.** Conditions are sampled with the same seed, so they share their random numbers.

**Provenance of every curated result.** Each file in `results/` has a source path and SHA-256 in a `PROVENANCE.csv`.

## Checks you can run

```bash
python -m unittest discover -s tests -t .                       # no dataset needed (some tests skip without checkpoints)
python inference/demo/run_demo.py --check                       # needs checkpoints and Saliency4ASD images; about 2.5 min on CPU
python scripts/collect_results.py --check                       # needs the local training/runs/ outputs
PYTHONIOENCODING=utf-8 python scripts/research_completion/z_consistency_check.py   # needs training/runs/research_completion/
```

| Check | What it verified in the 2026-11-13 cleanup |
|---|---|
| Unit tests | Repository hygiene, unchanged upstream TPP-Gaze, preprocessing rules, the Gaze4ASD mapping, the split, the metrics, the zero-embedding equivalence, checkpoint hashes and the 1,290 identical base tensors, and the headline numbers in the README |
| Demo check | All 35 saved demo scanpaths regenerated identically from the reorganised code. E1 sampling calls reproduced within 10⁻⁵. |
| Consistency check | 44 numbers cited in the research-audit documents recomputed from the saved outputs, with 0 problems |
| Trainer smoke tests | Phase 1 and Phase 2 trainers run from the new configuration paths |

## Known limits

**CPU vs GPU.**
- The research-completion re-scorings ran on CPU. Before trusting each one, a gate reproduced the saved GPU aggregates: within 7 × 10⁻⁵ nats in-domain and 2 × 10⁻⁴ externally.
- Per-trial values differ slightly between platforms (median 0.005 nats per trial), without bias.
- Generated samples agree in distribution, not sample by sample (E1c).

**Sampling across versions.** The same seed can give different scanpaths on a different PyTorch version or device, while the model outputs and likelihoods agree.

**Batch composition.**
- **Likelihood.** It can be affected slightly by the released transformer's batch-dimension attention (about 0.0001 nats).
- **Generation.** It depends on the number of scanpaths sampled together (20 in all experiments).

**DenseNet201 download.** It comes from `torch.hub` (`pytorch/vision:v0.10.0`), so a first run needs internet access.

## Reproducing everything from scratch

You need the datasets (see [`dataset_documentation.md`](dataset_documentation.md)) and the pretrained TPP-Gaze weights. Then:

1. Preprocess both datasets: [`preprocessing.md`](preprocessing.md).
2. Train Phase 1, then Phase 2: [`training_guide.md`](training_guide.md). Retrained models will differ slightly from the reported checkpoints, unless you use the original checkpoints.
3. Run the evaluations in the order given in [`evaluation_guide.md`](evaluation_guide.md).
4. Rebuild the summaries:
   ```bash
   python training/build_results_v2.py
   python scripts/collect_results.py
   ```
   Then compare them with `results/`.

## Path changes in the 2026-11 reorganisation

Raw outputs, the manuscript and older documents may still mention the old paths. The mapping is:

| Old path | New path |
|---|---|
| `inference.py` | `inference/inference.py` |
| `demo/` | `inference/demo/` |
| `test/input/`, `test/outputs/` | `inference/inputs/`, `inference/outputs/` (both Git-ignored) |
| `training/config.yaml` | `configs/phase1_baseline.yaml` |
| `training/config_phase2_embonly.yaml` | `configs/phase2_embedding_only.yaml` |
| `training/config_phase2.yaml` | `archive/configs/phase2_lr1e-3_no_saved_results.yaml` |
| `research_completion/scripts/` | `scripts/research_completion/` |
| `research_completion/outputs/` | `training/runs/research_completion/` (Git-ignored) |
| `research_completion/*.md` | `docs/research_audit/` |
| `results/external_gaze4asd_v2/` | `results/external_evaluation/` |
| `results/external_gaze4asd/`, `results/final_results/` | `archive/results/` |
| `analysis/run_sensitivity.py` | `preprocessing/run_sensitivity.py` |
| `analysis/` (other scripts) | `archive/code/early_gaze4asd_audits/` |
| `docs/*.txt` audits | `docs/audits/` |
| `docs/figures/make_architecture_diagram.py` | `scripts/make_architecture_diagram.py` |

The complete list is in [`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md).
