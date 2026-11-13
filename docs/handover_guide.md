# Technical Handover

**For:** a researcher taking over the ASD-ScanPath project.
**State:** 2026-11-13. Research analyses complete, with a final manuscript draft that is not in this repository (see "Decisions pending" below).

## 1. What has been completed

| Work | Status | Where |
|---|---|---|
| Saliency4ASD preprocessing | Done | `preprocessing/preprocess_dataset.py`, [`preprocessing.md`](preprocessing.md) |
| Gaze4ASD preprocessing with corrected mapping, plus two sensitivity versions | Done (mapping partly inferred) | `preprocessing/preprocess_gaze4asd_v2.py` |
| Phase 1 fine-tuning | Done | `training/train_tpp.py`, `configs/phase1_baseline.yaml` |
| Phase 2 embedding-only training | Done | `training/train_tpp_cond.py`, `configs/phase2_embedding_only.yaml` |
| In-domain likelihood, with image-level CI (E3) | Done: inconclusive | `results/in_domain/` |
| External likelihood under three versions (E2) | Done: Phase 2 worse | `results/external_evaluation/`, `results/sensitivity_analysis/` |
| Real ASD vs TD effects on Gaze4ASD, with robustness checks | Done | `results/external_evaluation/`, `results/sensitivity_analysis/` |
| Behavioural comparison against a seed-only baseline (E1), per-group realism, MultiMatch validation | Done | `results/behavioural_analysis/` |
| Temperature selection (validation) and external check | Done: τ = 1 | `results/temperature_analysis/`, `results/external_evaluation/temperature_sensitivity.txt` |
| Research-completion audit and unresolved issues | Done | [`research_audit/`](research_audit/README.md) |
| Inference tool and reproducible demo | Done | `inference/` |
| Supervisor presentation and speaking scripts | Done | `presentation/` |
| Final manuscript draft | Draft, ready for supervisor review | Local only (`paper/final/`); not published |

The findings in one paragraph are in the [README](../README.md#13-main-verified-results). The full numbers, with sources, are in [`../results/final/KEY_RESULTS.md`](../results/final/KEY_RESULTS.md).

## 2. Where things are

| Need | Go to |
|---|---|
| The model change | `training/conditioned.py` (wrapper), `training/model_utils.py::tppgaze_nll` (where the embedding is added), `training/generate_cond.py` (sampler) |
| Original TPP-Gaze | `tppgaze/`, unmodified upstream code |
| Training settings | `configs/` |
| Evaluation code | `training/eval_external_v2.py`, `compare_phase1_phase2.py`, `temperature_sensitivity.py`, `eval_gaze4asd_tau1.py`, `stats_external_v2.py`, and `scripts/research_completion/` |
| Behavioural measures and MultiMatch | `training/metrics.py`; `training/eval_external_v2.py::metric_row` |
| Numbers for a paper or slide | `results/final/KEY_RESULTS.md`, then the folder it cites |
| Why a decision was made | [`research_audit/`](research_audit/README.md), [`limitations.md`](limitations.md), `archive/README.md` for superseded work |
| Each folder | Its own `README.md` |

## 3. How to run the project

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .              # works without data
```

**With the datasets in place** ([`dataset_documentation.md`](dataset_documentation.md)):

```bash
python preprocessing/preprocess_dataset.py
python preprocessing/preprocess_gaze4asd_v2.py
python training/train_tpp.py train
python training/train_tpp_cond.py train
# evaluations: see docs/evaluation_guide.md
```

**With the checkpoints in place:**

```bash
python inference/inference.py --image <image> --condition both
python inference/demo/run_demo.py --check
```

## 4. Checkpoints required

| File | Needed for | SHA-256 |
|---|---|---|
| `tppgaze/data/model_transformer.pth` | Phase 1 training (initial weights) | Public: [TPP-Gaze release v1.0.0](https://github.com/phuselab/tppgaze/releases/tag/v1.0.0) |
| `training/runs/baseline/best_model.pth` | Phase 2 training, Phase 1 evaluation, `--compare-phase1` | `b613413027e775b98e449cbdbc9afa0253d6bd40f0cb2502230ed5bf1000b0bb` |
| `training/runs/phase2_embedding_only/best_model_conditioned.pth` | All Phase 2 evaluation, inference and the demo | `35319673382378487d23c1871cab0cf05b71b1e7963e180ec024e5d80901a9a4` |

The trained checkpoints are **not in the repository**; get them from the project owner. You also need DenseNet201 ImageNet weights, which `torch.hub` downloads automatically.

## 5. Datasets required

| Dataset | Needed for | Location expected by the code |
|---|---|---|
| Saliency4ASD | Preprocessing, training, in-domain evaluation, demo | `TrainingData/TrainingData/` |
| Gaze4ASD | External evaluation | `external_data/Gaze4ASD/` |

The data terms are unresolved. **Never commit dataset content, participant records or stimulus images.** `tests/test_repository_hygiene.py` guards against this.

## 6. Results already obtained

See [`../results/README.md`](../results/README.md). In short:
- **Phase 1 fine-tuning** improves the likelihood clearly.
- **The ASD/TD label:**
  - its in-domain gain is inconclusive;
  - its external fit is worse;
  - it changes generated behaviour, with some spatial directions correct at reduced size, several directions wrong, and no group more realistic.
- **Temperature** affects behaviour, not likelihood.

## 7. What remains to be investigated

1. **Ask the Gaze4ASD authors** about the display rectangle, the TD participant count (132 vs 133) and participant-level age.
2. **Model viewing time and stopping** to fix fixation over-production (about 10 generated vs 6 real per scanpath), the largest error.
3. **Model off-image gaze**, which differs by group and which the model cannot generate.
4. **Test why the TD embedding fits Gaze4ASD worse.** This needs data that separate age, tracker and stimuli.
5. **Compare richer conditioning against the same baseline:** other injection points or more parameters, judged against the same unconditioned control and per-group realism.
6. **Replicate** on independent data with preregistered tests.

## 8. Known issues to be aware of

- **Released-code quirk.** The released TPP-Gaze transformer attends across the batch dimension. It is kept for fidelity; do not "fix" it without retraining, or results will change.
- **Temperature.** τ = 1 is the default in our code; the released config uses 2. Always state τ when reporting generated behaviour.
- **Gaze4ASD mapping.** Treat the fit-to-height mapping as strongly supported but partly inferred.
- **Superseded results.** Anything based on the old native-size mapping is superseded (`archive/`).
- **Overwrite behaviour.** Most evaluation scripts refuse to overwrite their output folder. `build_results_v2.py` deletes and rewrites `results/external_evaluation/`.
- **Seeds across versions.** The same seed can give different samples on another PyTorch version or device; likelihoods agree.
- **Windows consoles.** Set `PYTHONIOENCODING=utf-8` for scripts that print "−".
- **Presentation builds.** The presentation build scripts need Node.js with `pptxgenjs` 3.12 and a local theme helper (`PPTX_SKILL_DIR`). They are not needed for the research code.
- **History rewrite.** Restricted material from the October 2026 commits has been removed from the **local** history ([`GIT_HISTORY_CLEANUP.md`](GIT_HISTORY_CLEANUP.md)). After the force-push, collaborators must re-clone rather than pull.

## 9. Decisions pending (repository owner)

- **Licence** for this project's own code. None has been chosen; TPP-Gaze stays MIT.
- **Citation file** (`CITATION.cff`), once manuscript authorship is decided.
- **Publishing the manuscript** (`paper/`, currently Git-ignored).
- **Releasing the trained checkpoints.**
- **Force-push of the locally rewritten history**, which removes the stimulus figures and record-level audits from the published history; see [`GIT_HISTORY_CLEANUP.md`](GIT_HISTORY_CLEANUP.md), Section 7.
