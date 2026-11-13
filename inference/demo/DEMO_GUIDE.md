# ASD-ScanPath Demo Guide

This demo shows how the ASD/TD-conditioned TPP-Gaze model (Phase 2) generates scanpaths for one image under the ASD and TD conditions. A scanpath is an ordered sequence of fixations, each with a location and a duration. The demo also shows the unconditioned Phase 1 model for comparison.

**This is a research demonstration. It is not a diagnostic or clinical tool.** A generated scanpath says nothing about any individual, and the model has no clinical validity.

## 1. Requirements

**Environment.** The demo was tested on Windows 11 with:
- Python 3.12.4
- torch 2.3.1+cpu and torchvision 0.18.1+cpu
- numpy 1.26.4
- matplotlib 3.8.4
- Pillow 11.0.0
- omegaconf 2.3.1
- scipy 1.13.0

It needs no GPU and no new dependencies.

**Network.** The network is needed only once. The image encoder (DenseNet201) is built through `torch.hub`, which must find `pytorch/vision v0.10.0` and the DenseNet weights in `~/.cache/torch/hub/`. Both are cached on this machine, and the demo was tested with the network blocked. On a new machine, run any command once while online.

**Checkpoints.** Both must be present and unchanged. They are git-ignored, so copy them separately to a new machine. `run_demo.py` refuses to run if a hash differs.

| Model | Path | SHA-256 |
|---|---|---|
| Phase 1 (unconditioned) | `training/runs/baseline/best_model.pth` | `b613413027e775b98e449cbdbc9afa0253d6bd40f0cb2502230ed5bf1000b0bb` |
| Phase 2 (ASD/TD-conditioned) | `training/runs/phase2_embedding_only/best_model_conditioned.pth` | `35319673382378487d23c1871cab0cf05b71b1e7963e180ec024e5d80901a9a4` |

These are the hashes recorded for the E1 experiment in the paper.

**Other files used:**
- `training/runs/phase2_embedding_only/summary_full.json`: duration normalization statistics. If it is missing, a warning is printed and the identical built-in values are used.
- For the curated examples only:
  - `TrainingData/TrainingData/Images/` (Saliency4ASD)
  - `processed_data/`
  - `training/runs/baseline/split.json`
  - `training/runs/research_completion/e1_phase_tau1_v2/`

## 2. Running it

Run all commands from the project root (`gaze-ag/`).

**Single image (`inference/inference.py`).** About 20 s per command on this laptop's CPU, most of it model loading.

```bash
# ASD and TD side by side (default tau = 1, seed 0)
python inference/inference.py --image inference/demo/inputs/synthetic_scene.png --condition both

# add the unconditioned Phase 1 model as a third panel, with the same seed and settings
python inference/inference.py --image inference/demo/inputs/synthetic_scene.png --compare-phase1

# one condition only
python inference/inference.py --image inference/demo/inputs/synthetic_scene.png --condition ASD

# four independent samples per condition (seeds 0-3), as a grid
python inference/inference.py --image inference/demo/inputs/synthetic_scene.png --n-samples 4

# temperature 2 (the official TPP-Gaze config value), another seed
python inference/inference.py --image inference/demo/inputs/synthetic_scene.png --temperature 2 --seed 5

# draw on a blank canvas instead of the photograph (for material you may share)
python inference/inference.py --image TrainingData/TrainingData/Images/60.png --no-image

# write somewhere other than inference/outputs/<image name>/
python inference/inference.py --image inference/demo/inputs/synthetic_scene.png --out-dir inference/demo/local_only/live
```

**Options:**

| Option | Default | Meaning |
|---|---|---|
| `--condition` | `both` | `ASD`, `TD` or `both` |
| `--compare-phase1` | off | Adds the Phase 1 model as a panel |
| `--temperature` | 1.0 | Spatial sampling temperature τ; durations are not tempered |
| `--seed` | 0 | Random seed; every panel uses the same seed |
| `--n-samples K` | 1 | K samples per condition with seeds `seed`…`seed+K−1` |
| `--t-end` | 3.0 | Viewing window in seconds, as in training |
| `--n-sim` | 1 | Batch size per sampling call; leave it at 1 (see §5) |
| `--no-image` | off | Draw on a blank canvas |
| `--checkpoint`, `--phase1-checkpoint` | paths in §1 | Override the checkpoint paths |

Any image format that Pillow reads works, at any size. The model sees a 512×512 resize, and the coordinates are mapped back to the original size.

**Curated examples (`inference/demo/run_demo.py`).** Loads both models once.

```bash
python inference/demo/run_demo.py                          # all 5 examples, about 2.5 minutes
python inference/demo/run_demo.py --example s4a_random_60  # one example
python inference/demo/run_demo.py --check                  # reproducibility check (about 2.5 minutes)
```

## 3. Output files

**`inference/inference.py`.** Written to `inference/outputs/<image name>/`, which is git-ignored:

| File | Contents |
|---|---|
| `asd_scanpath.png`, `td_scanpath.png`, `phase1_scanpath.png` | One panel each |
| `<image>_asd_vs_td.png` or `<image>_phase1_vs_asd_vs_td.png` | Side-by-side figure |
| `<image>_samples_grid.png` and `<condition>_samples.json` | With `--n-samples` above 1 |
| `<condition>_scanpath.json` | Generated scanpath (fields below) |

The JSON file contains:
- `scanpath_pixels`: `[x, y, duration_s]`, clipped to the image as in the official TPP-Gaze code.
- `scanpath_normalized`: unclipped model output in [−1, 1].
- `events`: one record per fixation, with order, onset, duration, pixel and normalized position, and whether it was clipped.
- `n_clipped`.
- `model`: phase, checkpoint path and SHA-256, and torch, numpy and device.
- `temperature`, `seed`, `t_end_s`, `n_sim` and the image size.

The original fields are unchanged, so older readers still work.

**`run_demo.py`.** Writes to two places:
- **`inference/demo/demo_results/<example>/`** (may be shared):
  - figures drawn on a blank canvas (`*_no_image.png`);
  - `samples.json`, holding every generated scanpath with its events, and numeric summaries;
  - for the synthetic image only, the figures drawn over the image as well;
  - `inference/demo/demo_results/manifest.json`: selection rule, checkpoint hashes, settings, environment and validation results;
  - `inference/demo/demo_results/reproducibility_check.json`.
- **`inference/demo/local_only/<example>/`** (this machine only): figures drawn over the Saliency4ASD photographs, and `real_vs_generated.png`, which plots real participants' fixations. The folder has its own `.gitignore` so that it is never committed.

## 4. The curated examples

The stimuli were selected before any demo output was generated, using a rule based only on saved E1 results (the rule is stored in `manifest.json`). They were not chosen because the model looked good on them.

| Example | Why it is included | Figures |
|---|---|---|
| `s4a_random_60` | Random held-out Saliency4ASD test image (typical case) | Phase comparison, seed grid, temperature, real vs generated |
| `s4a_random_246` | Second random test image | Phase comparison, real vs generated |
| `s4a_best_102` | Test image where the model's ASD−TD dispersion difference best matches the real one (real +0.013, generated +0.014) | Phase comparison, real vs generated |
| `s4a_worst_116` | Test image where it matches worst. Real ASD scanpaths are less dispersed than TD (−0.047), but generated ASD scanpaths are more dispersed (+0.015). | Phase comparison, real vs generated |
| `synthetic_scene` | Simple drawing made by `run_demo.py` (shapes, a schematic face, text). It is outside the training distribution and contains no third-party content, so its overlays can be shared. | Phase comparison, seed grid, temperature |

All Saliency4ASD test images were held out from training and model selection. Across all 45 test images, the model's generated ASD−TD dispersion difference is positive on every image (+0.002 to +0.020). The real difference ranges from −0.047 to +0.058. The conditioning therefore acts as a roughly uniform shift rather than an image-specific effect, although the real per-image differences are themselves noisy with only 6–14 viewers per group.

## 5. Interpreting the figures

**Encoding:**
- Numbers give the fixation order.
- Green is the first fixation and red the last.
- Circle area grows with fixation duration, and the exact durations are listed under each panel.
- Arrows show the movement to the next fixation.

**Shared random numbers.** All panels with the same seed use the same random numbers (common random numbers, as in experiment E1), so they differ only because of the model. In the phase comparison, Phase 1 and Phase 2-ASD are often almost identical. This shows that the learned group embedding (1,024 parameters added to a frozen model) makes only a small change to any single sample.

**Diverging sequences.** When a sequence does diverge (for example after fixation 2 in the TD panel for image 60), everything that follows differs, because each fixation depends on the previous ones. A visibly different single pair therefore does not mean a large group effect.

**Seed grid.** Samples within one condition vary much more from seed to seed than ASD and TD differ at a fixed seed. Never interpret one sample as "the ASD scanpath". Group differences exist only on average, over many samples and images.

**Temperature.** τ raises each spatial mixture component's spread σ to the power τ, so at τ = 2 the components are narrower. Durations are not tempered directly; they change only because each fixation depends on the earlier ones.
- **On average,** τ = 2 gives less dispersed scanpaths (see the paper).
- **In a single sample,** this need not be visible. In the curated temperature figures, spread is smaller at τ = 2 in 3 of 4 sample pairs, and larger in the fourth (TD, image 60).
- **Default.** τ = 1 was chosen on Saliency4ASD validation images for behavioural realism.
- **Likelihood.** Temperature never changes the model's likelihood, so it is not a model improvement.

**Real vs generated.** Each dot is one fixation. The figure compares all real scanpaths for that image with the 20 per group that E1 saved (replicate 1). Things to point out:
- Generated scanpaths contain more fixations than real ones (about 10–12 vs 6–9 per scanpath in these examples), as reported in the paper.
- The numbers come from one replicate and a few viewers, so they are noisy. Use the paper's tables for group statistics.

**Totals under 3 s.** Sampling stops at the 3 s window, and the fixation still in progress at 3 s is dropped. Totals are therefore usually a little below 3 s, and occasionally well below if the dropped fixation was long.

**Clipping.** Generated positions outside the image are clipped to its border in `scanpath_pixels` and the figures; `n_clipped` counts them. None of the 35 curated scanpaths needed clipping.

**Batch size (`--n-sim`).** Keep it at 1. The released TPP-Gaze transformer mixes information across the samples in a batch (the `batch_first=False` layout disclosed in the paper). A sample drawn in a batch therefore depends on the other samples in that batch. E1 drew batches of 20, so demo samples and paper samples are drawn the same way only when `n_sim = 20`; the `--check` cross-check uses exactly that setting.

## 6. Verification

Run on 2026-11-03 and recorded in `inference/demo/demo_results/`:
- **Checkpoints.** Both checkpoint hashes match the values recorded for E1. The Phase 1 checkpoint and the base of the Phase 2 checkpoint are bit-identical, so the two phases differ only in the group embedding, and both run through the same sampler.
- **Regression.** The updated `inference/inference.py` reproduces the previous version's outputs exactly (the existing `inference/outputs/beach` results, τ = 2, seed 0).
- **Determinism.** Identical settings give identical JSON in separate processes, and `run_demo.py --check` regenerated all 35 curated scanpaths identically. This was tested on the CPU listed in §1; a GPU or another torch build may give slightly different samples.
- **Same procedure as the paper.** Re-drawing two saved E1 sampling calls (Phase 2 ASD and Phase 1, test image 8, seed 101000, 20 samples) through the demo's loading code gave the same fixation counts and values within 1e-5. That is the rounding step of the saved E1 files.
- **Event checks** on every curated scanpath:
  - positions are inside the image;
  - durations are positive (0.057–1.20 s, median about 0.12–0.21 s per example);
  - onsets increase;
  - totals are at most 3 s;
  - fixation numbering is consecutive.
- **No other files changed.** No checkpoint, dataset, experiment output or manuscript file was modified. A SHA-256 manifest of 8,001 project files taken before the demo runs was unchanged afterwards.

## 7. Limitations to state when presenting

1. **Not diagnostic.** Generated scanpaths are samples from a statistical model trained on group labels. They cannot classify, screen or describe any person.
2. **Small, inconclusive likelihood gain.** On held-out Saliency4ASD images, conditioning improves likelihood by +0.0073 nats per fixation, with a 95% image-level CI of [−0.0012, +0.0163], which is inconclusive.
3. **Worse on the external dataset.** On Gaze4ASD, the conditioned model is worse than Phase 1 in likelihood under all three preprocessing versions, and worse for TD in every version. Conditioning is not generally superior to TPP-Gaze.
4. **Partial behavioural match:**
   - Conditioning moves generated ASD and TD scanpaths apart in the right direction for dispersion, saccade amplitude and revisits, at about 69%, 64% and 32% of the real differences.
   - Fixation count, total time and direction concentration move the wrong way.
   - Neither group's generated scanpaths become spatially closer to that group's real scanpaths.
5. **Too many fixations.** Generated scanpaths contain more fixations than real ones.
6. **Uniform shift.** The ASD−TD shift is roughly the same on every image (§4); the model does not reproduce image-specific group differences.
7. **Single samples are not group profiles.** Within-condition variability is larger than the condition effect.
8. **Confounds.** Group differences in the data are confounded with age and other factors that cannot be adjusted for. Generated differences do not establish any mechanism.
9. **Out-of-distribution images.** The model was trained on Saliency4ASD: natural photographs viewed by children aged 5–12, with scanpaths truncated to 3 s. Output on other kinds of image, such as the synthetic scene or screenshots, is outside its tested range.

## 8. Data and permissions

**Saliency4ASD.**
- The images come from MIT1003 and include photographs of people and children.
- The dataset's usage agreement and its Zenodo CC BY 4.0 record conflict, and this is still unresolved (`docs/research_audit/UNRESOLVED_ISSUES.md` §6).
- The photographs, their overlays and real gaze plots are therefore kept in `inference/demo/local_only/`.
- Show them only from this machine, and don't put them in slides, reports or the repository.

**Gaze4ASD.** Its stimuli include photographs of children and have no stated source; they are not used in the demo.

**Images in `inference/inputs/`.**
- `beach.png` and `cat.png`/`image.png`/`test_image.png` are copies of Gaze4ASD stimuli.
- `image1.png` and `image2.png` are web photographs of unknown licence (one shows identifiable people).
- Don't use them for anything that will be shared.

**`synthetic_scene.png`** was drawn by `run_demo.py`. It and its outputs can be shared.

**Committing.** If the demo is committed later, note that the repository's `.gitignore` ignores all `*.json` files. The JSON in `inference/demo/demo_results/` would need an explicit exception.

## 9. Suggested presentation sequence (about 10 minutes)

1. **The encoding (shareable).** Open `inference/demo/demo_results/synthetic_scene/phase_comparison.png`, explain the encoding (§5), and point out that the three panels share random numbers.
2. **Live run.** About 20 s:
   ```bash
   python inference/inference.py --image TrainingData/TrainingData/Images/60.png --compare-phase1 --out-dir inference/demo/local_only/live
   ```
   Open `inference/demo/local_only/live/60/60_phase1_vs_asd_vs_td.png`. It shows exactly the same scanpaths as the pre-generated `inference/demo/local_only/s4a_random_60/phase_comparison.png`, which demonstrates reproducibility. This was checked: all three JSON event lists are identical; only the figure title and footer differ.
3. **Variability.** `inference/demo/local_only/s4a_random_60/seeds_grid.png`: within-condition variation is larger than the ASD/TD difference.
4. **Temperature.** `inference/demo/local_only/s4a_random_60/temperature.png`: τ changes the sampled locations. It is a sampling choice, not a model change, and its average effect (less dispersion at τ = 2) shows only over many samples (§5).
5. **Real vs generated.**
   - `inference/demo/local_only/s4a_random_60/real_vs_generated.png`: generated scanpaths have too many fixations.
   - Then `s4a_best_102` and `s4a_worst_116`: the model's ASD/TD shift matches the real one on some images and contradicts it on others.
6. **Close** with the limitations in §7.

**Backup.** If a live command fails, every figure above is already generated. If the projector should not show dataset photographs, use the `*_no_image.png` versions in `inference/demo/demo_results/`.

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `checkpoint hash mismatch` (`run_demo.py`) | A checkpoint was replaced or corrupted. Restore the original files listed in §1. |
| `FileNotFoundError` for a `.pth` file | Checkpoints are not in the repository. Copy `training/runs/...` from the training machine. |
| Hangs or fails at "Using cache found in …" or while downloading | The torch hub cache is missing. Run once with internet access, or copy `~/.cache/torch/hub/` from this machine. |
| `ModuleNotFoundError: omegaconf` (or another package) | `pip install -r requirements.txt` |
| `image not found` | Check the path, which is relative to the project root. |
| `WARNING: could not read …summary_full.json` | The built-in statistics, identical to that file, are used. Outputs are unaffected. |
| `WARNING: … no fixation inside the 3 s window` | The first sampled fixation was longer than 3 s, which is rare. Try another seed and say so. |
| Results differ from the saved ones | Check the seed, τ and `--n-sim 1`. A different torch version, thread library or GPU can change samples; recheck with `python inference/demo/run_demo.py --check`. |
| IDE reports "Cannot find module tppgaze.utils" | This is static analysis only. The scripts add `tppgaze/` to the path at runtime. |
| Each command takes about 20 s | This is normal on CPU. Loading the models takes about 3 s; the rest is start-up, image encoding, sampling and drawing. For several images in a row, `run_demo.py` loads the models once. |
