# Inference and Demo

> Research demonstration only. Generated scanpaths are samples from a group-level model. They do not describe any individual and must not be used to assess anyone.

## Requirements

**Checkpoints.** They are not distributed; get them from the project owner, or retrain them ([`../docs/training_guide.md`](../docs/training_guide.md)).

```text
training/runs/phase2_embedding_only/best_model_conditioned.pth   Phase 2, ASD/TD-conditioned (always needed)
training/runs/baseline/best_model.pth                            Phase 1 (only with --compare-phase1)
```

The log-duration statistics are read from `training/runs/phase2_embedding_only/summary_full.json` when present; otherwise the training-split values (−1.4366, 0.6936) are used. DenseNet201 is downloaded by `torch.hub` on first use.

## `inference.py`

```bash
python inference/inference.py --image path/to/image.png --condition ASD
python inference/inference.py --image path/to/image.png --condition TD
python inference/inference.py --image path/to/image.png --condition both          # side by side
python inference/inference.py --image path/to/image.png --compare-phase1          # Phase 1, ASD and TD
python inference/inference.py --image path/to/image.png --n-samples 4             # four seeds per condition, as a grid
python inference/inference.py --image path/to/image.png --temperature 2.0 --seed 5
python inference/inference.py --help                                              # all options (--no-image, --out-dir, ...)
```

**Sampling settings.**
- **Temperature.** The default is τ = 1, the setting selected on Saliency4ASD validation images. τ = 2 is the released TPP-Gaze config value. Temperature changes sampling only, never the model or its likelihood.
- **Shared random numbers.** All conditions are sampled with the same seed, so they share their random numbers and differ only through the model.
- **Viewing window.** Each scanpath fills a 3 s window.
- **Run time.** One command takes about 15–20 s on a laptop CPU, mostly model loading.

**Outputs** go to `inference/outputs/<image name>/`, which Git ignores; `--out-dir` changes the parent folder.

| File | Contents |
|---|---|
| `<condition>_scanpath.png` | Numbered fixations over the image: circle area shows duration, arrows show order |
| `<condition>_scanpath.json` | `scanpath_pixels` as `[x, y, duration_s]` (clipped to the image), `scanpath_normalized` as sampled, per-fixation events, model, checkpoint hash and settings |
| `<image>_asd_vs_td.png` | With `--condition both` |
| `<image>_phase1_vs_asd_vs_td.png` | With `--compare-phase1` |
| `<condition>_samples.json`, `<image>_samples_grid.png` | With `--n-samples > 1` |

`inference/inputs/` is a Git-ignored place for your own test images. Do not commit photographs whose licence is unclear.

## Curated demo (`demo/`)

```bash
python inference/demo/run_demo.py                          # regenerate all examples (about 2.5 min on CPU)
python inference/demo/run_demo.py --example s4a_random_60  # one example
python inference/demo/run_demo.py --check                  # regenerate and compare with the saved results
```

**What it uses.**
- **Images.** Saliency4ASD test images (selected by a fixed rule from saved E1 results) and a synthetic scene drawn by the script.
- **Sampling.** It goes through the same sampler as `inference.py`.

**What is published.** `demo/demo_results/` holds figures on a blank canvas, generated scanpaths and summary statistics. It contains no stimulus pixels and no real gaze records.

**What stays local.** Figures drawn over the stimulus photographs, and real-vs-generated plots, are written to `demo/local_only/`, which is Git-ignored.

**Guide.** How to read the figures, the selection rule and the limitations are described in [`demo/DEMO_GUIDE.md`](demo/DEMO_GUIDE.md).
