# Training Guide

## Requirements

**Data and weights:**
- the processed Saliency4ASD data in `processed_data/` (see [`preprocessing.md`](preprocessing.md));
- the Saliency4ASD images in `TrainingData/TrainingData/Images/`;
- the pretrained TPP-Gaze weights at `tppgaze/data/model_transformer.pth`:
  ```bash
  curl -L -o tppgaze/data/model_transformer.pth https://github.com/phuselab/tppgaze/releases/download/v1.0.0/model_transformer.pth
  ```
- internet access on first use, to download DenseNet201 through `torch.hub`.

**Hardware.** A GPU is recommended. The reported runs used an NVIDIA GPU with PyTorch 2.12.1 and CUDA 13.0; the scripts fall back to CPU.

## Configurations

| File | Phase | Notes |
|---|---|---|
| [`../configs/phase1_baseline.yaml`](../configs/phase1_baseline.yaml) | 1 | Writes `training/runs/baseline/` |
| [`../configs/phase2_embedding_only.yaml`](../configs/phase2_embedding_only.yaml) | 2 | Reads the Phase 1 checkpoint and split; writes `training/runs/phase2_embedding_only/` |

Paths in the configs are relative to the repository root. The two trainers default to these configs.

**Archived Phase 2 configuration.** [`../archive/configs/phase2_lr1e-3_no_saved_results.yaml`](../archive/configs/phase2_lr1e-3_no_saved_results.yaml) is an earlier Phase 2 configuration (learning rate 10⁻³; the base is trained too, apart from the frozen backbone).
- **No results.** Its results could not be found, and it is not reported.
- **Was the default.** It was the trainer's default before the cleanup; the trainer now defaults to the reported configuration.

## Phase 1: fine-tune TPP-Gaze (no label)

```bash
python training/train_tpp.py smoke    # one batch: finite loss, non-zero gradients, frozen backbone gets none
python training/train_tpp.py short --epochs 3 --limit-images 12 --batch-size 16   # quick end-to-end run
python training/train_tpp.py train    # full run
```

**Settings:**
- AdamW, learning rate 10⁻³, weight decay 0.1;
- batch 128, gradient clipping at 5;
- early stopping after 20 epochs without validation improvement (at most 300 epochs);
- seed 0;
- image-level split 70/15/15.

**What is trained.** DenseNet201 is frozen and its features are cached, so 3,881,784 parameters are trained.

**Log-duration normalisation.** It uses the training split only (mean −1.4366, SD 0.6936).

**Outputs** (in `training/runs/baseline/`):
- `best_model.pth`;
- `split.json` (the image split);
- logs and `summary_full.json`;
- `model_config_used.yaml`.

**Reported run:**
- best validation NLL −0.5845, at epoch 6;
- stopped at epoch 26, when validation NLL had reached −0.384 against a training NLL of −0.969. Early stopping mattered.

Summary: [`../results/in_domain/phase1_training_summary.json`](../results/in_domain/phase1_training_summary.json).

## Phase 2: train only the ASD/TD embedding

```bash
python training/train_tpp_cond.py smoke   # zero embedding: ASD and TD NLL identical; non-zero: they differ; base fully frozen
python training/train_tpp_cond.py train
```

**Settings:**
- starts from `training/runs/baseline/best_model.pth`, with the same split (`split_json`) and objective;
- the whole base is frozen (`train_embedding_only: true`) and kept in evaluation mode, so dropout cannot change it;
- only the 2 × 512 embedding (1,024 parameters) is trained, at learning rate 10⁻⁴.

**Outputs** (in `training/runs/phase2_embedding_only/`):
- `best_model_conditioned.pth`, containing the base (`base.*`) and `group_emb.weight`;
- `summary_full.json`.

**Reported run.**
- **Validation NLL:** went from −0.5845 to −0.5964, with the best epoch at 8 of 28. This gain is optimistic, because validation also chose the checkpoint; the fair estimate is the test-set result (+0.0073, inconclusive).
- **Base unchanged:** all 1,290 base tensors are identical to Phase 1.

Summary: [`../results/in_domain/phase2_training_summary.json`](../results/in_domain/phase2_training_summary.json).

## Model selection

Validation images selected both checkpoints and the sampling temperature. The test images and Gaze4ASD were never used for training or selection.

## Checkpoints used for the reported results

| File | SHA-256 |
|---|---|
| `training/runs/baseline/best_model.pth` | `b613413027e775b98e449cbdbc9afa0253d6bd40f0cb2502230ed5bf1000b0bb` |
| `training/runs/phase2_embedding_only/best_model_conditioned.pth` | `35319673382378487d23c1871cab0cf05b71b1e7963e180ec024e5d80901a9a4` |

**Not distributed.** The checkpoints are not in the repository. Releasing them is the project owner's decision; they are derived from data whose terms are unresolved. `tests/test_checkpoints.py` verifies them when they are present.
