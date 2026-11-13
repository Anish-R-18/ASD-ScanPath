# Configurations

| File | Used by | Purpose |
|---|---|---|
| `phase1_baseline.yaml` | `training/train_tpp.py` (default) | Phase 1: fine-tune TPP-Gaze on combined ASD + TD Saliency4ASD data, no label. AdamW, lr 10⁻³, weight decay 0.1, batch 128, gradient clip 5, patience 20, DenseNet frozen and cached, seed 0, image split 70/15/15. |
| `phase2_embedding_only.yaml` | `training/train_tpp_cond.py` (default) and the evaluation scripts (for `images_dir`) | Phase 2: start from the best Phase 1 checkpoint, reuse its split, freeze the whole base, and train only the 2 × 512 ASD/TD embedding at lr 10⁻⁴ |

**Paths.** All paths are relative to the repository root. Set `images_dir` if your Saliency4ASD images are elsewhere.

**Model architecture configuration.** It is the original TPP-Gaze file [`../tppgaze/data/config.yaml`](../tppgaze/data/config.yaml), used unchanged. The training scripts override only the log-duration mean and standard deviation, using values computed on our training split.

**Superseded configuration.** An earlier Phase 2 configuration (learning rate 10⁻³, no saved results) is kept in [`../archive/configs/`](../archive/README.md).
