# ASD-ScanPath: Modelling Gaze Dynamics in Space and Time for Autism Screening

This repository contains the code and preprocessing pipelines for our research on **Autism Spectrum Disorder (ASD) screening using Neural Temporal Point Processes**.

By combining deep neural networks with Temporal Point Processes (TPP), we model both the *spatial* distribution and the *temporal* dynamics (fixation durations) of eye gaze. Our architecture, **TPP-Gaze**, captures intricate behavioral differences between ASD and Typically Developing (TD) children when viewing complex visual stimuli.

> **⚠️ NOTE ON DATASETS:**  
> The datasets (Saliency4ASD, Gaze4ASD) used in this research are **NOT distributed in this repository** due to privacy and licensing restrictions. They must be obtained directly from their official sources.

---

## 1. Overview and Motivation
Autism screening often relies on identifying atypical gaze behaviors. Traditional methods rely on static saliency maps or aggregate statistics (e.g., total fixation time), which destroy the rich temporal flow of a child's visual attention. 

Our approach uses **TPP-Gaze**, an architecture that:
1. Extracts semantic visual features using CNNs (DenseNet).
2. Models the spatial locations of fixations using a discrete categorical distribution.
3. Models the continuous duration of fixations (inter-event times) using Log-Normal Mixture distributions.
4. Uses a Transformer Encoder to capture the sequential dependencies between fixations.

We achieve ASD/TD conditioning by injecting group embeddings (ASD vs. TD) directly into the model, allowing us to explicitly train and evaluate the behavioral divergences between the two groups on the same images.

---

## 2. Repository Structure

```
ASD-ScanPath/
├── configs/            # YAML configuration files (e.g., config.yaml)
├── tppgaze/            # Core TPP-Gaze architecture (PyTorch)
├── training/           # Phase 1 & Phase 2 training scripts
├── preprocessing/      # Scripts to process raw eye-tracking data
├── analysis/           # Behavioral and statistical analysis code
├── results/            # Checkpoints, figures, and numerical tables
├── docs/               # Audits, feasibility reports, and documentation
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 3. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anish-R-18/ASD-ScanPath.git
   cd ASD-ScanPath
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
*(Note: Code is written for Python 3.9+ and PyTorch 1.9+).*

---

## 4. Dataset Acquisition
To replicate our experiments, you must request the following datasets from their respective authors:

1. **Saliency4ASD** (Primary Dataset): Used for Phase 1 & 2 training.
2. **Gaze4ASD** (External Validation): Used for zero-shot testing.

Place the downloaded raw data in an `external_data/` or `AdditionalData/` directory.

---

## 5. Preprocessing Pipeline

### 5.1 Saliency4ASD
Run the provided pipeline to clean the raw discrete coordinates:
```bash
python preprocessing/preprocess_dataset.py
```
**Key Cleaning Steps:**
- Removes the initial centering fixation.
- Filters fixations with duration `< 60ms`.
- Truncates scanpaths to a `3.0s` maximum cumulative duration.
- Enforces a minimum length of 3 fixations.
- Normalizes coordinates to `[-1, 1]`.

### 5.2 Gaze4ASD (External Validation)
Run the script to map screen coordinates to image-local space:
```bash
python preprocessing/preprocess_gaze4asd.py
```
This correctly offsets the `2560x1440` Tobii screen coordinates to the centered image dimensions and applies the `[-1, 1]` bounding clip to match the Phase 1 training constraints perfectly.

---

## 6. Training Instructions

### Phase 1: Base TPP-Gaze Training
Pre-train the temporal point process model on general gaze data to learn basic visual-temporal alignments:
```bash
python training/train_phase1.py --config configs/config.yaml
```

### Phase 2: ASD/TD Conditioning
Fine-tune the model by appending an `ASD/TD` group embedding to the input sequence:
```bash
python training/train_phase2.py --config configs/config.yaml
```

---

## 7. Evaluation & Main Findings

Evaluate the trained models using the analysis scripts:
```bash
python analysis/run_sensitivity.py
```

### Main Behavioral Findings
- **Spatial Edge-Piling:** We found that dropping out-of-bounds fixations (looking off-screen) destroys the temporal integrity of the scanpath. ASD children exhibited a higher propensity for out-of-bounds looks. Clipping these to the image boundary `[-1, 1]` safely preserves temporal rhythms and provides the model with critical behavioral cues.
- **Duration Differences:** ASD fixations often exhibited longer tails in duration, confirming hypotheses regarding visual hyper-focus on non-social elements.

### Important Limitations
- **Data Sparsity:** Strict temporal filtering (e.g., removing any scanpath with `< 3` valid fixations) disproportionately affects ASD samples due to higher rates of off-screen glances.
- **Hardware Variation:** Gaze4ASD and Saliency4ASD were recorded on different screens. Precise coordinate mapping is critical, as failing to offset raw screen coordinates artificially shifts the spatial distribution relative to the image features.

---

## 8. Inference / Demo

We provide a standalone inference tool to generate predicted scanpaths for any input image conditioned on either ASD or TD behavior. The model outputs predicted fixation coordinates along with predicted continuous durations.

The visualizer renders the sequence as numbered points connected by arrows, where circle size corresponds to fixation duration and explicit time labels are attached.

**Example Commands:**
Generate an ASD-conditioned scanpath:
```bash
python inference.py --image path/to/image.png --condition ASD
```
Generate a TD-conditioned scanpath:
```bash
python inference.py --image path/to/image.png --condition TD
```
Generate both conditions side-by-side for visual comparison:
```bash
python inference.py --image path/to/image.png --condition both
```

Outputs (saved to `outputs/` by default):
- `<condition>_scanpath.png`: Visualized scanpath over the image.
- `<condition>_scanpath.json`: Raw sequential predictions formatted as `[x, y, duration_in_seconds]`.
- `<image_name>_asd_vs_td.png`: A side-by-side comparison figure (if `--condition both` is used).

---

## 9. Reproducing Figures & Tables
All final figures and numerical tables can be reproduced by running the Jupyter notebooks or python scripts located in the `results/` and `analysis/` directories. Pre-computed summary files are located in `docs/`.

---

## 10. Citation
If you use this code in your research, please cite:

```bibtex
@article{ASDScanPath2026,
  title={ASD-ScanPath: Modelling Gaze Dynamics in Space and Time for Autism Screening},
  author={Anish, R. and et al.},
  journal={TBD},
  year={2026}
}
```

*(Placeholder: Update with official publication details once accepted).*
