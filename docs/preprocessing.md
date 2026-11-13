# Preprocessing

Run all commands from the repository root. Raw datasets are read only; each script writes to a Git-ignored folder.

## Saliency4ASD → `processed_data/`

```bash
python preprocessing/preprocess_dataset.py     # writes asd_scanpaths.json, td_scanpaths.json and a report
python preprocessing/validate_dataset.py       # checks the processed files (all 300 images present in both groups, ...)
python preprocessing/run_sensitivity.py        # optional: retention by group for minimum lengths 4, 3 and 2 (docs/audits/preprocessing_sensitivity.txt)
```

**Source.** [`../preprocessing/preprocess_dataset.py`](../preprocessing/preprocess_dataset.py). The script checks raw files with SHA-256 before and after the run, so it never modifies them.

**Pipeline.** Applied to each scanpath, in this order:
1. Split each raw file into sequences at every index reset (`Idx == 0`).
2. Drop the whole sequence if any fixation lasts longer than 3,000 ms.
3. Remove the first fixation, which is usually the central fixation before the image.
4. Build a timeline that starts at 0 after the removed fixation.
5. Remove fixations shorter than 60 ms.
6. Truncate at 3 s. A final fixation that crosses 3 s is clipped, or dropped if the clipped part would be shorter than 60 ms.
7. Require at least 3 fixations.
   - **Our choice.** The TPP-Gaze paper used 4.
   - **Why.** With a minimum of 4, retention differs strongly by group (TD 77% vs ASD 62%), and a minimum of 3 reduces this gap. See [`audits/preprocessing_sensitivity.txt`](audits/preprocessing_sensitivity.txt).
8. Normalise coordinates to [−1, 1] with 2(*x* − 1)/(*W* − 1) − 1, and convert durations to seconds.

**Output.** 2,705 ASD and 3,324 TD scanpaths. The aggregate report is copied to [`../results/in_domain/saliency4asd_preprocessing_report.txt`](../results/in_domain/saliency4asd_preprocessing_report.txt). Earlier audits of the raw data are in [`audits/dataset_audit.txt`](audits/dataset_audit.txt) and [`audits/duration_audit.txt`](audits/duration_audit.txt).

## Gaze4ASD → `external_data/Gaze4ASD_processed_v2/`

```bash
python preprocessing/preprocess_gaze4asd_v2.py
```

**Source.** [`../preprocessing/preprocess_gaze4asd_v2.py`](../preprocessing/preprocess_gaze4asd_v2.py). It writes three versions and validation reports, under `external_data/Gaze4ASD_processed_v2/reports/`.

### Coordinate mapping

Gaze is recorded in screen pixels on a 2560 × 1440 display, and the release does not record where each image was drawn. For an image of size *W* × *H*:

```text
fit-to-height (primary):  scale = 1440 / H;  offset_x = (2560 − scale·W) / 2
                          image_x = (screen_x − offset_x) / scale;   image_y = screen_y / scale
stretch (sensitivity):    image_x = screen_x · W / 2560;              image_y = screen_y · H / 1440
normalised:               norm = 2 · image / size − 1
```

**Evidence for fit-to-height.** It is strongly supported but partly inferred: the original presentation code is not available.

| Candidate mapping | Match to the authors' own saliency maps (median *r*, 4,533 maps) | Fixations inside the image |
|---|---|---|
| Fit-to-height | 0.9999 | 96.5% |
| Stretch | 0.91 | — |
| Native-size centred | 0.23 | 81.4% |

Gaze alone does not exclude the stretch mapping, so stretch is kept as a sensitivity version. The full methodology is in [`../results/external_evaluation/corrected_mapping_methodology.txt`](../results/external_evaluation/corrected_mapping_methodology.txt).

### Off-image fixations and versions

All versions apply the same basic steps:
1. remove the first fixation;
2. remove fixations shorter than 60 ms;
3. apply the 3 s window;
4. map the coordinates;
5. require at least 3 fixations.

| Version | Mapping | Off-image fixations | Role |
|---|---|---|---|
| `fit_height_policy_a` | fit-to-height | clipped to the image edge (Policy A); 3.52% of fixations affected | **Primary**, fixed before any model was evaluated on Gaze4ASD |
| `stretch_policy_a` | stretch | clipped | Sensitivity |
| `fit_height_policy_b` | fit-to-height | removed (Policy B) | Sensitivity |

**Off-image gaze differs by group.**
- **How much.** It accounts for 7.35% of ASD fixations against 2.72% of TD fixations, about 2.7×.
- **What Policy B therefore removes.** More ASD data than TD data: 8.07% of ASD fixations against 3.03% of TD fixations.
- **Consequence.** Both policies bias group comparisons in different ways, so both are reported.

### Superseded mapping

`preprocess_gaze4asd.py` assumed native-size, centred stimuli, which the data contradicted.
- The script is kept in [`../archive/code/`](../archive/README.md).
- Its results are in [`../archive/results/external_gaze4asd_native_mapping/`](../archive/results/external_gaze4asd_native_mapping/SUPERSEDED.txt).
- Do not use any value derived from it.
