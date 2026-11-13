# Datasets

**Not included.** Neither dataset is in this repository. You need both to preprocess, train or evaluate; inference on your own images needs only the checkpoints.

## Saliency4ASD (training, validation and held-out test)

| Item | Value |
|---|---|
| Content | Eye movements of children with ASD and TD controls viewing 300 natural images taken from MIT1003 (Judd et al., 2009) |
| References | Duan et al., ACM MMSys 2019; Gutiérrez et al., *Signal Processing: Image Communication* 92 (2021) |
| Access | [Zenodo record 13960426](https://zenodo.org/records/13960426) |
| Eye tracker and display | Tobii T120, 1280 × 1024; 3 s per image |
| Age | ASD 5–12 years (mean 8); controls mean 8. No participant-level age. |
| Participants | The README reports 14 per group per image, but the files hold 8–17 (ASD) and 9–15 (TD) sequences per image. There are no participant IDs. |
| After preprocessing | 2,705 ASD and 3,324 TD scanpaths; 18,475 and 25,504 fixations |
| Split | By image with seed 0: 210 / 45 / 45 images, 4,229 / 897 / 903 scanpaths. The split is in [`../results/in_domain/saliency4asd_image_split_seed0.json`](../results/in_domain/saliency4asd_image_split_seed0.json). |

**Expected layout.** `TrainingData/TrainingData/{ASD,TD,Images,...}`, as released.

**Sequences vs participants.** We treat each index-reset sequence as one scanpath. Because some images have more than 14 sequences, sequences may not correspond one-to-one to participants. In-domain uncertainty is therefore estimated across images only.

## Gaze4ASD (external evaluation only)

| Item | Value |
|---|---|
| Content | 33 ASD and 133 TD children (README: 132 TD), aged 3–7, viewing 30 images selected for large ASD/TD differences (mostly smiling faces) |
| Reference | Yang et al., IEEE ICME 2025 |
| Access | [github.com/yyzyyz001/Gaze4ASD](https://github.com/yyzyyz001/Gaze4ASD) |
| Eye tracker and display | Tobii Pro Fusion; 27-inch 2560 × 1440 at 70 cm; 3 s per image |
| Age | Group means 5.04 (ASD) and 4.54 (TD); no participant-level age |
| After preprocessing (primary) | 771 ASD and 3,605 TD trials; 4,830 and 23,161 fixations |

**Expected layout.** `external_data/Gaze4ASD/{ASD_ScanPath,TD_ScanPath,ASD_rawData,TD_rawData,Image Stimuli}`.

**Known issues**
- **Participant count.** The README states 132 TD children, but the release contains 133 TD folders, and all of them contribute trials. We use the data as released and report the mismatch.
- **Coordinate mapping.** Gaze is recorded in screen pixels, and the release does not record where each image was drawn. Our fit-to-height mapping is strongly supported but partly inferred; see [`preprocessing.md`](preprocessing.md).
- **Independence from Saliency4ASD.** The supplement describes fine-tuning a saliency model on Saliency4ASD during stimulus selection, so the external stimuli may not be fully independent of our training data.
- **Literature access.** We could not access the full text of the main Gaze4ASD paper. Statements here come from the README and the supplement.

## Data terms

These are **unresolved**, so the repository publishes no dataset content.

- **Saliency4ASD.**
  - **The licence record.** The Zenodo record lists CC BY 4.0.
  - **The usage agreement.** `UsageAgreement.txt`, distributed with the data, limits use to the ICME 2019 Grand Challenge "until the dataset is later published". It also states that no part of the database "shall be … shared or disclosed in any way … to any third party".
  - **How we treat the conflict.** We do **not** assume that the CC BY record overrides the agreement.
  - **Image origin.** The images come from MIT1003, which may carry its own terms.
- **Gaze4ASD.**
  - **The licence statement.** The README states "This dataset is provided under the MIT License", but there is no LICENSE file and no named copyright holder.
  - **Stimulus source.** Not stated.
  - **How we treat it.** We read the statement as permission to use the gaze data with attribution, not as clearance to redistribute the stimulus photographs.

**What this means in practice:**
- No dataset files, participant records, processed scanpaths or stimulus photographs are committed.
- Published results are group-level statistics, model samples and plots. Demo figures for dataset images are drawn on a blank canvas.
- Figures drawn over stimulus photographs, and real-vs-generated plots, are written only to Git-ignored `local_only` folders and must not be shared.
- **Earlier commits.** Three older figures showing dataset stimuli, one of which shows children, the usage agreement and two audits listing individual fixation records were committed in October 2026.
  - **Locally:** they have been removed from (or redacted in) every commit of the local history.
  - **On GitHub:** they remain in the published history until the rewritten history is force-pushed.
  - **Details:** see [`GIT_HISTORY_CLEANUP.md`](GIT_HISTORY_CLEANUP.md).

**Who can resolve this.** The dataset maintainers, the supervisor or the institution. The open questions are listed in [`research_audit/UNRESOLVED_ISSUES.md`](research_audit/UNRESOLVED_ISSUES.md) (Section 6).

## Other local data

`external_data/Eye-Tracking Dataset/` exists on the original workstation but is **not used** by any current code (its references were removed in commit `be5989e`).
