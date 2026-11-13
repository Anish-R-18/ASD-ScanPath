# Unresolved Issues

These issues cannot be settled with local data and code.
- **C:** resolvable with external information or confirmation.
- **D:** cannot be resolved with the released data.

Each entry gives the current evidence, why it matters, and who could resolve it.

## 1. Literature access (C)

| Item | What was attempted | Result | Consequence |
|---|---|---|---|
| Zhong et al., *Pattern Recognition* 180:114305 (2026), doi 10.1016/j.patcog.2026.114305 | Publisher page and DOI | HTTP 403 or redirect stub. **Institutional access was not available from this environment.** Only the abstract and metadata were read. | The paper's comparison with this work rests on the abstract. Datasets, metrics, numbers and whether code exists are unverified. |
| Gaze4ASD main paper, ICME 2025, doi 10.1109/ICME59968.2025.11210055 | IEEE Xplore and DOI | Not accessible; metadata only. The local supplement was read. | The published participant count, stimulus source, display procedure and coordinate convention are unverified. |
| Jarodzka et al. 2010; Dewhurst et al. 2012 (MultiMatch) | Not retrieved in full | Method checked against the reference implementation instead | The method description in the paper should cite the papers and the implementation actually used. |

**Who can resolve this:** a supervisor or the library, with institutional access. The Gaze4ASD authors can answer the dataset questions.

**Renewed attempt (final manuscript phase, 2026-11-03).**
- Both DOIs were tried again. They redirect to ScienceDirect (`linkinghub.elsevier.com`) and IEEE Xplore, which returned empty pages, so the full texts are still unavailable.
- The final manuscript (`paper/final/`) states this in Section 2.4 and in the Limitations, and limits its statements to the abstract, the dataset README and the supplement.
- One new reference was added and verified against Crossref: `multimatch-gaze` (Wagner, Halchenko and Hanke, *Journal of Open Source Software* 4(40):1525, 2019, doi 10.21105/joss.01525).

## 2. Gaze4ASD dataset facts (C)

- **132 vs 133 TD participants.** The README says 132. Raw (`.mat`) and scanpath folders both contain 133 TD participants, and all 133 contribute trials. Possible explanations include a late addition or a count error. Ask the authors, or check the main paper.
- **Coordinate mapping.** Fit-to-height (1440/H scaling, centred horizontally) is strongly supported, but it is not confirmed by the presentation code. It must be called "strongly supported but partly inferred" until the authors confirm it. E2 and A3 show which conclusions depend on it (see the additional results).

  Source: `external_data/Gaze4ASD_processed_v2/reports/mapping_methodology.txt`.

  | Candidate | Authors' saliency maps, median r | Fixations inside the image |
  |---|---|---|
  | Fit-to-height | 0.9999 | 96.5% |
  | Stretch | 0.91 | — |
  | Native centred placement | 0.23 | 81.4% |

  Gaze alone does not exclude stretch, so it is a weaker alternative, not a refuted one.
- **Stimulus source.** The README says only that the 30 images were "selected by our Image Selection Module". Their origin, and therefore their copyright, is not stated.

## 3. Age (D, with C for summary statistics)

- The released Gaze4ASD raw files contain system information, calibration, gaze samples and the display rectangle, but **no age, sex or date-of-birth field**. The README gives only group means (ASD 5.04, TD 4.54 years, range 3–7). Saliency4ASD gives a range (5–12) and a mean (8), with no per-participant values.
- **Consequence:** ASD vs TD effects cannot be age-adjusted, and dataset differences (Saliency4ASD is older) are confounded with age, eye tracker, display, stimuli and site. Only the authors could supply participant-level age, and that would probably need an ethics amendment.
- **What is possible now:** state the confound plainly, and report the 0.5-year group difference in mean age as a known imbalance with unknown spread.

## 4. Saliency4ASD sequences (C)

The README describes 14 participants per group per image. The raw files contain 8–17 (ASD) and 9–15 (TD) index-reset sequences per image, so some images exceed 14. Whether one sequence is one participant's viewing cannot be determined from the release. This affects how "scanpath" maps to "participant" for in-domain evaluation. There are no participant IDs, so in-domain clustering can only be by image (see E3).

## 5. Model-selection history (C, internal)

`archive/configs/phase2_lr1e-3_no_saved_results.yaml` (learning rate 1e-3) and comments in `config_phase2_embonly.yaml` ("RETRY", "lowered from 1e-3") show that an earlier Phase 2 configuration existed. Its output folder (`training/runs/phase2_conditioned/`) does not exist locally, and no metrics were found. Whoever ran it should state whether it was evaluated, and on which split. If it was evaluated on validation data only, the paper should report one discarded configuration. Nothing indicates the test split was used.

## 6. Data terms (C)

**Saliency4ASD.** The Zenodo record (13960426) lists **CC BY 4.0**. The `UsageAgreement.txt` in the release restricts use and sharing "until the dataset is later published". The Zenodo licence is **not** assumed to override the agreement automatically. Points to consider:
- the agreement may predate the Zenodo publication;
- the images come from MIT1003, which may carry separate terms.

Ask the dataset maintainers, or the supervisor or institution, which terms apply to (a) research use, (b) publishing derived statistics and figures, and (c) redistributing stimuli.

**Gaze4ASD.** The README states "This dataset is provided under the MIT License". However:
- there is no LICENSE file and no named copyright holder;
- MIT is a software licence and is unusual for a dataset of children's gaze;
- the stimulus images' origin is not stated, so the README alone cannot establish redistribution rights for the stimuli.

Treat the MIT statement as permission to use the gaze data with attribution, but **not** as clearance to redistribute stimulus photographs.

**Practice in this audit.** No dataset contents or stimulus photographs were redistributed. All new outputs are statistics, model samples and normalized coordinates, and they stay local.

## 7. Stimulus figures in the public repository (decision needed)

Three earlier tracked figures show dataset stimuli, including photographs of children:
- `archive/results/final_results_2026-11-03/figures/fig4a_*.png`
- `archive/results/final_results_2026-11-03/figures/fig4b_*.png`
- `archive/results/external_gaze4asd_native_mapping/fig_representative_scanpaths.png`

They are in the public repository and its history. This audit did not change them (no commits were allowed). The options are:
1. keep them, once rights are confirmed;
2. remove them in a new commit (they remain in history);
3. rewrite history (disruptive).

This is the repository owner's decision. Because the photographs show children, caution is warranted whatever the licence says.

## 8. Fundamental limits (D)

These cannot be resolved with any further analysis of the existing data:
- **Cross-dataset shift is confounded.** Saliency4ASD and Gaze4ASD differ in age, eye tracker (Tobii T120 at 1280×1024 vs Tobii Pro Fusion at 2560×1440), viewing distance, stimuli, site and diagnostic procedure at once. Why the external TD fit worsens cannot be attributed to any single factor.
- **Small ASD sample.** There are 33 ASD participants externally, and in-domain samples per image are small. Subgroup intervals are wide.
- **No clinical validity.** Neither dataset supports diagnostic claims, and none are made.
- **Saliency4ASD has no participant identity**, so in-domain uncertainty can only be estimated across images.
