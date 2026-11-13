# Asset sources: 8-slide supervisor presentation (research progress report)

There are two versions, built from one script (`build/build_supervisor.js`). Only slide 4 differs.

| Version | File | Photographs |
|---|---|---|
| **Shareable** | `ASD_ScanPath_Supervisor_Presentation.pptx` / `.pdf` | None. Slide 4 shows blank-canvas scanpaths and statistics. |
| **Restricted, local use only** | `local_only_restricted/ASD_ScanPath_Supervisor_Presentation_RESTRICTED.pptx` / `.pdf` | Saliency4ASD test photographs 102 and 116, with their generated scanpaths and real-vs-generated fixation plots. Git-ignored; "RESTRICTED · LOCAL USE ONLY" on every slide. |

**Revision note.** This revision rewrote the deck as a research progress report and dropped the illustrations on slide 1: test image 60 (restricted version) and the synthetic scene (shareable version). They are no longer used.

## Permission check for the photographs

- **Usage agreement** (`UsageAgreement.txt (distributed with Saliency4ASD; not in this repository)`, distributed with Saliency4ASD):
  - Use is restricted to the ICME 2019 Grand Challenge "until the dataset is later published".
  - "No part of the database nor any copy thereof shall be… shared or disclosed in any way… to any third party."
- **Zenodo record 13960426** lists CC BY 4.0. We do not assume the CC BY record overrides the agreement; the conflict is unresolved (`docs/research_audit/UNRESOLVED_ISSUES.md` §6; manuscript §4.2).
- **Origin.** The images come from MIT1003 (Judd et al., ICCV 2009), which may carry separate terms.
- **Decision.** Permission for a file that can be passed on is **not established**. The photographs appear only in the restricted local version.
- **Checked.** Hash comparison found that the shareable PPTX embeds no photograph and no file from `inference/demo/local_only/` or `local_only_restricted/`.

## Photographs and experimental figures (restricted version, slide 4)

Each photograph is the exact file that was tested: its SHA-256 equals the `image_sha256` recorded in `inference/demo/demo_results/<example>/samples.json`. The crops cut whole panels out of the existing demo figures, removing only titles, captions and legends. There was no resizing or redrawing, and each crop's aspect ratio equals the photograph's, so fixation positions are unchanged.

| Asset | Source | Content and restriction |
|---|---|---|
| `photo_102.png` | `TrainingData/TrainingData/Images/102.png` (byte-identical) | Test image 102, best-match example: a street with a parked car and a tree. Restricted. |
| `overlay_102_{phase1,asd,td}.png` | Panels of `inference/demo/local_only/s4a_best_102/phase_comparison.png` | Phase 1, Phase 2 ASD and Phase 2 TD samples over the photo (seed 0, τ = 1). Restricted. |
| `rvg_102_{real,gen}_{asd,td}.png` | The four panels of `inference/demo/local_only/s4a_best_102/real_vs_generated.png` | All real fixations (6 ASD, 10 TD scanpaths) and E1 replicate 1 generated fixations (20 per label). Real participant gaze, so restricted. |
| `photo_116.png` | `TrainingData/TrainingData/Images/116.png` (byte-identical) | Test image 116, worst-match example: three identifiable adults opening a gift. Restricted. |
| `overlay_116_{phase1,asd,td}.png` | Panels of `inference/demo/local_only/s4a_worst_116/phase_comparison.png` | As for 102. Restricted. |
| `rvg_116_{real,gen}_{asd,td}.png` | Panels of `inference/demo/local_only/s4a_worst_116/real_vs_generated.png` | Real (7 ASD, 14 TD scanpaths) and generated (20 per label) fixations. Restricted. |

**Selection.** The best and worst agreement, among the 45 Saliency4ASD test images, between generated and real ASD−TD dispersion. The rule was fixed before any demo output was generated (`inference/demo/demo_results/manifest.json`).

**Values shown.**
- **Row labels:** 10-seed E1 averages from `manifest.json` (+0.0133 vs +0.014; −0.0473 vs +0.0151).
- **Statistics:** replicate-1 values from `real_vs_generated_summary` in `samples.json`: 0.132 / 0.118 / 0.150 / 0.138 for 102, and 0.117 / 0.165 / 0.162 / 0.109 for 116.

## Shareable assets (slide 4)

| Asset | Source | Processing |
|---|---|---|
| `assets/blank_{102,116}_{phase1,asd,td}.png` | Panels of `inference/demo/demo_results/s4a_{best_102,worst_116}/phase_comparison_no_image.png` | Panel crop: a blank canvas of the image size, no photograph |

In the shareable version, slide 4 has a grey "Photograph withheld" box in place of each photograph, and the dispersion statistics in place of the real-gaze plots.

## Native diagrams and charts (both versions)

| Slide | Element | Source |
|---|---|---|
| 1 | Gap, research questions, approach steps and short answer | Manuscript §1, §2.4, §3.5, §6.2–6.7 |
| 2 | Kept vs new comparison | Manuscript §3.2, Table 1, §3.5, §4.4; `training/conditioned.py`; `training/runs/phase2_embedding_only/summary_full.json` (23,896,736 total, 1,024 trainable) |
| 3 | Architecture diagram and callouts | Manuscript §3.3–3.6, Eq. 2, Figure 1; `training/conditioned.py`, `train_tpp_cond.py`, `generate_cond.py` |
| 4 | Dataset counts | Manuscript §4.1–4.2, Table 2 |
| 5 | Pipeline steps and Phase 1 vs Phase 2 table | `configs/phase1_baseline.yaml`, `configs/phase2_embedding_only.yaml`, `training/runs/baseline/split.json`, `training/runs/*/summary_full.json` (best and last-epoch validation NLL); manuscript §4.3–4.4 |
| 6 | Evaluation design | Manuscript §3.8, §4.3 (mapping evidence), §5.1–5.5, §6.4 (7.35% vs 2.72% off-image) |
| 7 | Forest plot | Table 3 (in-domain) and Table 5 (three external versions, CPU re-scoring; the primary −0.0172 matches the Table 4 GPU value) |
| 7 | Bar chart | Table 7 / §6.6 (69, 64, 32%); independent run 60 / 50 / 29% (§6.6) |
| 7 | Results table | §6.1 (pretrained −0.1665 → −0.4957), Tables 3–4, 7–8, §6.7, §6.8 (τ = 2 shares 46 / 33 / 15%) |
| 8 | Conclusions, open questions, limitations, next experiments | Manuscript §4.2, §6.3–6.7, §7–9; `inference/demo/demo_results/manifest.json` (positive generated contrast on 45 of 45 test images); `docs/research_audit/UNRESOLVED_ISSUES.md` |

## Number check

Every number on the slides of both versions, and in `SPEAKING_SCRIPT.md`, was matched against the manuscript, the training configs and run summaries, and the demo manifest and `samples.json` files. The exceptions are deliberate:
- "about 3.9 million" = 3,881,784, rounded;
- "almost 24 million" = 23,895,712, rounded;
- "0.053" = 0.162 − 0.109, from `samples.json`.

The manuscript gives dispersion ≈ 69% and saccade amplitude ≈ 64% (Table 7), and the slides follow it.

## Not used, and why

| Item | Reason |
|---|---|
| Gaze4ASD stimuli | Mostly face photographs, including children; their source is not stated, and the README's MIT statement does not establish redistribution rights |
| `inference/inputs/*.png` | Copies of Gaze4ASD stimuli or web photographs of unknown licence |
| Test images 60 and 246, the seed-grid and the temperature figures | Not needed for the research-report structure; available in `inference/demo/local_only/` for questions |
| Synthetic scene | Illustrated the general concept only, which this audience does not need |

## Asset hashes (SHA-256)

**Restricted** (`local_only_restricted/assets/`):
```
b9171de0c851da3f0bb1583cf36e7aacf3859145041cc6139a05c9b4b993d3ce  photo_102.png
1bcddc7973a6cf340c48b0098f77e4a7a313dd0cc549e618789bb169a4720abd  photo_116.png
278d93cd7fa661f8a955d53d2461fd905fa2e67c1a3ff747b12d6632e6475b2a  overlay_102_phase1.png
a84629442064701bd56b3d39d214e38e204369cea6e7026b0b0b94a687a359fe  overlay_102_asd.png
385915ce308ab9b143026b3aed494c7cad34dda8ae026918f0f81156a51872f7  overlay_102_td.png
9026367798bf60634c4e58317e5daedeffa409b15cbf4b8b01851492fcf09554  rvg_102_real_asd.png
ad79c17d80c80d3ad671637af1c965256ae5561d47901226d59a46a263b2f53d  rvg_102_real_td.png
3fbfde7ceb22e022cecb0e55e9c81b67fd103db220468095c588761147482097  rvg_102_gen_asd.png
dcde51e6f5714462f0feac88f25e4acb8d8a433312579c695711d1c14c812a7c  rvg_102_gen_td.png
49f171d29b79d3f0db3593e72415175e3e4296c4dcdbcd96d4ee6e482daee34f  overlay_116_phase1.png
5cc2cde06ef492057423b24195cde1a2fa1f772052e248aa50fb4c6ff576a8ea  overlay_116_asd.png
3e943937fb8f6a2c75e269abef354cc73f0a77c093cd3bc76447965bb82e8017  overlay_116_td.png
da87085d9326a1acc721208f28e017584b47d268763df3da87725e23f1c540c6  rvg_116_real_asd.png
202e3e9dfd6519fa6930bc91314d7af25c59fecf0bae11c610690ea5db8a0ce8  rvg_116_real_td.png
52b2d6ec6b302d0955e2fa2c3a55de7c063d32275570f0bfe1bfe8e9c6e13805  rvg_116_gen_asd.png
3ab2bb5a9cf07185e5aad939ebb7ffe605c0515469961baba001079debc99df3  rvg_116_gen_td.png
```
**Shareable** (`assets/`):
```
0376e6d88d7411efa9d9d27c5c22013cf6b3ab6bc06b80c6213cdeb26dcea654  blank_102_phase1.png
71cfefbe3c75b7e6aa05fa60cea24ac7d0ebe2a4c378fb724e3040b707218fba  blank_102_asd.png
7644c8eb65b48a4b8d6b4ecf5ee8f500c9ac37278862339a2cb37536767042c5  blank_102_td.png
1322e3dead2097a9d2fc34a0b4431b0187502521271999a5d6d03d22d0d4dc47  blank_116_phase1.png
31a767362efc7ddd7e112c33b8246a295aedb3b5068e14878e5f8547d13ffece  blank_116_asd.png
146b0819f6570835a73d96243f328c2fa7a8798b007c63b62387bfd755b42d13  blank_116_td.png
```

## Build

- **Generator.** `build/build_supervisor.js` (pptxgenjs 3.12.0). The default build is the shareable deck; `DECK=restricted` builds the restricted one, and `OUT_PATH` overrides the output file.
- **Speaker notes.** Taken from `SPEAKING_SCRIPT.md`; each version gets only its own slide 4 paragraph.
- **Export.** `build/export.ps1` produces the PDF through Microsoft PowerPoint and never closes a presentation you have open.
- **Slide content.** `build/slide_content.py [restricted]` writes the slide-content file.
- **Checks.** The OOXML validator passed for both decks; all 8 slides were rendered and inspected; each PDF matches its PPTX.
