# Asset sources: ASD-ScanPath research presentation

This file lists every figure, chart, diagram and data value used in `ASD_ScanPath_Research_Presentation.pptx` and its PDF, with its source.

## Sources of numbers

| Source | Used for |
|---|---|
| `paper/final/ASD_ScanPath_Final.md` (identical in content to the final PDF and DOCX) | Every result, dataset property and method detail on slides 1–13 and 15–17 |
| `inference/demo/demo_results/manifest.json` | Slide 14 only: real and generated ASD−TD dispersion contrasts for test images 102 and 116, the range of real contrasts (−0.0473 to +0.0576), and "positive on all 45 test images" |
| `research_completion/*.md`, `paper/final/FINAL_MANUSCRIPT_AUDIT.md` | Cross-checks only. Their conclusions agree with the manuscript; no number is taken from them that is not also in the manuscript. |

**Automated check.** Every numeric token in the slide text, tables, chart data and speaker notes was matched against these two source files. All tokens were found, apart from layout constants such as axis ticks and step numbers.

## Embedded images

| Slide | File in deck | Source | Processing |
|---|---|---|---|
| 10 | `assets/fig3_likelihood_gains.png` | `paper/final/figures/fig3_likelihood_gains.png` (manuscript Figure 3) | Byte-identical copy |
| 11 | `assets/fig5_e1_contrast_shares.png` | `paper/final/figures/fig5_e1_contrast_shares.png` (manuscript Figure 5) | Byte-identical copy |
| 14 | `assets/demo_best_102_panels.png` | `inference/demo/demo_results/s4a_best_102/phase_comparison_no_image.png` | Cropped to the three panels (rows 105–649, columns 6–2238). The in-image title, panel titles, captions and legend were removed, and native text was added on the slide instead. |
| 14 | `assets/demo_worst_116_panels.png` | `inference/demo/demo_results/s4a_worst_116/phase_comparison_no_image.png` | Cropped in the same way (rows 106–591) |

SHA-256 of the assets:
- `fig3_likelihood_gains.png`: `df7e3b93013dada811218f8e733c0901052116bbd8b7c91a6b3adaf12b40049c` (same as the source)
- `fig5_e1_contrast_shares.png`: `23e318eaefd6752fb7d2cdf5dc65b858c52551698d456c2110c85240f9bd4053` (same as the source)
- `demo_best_102_panels.png`: `7411f3b634ad3dd74e658a2cca289afecf8ec81d8702bcb98753b8f63d0c7400`
- `demo_worst_116_panels.png`: `228ec88590fca36b00c9a571933f1c9297272a28492275052da356a07fdefa0e`

The demo panels are model-generated scanpaths drawn on a blank grey canvas the size of the image. They contain no stimulus photograph and no real participant gaze.

## Native (editable) charts and diagrams

| Slide | Element | Data source |
|---|---|---|
| 1 | Abstract scanpath motif | Illustration only; no data |
| 2 | Location / order / duration drawings | Illustration only; no data |
| 5 | Marks and inter-event-time illustration | Illustration only; durations are arbitrary and labelled "Illustration (not data)" |
| 5 | Component list | §3.2–3.4, §3.6–3.7, Table 1 |
| 6 | Architecture diagram | Redrawn from manuscript Figures 1 and 2 and §3.5. Teal = original TPP-Gaze; orange = the added group embedding. |
| 6 | Stat cards (1,024; 0; 1,290) | §3.5 |
| 7 | Dataset table | Table 2 |
| 8 | Common-random-numbers 2 × 2 diagram | §5.3 |
| 9 | Forest plot (5 rows) | Table 3: gains and 95% image-level intervals |
| 12 | Spatial-dispersion dot plot | §6.7: ASD 0.113 vs 0.113; TD 0.112 → 0.101; real TD 0.108 |
| 12 | Bar chart, fixations per scanpath | §6.7: real 6.25 / 6.41; generated 10.19 / 9.60 |
| 12 | Table 8 excerpt | Table 8 (6 of 9 rows) |
| 13 | Bar chart, error reduction at τ = 1 vs τ = 2 | Table 9: 52.5%, 49.9%, 66.1%, 56.5% |
| 13 | Bar chart, contrast share of real | §6.6 (τ = 1: 69, 64, 32%) and §6.8 (τ = 2: 46, 33, 15%) |

## Considered and not used

| Item | Reason |
|---|---|
| `inference/demo/local_only/` (all files) | Overlays on Saliency4ASD photographs and plots of real gaze. Not for sharing; never opened for this deck. |
| Any stimulus photograph (Saliency4ASD, Gaze4ASD, `inference/inputs/`) | Dataset terms unresolved; not reproduced |
| `paper/final/figures/fig1_architecture.png`, `fig2_pipeline.png` | Replaced by the editable native diagram on slide 6 |
| `paper/final/figures/fig4_real_effects_forest.png` | Its five significant effects are summarized as text on slide 11 to keep that slide readable |
| `paper/final/figures/fig6_temperature_tau1_vs_tau2.png` | Ten panels are too dense for a slide. Replaced by native charts of Table 9 and the §6.8 shares. |
| `inference/demo/demo_results/synthetic_scene/*`, `s4a_random_*` | The best and worst examples cover the required comparisons. The synthetic scene and the random examples remain available for the live demo. |

## Build and verification

- **Build.** The deck was built with pptxgenjs 3.12.0 from `build/build.js`, using the fonts Cambria (headings) and Calibri (body), in 16:9 at 13.33 × 7.5 in.
  - The PDF and slide renders were exported by Microsoft PowerPoint 16 (`build/export.ps1`).
  - `SLIDE_CONTENT.md` is generated from the built deck by `build/slide_content.py`.
  - `build.js` uses absolute paths and the pptx skill's `apply_theme.js` helper; adjust them to rebuild elsewhere.
- **Validation.** The OOXML validator passed.
- **Visual check.** All 17 slides were rendered and inspected for clipping and overlap.
- **PDF vs PPTX.** The PDF has 17 pages of 13.33 × 7.5 in, and every slide text token is present on the matching PDF page.
- **File hashes.**
  - PPTX `2df7aedc18baadab7629ce50c929f668636295f8c7fd4ac99519a211b7f924ff`
  - PDF `0ad6f99b587e7f0c83f8eb1fe53b70e4d8cf1de5bc6439f420745b95947aa24f`
  - These are the hashes at the time of writing; they change if the deck is edited.
