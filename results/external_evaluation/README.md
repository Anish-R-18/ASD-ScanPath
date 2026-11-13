# Corrected Gaze4ASD Results

Publication summaries for the corrected Gaze4ASD evaluation. Built by `training/build_results_v2.py` from local outputs that Git ignores. `tables/provenance.csv` lists each output's source file and its SHA-256 hash.

**Status.** The fit-to-height mapping is strongly supported but partly inferred, and not experimentally confirmed. Statistics are exploratory unless marked. No clinical or diagnostic validity is claimed.

## Main results

| Measure | Result |
|---|---|
| External NLL, Phase 1 vs Phase 2 | -0.5568 vs -0.5396: Phase 1 is better overall by 0.0172 |
| In-domain NLL, Phase 1 vs Phase 2 | -0.4957 vs -0.5030: gain +0.0073 for Phase 2, 95% image-level CI [-0.0012, +0.0163]; statistically inconclusive |
| Selected temperature | τ=1, chosen on Saliency4ASD validation images |
| Fixation count at τ=1, ASD and TD | 10.24 and 9.62, against 6.25 and 6.41 real |

Temperature changes sampling only, not the model or its likelihood.

## Contents

| Path | Contents |
|---|---|
| `final_external_comparison.txt` | Phase 1 vs Phase 2, ASD and TD NLL, participant-level statistics, generated behaviour, and the full comparison report |
| `temperature_sensitivity.txt` | Validation-set selection and Gaze4ASD evaluation at τ=1, with methods |
| `corrected_mapping_methodology.txt` | Display mapping, its evidence and uncertainty, and dataset versions |
| `tables/` | Group-level and image-pooled result tables |
| `figures/` | Selected figures |

## Deliberately excluded

Raw or processed datasets; per-scanpath and per-participant records; generated scanpaths; checkpoints; training logs; figures that show dataset stimuli; and anything based on the old native-size mapping, which is archived in [`../../archive/results/external_gaze4asd_native_mapping/`](../../archive/results/external_gaze4asd_native_mapping/SUPERSEDED.txt).

Other result folders: [`../README.md`](../README.md) (index of all results).
