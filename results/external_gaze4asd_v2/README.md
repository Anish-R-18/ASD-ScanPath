# Corrected Gaze4ASD Results

Publication summaries for the corrected Gaze4ASD evaluation. Built by `training/build_results_v2.py` from local outputs that Git ignores. `tables/provenance.csv` lists each output's source file and its SHA-256 hash.

**Status.** The fit-to-height mapping is strongly supported but partly inferred, and not experimentally confirmed. Statistics are exploratory unless marked. No clinical or diagnostic validity is claimed.

## Main results

| Measure | Result |
|---|---|
| External NLL, Phase 1 vs Phase 2 | -0.5568 vs -0.5396: Phase 1 is better overall by 0.0172 |
| In-domain NLL, Phase 1 vs Phase 2 | -0.4957 vs -0.5030: Phase 2 is better by 0.0073 |
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

Raw or processed datasets; per-scanpath and per-participant records; generated scanpaths; checkpoints; training logs; figures that show dataset stimuli; and anything based on the old native-size mapping, which is marked superseded in [`../external_gaze4asd/SUPERSEDED.txt`](../external_gaze4asd/SUPERSEDED.txt).
