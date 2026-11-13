# Meeting Walkthrough

**What this is.** The supervisor meeting was run by voice, opening the files in this folder in order, rather than presenting slides. The script, with what to open and what to say at each step, is [`00_SCRIPT.md`](00_SCRIPT.md). It takes about 12 minutes.

## Contents

| Folder | Contents | Source |
|---|---|---|
| `01_example_image/` | **Local only (Git-ignored).** An outside photograph (`inference/inputs/beach.png`, licence unknown) and its ASD vs TD scanpaths. These were made by `inference/inference.py` on 2026-10-04 with the reported Phase 2 checkpoint, at τ = 2, seed 0, one sample per label. They are an illustration, not evidence. | Copies of the local files |
| `02_what_we_did/` | `1_our_approach.png`; `2_architecture.png` (manuscript Figure 1) | Drawn from reported values; Figure 1 is copied unchanged |
| `03_training/`, `04_testing/`, `05_novelty/` | One summary figure each | Drawn from reported values |
| `06_results/` | Results at a glance, likelihood, generated behaviour, conclusion and next steps | Drawn from reported values |

**Where the numbers come from.** Every number in the figures comes from the final manuscript (§6.1 and Tables 3, 5, 7 and 8), the training configurations and run summaries, or the demo results. No experiment was re-run.

**Redrawing the figures.** They are drawn by `_build/make_meeting_figs.py`; run it from the repository root:

```bash
python presentation/supporting_material/meeting_walkthrough/_build/make_meeting_figs.py presentation/supporting_material/meeting_walkthrough
```
