# Presentation Material

All published material here is shareable: it contains no dataset stimulus photographs and no real gaze plots.

| Folder | Contents |
|---|---|
| [`supervisor_8_slide/`](supervisor_8_slide/ASSET_SOURCES.md) | **Latest supervisor presentation.** An 8-slide research progress report (PPTX and PDF), with its speaking script (`SPEAKING_SCRIPT.md`), slide text (`SLIDE_CONTENT.md`), asset sources and build scripts. The shareable version shows blank-canvas scanpaths. |
| [`supporting_material/meeting_walkthrough/`](supporting_material/meeting_walkthrough/README.md) | The voice-led walkthrough used in the supervisor meeting: numbered figures to open in order, and the script to read (`00_SCRIPT.md`) |
| [`supporting_material/research_presentation_17_slide/`](supporting_material/research_presentation_17_slide/ASSET_SOURCES.md) | Earlier 17-slide presentation, with speaking script, 43-question Q&A (`RESEARCH_QA.md`) and rehearsal checklist |

## Local-only versions

These folders exist only on the original workstation and are Git-ignored:
- `supervisor_8_slide/local_only_restricted/`: the version of the 8-slide deck with the actual Saliency4ASD test photographs;
- `supporting_material/meeting_walkthrough/01_example_image/`: a photograph whose licence is unknown.

**Do not add them to the repository.** The Saliency4ASD usage agreement forbids sharing the data, and its relation to the CC BY record is unresolved.

## Notes

- **Where the numbers come from.** The slides were written from the final manuscript, the training configurations and the demo results. Paths quoted on the slides use the layout before the 2026-11 reorganisation; see [`../docs/reproducibility.md`](../docs/reproducibility.md) for the mapping.
- **Rebuilding the decks** needs:
  - Node.js with `pptxgenjs` 3.12;
  - the theme helper of the local "pptx" skill (`PPTX_SKILL_DIR`);
  - Microsoft PowerPoint, for the PDF export (`build/export.ps1`).

  These tools are not needed for the research code. `build/slide_content.py` (python-pptx) regenerates `SLIDE_CONTENT.md` from a deck.
