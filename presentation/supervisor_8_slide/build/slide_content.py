# Writes presentation/supervisor_8_slide/SLIDE_CONTENT.md from the built deck,
# so the document always matches the slides (text, tables, chart data, images).
import os
import re
from pptx import Presentation

import sys
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/") + "/"  # presentation/supervisor_8_slide/
# usage: slide_content.py [restricted]
RESTRICTED = len(sys.argv) > 1 and sys.argv[1] == "restricted"
DECK = DIR + ("local_only_restricted/ASD_ScanPath_Supervisor_Presentation_RESTRICTED.pptx" if RESTRICTED else "ASD_ScanPath_Supervisor_Presentation.pptx")
OUTMD = DIR + ("local_only_restricted/SLIDE_CONTENT_RESTRICTED.md" if RESTRICTED else "SLIDE_CONTENT.md")
prs = Presentation(DECK)

PURPOSE = {
    1: "Research gap, the three research questions, our two-phase approach and the short answer",
    2: "What we kept unchanged from TPP-Gaze vs what is new in our implementation, and why the design is minimal",
    3: "Implementation: where the group embedding enters, and that it is used in training, scoring and sampling",
    4: "Data counts and preparation, plus test images 102 and 116 with generated and real scanpaths",
    5: "Phase 1 and Phase 2 training decisions, settings and the overfitting observation",
    6: "Evaluation design: likelihood paths, three external preprocessing variants, E1 and the temperature run",
    7: "Results: likelihood vs generated behaviour, with plain verdicts",
    8: "What we can conclude, what is unresolved, limitations and next experiments",
}


def lines_of(sh):
    out = []
    if sh.has_text_frame:
        for p in sh.text_frame.paragraphs:
            t = "".join(r.text for r in p.runs).strip()
            if t and not re.fullmatch(r"\d|\+", t):  # skip numbers inside step markers and the add node
                out.append("- " + t)
    if sh.has_table:
        rows = [[c.text.replace("\n", " ") for c in r.cells] for r in sh.table.rows]
        out += ["", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        out += ["| " + " | ".join(r) + " |" for r in rows[1:]] + [""]
    if sh.has_chart:
        ch = sh.chart
        out.append(f"- **Chart:** {ch.chart_title.text_frame.text if ch.has_title else ''}")
        cats = list(ch.plots[0].categories)
        for s in ch.plots[0].series:
            out.append("  - " + ", ".join(f"{c}: {v:g}%" for c, v in zip(cats, s.values)))
    if sh.shape_type == 13:
        alt = sh._element.xpath("./p:nvPicPr/p:cNvPr/@descr")
        out.append(f"- *[Image: {alt[0] if alt else sh.name}]*")
    return out


L = ["# Slide content: 8-slide supervisor presentation" + (" (RESTRICTED version, local use only)" if RESTRICTED else " (shareable version)"), "",
     "Generated from `" + DECK.split("/")[-1] + "`. The spoken explanation for each slide is in `SPEAKING_SCRIPT.md` "
     "(and in the PowerPoint speaker notes); sources for figures and charts are in `ASSET_SOURCES.md`.", ""]
for i, sl in enumerate(prs.slides, 1):
    title = next(sh.text_frame.text for sh in sl.placeholders if "TITLE" in str(sh.placeholder_format.type))
    L += [f"## Slide {i}: {title}", "", f"*Purpose:* {PURPOSE[i]}", ""]
    for sh in sl.shapes:
        if sh.is_placeholder and "TITLE" in str(sh.placeholder_format.type):
            continue
        L += lines_of(sh)
    L.append("")
open(OUTMD, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote SLIDE_CONTENT.md for", len(prs.slides), "slides")
