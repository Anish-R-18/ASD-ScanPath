# Writes this deck's SLIDE_CONTENT.md from the built deck, so the document
# always matches the slides (on-slide text, tables, chart data and speaker notes).
import os
import re
from pptx import Presentation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/") + "/"  # this deck's folder
prs = Presentation(ROOT + "ASD_ScanPath_Research_Presentation.pptx")

META = {
    1: ("Title", "Manuscript title and abstract", "Abstract scanpath motif (illustration, not data)"),
    2: ("Motivation and background", "§1, §2.3; refs [1]–[3]", "Three drawn cards (location, order, duration), illustrations only"),
    3: ("Problem statement and objectives", "§1 (research question, contributions, what is not claimed)", "Four numbered research-question cards; 'not claimed' panel"),
    4: ("Related work and research gap", "§2.1–2.4", "Four literature cards and a gap statement"),
    5: ("TPP-Gaze background", "§3.1–3.4, §3.6–3.7, Table 1", "Native illustration of marks and inter-event times (not data); component list"),
    6: ("Proposed ASD/TD conditioning", "§3.5, Figure 1, Figure 2, §4.4", "Native, editable architecture diagram redrawn from Figures 1–2; original model in teal, our addition in orange"),
    7: ("Datasets and preprocessing", "Table 2, §4.1–4.3", "Native table (Table 2) and preprocessing steps"),
    8: ("Experimental setup and metrics", "§4.4, §5.1, §5.3, §5.5", "Three-column design; native 2 × 2 diagram of the common-random-numbers design"),
    9: ("In-domain likelihood", "§6.1, §6.2, Table 3", "Native forest plot of Table 3 gains with 95% intervals"),
    10: ("External Gaze4ASD evaluation", "§4.3 (mapping evidence), §6.3, §6.4, Tables 4–5, Figure 3", "Manuscript Figure 3; verdict blocks"),
    11: ("Generated behavioural differences (E1)", "§5.3, §6.5, §6.6, Tables 6–7, Figure 5", "Manuscript Figure 5; share callouts"),
    12: ("Per-group realism", "§6.7, Table 8, MultiMatch paragraph", "Native dot plot (dispersion), native bar chart (fixation counts), native excerpt of Table 8"),
    13: ("Sampling temperature and sensitivity", "§3.8, §6.8, Table 9, Figure 5 (τ = 2 shares)", "Two native bar charts (Table 9 reductions; contrast shares at τ = 1 and τ = 2)"),
    14: ("Qualitative demo examples", "demo/demo_results/manifest.json; demo/DEMO_GUIDE.md", "Blank-canvas demo strips for test images 102 (best match) and 116 (worst match)"),
    15: ("Limitations and threats to validity", "§8, §6.4 (off-image gaze)", "3 × 3 grid of limitation cards"),
    16: ("Conclusions and future work", "§9, §7", "Verdict list and future-work panel (dark slide)"),
    17: ("References (backup)", "Manuscript reference list", "Two-column reference list, numbering as in the manuscript"),
}


def shape_lines(sh):
    out = []
    if sh.has_text_frame:
        for p in sh.text_frame.paragraphs:
            t = "".join(r.text for r in p.runs).strip()
            if t:
                out.append(t)
    if sh.has_table:
        rows = [[c.text.replace("\n", " ") for c in r.cells] for r in sh.table.rows]
        out.append("")
        out.append("| " + " | ".join(rows[0]) + " |")
        out.append("|" + "---|" * len(rows[0]))
        for r in rows[1:]:
            out.append("| " + " | ".join(r) + " |")
        out.append("")
    if sh.has_chart:
        ch = sh.chart
        title = ch.chart_title.text_frame.text if ch.has_title else "Chart"
        out.append(f"**Chart: {title}**")
        cats = list(ch.plots[0].categories)
        for s in ch.plots[0].series:
            vals = ", ".join(f"{c}: {v:g}" for c, v in zip(cats, s.values))
            out.append(f"- {s.name}: {vals}")
    if sh.shape_type == 13:  # picture
        alt = sh._element.xpath("./p:nvPicPr/p:cNvPr/@descr")
        out.append(f"*[Image: {alt[0] if alt else sh.name}]*")
    return out


L = []
L.append("# ASD-ScanPath research presentation: slide content")
L.append("")
L.append("Generated from `ASD_ScanPath_Research_Presentation.pptx` (17 slides: 16 main slides and 1 backup). "
         "Every number on the slides comes from `paper/final/ASD_ScanPath_Final.md`; the demo selection values come from "
         "`demo/demo_results/manifest.json`. See `ASSET_SOURCES.md` for figures and charts.")
L.append("")
L.append("## Narrative")
L.append("")
L.append("1. **Why** (slides 2–4): gaze differs between autistic and typically developing children in where, in what order and for how long they look. TPP-Gaze models location and duration jointly. We ask what a minimal group label changes.")
L.append("2. **How** (slides 5–8): TPP-Gaze as a marked point process; one zero-initialized 2 × 512 embedding added before both heads of the frozen model; Saliency4ASD for training and testing, Gaze4ASD only for external evaluation; likelihood and a common-random-numbers behavioural comparison.")
L.append("3. **What we found** (slides 9–13): in-domain gain inconclusive; external fit worse; generated spatial contrasts right in direction at reduced size, temporal and direction-concentration contrasts wrong; neither group more realistic; temperature changes generated behaviour, not likelihood.")
L.append("4. **What it looks like** (slide 14): same image and seed, Phase 1 vs Phase 2 ASD vs Phase 2 TD; a best-match and a worst-match image; one sample is not a child.")
L.append("5. **What it means** (slides 15–16): limitations, conclusions and future work. Not a diagnostic tool.")
L.append("")
total = 0
L.append("## Timing")
L.append("")
L.append("| Slide | Topic | Suggested time |")
L.append("|---|---|---|")
times = {}
for i, sl in enumerate(prs.slides, 1):
    notes = sl.notes_slide.notes_text_frame.text if sl.has_notes_slide else ""
    m = re.search(r"about (\d+) s", notes)
    times[i] = int(m.group(1)) if m else 0
    total += times[i]
    L.append(f"| {i} | {META[i][0]} | {str(times[i]) + ' s' if times[i] else 'backup'} |")
L.append("")
L.append(f"Total for the main slides: about {total // 60} min {total % 60} s, plus about 20 s if the optional live demo command is run on slide 14. Questions are extra.")
L.append("")
for i, sl in enumerate(prs.slides, 1):
    title = next((sh.text_frame.text for sh in sl.placeholders if sh.placeholder_format.type is not None and "TITLE" in str(sh.placeholder_format.type)), META[i][0])
    L.append(f"## Slide {i}: {title.strip()}")
    L.append("")
    L.append(f"- **Section:** {META[i][0]}")
    L.append(f"- **Source:** {META[i][1]}")
    L.append(f"- **Visual:** {META[i][2]}")
    L.append("")
    L.append("**On-slide content**")
    L.append("")
    for sh in sl.shapes:
        if sh.is_placeholder and "TITLE" in str(sh.placeholder_format.type):
            continue
        for line in shape_lines(sh):
            if line.startswith("|") or line == "" or line.startswith("**Chart") or line.startswith("- ") or line.startswith("*[Image"):
                L.append(line)
            elif re.fullmatch(r"\d+|[+]", line):
                continue  # numbers inside fixation circles and the add node
            else:
                L.append("- " + line)
    L.append("")
    notes = sl.notes_slide.notes_text_frame.text.strip() if sl.has_notes_slide else ""
    if notes:
        L.append("**Speaker notes**")
        L.append("")
        for para in notes.split("\n"):
            if para.strip():
                L.append("> " + para.strip())
                L.append(">")
        if L[-1] == ">":
            L.pop()
        L.append("")

open(ROOT + "SLIDE_CONTENT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote SLIDE_CONTENT.md,", total, "s of main-slide time")
