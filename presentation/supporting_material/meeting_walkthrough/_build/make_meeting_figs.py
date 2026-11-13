"""Builds the meeting-walkthrough figures. Every number is copied from
paper/final/ASD_ScanPath_Final.md, training/runs/*/summary_full.json or
demo/demo_results/manifest.json. No new experiment is run."""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1]
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

INK = "#1F2328"; MUTED = "#5A6270"; LINE = "#D0D5DD"; BG = "#FFFFFF"
TEAL = "#1F7A7A"; TEAL_L = "#E3F2F2"
OURS = "#C46A00"; OURS_L = "#FFF0DC"   # our addition (matches manuscript fig1 orange)
GOOD = "#1E7B34"; GOOD_L = "#E3F4E7"
BAD = "#B42318"; BAD_L = "#FDE8E6"
WARN = "#9A6700"; WARN_L = "#FFF6D6"
NEUT = "#57606A"; NEUT_L = "#EEF0F3"
ASD = "#D9730D"; TD = "#2F6DB5"


def canvas(title, subtitle=None):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(0.5, 8.45, title, fontsize=27, fontweight="bold", color=INK, va="center")
    if subtitle:
        ax.text(0.5, 7.95, subtitle, fontsize=15, color=MUTED, va="center")
    return fig, ax


def box(ax, x, y, w, h, fc, ec, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw))


def card(ax, x, y, w, h, head, lines, fc, ec, hc=None, fs=14.5, hfs=17, gap=0.40):
    box(ax, x, y, w, h, fc, ec)
    ax.text(x + 0.25, y + h - 0.38, head, fontsize=hfs, fontweight="bold", color=hc or ec, va="center")
    yy = y + h - 0.95
    for ln in lines:
        if ln == "":
            yy -= gap * 0.45
            continue
        bold = ln.startswith("**")
        t = ln.strip("*")
        ax.text(x + 0.25, yy, t, fontsize=fs, color=INK, va="center",
                fontweight="bold" if bold else "normal")
        yy -= gap


def chip(ax, x, y, text, fc, ec, w=2.3, h=0.46, fs=12.5):
    box(ax, x, y - h / 2, w, h, fc, ec, lw=1.4)
    ax.text(x + w / 2, y, text, fontsize=fs, fontweight="bold", color=ec, ha="center", va="center")


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=2.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22,
                                 color=color, lw=lw))


def footer(ax, text):
    ax.text(0.5, 0.3, text, fontsize=11.5, color=MUTED, va="center", style="italic")


def save(fig, rel):
    p = os.path.join(OUT, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, facecolor=BG); plt.close(fig); print("wrote", rel)


# ---------------------------------------------------------------- 02 results / 1 at a glance
fig, ax = canvas("Results at a glance",
                 "Phase 1 = TPP-Gaze fine-tuned on our data, no label.   Phase 2 = same model + a small ASD/TD label.")
rows = [
    ("LIKELIHOOD (does it fit real gaze better?)", None, None, None),
    ("Phase 1 fine-tuning vs released TPP-Gaze", "Test NLL  −0.1665  →  −0.4957  (lower is better)", "IMPROVED", "good"),
    ("Phase 2 vs Phase 1, our 45 test images", "+0.0073 nats/fixation, 95% CI [−0.0012, +0.0163]", "INCONCLUSIVE", "neut"),
    ("Phase 2 vs Phase 1, Gaze4ASD (never seen)", "−0.0172; Phase 1 better in all 3 preprocessing versions", "WORSE", "bad"),
    ("GENERATED BEHAVIOUR (Gaze4ASD, 10 seeds)", None, None, None),
    ("Does the label change the scanpaths?", "8 of 9 measures beyond seed-only baseline; ~1% of image diagonal", "YES, SMALL", "warn"),
    ("Spatial ASD−TD differences", "Right direction: dispersion 69%, amplitude 64%, revisit 32% of real", "PARTIAL", "warn"),
    ("Fixation count, total time, direction", "Go the opposite way to the real ASD−TD difference", "WRONG WAY", "bad"),
    ("Is either group more realistic?", "No spatial measure moves closer to its own real group", "NOT IMPROVED", "bad"),
    ("SAMPLING", None, None, None),
    ("Temperature 2 instead of 1", "Same directions, smaller contrasts (46 / 33 / 15%); likelihood unchanged", "CAVEAT", "neut"),
]
pal = {"good": (GOOD_L, GOOD), "bad": (BAD_L, BAD), "warn": (WARN_L, WARN), "neut": (NEUT_L, NEUT)}
y = 7.35
for q, a, v, k in rows:
    if a is None:
        ax.text(0.5, y, q, fontsize=13.5, fontweight="bold", color=MUTED, va="center")
        ax.plot([0.5, 15.5], [y - 0.24, y - 0.24], color=LINE, lw=1.2)
        y -= 0.56
        continue
    ax.text(0.65, y, q, fontsize=15, color=INK, va="center", fontweight="bold")
    ax.text(5.65, y, a, fontsize=14.5, color=INK, va="center")
    chip(ax, 13.15, y, v, *pal[k])
    y -= 0.6
footer(ax, "Sources: manuscript §6.1, Tables 3–5, 7, 8, §6.8. NLL in nats per fixation; positive gain = Phase 2 better.")
save(fig, "06_results/1_results_at_a_glance.png")

# ---------------------------------------------------------------- 02 results / 2 likelihood
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
fig.text(0.031, 0.94, "Likelihood: fine-tuning helped, the label did not",
         fontsize=27, fontweight="bold", color=INK, va="center")
fig.text(0.031, 0.885, "Joint negative log-likelihood (NLL), nats per fixation. Lower NLL = better fit.",
         fontsize=15, color=MUTED, va="center")
a1 = fig.add_axes([0.06, 0.16, 0.25, 0.62])
vals = [-0.1665, -0.4957]
bars = a1.bar([0, 1], vals, color=[NEUT, TEAL], width=0.55)
for b, v in zip(bars, vals):
    a1.text(b.get_x() + b.get_width() / 2, v - 0.02, f"{v:.4f}".replace("-", "−"), ha="center", va="top", fontsize=15,
            fontweight="bold", color=INK)
a1.set_xticks([0, 1]); a1.set_xticklabels(["Released\nTPP-Gaze", "Phase 1\n(fine-tuned)"], fontsize=14)
a1.set_ylim(-0.6, 0.02); a1.axhline(0, color=INK, lw=1)
a1.set_title("Saliency4ASD test set (45 images)", fontsize=15, color=INK, pad=12)
a1.set_ylabel("test NLL (lower = better)", fontsize=13, color=MUTED)
for s in ["top", "right"]: a1.spines[s].set_visible(False)
a1.tick_params(labelsize=12, colors=MUTED)

a2 = fig.add_axes([0.47, 0.16, 0.49, 0.62])
lab = ["Our test images\n(in-domain)", "Gaze4ASD, primary\n(fit-to-height, clip)",
       "Gaze4ASD\n(stretch, clip)", "Gaze4ASD\n(fit-to-height, remove)"]
mid = [0.0073, -0.0172, -0.0274, -0.0087]
lo = [-0.0012, -0.0225, -0.0322, -0.0130]
hi = [0.0163, -0.0122, -0.0226, -0.0042]
col = [NEUT, BAD, BAD, BAD]
yy = [3, 2, 1, 0]
a2.axvspan(0, 0.025, color=GOOD_L, zorder=0); a2.axvspan(-0.04, 0, color=BAD_L, zorder=0)
a2.axvline(0, color=INK, lw=1.3)
for m, l, h, c, yv in zip(mid, lo, hi, col, yy):
    a2.plot([l, h], [yv, yv], color=c, lw=3.5, solid_capstyle="round")
    a2.plot(m, yv, "o", ms=13, color=c, mec="white", mew=2)
    fm = lambda v: f"{v:+.4f}".replace("-", "−")
    a2.text(0.0275, yv, f"{fm(m)}  [{fm(l)}, {fm(h)}]", va="center", fontsize=13, color=INK,
            fontweight="bold" if yv == 3 else "normal")
a2.set_yticks(yy); a2.set_yticklabels(lab, fontsize=13.5)
a2.set_xlim(-0.04, 0.052); a2.set_ylim(-0.6, 3.6)
a2.set_title("Phase 2 − Phase 1 gain with 95% CI", fontsize=15, color=INK, pad=12)
a2.set_xlabel("← Phase 1 better          Phase 2 better →", fontsize=13.5, color=MUTED)
a2.text(-0.039, 3.45, "worse with label", fontsize=12, color=BAD, va="center")
a2.text(0.0005, 3.45, "better with label", fontsize=12, color=GOOD, va="center")
for s in ["top", "right", "left"]: a2.spines[s].set_visible(False)
a2.tick_params(labelsize=12, colors=MUTED); a2.tick_params(axis="y", length=0, colors=INK)
fig.text(0.031, 0.05, "In-domain CI includes zero (27 of 45 images favour Phase 2, p = 0.11) → inconclusive.   "
         "Gaze4ASD: all three CIs are below zero → worse.", fontsize=13, color=INK)
fig.text(0.031, 0.018, "Sources: manuscript §6.1, Table 3 (in-domain), Table 5 (three external versions, CPU re-scoring).",
         fontsize=11.5, color=MUTED, style="italic")
save(fig, "06_results/2_likelihood.png")

# ---------------------------------------------------------------- 02 results / 3 behaviour
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
fig.text(0.031, 0.94, "Generated behaviour: partly right, partly wrong",
         fontsize=27, fontweight="bold", color=INK, va="center")
fig.text(0.031, 0.885, "Gaze4ASD, 30 images, 10 seeds, τ = 1. Generated ASD−TD difference as a share of the real ASD−TD difference.",
         fontsize=15, color=MUTED, va="center")
a1 = fig.add_axes([0.20, 0.17, 0.42, 0.62])
meas = ["Spatial dispersion", "Saccade amplitude", "Revisit rate", "Direction concentration", "Total fixation time"]
real = [0.0169, 0.0151, -0.0805, 0.0270, -0.0822]
gen = [0.0117, 0.0096, -0.0261, -0.0052, 0.0210]
share = [g / r for g, r in zip(gen, real)]
yv = list(range(len(meas)))[::-1]
for s, y_, m in zip(share, yv, meas):
    c = GOOD if s > 0 else BAD
    a1.barh(y_, s, color=c, height=0.55)
    a1.text(s + (0.03 if s > 0 else -0.03), y_, f"{s*100:.0f}%".replace("-", "−"), va="center",
            ha="left" if s > 0 else "right", fontsize=14, fontweight="bold", color=INK)
a1.axvline(0, color=INK, lw=1.3); a1.axvline(1, color=INK, lw=1.3, ls=":")
a1.text(0.985, 4.62, "real size", ha="right", fontsize=12, color=MUTED)
a1.set_yticks(yv); a1.set_yticklabels(meas, fontsize=14)
a1.set_xlim(-0.5, 1.15); a1.set_ylim(-0.6, 4.85)
a1.set_xlabel("← wrong direction        right direction →", fontsize=13.5, color=MUTED)
for s in ["top", "right", "left"]: a1.spines[s].set_visible(False)
a1.tick_params(axis="y", length=0); a1.tick_params(labelsize=12, colors=MUTED)
a1.set_xticks([-0.5, 0, 0.5, 1.0]); a1.set_xticklabels(["−50%", "0", "50%", "100%"])

a2 = fig.add_axes([0.71, 0.17, 0.26, 0.62])
x = [0, 1]
a2.bar([v - 0.19 for v in x], [6.25, 6.41], width=0.36, color=[ASD, TD], alpha=0.45, label="real")
a2.bar([v + 0.19 for v in x], [10.19, 9.60], width=0.36, color=[ASD, TD], label="generated")
for xv, r, g in zip(x, [6.25, 6.41], [10.19, 9.60]):
    a2.text(xv - 0.19, r + 0.15, f"{r:.2f}", ha="center", fontsize=12.5, color=INK)
    a2.text(xv + 0.19, g + 0.15, f"{g:.2f}", ha="center", fontsize=12.5, color=INK, fontweight="bold")
    a2.text(xv - 0.19, 0.3, "real", ha="center", fontsize=11, color=INK)
    a2.text(xv + 0.19, 0.3, "gen.", ha="center", fontsize=11, color="white", fontweight="bold")
a2.set_xticks(x); a2.set_xticklabels(["ASD", "TD"], fontsize=14)
a2.set_ylim(0, 12); a2.set_title("Fixations per scanpath", fontsize=15, color=INK, pad=12)
for s in ["top", "right"]: a2.spines[s].set_visible(False)
a2.tick_params(labelsize=12, colors=MUTED)
fig.text(0.031, 0.075, "Right direction but smaller (≈69 / 64 / 32%; an independent run gave 60 / 50 / 29%). Neither group gets closer to its own real group:",
         fontsize=13, color=INK)
fig.text(0.031, 0.045, "TD dispersion contracts 0.112 → 0.101, past real TD 0.108; ASD stays 0.113. Both labels over-produce fixations (~10 vs ~6).",
         fontsize=13, color=INK)
fig.text(0.031, 0.015, "Sources: manuscript Table 7 (shares = Phase 2 contrast ÷ real contrast), Table 8, §6.6–6.7.",
         fontsize=11.5, color=MUTED, style="italic")
save(fig, "06_results/3_generated_behaviour.png")

# ---------------------------------------------------------------- 03 what we did / 1 approach
fig, ax = canvas("What we did: one controlled comparison",
                 "Question: if one shared TPP-Gaze model gets only a small ASD/TD label, what changes?")
steps = [
    ("Released TPP-Gaze", ["pretrained weights,", "no group input"], NEUT_L, NEUT),
    ("Phase 1", ["fine-tune on Saliency4ASD", "ASD + TD together,", "no label"], TEAL_L, TEAL),
    ("Phase 2", ["add ASD/TD embedding", "(starts at zero)", "train only 1,024 numbers"], OURS_L, OURS),
    ("Compare P1 vs P2", ["same images,", "same random seeds,", "same settings"], NEUT_L, INK),
]
x0 = 0.5; w = 3.3; gapx = 0.6
for i, (h, ls, fc, ec) in enumerate(steps):
    xx = x0 + i * (w + gapx)
    card(ax, xx, 4.6, w, 2.7, h, ls, fc, ec, fs=15, hfs=19, gap=0.48)
    if i < 3:
        arrow(ax, xx + w + 0.05, 5.95, xx + w + gapx - 0.05, 5.95)
qs = [
    ("Q1", "Better fit on our held-out test images?", "likelihood, in-domain"),
    ("Q2", "Does that hold on Gaze4ASD?", "likelihood, external, never trained on"),
    ("Q3", "Do generated ASD vs TD scanpaths differ like real children?", "generated behaviour"),
]
for i, (k, q, s) in enumerate(qs):
    yy = 3.6 - i * 0.95
    box(ax, 0.5, yy - 0.38, 15, 0.76, "#F7F8FA", LINE, lw=1.2)
    ax.text(0.8, yy, k, fontsize=17, fontweight="bold", color=OURS, va="center")
    ax.text(1.6, yy, q, fontsize=16, color=INK, va="center", fontweight="bold")
    ax.text(15.2, yy, s, fontsize=14, color=MUTED, va="center", ha="right")
footer(ax, "Why this way: two separate ASD and TD models would differ in every weight, so we could not tell which differences come from the group.")
save(fig, "02_what_we_did/1_our_approach.png")

# ---------------------------------------------------------------- 04 training
fig, ax = canvas("How we trained",
                 "Saliency4ASD, split by image (seed 0): 210 train / 45 validation / 45 test images.")
# split bar
tot = 15.0
segs = [(210, "Train: 210 images, 4,229 scanpaths", TEAL), (45, "Val: 45 / 897", NEUT), (45, "Test: 45 / 903", INK)]
xx = 0.5
for n, t, c in segs:
    ww = tot * n / 300
    box(ax, xx, 6.95, ww - 0.06, 0.62, c, c)
    ax.text(xx + (ww - 0.06) / 2, 7.26, t, fontsize=14, color="white", ha="center", va="center", fontweight="bold")
    xx += ww
rows = [
    ("", "Phase 1: fine-tune", "Phase 2: label only"),
    ("Why", "adapt TPP-Gaze to these children first", "isolate what the label adds"),
    ("Starts from", "released TPP-Gaze weights", "best Phase 1 checkpoint"),
    ("Trained", "≈3.9 million (DenseNet201 frozen)", "1,024 (2 × 512 embedding); base frozen"),
    ("Optimiser", "AdamW, lr 1e-3, wd 0.1, batch 128", "lr 1e-4; base kept in eval mode"),
    ("Stopping", "patience 20; best epoch 6 of 26", "best epoch 8 of 28"),
    ("Validation NLL", "best −0.5845 (−0.384 by epoch 26)", "−0.5845 → −0.5964"),
]
cx = [0.5, 3.6, 9.7]; cw = [3.1, 6.1, 5.8]
y = 6.35
for i, r in enumerate(rows):
    h = 0.62
    for j, t in enumerate(r):
        if i == 0:
            fc = [BG, TEAL_L, OURS_L][j]; ec = [BG, TEAL, OURS][j]
        else:
            fc = "#F7F8FA" if j == 0 else BG; ec = LINE
        if not (i == 0 and j == 0):
            box(ax, cx[j], y - h, cw[j] - 0.08, h - 0.06, fc, ec, lw=1.1)
        ax.text(cx[j] + 0.2, y - h / 2 - 0.03, t, fontsize=15 if i else 17, va="center",
                color=(ec if i == 0 else INK), fontweight="bold" if (i == 0 or j == 0) else "normal")
    y -= h
ax.text(0.5, 1.65, "Same for both phases:", fontsize=15, fontweight="bold", color=INK)
ax.text(0.5, 1.2, "joint location + duration NLL • log-duration normalised with our training split (mean −1.4366, SD 0.6936) •",
        fontsize=14, color=INK)
ax.text(0.5, 0.82, "scanpaths with ≥ 3 fixations • all checkpoint and temperature choices made on validation only, never on test.",
        fontsize=14, color=INK)
footer(ax, "Sources: training/config.yaml, config_phase2_embonly.yaml, runs/*/summary_full.json, runs/baseline/split.json. Phase 1 overfits after epoch 6, so early stopping mattered.")
save(fig, "03_training/training.png")

# ---------------------------------------------------------------- 05 testing
fig, ax = canvas("How we tested",
                 "Two separate questions, two separate evaluations. All on data never used for training or selection.")
card(ax, 0.5, 3.15, 7.3, 4.45, "A. Likelihood: does the label fit real gaze better?",
     ["Both models score the same real scanpaths.",
      "Gain = Phase 1 NLL − Phase 2 NLL (positive = label helps).",
      "",
      "**In-domain: Saliency4ASD test, 45 images",
      "  resample images (no participant IDs in release)",
      "**External: Gaze4ASD, 30 images, 33 ASD + 133 TD",
      "  resample children, images, or both",
      "  3 preprocessing versions (screen mapping inferred)"],
     TEAL_L, TEAL, fs=14.5, hfs=17, gap=0.45)
card(ax, 8.2, 3.15, 7.3, 4.45, "B. Experiment E1: generated behaviour",
     ["Per test image: 20 scanpaths × 10 seeds, τ = 1",
      "(30 Gaze4ASD + 45 Saliency4ASD test images).",
      "",
      "**Generate: ASD label, TD label, and Phase 1 twice",
      "  (Phase 1 on the same two seed streams)",
      "**Label effect = (ASD − TD) − (Phase 1 A − Phase 1 B)",
      "  removes the difference seeds alone create",
      "**Realism: each label vs its own real group"],
     OURS_L, OURS, fs=14.5, hfs=17, gap=0.45)
card(ax, 0.5, 0.75, 15.0, 2.05, "Gaze4ASD preprocessing versions, and why",
     ["Fit-to-height + clip off-image fixations (primary; matches authors' saliency maps, r = 0.9999 over 4,533 maps)",
      "Stretch + clip (in case the inferred mapping is wrong)   •   Fit-to-height + remove off-image fixations",
      "Off-image gaze: ASD 7.35% vs TD 2.72% of fixations, so how we treat it matters. Temperature tested in a separate run."],
     "#F7F8FA", NEUT, hc=INK, fs=14, hfs=16, gap=0.42)
footer(ax, "Sources: manuscript §3.8, §4.3, §5.1–5.5, §6.4. Primary version fixed before any model was scored on Gaze4ASD. All analyses exploratory.")
save(fig, "04_testing/testing.png")

# ---------------------------------------------------------------- 06 novelty
fig, ax = canvas("What is new in our work",
                 "We kept TPP-Gaze as released and added the smallest possible group input.")
card(ax, 0.5, 2.55, 7.3, 5.05, "Kept from TPP-Gaze (unchanged)",
     ["Code, pretrained weights",
      "DenseNet201 + CoordConv image encoder",
      "Transformer + LSTM gaze-history encoder",
      "16-component location head,",
      "4-component log-normal duration head",
      "Joint loss and sampler",
      "Released batch layout (disclosed quirk)"],
     TEAL_L, TEAL, fs=15, hfs=18, gap=0.5)
card(ax, 8.2, 2.55, 7.3, 5.05, "New in our work",
     ["**ASD/TD embedding (2 × 512), zero-initialised,",
      "**added to the context before both heads",
      "Same in training, scoring and sampling",
      "Only 1,024 trainable numbers in Phase 2",
      "Phase 1 seed-only baseline for the label effect",
      "External test on Gaze4ASD (3 preprocessings)",
      "Per-group realism, not just ASD−TD contrasts"],
     OURS_L, OURS, fs=15, hfs=18, gap=0.5)
box(ax, 0.5, 0.75, 15, 1.45, INK, INK)
ax.text(0.8, 1.72, "Why this matters:", fontsize=16, fontweight="bold", color="white", va="center")
ax.text(0.8, 1.2, "All 1,290 base weight tensors are identical in both checkpoints, so Phase 1 = Phase 2 with a zero embedding.\n"
        "Every difference we measure comes from the label. Not claimed: better than other ASD scanpath models, or any diagnostic use.",
        fontsize=14.5, color="white", va="center", linespacing=1.5)
save(fig, "05_novelty/novelty.png")

# ---------------------------------------------------------------- 07 conclusion
fig, ax = canvas("What we conclude, and what comes next")
box(ax, 0.5, 6.2, 15, 1.35, OURS_L, OURS)
ax.text(0.8, 6.875, "A minimal ASD/TD embedding changes generated gaze behaviour and reproduces some spatial group-contrast\n"
        "directions, but does not consistently improve likelihood or individual-group realism.",
        fontsize=17, color=INK, va="center", fontweight="bold", linespacing=1.45)
card(ax, 0.5, 1.0, 4.8, 4.85, "Still open",
     ["Why the TD embedding fits", "Gaze4ASD worse (cause untested)", "",
      "How much the extra fixations", "and missing off-image gaze", "drive the wrong timing results"],
     NEUT_L, NEUT, fs=14.5, hfs=17, gap=0.44)
card(ax, 5.6, 1.0, 4.8, 4.85, "Limitations",
     ["Gaze4ASD screen mapping inferred", "(not confirmed by authors)", "",
      "No participant ages: cannot adjust", "Ages 3–7 vs 5–12 across datasets", "",
      "Label acts as one shift everywhere:", "generated ASD−TD dispersion is", "positive on all 45 test images"],
     BAD_L, BAD, fs=14.5, hfs=17, gap=0.44)
card(ax, 10.7, 1.0, 4.8, 4.85, "Next steps",
     ["1. Ask Gaze4ASD authors: display", "   geometry, participant count, ages", "",
      "2. Model stopping / viewing time", "   (fix ~10 vs ~6 fixations)", "",
      "3. Model off-image gaze", "", "4. Re-run E1 with same baseline"],
     GOOD_L, GOOD, fs=14.5, hfs=17, gap=0.44)
footer(ax, "Group-level research model only. Not a diagnostic or screening tool. Sources: manuscript §6.3–6.7, §7–9; demo/demo_results/manifest.json.")
save(fig, "06_results/4_conclusion_next_steps.png")
