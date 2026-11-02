"""
Presentation architecture diagram for the ASD-ScanPath project (Phase-2
ASD/TD-conditioned TPP-Gaze). Visualization/documentation only -- it reflects the
implemented code: original TPP-Gaze base (image encoder = DenseNet201+CoordConv,
history encoder = transformer+LSTM, cat fusion -> 512-d, LGMM duration head +
2D-GMM position head) with our learnable 2-class group embedding (nn.Embedding(2,512))
ADDED to the fused context before BOTH heads, sampled autoregressively.

Outputs: architecture_diagram.png (hi-res) and architecture_diagram.svg (editable).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))

C_INPUT = "#eef1f4"; C_IMG = "#d9e7f6"; C_HIST = "#dcefe1"; C_FUSE = "#e7e0f2"
C_COND = "#ffe1ad"; ORANGE = "#e07b00"; C_SPA = "#d7eef0"; C_TMP = "#fde6cf"
C_OUT = "#eef1f4"; BLUE = "#2e74b5"; EC = "#3a3f44"; TXT = "#15191c"; GRAY = "#6b7280"


def box(ax, cx, cy, w, h, title, sub=None, fc=C_INPUT, ec=EC, lw=1.6,
        tsize=11, ssize=9.3, tw="bold", tcol=TXT):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.09", fc=fc, ec=ec, lw=lw, zorder=3))
    if sub:
        ax.text(cx, cy + h / 2 - 0.30, title, ha="center", va="center",
                fontsize=tsize, fontweight=tw, color=tcol, zorder=4)
        ax.text(cx, cy - 0.12, sub, ha="center", va="center", fontsize=ssize,
                color=tcol, linespacing=1.35, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tsize,
                fontweight=tw, color=tcol, zorder=4, linespacing=1.3)


def arrow(ax, p1, p2, color=EC, lw=2.0, rad=0.0, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                 color=color, lw=lw, ls=ls, shrinkA=3, shrinkB=3,
                 connectionstyle=f"arc3,rad={rad}", zorder=2))


fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(8, 8.62, "ASD-ScanPath: ASD/TD-Conditioned TPP-Gaze for Scanpath Generation",
        ha="center", va="center", fontsize=16, fontweight="bold", color=TXT)
ax.text(8, 8.22, "Blue / green = TPP-Gaze base (Phase 1 weights, frozen in Phase 2)      "
        "Orange = our ASD/TD conditioning (added)", ha="center", va="center",
        fontsize=10.5, color=GRAY)

YM = 6.25

# 1. INPUT
ax.add_patch(FancyBboxPatch((0.22, 4.75), 2.26, 3.0, boxstyle="round,pad=0.02,rounding_size=0.09",
             fc=C_INPUT, ec=EC, lw=1.6, zorder=3))
ax.text(1.35, 7.35, "1. INPUT", ha="center", va="center", fontsize=12.5, fontweight="bold", color=TXT, zorder=4)
ax.text(1.35, 6.55, "Stimulus image", ha="center", va="center", fontsize=9.3, color=TXT, zorder=4)
ax.text(1.35, 6.05, "Previous gaze history", ha="center", va="center", fontsize=9.3, color=TXT, zorder=4)
ax.text(1.35, 5.45, "ASD / TD condition", ha="center", va="center", fontsize=9.6,
        color=ORANGE, fontweight="bold", zorder=4)

# 2 / 3 encoders
box(ax, 4.35, 7.25, 2.7, 1.45, "2. Image Encoder",
    "DenseNet201 + CoordConv\n-> image features (256-d)", fc=C_IMG)
box(ax, 4.35, 5.25, 2.7, 1.45, "3. Gaze-History Encoder",
    "TPP-Gaze transformer + LSTM\n-> history features (256-d)", fc=C_HIST)
# 4 fusion
box(ax, 7.3, YM, 2.1, 1.7, "4. Fusion (concat)", "image + history\n-> 512-d context", fc=C_FUSE)
# 5 conditioning (highlight)
box(ax, 10.15, YM, 2.75, 2.1, "5. ASD / TD Conditioning",
    "learnable 2-class embedding\nASD = 0,  TD = 1  (2 x 512)\nADDED to 512-d context\n(+1,024 params)",
    fc=C_COND, ec=ORANGE, lw=2.8, tsize=11, ssize=9.0)
ax.text(10.15, YM + 1.26, "OUR RESEARCH ADDITION", ha="center", va="center",
        fontsize=10, fontweight="bold", color=ORANGE, zorder=5)
ax.text(10.15, YM - 1.28, "embedding selected by the ASD / TD input label", ha="center",
        va="center", fontsize=7.8, color=ORANGE, style="italic", zorder=5)
# 6 heads
box(ax, 13.1, 7.15, 2.3, 1.3, "6a. Spatial head", "2D GMM\n-> next (x, y)", fc=C_SPA, ssize=9)
box(ax, 13.1, 5.35, 2.3, 1.3, "6b. Temporal head", "Log-Gaussian MM\n-> next duration", fc=C_TMP, ssize=9)
# output node
box(ax, 15.28, YM, 1.3, 1.6, "Predicted\nfixation\n(x, y, d)", fc=C_OUT, tsize=9.5)

# main arrows
arrow(ax, (2.48, 6.75), (2.98, 7.1), rad=-0.08)
arrow(ax, (2.48, 5.75), (2.98, 5.4), rad=0.08)
arrow(ax, (5.70, 7.0), (6.25, 6.6))
arrow(ax, (5.70, 5.5), (6.25, 5.9))
arrow(ax, (8.35, YM), (8.775, YM))
arrow(ax, (11.525, 6.5), (11.95, 7.0))
arrow(ax, (11.525, 6.0), (11.95, 5.5))
arrow(ax, (14.25, 7.0), (14.63, 6.55))
arrow(ax, (14.25, 5.5), (14.63, 5.95))

# 7. autoregressive loop: output -> back into history encoder, routed below the band
ax.plot([15.28, 15.28, 4.35], [5.45, 4.2, 4.2], color=GRAY, lw=1.8, ls="--", zorder=1)
ax.add_patch(FancyArrowPatch((4.35, 4.2), (4.35, 4.525), arrowstyle="-|>", mutation_scale=16,
             color=GRAY, lw=1.8, zorder=1, shrinkA=0, shrinkB=2))
ax.text(9.7, 3.82, "7. autoregressive loop:  append predicted fixation  ->  predict next,  "
        "until the 3 s viewing window ends", ha="center", va="center", fontsize=9.6,
        color=GRAY, style="italic")

# ---- divider ----
ax.plot([0.5, 15.5], [3.35, 3.35], color="#d0d4d8", lw=1.2, zorder=0)
ax.text(0.55, 3.02, "8. FINAL OUTPUT  —  predicted scanpath  +  visualization (numbered fixations, arrows, durations)",
        ha="left", va="center", fontsize=11, fontweight="bold", color=TXT)

# ---- bottom concept: same image -> ASD / TD scanpaths ----
yb = 1.45
box(ax, 2.6, yb, 2.0, 1.1, "Same image", fc=C_INPUT, tsize=11.5)

def scan_box(cx, cy, label, col, pts):
    box(ax, cx, cy, 3.1, 1.0, label, fc="white", ec=col, lw=2.0, tsize=10.5, tcol=col)
    xs = [cx - 1.25 + 0.52 * i for i in range(len(pts))]
    ys = [cy - 0.33 + p for p in pts]
    for i in range(1, len(xs)):
        arrow(ax, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), color=col, lw=1.2, style="-|>")
    for x, y in zip(xs, ys):
        ax.add_patch(Circle((x, y), 0.065, fc=col, ec="k", lw=0.5, alpha=0.85, zorder=6))

box(ax, 6.15, yb + 0.6, 1.2, 0.62, "ASD", fc=C_COND, ec=ORANGE, lw=2.0, tsize=11.5)
box(ax, 6.15, yb - 0.6, 1.2, 0.62, "TD", fc=C_IMG, ec=BLUE, lw=2.0, tsize=11.5)
arrow(ax, (3.6, yb + 0.15), (5.55, yb + 0.55), color=ORANGE, lw=1.9)
arrow(ax, (3.6, yb - 0.15), (5.55, yb - 0.55), color=BLUE, lw=1.9)
scan_box(9.8, yb + 0.6, "ASD-conditioned scanpath", ORANGE, [0.12, -0.05, 0.2, 0.0, 0.12])
scan_box(9.8, yb - 0.6, "TD-conditioned scanpath", BLUE, [0.05, 0.13, 0.0, 0.11, 0.03])
arrow(ax, (6.75, yb + 0.6), (8.25, yb + 0.6), color=ORANGE, lw=1.9)
arrow(ax, (6.75, yb - 0.6), (8.25, yb - 0.6), color=BLUE, lw=1.9)
ax.text(12.4, yb, "same stimulus,\ntwo group-conditioned\ngaze predictions",
        ha="left", va="center", fontsize=9.6, color=GRAY, style="italic", linespacing=1.3)

fig.savefig(os.path.join(HERE, "architecture_diagram.png"), dpi=200, facecolor="white", bbox_inches="tight")
fig.savefig(os.path.join(HERE, "architecture_diagram.svg"), facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved architecture_diagram.png and architecture_diagram.svg to", HERE)
