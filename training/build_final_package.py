"""
Assemble the final TPP-Gaze ASD/TD results package from completed experiments.
NO model work. Reads saved JSONs, writes tables/figures/text under
training/runs/final_results/. Reports only experiment-supported numbers.
"""
from __future__ import annotations
import json, csv, os, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "training/runs"
OUT = os.path.join(R, "final_results"); FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

def load(p):
    return json.load(open(p))

sal = load(f"{R}/phase2_embedding_only/test_eval_phase2.json")   # phase1 vs phase2 (Sal4ASD test)
p1sum = load(f"{R}/baseline/summary_full.json")
p2sum = load(f"{R}/phase2_embedding_only/summary_full.json")
p1test = load(f"{R}/baseline/test_eval.json")                    # has pretrained + fine_tuned
extnll = load(f"{R}/external_gaze4asd/external_nll.json")
extagg = load(f"{R}/external_gaze4asd/external_aggregate_summary.json")
extmm = load(f"{R}/external_gaze4asd/external_similarity_multimatch.json")
salmm = load(f"{R}/phase2_embedding_only/analysis/similarity_multimatch.json")
stats = load(f"{R}/external_gaze4asd/results_statistics.json")

# ---------------- 1. Main results table (Sal4ASD held-out test) ----------------
with open(f"{OUT}/1_main_results_saliency4asd_test.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "group", "NLL", "duration_NLL", "position_NLL", "n_scanpaths"])
    # pretrained reference (overall only, from Phase-1 test eval)
    pre = p1test["pretrained_same_stats"]
    w.writerow(["Pretrained (ref)", "overall", f"{pre['nll']:.4f}", f"{pre['nll_duration']:.4f}",
                f"{pre['nll_position']:.4f}", pre.get("n_scanpaths", sal["phase1"]["overall"]["n_scanpaths"])])
    for model, key in [("Phase1 (fine-tuned)", "phase1"), ("Phase2 (ASD/TD cond.)", "phase2")]:
        for g in ("overall", "ASD", "TD"):
            d = sal[key][g]
            w.writerow([model, g, f"{d['nll']:.4f}", f"{d['nll_duration']:.4f}",
                        f"{d['nll_position']:.4f}", d["n_scanpaths"]])
    w.writerow([])
    w.writerow(["improvement Phase2 vs Phase1", "group", "abs_NLL", "rel_%", "", ""])
    for g, key in [("overall", "improvement_overall"), ("ASD", "improvement_ASD"), ("TD", "improvement_TD")]:
        im = sal[key]; w.writerow(["", g, f"{im['absolute']:+.4f}", f"{im['relative_pct']:+.2f}", "", ""])

# ---------------- 2. External validation table (Gaze4ASD) ----------------
with open(f"{OUT}/2_external_gaze4asd_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["# NLL (per event)"])
    w.writerow(["policy/group", "NLL", "duration_NLL", "position_NLL", "n_trials"])
    for pol in ("A", "B"):
        for g in ("overall", "ASD", "TD"):
            d = extnll[pol][g]
            w.writerow([f"Policy{pol} {g}", f"{d['nll']:.4f}", f"{d['nll_duration']:.4f}",
                        f"{d['nll_position']:.4f}", d["n_trials"]])
    s = extnll["saliency4asd_phase2_test"]
    for g in ("overall", "ASD", "TD"):
        w.writerow([f"Sal4ASD test {g}", f"{s[g]['nll']:.4f}", f"{s[g]['nll_duration']:.4f}",
                    f"{s[g]['nll_position']:.4f}", s[g].get("n_scanpaths", "")])
    w.writerow([]); w.writerow(["# Behavioral means (Policy A)"])
    bkeys = ["n_fix", "dur_mean", "sacc_amp_norm_mean", "dispersion", "revisit_rate"]
    w.writerow(["metric", "real_ASD", "real_TD", "gen_ASD", "gen_TD"])
    for k in bkeys:
        w.writerow([k] + [f"{extagg[p][k]['mean']:.4f}" for p in ("real_ASD", "real_TD", "gen_ASD", "gen_TD")])

# ---------------- 3. Participant-level statistics table ----------------
with open(f"{OUT}/3_participant_statistics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "real_ASD_mean", "real_TD_mean", "real_diff(ASD-TD)", "real_diff_CI95",
                "real_Hedges_g", "real_g_CI95", "real_perm_p", "real_sig",
                "gen_ASD_mean", "gen_TD_mean", "gen_diff(ASD-TD)", "gen_Cohen_dz", "gen_dz_CI95",
                "gen_perm_p", "gen_sig", "direction_match"])
    def sig(p): return "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
    for m, d in stats["metrics"].items():
        r, g, c = d["real"], d["generated"], d["comparison"]
        w.writerow([d["label"], f"{r['ASD']['mean']:.4f}", f"{r['TD']['mean']:.4f}",
                    f"{r['mean_diff_ASD_minus_TD']:+.4f}", f"[{r['mean_diff_CI95'][0]:.4f},{r['mean_diff_CI95'][1]:.4f}]",
                    f"{r['hedges_g']:+.3f}", f"[{r['hedges_g_CI95'][0]:.3f},{r['hedges_g_CI95'][1]:.3f}]",
                    f"{r['perm_p']:.4f}", sig(r['perm_p']),
                    f"{g['ASD']['mean']:.4f}", f"{g['TD']['mean']:.4f}", f"{g['mean_diff_ASD_minus_TD']:+.4f}",
                    f"{g['cohen_dz_paired']:+.3f}", f"[{g['cohen_dz_CI95'][0]:.3f},{g['cohen_dz_CI95'][1]:.3f}]",
                    f"{g['perm_p']:.4f}", sig(g['perm_p']), c["direction_match"]])

# ---------------- 4. Figures ----------------
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 150})

# (a) Phase1 vs Phase2 NLL on Sal4ASD test
fig, ax = plt.subplots(figsize=(7, 4.5))
groups = ["overall", "ASD", "TD"]; x = np.arange(3); wd = 0.38
p1 = [sal["phase1"][g]["nll"] for g in groups]; p2 = [sal["phase2"][g]["nll"] for g in groups]
ax.bar(x - wd/2, p1, wd, label="Phase 1", color="#8aa0b5")
ax.bar(x + wd/2, p2, wd, label="Phase 2 (ASD/TD)", color="#2e74b5")
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("Test NLL (lower = better)")
ax.set_title("Saliency4ASD held-out test: Phase 1 vs Phase 2"); ax.legend()
for xi, (a, b) in enumerate(zip(p1, p2)):
    ax.text(xi - wd/2, a, f"{a:.3f}", ha="center", va="top" if a < 0 else "bottom", fontsize=8)
    ax.text(xi + wd/2, b, f"{b:.3f}", ha="center", va="top" if b < 0 else "bottom", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_phase1_vs_phase2_nll.png"); plt.close(fig)

# (e) real vs generated scanpath similarity (MultiMatch, Saliency4ASD in-domain)
dims = ["mm_shape", "mm_length", "mm_direction", "mm_position", "mm_duration"]
dlab = ["shape", "length", "direction", "position", "duration"]
def mm_avg(d, keys): return [np.mean([d[k][dim] for k in keys]) for dim in dims]
rvr = mm_avg(salmm, ["RvR_ASD", "RvR_TD"]); rvg = mm_avg(salmm, ["RvG_ASD", "RvG_TD"])
fig, ax = plt.subplots(figsize=(7.5, 4.5)); x = np.arange(len(dims)); wd = 0.38
ax.bar(x - wd/2, rvr, wd, label="Real-vs-Real (baseline)", color="#9c9c9c")
ax.bar(x + wd/2, rvg, wd, label="Real-vs-Generated", color="#e07b00")
ax.set_xticks(x); ax.set_xticklabels(dlab); ax.set_ylim(0, 1); ax.set_ylabel("MultiMatch similarity (1=identical)")
ax.set_title("Scanpath similarity: real-vs-generated vs real-vs-real (Saliency4ASD)"); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_real_vs_generated_similarity.png"); plt.close(fig)

# (b,c,d) copy existing figures
copies = [
    (f"{R}/external_gaze4asd/fig_stat_distributions.png", f"{FIG}/fig2_real_vs_gen_ASD_TD_distributions.png"),
    (f"{R}/external_gaze4asd/fig_stat_effect_sizes.png", f"{FIG}/fig3_effect_size_forest.png"),
    (f"{R}/phase2_embedding_only/analysis/fig_representative_scanpaths.png", f"{FIG}/fig4a_representative_scanpaths_saliency4asd.png"),
    (f"{R}/external_gaze4asd/fig_representative_scanpaths.png", f"{FIG}/fig4b_representative_scanpaths_gaze4asd.png"),
]
for src, dst in copies:
    if os.path.exists(src):
        shutil.copy(src, dst)

# ---------------- 5. MAIN_FINDINGS.txt ----------------
im = sal["improvement_overall"]; imA = sal["improvement_ASD"]; imT = sal["improvement_TD"]
MF = f"""TPP-GAZE ASD/TD - MAIN FINDINGS
(Only results directly supported by the experiments. Categories kept distinct.)

A) MODEL LIKELIHOOD IMPROVEMENT (held-out test negative log-likelihood; lower=better)
   Saliency4ASD test (45 images, {sal['phase1']['overall']['n_scanpaths']} scanpaths):
     Pretrained (ref, overall):        {p1test['pretrained_same_stats']['nll']:+.4f}
     Phase 1 (fine-tuned):   overall {sal['phase1']['overall']['nll']:+.4f}  ASD {sal['phase1']['ASD']['nll']:+.4f}  TD {sal['phase1']['TD']['nll']:+.4f}
     Phase 2 (ASD/TD cond.): overall {sal['phase2']['overall']['nll']:+.4f}  ASD {sal['phase2']['ASD']['nll']:+.4f}  TD {sal['phase2']['TD']['nll']:+.4f}
   Phase 2 improves over Phase 1: overall {im['absolute']:+.4f} ({im['relative_pct']:+.2f}%),
     ASD {imA['absolute']:+.4f} ({imA['relative_pct']:+.2f}%), TD {imT['absolute']:+.4f} ({imT['relative_pct']:+.2f}%).
   Phase 2 adds only 1,024 trainable params (2x512 group embedding); base frozen.
   -> Conditioning gives a small but consistent in-domain likelihood gain.

B) BEHAVIORAL DIRECTION MATCHING (generated ASD vs TD vs real ASD vs TD)
   Generated scanpaths reproduce the DIRECTION of the real ASD-vs-TD difference in
   4 of 5 metrics: fixation duration (ASD<TD), saccade amplitude (ASD>TD),
   spatial dispersion (ASD>TD), revisit rate (ASD<TD).
   It does NOT reproduce fixation-count direction (real ASD<TD, generated ASD>TD),
   and it over-produces fixation counts overall.

C) STATISTICAL SIGNIFICANCE (Gaze4ASD, participant-level REAL; image-level paired GEN; EXPLORATORY)
   Real significant group effects: spatial dispersion (Hedges g=+0.65, p<.01),
     revisit rate (g=-1.04, p<.01, largest), saccade amplitude (g=+0.46, p<.05).
   Real NOT significant: fixation duration (g=-0.14) and fixation count (g=-0.21).
   Generated, matching direction AND significant: spatial dispersion (dz=+0.80, p<.01),
     saccade amplitude (dz=+0.38, p<.05).
   Generated mismatch/weak: duration dz=-0.58 (p<.01) but real ns (model exaggerates);
     revisit dz=-0.14 (ns) vs real large (model under-reproduces);
     fixation count wrong direction.
   p-values are exploratory (Bonferroni alpha for 5 tests = 0.01; no pre-registration).

D) EXTERNAL GENERALIZATION (Gaze4ASD, frozen model, no tuning)
   Duration likelihood transfers (duration NLL ~ -0.40 vs -0.46 in-domain).
   Position likelihood does NOT transfer (position NLL +1.44 Policy A / +0.91 Policy B
     vs -0.04 in-domain) -> spatial domain gap.
   Overall NLL: Policy A {extnll['A']['overall']['nll']:+.4f}, Policy B {extnll['B']['overall']['nll']:+.4f}
     vs Saliency4ASD test {extnll['saliency4asd_phase2_test']['overall']['nll']:+.4f}.
   Scanpath structure stays plausible (MultiMatch real-vs-generated ~ real-vs-real).
   ASD-vs-TD behavioral direction partly generalizes (spatial effects strongest).

No clinical validity is claimed. These are computational modeling results on
group-labeled gaze data; they are not a diagnostic or individual-level tool.
"""
open(f"{OUT}/MAIN_FINDINGS.txt", "w").write(MF)

# ---------------- 6. LIMITATIONS.txt ----------------
LIM = """TPP-GAZE ASD/TD - LIMITATIONS

1. No participant IDs in Saliency4ASD.
   The Saliency4ASD processing has no per-observer identifiers, so conditioning
   and all in-domain analyses use the GROUP label only (ASD/TD). Participant-level
   statistics were therefore only possible on the external Gaze4ASD data.

2. Gaze4ASD Policy-A clipping.
   Policy A clips out-of-bounds fixations to the image border (+/-1), piling points
   on edges that are very unlikely under the model and inflating the position NLL
   (Policy A overall 1.04 vs Policy B, which removes out-of-bounds, 0.53).
   Policy A is reported as primary per the request, but Policy B is the fairer NLL.

3. Generated fixation-count overproduction.
   The TPP sampler fills the 3 s window and generates more fixations than humans
   (generated ~10 vs real ~6-7), and it reverses the (weak, non-significant) real
   ASD-vs-TD fixation-count direction. Fixation count is the least reliable output.

4. Weak revisit-rate reproduction.
   Revisit rate is the LARGEST real ASD-vs-TD effect (g=-1.04) but the generated
   model barely separates the groups on it (dz=-0.14, ns). Direction matches,
   magnitude does not.

5. External spatial domain gap.
   Position likelihood and absolute spatial scale do not transfer to Gaze4ASD
   (different stimuli; DenseNet/positional priors tuned to Saliency4ASD). Generated
   saccade amplitude / dispersion are ~half of real on Gaze4ASD. Temporal (duration)
   behavior transfers better than spatial behavior.

6. Exploratory statistical analysis.
   Permutation/bootstrap tests are two-sided and exploratory; there was no
   pre-registration and no formal multiple-comparison control beyond a reported
   Bonferroni reference (alpha=0.01 for 5 tests). The ASD/TD groups are also
   imbalanced (Gaze4ASD: 33 ASD vs 133 TD participants). Results should be read as
   effect-direction/size evidence, not confirmatory hypothesis tests.

Additional notes:
- Architectural quirk (documented): the released TPP-Gaze transformer encoder mixes
  over the batch dimension, so results depend on batch composition; this is inherent
  to the pretrained weights and was reproduced faithfully, not "fixed".
- The ASD/TD conditioning is a 1,024-parameter additive embedding; its effect is
  small by design, which bounds how strongly the model can separate the groups.
"""
open(f"{OUT}/LIMITATIONS.txt", "w").write(LIM)

# ---------------- README index ----------------
README = """FINAL RESULTS PACKAGE - TPP-Gaze ASD/TD
Generated from completed Phase-1/Phase-2/Saliency4ASD/Gaze4ASD/statistics runs.
No model was retrained or modified to build this package.

Tables:
  1_main_results_saliency4asd_test.csv  Phase 1 vs Phase 2 (held-out test NLL + improvement)
  2_external_gaze4asd_table.csv         Gaze4ASD NLL (Policy A/B) + behavioral means
  3_participant_statistics.csv          Hedges g / Cohen dz / permutation p / 95% CIs / direction match

Figures (figures/):
  fig1_phase1_vs_phase2_nll.png                     model likelihood (Phase1 vs Phase2)
  fig2_real_vs_gen_ASD_TD_distributions.png         real ASD/TD vs generated ASD/TD (5 metrics)
  fig3_effect_size_forest.png                       real g vs generated dz (95% CI)
  fig4a_representative_scanpaths_saliency4asd.png    same-image ASD vs TD generated (in-domain)
  fig4b_representative_scanpaths_gaze4asd.png        same-image ASD vs TD generated (external)
  fig5_real_vs_generated_similarity.png             MultiMatch real-vs-gen vs real-vs-real

Text:
  MAIN_FINDINGS.txt   supported results, split into likelihood / direction / significance / generalization
  LIMITATIONS.txt     6 documented limitations; no clinical-validity claims
"""
open(f"{OUT}/README.txt", "w").write(README)

print("FINAL PACKAGE CONTENTS:")
for root, _, fs in os.walk(OUT):
    for fn in sorted(fs):
        print("  ", os.path.relpath(os.path.join(root, fn), OUT))
print("BUILD_DONE")
