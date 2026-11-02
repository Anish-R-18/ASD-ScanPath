"""
Build the tracked publication summary in results/external_gaze4asd_v2/ from local, Git-ignored outputs.
No model work. Tables are derived from machine-readable results, reports are reproduced verbatim, and figures
come from an explicit list. Nothing based on the old native-size Gaze4ASD mapping is included, nor raw data,
per-scanpath or per-participant records, checkpoints, logs, or images that show dataset stimuli.

  python training/build_results_v2.py
"""
from __future__ import annotations
import csv, hashlib, json, os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/external_gaze4asd_v2"
TAB, FIG = f"{OUT}/tables", f"{OUT}/figures"
V2 = "training/runs/external_gaze4asd_v2"
P12 = f"{V2}/phase1_vs_phase2"
TAU1 = f"{V2}/temperature_tau1"
VAL = "training/runs/temperature_sensitivity"
REP = "external_data/Gaze4ASD_processed_v2/reports"
SAL = "training/runs/phase2_embedding_only/test_eval_phase2.json"
PRIMARY = "fit_height_policy_a"
CORRECTED = (PRIMARY, "stretch_policy_a", "fit_height_policy_b")
FIGURES = {
    f"{P12}/comparison_figures/fig1_nll_improvement.png": "phase1_vs_phase2_nll_gain.png",
    f"{P12}/comparison_figures/fig5_embedding_by_group.png": "phase2_embedding_by_group.png",
    f"{TAU1}/figures/fig1_metrics_tau1_vs_tau2.png": "generated_metrics_tau1_vs_tau2.png",
    f"{TAU1}/figures/fig2_group_differences.png": "asd_td_differences_tau1_vs_tau2.png",
    f"{TAU1}/figures/fig3_multimatch.png": "multimatch_tau1_vs_tau2.png",
    f"{TAU1}/figures/fig4_seed_stability.png": "tau1_seed_stability.png",
    f"{TAU1}/figures/fig5_transfer_from_validation.png": "tau1_transfer_validation_to_gaze4asd.png",
    f"{VAL}/figures/fig2_mechanism.png": "temperature_mechanism.png",
    f"{VAL}/figures/fig1_metrics_by_temperature.png": "validation_metrics_by_temperature.png",
}
provenance = []


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def read_text(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def note(out, *sources):
    for s in sources:
        provenance.append(dict(output=out, source=s, source_sha256=sha(s)))


def table(name, rows, *sources):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(f"{TAB}/{name}", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r.get(k), (list, dict)) else r.get(k, "")) for k in keys})
    note(f"tables/{name}", *sources)


def verbatim(title, path):
    return f"\n{'=' * 88}\n{title}\nReproduced verbatim from {path}\n{'=' * 88}\n\n{read_text(path).rstrip()}\n"


for folder, _, files in os.walk(OUT):
    for name in files:
        os.remove(os.path.join(folder, name))
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

G = read_csv(f"{P12}/group_nll_comparison.csv")
OV = read_json(f"{P12}/overall_nll_comparison.json")
Sx = read_json(f"{TAU1}/statistical_results.json")
PL = [r for r in read_csv(f"{V2}/statistics/participant_level_real.csv") if r["version"] == PRIMARY]
SUM = [r for r in read_csv(f"{V2}/comparison/nll_summary.csv") if r["version"] in CORRECTED]
CB = read_json(f"{V2}/comparison/nll_uncertainty.json")["model_vs_centre_bias"]
Tt = read_json(SAL)
DS = read_csv(f"{REP}/dataset_summary.csv")

# ---------------- tables
table("dataset_summary.csv", DS, f"{REP}/dataset_summary.csv")
table("saliency4asd_test_nll.csv", [dict(model=m, group=g, **Tt[k][g]) for m, k in (("Phase 1", "phase1"), ("Phase 2", "phase2")) for g in ("overall", "ASD", "TD")], SAL)
table("external_nll_phase1_vs_phase2.csv", G, f"{P12}/group_nll_comparison.csv")
spec = OV["embedding_specificity"]
table("external_embedding_by_group.csv", [dict(component=k, comparison=c, **v) for k, x in spec["fixed_embedding_vs_phase1"].items() for c, v in x.items()], f"{P12}/overall_nll_comparison.json")
table("external_phase2_by_mapping.csv", SUM, f"{V2}/comparison/nll_summary.csv")
table("external_position_vs_centre_bias.csv", [dict(version=k, **CB[k]) for k in CORRECTED], f"{V2}/comparison/nll_uncertainty.json")
table("real_asd_vs_td_participant_level.csv", PL, f"{V2}/statistics/participant_level_real.csv")
table("tau1_vs_tau2_comparison.csv", read_csv(f"{TAU1}/tau1_vs_tau2_comparison.csv"), f"{TAU1}/tau1_vs_tau2_comparison.csv")
table("tau1_prespecified_tests.csv", Sx["prespecified"], f"{TAU1}/statistical_results.json")
table("tau1_vs_tau2_realism.csv", Sx["realism"], f"{TAU1}/statistical_results.json")
table("tau1_vs_tau2_distribution_distances.csv", Sx["distances"], f"{TAU1}/statistical_results.json")
table("tau1_vs_tau2_asd_td_differences.csv", Sx["group_differences"], f"{TAU1}/statistical_results.json")
table("tau1_vs_tau2_multimatch.csv", Sx["multimatch"], f"{TAU1}/statistical_results.json")
table("tau1_vs_tau2_window_matched.csv", Sx["window_matched"]["realism"], f"{TAU1}/statistical_results.json")
table("temperature_transfer_validation_to_gaze4asd.csv", Sx["transfer_from_validation"]["realism"], f"{TAU1}/statistical_results.json")
table("validation_temperature_comparison.csv", [r for r in read_csv(f"{VAL}/temperature_comparison.csv") if r["analysis"] == "clipped_primary"], f"{VAL}/temperature_comparison.csv")

# ---------------- figures
for src, dst in FIGURES.items():
    shutil.copyfile(src, f"{FIG}/{dst}")
    note(f"figures/{dst}", src)
order = [r for r in sorted(PL, key=lambda r: float(r["effect"]))]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#898781", "text.color": "#0b0b0b", "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "savefig.dpi": 200, "axes.axisbelow": True})
fig, ax = plt.subplots(figsize=(7, 3.6))
for i, r in enumerate(order):
    c = json.loads(r["effect_ci"])
    sig = float(r["p_holm"]) < .05
    ax.plot(c, [i, i], color="#0b0b0b" if sig else "#898781", lw=1.6, solid_capstyle="round")
    ax.plot(float(r["effect"]), i, "o", mfc="#0b0b0b" if sig else "#fcfcfb", mec="#0b0b0b" if sig else "#898781", ms=6.5, mew=1.3)
ax.axvline(0, color="#c3c2b7", lw=1)
ax.set_yticks(range(len(order))); ax.set_yticklabels([r["label"] for r in order], color="#52514e")
ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#e1e0d9", lw=.6)
ax.set_xlabel("Hedges' g (positive: ASD higher), with 95% bootstrap interval")
ax.set_title("Real ASD vs TD on corrected Gaze4ASD, participant level", loc="left", fontsize=10)
ax.plot([], [], "o", color="#0b0b0b", label="Holm p < .05"); ax.plot([], [], "o", mfc="#fcfcfb", mec="#898781", label="Not significant")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.35, -.42), ncol=2)
fig.savefig(f"{FIG}/real_asd_vs_td_participant_level.png", bbox_inches="tight"); plt.close(fig)
note("figures/real_asd_vs_td_participant_level.png", f"{V2}/statistics/participant_level_real.csv")

# ---------------- text
g = {(r["group"], r["component"]): r for r in G}
gd = {d["metric"]: d for d in Sx["group_differences"]}
rl = {(r["group"], r["metric"]): r for r in Sx["realism"]}
s = ["FINAL EXTERNAL COMPARISON - CORRECTED GAZE4ASD",
     "",
     "Corrected dataset: fit-to-height mapping, Policy A. The mapping is strongly supported but partly inferred, and NOT",
     "experimentally confirmed. Frozen models; no retraining. All statistics are exploratory unless marked. No clinical or",
     "diagnostic validity is claimed. The summary tables below are generated directly from the result files; see tables/.",
     "",
     "1. PHASE 1 VS PHASE 2, NLL PER FIXATION (lower is better; gain > 0 favours Phase 2)",
     f"{'':19s}{'Phase 1':>9s}{'Phase 2':>9s}{'Gain':>9s}{'Rel.':>8s}   {'95% CI, participants':24s}{'95% CI, images':24s}"]
fmt_ci = lambda c: f"[{float(c[0]):+.4f}, {float(c[1]):+.4f}]"
for gr in ("overall", "ASD", "TD"):
    for k in ("joint", "duration", "position"):
        r = g[(gr, k)]
        s.append(f"{gr + ' ' + k:19s}{float(r['phase1_nll']):9.4f}{float(r['phase2_nll']):9.4f}{float(r['improvement']):+9.4f}{float(r['relative_pct']):+7.1f}%   {fmt_ci(json.loads(r['participant_ci'])):24s}{fmt_ci(json.loads(r['image_ci'])):24s}")
s += ["", "Phase 1 is better overall. Conditioning does not improve overall external NLL. The Saliency4ASD gain did not transfer.",
      "For reference, Saliency4ASD held-out test NLL: Phase 1 " + f"{Tt['phase1']['overall']['nll']:.4f}, Phase 2 {Tt['phase2']['overall']['nll']:.4f}.",
      f"Phase 2 position NLL beats a centre-bias reference by {-CB[PRIMARY]['model_minus_centre']:.2f} [{-CB[PRIMARY]['model_minus_centre_ci'][1]:.2f}, {-CB[PRIMARY]['model_minus_centre_ci'][0]:.2f}] on {CB[PRIMARY]['images_model_better']} of {CB[PRIMARY]['images']} images.",
      "", "2. REAL ASD VS TD, PARTICIPANT LEVEL (33 ASD, 133 TD; one mean per participant; Holm within the family)",
      f"{'Measure':36s}{'ASD':>8s}{'TD':>8s}{'g':>8s}   {'95% CI':18s}{'Holm p':>8s}"]
for r in sorted(PL, key=lambda r: -abs(float(r["effect"]))):
    c = json.loads(r["effect_ci"])
    s.append(f"{r['label']:36s}{float(r['mean_a']):8.3f}{float(r['mean_b']):8.3f}{float(r['effect']):+8.2f}   [{c[0]:+.2f}, {c[1]:+.2f}]    {float(r['p_holm']):8.3f}")
s += ["", "3. GENERATED ASD MINUS TD (Phase 2; image-level; pooled over 10 seeds)",
      f"{'Measure':36s}{'Real':>9s}{'tau=1':>9s}{'tau=2':>9s}{'t=1 % of real':>15s}{'t=2 % of real':>15s}"]
for m in ("dispersion", "sacc_amp_norm_mean", "revisit_rate", "sacc_dir_resultant", "total_dur", "n_fix"):
    d = gd[m]
    same = d["tau1"] * d["real"] > 0 and d["tau2"] * d["real"] > 0
    s.append(f"{d['label']:36s}{d['real']:+9.4f}{d['tau1']:+9.4f}{d['tau2']:+9.4f}" + (f"{100 * d['ratio_tau1']:14.0f}%{100 * d['ratio_tau2']:14.0f}%" if same else f"{'wrong direction':>30s}"))
s += ["", "4. TAU=1 VS TAU=2 ON GAZE4ASD (per-image mean absolute error; four prespecified tests, not preregistered)",
      f"{'Test':30s}{'tau=1':>8s}{'tau=2':>8s}{'Reduction':>11s}  {'95% CI of gain':22s}{'Holm p':>8s}{'Seeds':>7s}"]
for r in Sx["prespecified"]:
    s.append(f"{r['label'].split(' (')[0] + ', ' + r['group']:30s}{r['mae_tau1']:8.4f}{r['mae_tau2']:8.4f}{r['rel_pct_of_tau2']:10.1f}%  {fmt_ci(r['ci']):22s}{r['p_holm_prespecified']:8.4f}{r['seeds']['n_pos']:5d}/10")
s += ["", "Temperature changes sampling only. It does not change the model or its NLL. tau=1 was selected on Saliency4ASD validation images.",
      f"Fixation-count overproduction remains at tau=1: {rl[('ASD', 'n_fix')]['tau1']:.2f} (ASD) and {rl[('TD', 'n_fix')]['tau1']:.2f} (TD), against real {rl[('ASD', 'n_fix')]['real']:.2f} and {rl[('TD', 'n_fix')]['real']:.2f}.",
      "Full tau=1 and temperature reports: temperature_sensitivity.txt."]
with open(f"{OUT}/final_external_comparison.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(s) + "\n" + verbatim("FULL REPORT: PHASE 1 VS PHASE 2 ON CORRECTED GAZE4ASD", f"{P12}/final_comparison_report.txt"))
note("final_external_comparison.txt", f"{P12}/final_comparison_report.txt", f"{P12}/group_nll_comparison.csv", f"{V2}/statistics/participant_level_real.csv", f"{TAU1}/statistical_results.json", f"{V2}/comparison/nll_uncertainty.json", SAL)
with open(f"{OUT}/temperature_sensitivity.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write("TEMPERATURE SENSITIVITY AND GAZE4ASD EVALUATION AT TAU=1\n\nTemperature was selected on Saliency4ASD validation images (Parts A and B) and then evaluated on Gaze4ASD (Parts C and D).\nIt changes sampling only, not the model or its NLL. Paths inside these reports refer to local, Git-ignored outputs.\n"
            + verbatim("PART A: VALIDATION-SET REPORT", f"{VAL}/final_report.txt") + verbatim("PART B: VALIDATION-SET METHODOLOGY", f"{VAL}/temperature_methodology.txt")
            + verbatim("PART C: GAZE4ASD TAU=1 REPORT", f"{TAU1}/final_report.txt") + verbatim("PART D: GAZE4ASD TAU=1 METHODOLOGY", f"{TAU1}/methodology_notes.txt"))
note("temperature_sensitivity.txt", f"{VAL}/final_report.txt", f"{VAL}/temperature_methodology.txt", f"{TAU1}/final_report.txt", f"{TAU1}/methodology_notes.txt")
m = ["CORRECTED GAZE4ASD MAPPING - METHODOLOGY AND UNCERTAINTY", "",
     "Status: fit-to-height is strongly supported but partly inferred, and NOT experimentally confirmed.",
     read_text(f"{REP}/mapping_methodology.txt").rstrip(), "",
     "DATASET VERSIONS (from dataset_summary.csv)",
     f"{'Version':22s}{'Role':13s}{'ASD/TD part.':>13s}{'ASD/TD trials':>15s}{'Fixations':>11s}{'Out of image':>14s}"]
for r in DS:
    m.append(f"{r['version']:22s}{r['role']:13s}{r['participants_ASD'] + '/' + r['participants_TD']:>13s}{r['trials_ASD'] + '/' + r['trials_TD']:>15s}{int(r['fixations_ASD']) + int(r['fixations_TD']):11d}{float(r['out_of_image_pct']):13.2f}%")
m += ["", "PHASE 2 NLL BY CORRECTED VERSION (overall; position NLL is not comparable between mappings without scale correction)",
      f"{'Version':22s}{'Joint':>9s}{'Duration':>10s}{'Position':>10s}"]
for r in SUM:
    if r["group"] == "overall":
        m.append(f"{r['version']:22s}{float(r['nll']):9.4f}{float(r['nll_duration']):10.4f}{float(r['nll_position']):10.4f}")
m += ["", "The older native-size centred mapping is SUPERSEDED and is not part of this package. See ../external_gaze4asd/SUPERSEDED.txt."]
with open(f"{OUT}/corrected_mapping_methodology.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(m) + "\n")
note("corrected_mapping_methodology.txt", f"{REP}/mapping_methodology.txt", f"{REP}/dataset_summary.csv", f"{V2}/comparison/nll_summary.csv")

readme = """# Corrected Gaze4ASD Results

Publication summaries for the corrected Gaze4ASD evaluation. Built by `training/build_results_v2.py` from local outputs that Git ignores. `tables/provenance.csv` lists each output's source file and its SHA-256 hash.

**Status.** The fit-to-height mapping is strongly supported but partly inferred, and not experimentally confirmed. Statistics are exploratory unless marked. No clinical or diagnostic validity is claimed.

## Main results

| Measure | Result |
|---|---|
| External NLL, Phase 1 vs Phase 2 | {p1} vs {p2}: Phase 1 is better overall by {gain} |
| In-domain NLL, Phase 1 vs Phase 2 | {sp1} vs {sp2}: Phase 2 is better by {sgain} |
| Selected temperature | τ=1, chosen on Saliency4ASD validation images |
| Fixation count at τ=1, ASD and TD | {ca} and {ct}, against {ra} and {rt} real |

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
""".format(p1=f"{float(g[('overall', 'joint')]['phase1_nll']):.4f}", p2=f"{float(g[('overall', 'joint')]['phase2_nll']):.4f}", gain=f"{-float(g[('overall', 'joint')]['improvement']):.4f}", sp1=f"{Tt['phase1']['overall']['nll']:.4f}", sp2=f"{Tt['phase2']['overall']['nll']:.4f}", sgain=f"{Tt['phase1']['overall']['nll'] - Tt['phase2']['overall']['nll']:.4f}", ca=f"{rl[('ASD', 'n_fix')]['tau1']:.2f}", ct=f"{rl[('TD', 'n_fix')]['tau1']:.2f}", ra=f"{rl[('ASD', 'n_fix')]['real']:.2f}", rt=f"{rl[('TD', 'n_fix')]['real']:.2f}")
with open(f"{OUT}/README.md", "w", encoding="utf-8", newline="\n") as f:
    f.write(readme)
table("provenance.csv", provenance, SAL)
print("built", OUT, "files:", sum(len(f) for _, _, f in os.walk(OUT)))
