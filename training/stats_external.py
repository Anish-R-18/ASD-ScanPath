"""
Formal statistical analysis of Gaze4ASD external validation (participant-level).
Reads the saved per-scanpath metrics; does NOT touch any model/raw data.

Units of analysis:
  REAL      -> one value per PARTICIPANT (mean over that participant's trials);
               ASD (n=33) vs TD (n=133) are INDEPENDENT groups.
  GENERATED -> one value per IMAGE (mean over the 20 sims per condition);
               ASD-cond vs TD-cond share the 30 images -> PAIRED by image.

Tests are exploratory (no pre-registration); a Bonferroni note is included.
"""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "training/runs/external_gaze4asd"
CSV = os.path.join(OUT, "external_per_scanpath_metrics.csv")
METRICS = {
    "dur_mean": "Fixation duration (s)",
    "sacc_amp_norm_mean": "Saccade amplitude (norm.)",
    "dispersion": "Spatial dispersion (norm.)",
    "n_fix": "Fixation count",
    "revisit_rate": "Revisit rate",
}
NPERM, NBOOT, SEED = 10000, 10000, 0
rng = np.random.default_rng(SEED)

rows = list(csv.DictReader(open(CSV)))
def val(r, k):
    try:
        v = float(r[k]); return v if np.isfinite(v) else np.nan
    except (ValueError, KeyError):
        return np.nan

def group_means(pop, unit_key, metric):
    """mean(metric) per unit (participant or image) for a population."""
    buckets = {}
    for r in rows:
        if r["population"] != pop:
            continue
        u = r[unit_key]
        if u in ("", "None"):
            continue
        buckets.setdefault(u, []).append(val(r, metric))
    out = {}
    for u, vs in buckets.items():
        vs = [v for v in vs if np.isfinite(v)]
        if vs:
            out[u] = float(np.mean(vs))
    return out

# ---- effect sizes / CIs / tests ----
def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp
    J = 1 - 3 / (4 * (na + nb) - 9)
    return d * J

def perm_indep(a, b, nperm=NPERM):
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); n = len(a)
    cnt = 0
    for _ in range(nperm):
        p = rng.permutation(pool)
        if abs(p[:n].mean() - p[n:].mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (nperm + 1)

def boot_indep(a, b, nboot=NBOOT):
    diffs, gs = [], []
    for _ in range(nboot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        diffs.append(aa.mean() - bb.mean()); gs.append(hedges_g(aa, bb))
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return ci(diffs), ci(gs)

def perm_paired(diff, nperm=NPERM):
    obs = diff.mean()
    signs = rng.choice([-1.0, 1.0], size=(nperm, len(diff)))
    pm = (signs * diff).mean(axis=1)
    return obs, (np.sum(np.abs(pm) >= abs(obs) - 1e-12) + 1) / (nperm + 1)

def boot_paired(diff, nboot=NBOOT):
    md, dz = [], []
    for _ in range(nboot):
        s = diff[rng.integers(0, len(diff), len(diff))]
        md.append(s.mean()); dz.append(s.mean() / s.std(ddof=1) if s.std(ddof=1) > 0 else 0.0)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return ci(md), ci(dz)

def desc(x):
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "median": float(np.median(x))}

results = {"metrics": {}, "notes": {
    "real_unit": "participant (ASD n, TD n independent)",
    "generated_unit": "image (ASD-cond vs TD-cond, paired by image)",
    "effect_convention": "ASD minus TD",
    "tests": "two-sided permutation (10000); 95% CI via bootstrap (10000)",
    "exploratory": True,
    "bonferroni_alpha_5_tests": 0.01}}

plot_data = {}
for m, label in METRICS.items():
    # REAL (participant-level, independent)
    ra = np.array(list(group_means("real_ASD", "participant", m).values()))
    rt = np.array(list(group_means("real_TD", "participant", m).values()))
    r_obs, r_p = perm_indep(ra, rt); r_g = hedges_g(ra, rt)
    r_dci, r_gci = boot_indep(ra, rt)
    # GENERATED (image-level, paired)
    ga = group_means("gen_ASD", "image_id", m); gt = group_means("gen_TD", "image_id", m)
    imgs = sorted(set(ga) & set(gt))
    gaa = np.array([ga[i] for i in imgs]); gtt = np.array([gt[i] for i in imgs])
    diff = gaa - gtt
    g_obs, g_p = perm_paired(diff); g_dz = float(diff.mean() / diff.std(ddof=1))
    g_dci, g_dzci = boot_paired(diff)

    results["metrics"][m] = {
        "label": label,
        "real": {"ASD": desc(ra), "TD": desc(rt), "mean_diff_ASD_minus_TD": float(r_obs),
                 "mean_diff_CI95": r_dci, "hedges_g": float(r_g), "hedges_g_CI95": r_gci,
                 "perm_p": float(r_p), "significant_0.05": bool(r_p < 0.05),
                 "significant_bonf_0.01": bool(r_p < 0.01)},
        "generated": {"ASD": desc(gaa), "TD": desc(gtt), "mean_diff_ASD_minus_TD": float(g_obs),
                      "mean_diff_CI95": g_dci, "cohen_dz_paired": g_dz, "cohen_dz_CI95": g_dzci,
                      "perm_p": float(g_p), "significant_0.05": bool(g_p < 0.05),
                      "significant_bonf_0.01": bool(g_p < 0.01)},
        "comparison": {
            "real_direction": "ASD>TD" if r_obs > 0 else "ASD<TD",
            "generated_direction": "ASD>TD" if g_obs > 0 else "ASD<TD",
            "direction_match": bool(np.sign(r_obs) == np.sign(g_obs)),
            "real_hedges_g": float(r_g), "generated_cohen_dz": g_dz},
    }
    plot_data[m] = {"label": label, "ra": ra, "rt": rt, "ga": gaa, "gt": gtt,
                    "r_g": r_g, "r_gci": r_gci, "g_dz": g_dz, "g_dzci": g_dzci}

json.dump(results, open(os.path.join(OUT, "results_statistics.json"), "w"), indent=2)

# ---- text report ----
def star(p): return "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
T = ["GAZE4ASD EXTERNAL VALIDATION - PARTICIPANT-LEVEL STATISTICS (exploratory)",
     "Effect convention: ASD minus TD. REAL=participant-level (independent), "
     "GENERATED=image-level (paired). Permutation p (two-sided, 10k); 95% CI bootstrap (10k).",
     "Bonferroni alpha for 5 tests = 0.01.", ""]
for m, label in METRICS.items():
    d = results["metrics"][m]; r, g, c = d["real"], d["generated"], d["comparison"]
    T.append(f"=== {label} ===")
    T.append(f"  REAL  ASD(n={r['ASD']['n']}): mean={r['ASD']['mean']:.4f} sd={r['ASD']['std']:.4f} med={r['ASD']['median']:.4f}")
    T.append(f"        TD (n={r['TD']['n']}): mean={r['TD']['mean']:.4f} sd={r['TD']['std']:.4f} med={r['TD']['median']:.4f}")
    T.append(f"        diff(ASD-TD)={r['mean_diff_ASD_minus_TD']:+.4f} CI95={np.round(r['mean_diff_CI95'],4).tolist()} "
             f"Hedges_g={r['hedges_g']:+.3f} CI95={np.round(r['hedges_g_CI95'],3).tolist()} p={r['perm_p']:.4f} [{star(r['perm_p'])}]")
    T.append(f"  GEN   ASD(n={g['ASD']['n']} img): mean={g['ASD']['mean']:.4f}  TD: mean={g['TD']['mean']:.4f}")
    T.append(f"        diff(ASD-TD)={g['mean_diff_ASD_minus_TD']:+.4f} CI95={np.round(g['mean_diff_CI95'],4).tolist()} "
             f"dz(paired)={g['cohen_dz_paired']:+.3f} CI95={np.round(g['cohen_dz_CI95'],3).tolist()} p={g['perm_p']:.4f} [{star(g['perm_p'])}]")
    T.append(f"  COMPARE: real {c['real_direction']} | gen {c['generated_direction']} | "
             f"direction_match={c['direction_match']}")
    T.append("")
open(os.path.join(OUT, "results_statistics.txt"), "w").write("\n".join(T) + "\n")
print("\n".join(T))

# ---- summary CSV (one row per metric) ----
with open(os.path.join(OUT, "results_statistics_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "real_ASD_mean", "real_TD_mean", "real_diff", "real_diff_CI_lo", "real_diff_CI_hi",
                "real_hedges_g", "real_p", "gen_ASD_mean", "gen_TD_mean", "gen_diff",
                "gen_dz", "gen_p", "direction_match"])
    for m, label in METRICS.items():
        d = results["metrics"][m]; r, g, c = d["real"], d["generated"], d["comparison"]
        w.writerow([label, f"{r['ASD']['mean']:.4f}", f"{r['TD']['mean']:.4f}", f"{r['mean_diff_ASD_minus_TD']:.4f}",
                    f"{r['mean_diff_CI95'][0]:.4f}", f"{r['mean_diff_CI95'][1]:.4f}", f"{r['hedges_g']:.3f}",
                    f"{r['perm_p']:.4f}", f"{g['ASD']['mean']:.4f}", f"{g['TD']['mean']:.4f}",
                    f"{g['mean_diff_ASD_minus_TD']:.4f}", f"{g['cohen_dz_paired']:.3f}", f"{g['perm_p']:.4f}",
                    c["direction_match"]])

# ---- plots (publication quality) ----
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 150})
ASD_C, TD_C = "#d1495b", "#2e74b5"

# distributions: 5 metrics x (real | generated)
fig, axs = plt.subplots(len(METRICS), 2, figsize=(10, 3.0 * len(METRICS)))
for i, (m, pd) in enumerate(plot_data.items()):
    for j, (A, B, title) in enumerate([(pd["ra"], pd["rt"], "REAL (participant)"),
                                       (pd["ga"], pd["gt"], "GENERATED (image)")]):
        ax = axs[i, j]
        bp = ax.boxplot([A, B], widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color="k"))
        for patch, col in zip(bp["boxes"], [ASD_C, TD_C]):
            patch.set_facecolor(col); patch.set_alpha(0.35)
        for k, (data, col) in enumerate([(A, ASD_C), (B, TD_C)], start=1):
            jit = (rng.random(len(data)) - 0.5) * 0.18 + k
            ax.scatter(jit, data, s=14, color=col, alpha=0.6, edgecolors="none", zorder=3)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["ASD", "TD"])
        if j == 0:
            ax.set_ylabel(pd["label"])
        ax.set_title(title if i == 0 else "", fontsize=11)
fig.suptitle("Gaze4ASD: ASD vs TD  (real participants | model-generated)", y=1.002, fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_stat_distributions.png"), bbox_inches="tight"); plt.close(fig)

# effect-size forest plot
fig, ax = plt.subplots(figsize=(8, 5))
ys = np.arange(len(METRICS))[::-1]
labels = [plot_data[m]["label"] for m in METRICS]
for k, m in enumerate(METRICS):
    pd = plot_data[m]; y = ys[k]
    ax.errorbar(pd["r_g"], y + 0.15, xerr=[[pd["r_g"] - pd["r_gci"][0]], [pd["r_gci"][1] - pd["r_g"]]],
                fmt="o", color="#444", capsize=3, label="real Hedges g" if k == 0 else "")
    ax.errorbar(pd["g_dz"], y - 0.15, xerr=[[pd["g_dz"] - pd["g_dzci"][0]], [pd["g_dzci"][1] - pd["g_dz"]]],
                fmt="s", color="#e07b00", capsize=3, label="generated dz" if k == 0 else "")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_yticks(ys); ax.set_yticklabels(labels); ax.set_xlabel("Effect size (ASD - TD)")
ax.set_title("ASD-vs-TD effect size: real vs generated"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_stat_effect_sizes.png"), bbox_inches="tight"); plt.close(fig)

print("\nARTIFACTS:", sorted(f for f in os.listdir(OUT) if f.startswith("results_") or f.startswith("fig_stat")))
print("STATS_DONE")
