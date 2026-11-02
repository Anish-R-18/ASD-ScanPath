"""
Exploratory statistics, sensitivity tables and figures for the corrected Gaze4ASD evaluation.
Reads only what eval_external_v2.py wrote. No model is loaded.

REAL ASD vs TD: one mean per participant (independent groups), Hedges' g.
GENERATED ASD vs TD and REAL vs GENERATED: one mean per image (paired), Cohen's dz.
Generated dz depends on the number of samples per image and is not on the same scale as g.
"""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import gaussian_filter

OUT = "training/runs/external_gaze4asd_v2"
IMG = "external_data/Gaze4ASD/Image Stimuli"
SEED, N = 0, 10000
VERS = ["fit_height_policy_a", "stretch_policy_a", "fit_height_policy_b"]
REFS = ["old_native_policy_a", "old_native_policy_b"]
LABEL = {"fit_height_policy_a": "Fit-to-height, Policy A (primary)", "stretch_policy_a": "Stretch, Policy A", "fit_height_policy_b": "Fit-to-height, Policy B",
         "old_native_policy_a": "Old native, Policy A", "old_native_policy_b": "Old native, Policy B"}
METRICS = {"dur_mean": "Fixation duration (s)", "dur_median": "Median fixation duration (s)", "n_fix": "Fixation count", "total_dur": "Total viewing duration (s)",
           "fix_rate_per_s": "Fixation rate (per s of fixation)", "sacc_amp_norm_mean": "Saccade amplitude (/diagonal)", "dispersion": "Spatial dispersion (/diagonal)",
           "revisit_rate": "Revisit rate", "sacc_dir_resultant": "Saccade direction concentration", "sacc_horizontal_frac": "Horizontal saccade fraction"}
SPATIAL = ["sacc_amp_norm_mean", "dispersion", "revisit_rate", "sacc_dir_resultant", "sacc_horizontal_frac"]
TEMPORAL = ["dur_mean", "dur_median", "n_fix", "total_dur", "fix_rate_per_s"]
MM = ["mm_shape", "mm_length", "mm_direction", "mm_position", "mm_duration"]
ASD, TD, INK, INK2, MUTED, GRID, AXIS, SURF = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titlecolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "savefig.dpi": 200, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True})
rng = np.random.default_rng(SEED)


def dirof(v):
    return f"{OUT}/comparison/{v}" if v.startswith("old") else f"{OUT}/{v}"


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def means_by(rows, pop, unit, m):
    b = {}
    for r in rows:
        if r["population"] == pop:
            v = num(r.get(m))
            if np.isfinite(v):
                b.setdefault(r[unit], []).append(v)
    return {k: float(np.mean(v)) for k, v in b.items()}


def hedges(a, b):
    na, nb = a.shape[-1], b.shape[-1]
    sp = np.sqrt(((na - 1) * a.var(-1, ddof=1) + (nb - 1) * b.var(-1, ddof=1)) / (na + nb - 2))
    return (a.mean(-1) - b.mean(-1)) / sp * (1 - 3 / (4 * (na + nb) - 9))


def independent(a, b):
    obs = a.mean() - b.mean()
    pool = np.r_[a, b]
    perm = rng.permuted(np.tile(pool, (N, 1)), axis=1)
    p = (np.sum(np.abs(perm[:, :len(a)].mean(1) - perm[:, len(a):].mean(1)) >= abs(obs) - 1e-12) + 1) / (N + 1)
    ba = a[rng.integers(0, len(a), (N, len(a)))]
    bb = b[rng.integers(0, len(b), (N, len(b)))]
    return dict(n_a=len(a), n_b=len(b), mean_a=float(a.mean()), sd_a=float(a.std(ddof=1)), median_a=float(np.median(a)), mean_b=float(b.mean()), sd_b=float(b.std(ddof=1)), median_b=float(np.median(b)),
                diff=float(obs), diff_ci=ci(ba.mean(1) - bb.mean(1)), effect=float(hedges(a, b)), effect_ci=ci(hedges(ba, bb)), effect_name="hedges_g", p=float(p))


def paired(a, b):
    d = a - b
    obs = d.mean()
    pm = (rng.choice([-1.0, 1.0], (N, len(d))) * d).mean(1)
    p = (np.sum(np.abs(pm) >= abs(obs) - 1e-12) + 1) / (N + 1)
    bd = d[rng.integers(0, len(d), (N, len(d)))]
    sd = bd.std(1, ddof=1)
    return dict(n_a=len(a), n_b=len(b), mean_a=float(a.mean()), sd_a=float(a.std(ddof=1)), median_a=float(np.median(a)), mean_b=float(b.mean()), sd_b=float(b.std(ddof=1)), median_b=float(np.median(b)),
                diff=float(obs), diff_ci=ci(bd.mean(1)), effect=float(obs / d.std(ddof=1)), effect_ci=ci(bd.mean(1) / np.where(sd > 0, sd, np.nan)), effect_name="cohen_dz", p=float(p))


def holm(rows):
    o = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m, run = len(rows), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * rows[i]["p"]))
        rows[i]["p_holm"] = run
    return rows


def contrasts(v):
    rows = read_csv(f"{dirof(v)}/per_scanpath_metrics.csv")
    res = {}
    imgs = sorted({r["image_id"] for r in rows}, key=lambda s: int(s.split("_")[-1].split(".")[0]))
    def img(pop, m):
        d = means_by(rows, pop, "image_id", m)
        return np.array([d[i] for i in imgs])
    for name, fn in (("real_ASD_vs_TD_participant", lambda m: independent(np.array(list(means_by(rows, "real_ASD", "participant", m).values())), np.array(list(means_by(rows, "real_TD", "participant", m).values())))),
                     ("real_ASD_vs_TD_image_paired", lambda m: paired(img("real_ASD", m), img("real_TD", m))),
                     ("gen_ASD_vs_TD_image_paired", lambda m: paired(img("gen_ASD", m), img("gen_TD", m))),
                     ("real_ASD_vs_gen_ASD_image_paired", lambda m: paired(img("real_ASD", m), img("gen_ASD", m))),
                     ("real_TD_vs_gen_TD_image_paired", lambda m: paired(img("real_TD", m), img("gen_TD", m)))):
        L = [dict(metric=m, label=METRICS[m], contrast=name, version=v, **fn(m)) for m in METRICS]
        res[name] = {x["metric"]: x for x in holm(L)}
    # does the generated group difference match the real one, on the same image-level unit?
    rep = []
    for m in METRICS:
        dr, dg = img("real_ASD", m) - img("real_TD", m), img("gen_ASD", m) - img("gen_TD", m)
        idx = rng.integers(0, len(imgs), (N, len(imgs)))
        rep.append(dict(metric=m, label=METRICS[m], version=v, real_diff=float(dr.mean()), gen_diff=float(dg.mean()), real_participant_g=res["real_ASD_vs_TD_participant"][m]["effect"],
                        real_participant_p_holm=res["real_ASD_vs_TD_participant"][m]["p_holm"], real_image_dz=res["real_ASD_vs_TD_image_paired"][m]["effect"], gen_image_dz=res["gen_ASD_vs_TD_image_paired"][m]["effect"],
                        gen_image_p_holm=res["gen_ASD_vs_TD_image_paired"][m]["p_holm"], same_direction=bool(np.sign(dr.mean()) == np.sign(dg.mean())), gen_minus_real=float(dg.mean() - dr.mean()),
                        gen_minus_real_ci=ci((dg - dr)[idx].mean(1)), real_diff_ci=ci(dr[idx].mean(1)), gen_diff_ci=ci(dg[idx].mean(1)), image_corr=float(np.corrcoef(dr, dg)[0, 1])))
    res["effect_reproduction"] = {x["metric"]: x for x in rep}
    # descriptive: circular mean direction, reported but not tested
    res["circular_mean_direction_deg"] = {}
    for p in ("real_ASD", "real_TD", "gen_ASD", "gen_TD"):
        a = np.array([num(r["sacc_dir_circ_mean"]) for r in rows if r["population"] == p]); a = a[np.isfinite(a)]
        res["circular_mean_direction_deg"][p] = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))
    # counts: raw, within the real viewing window, and rate
    cnt = {}
    for g in ("ASD", "TD"):
        cnt[g] = {"real_n_fix": img(f"real_{g}", "n_fix"), "gen_n_fix": img(f"gen_{g}", "n_fix"), "gen_n_fix_in_real_window": img(f"gen_{g}", "n_fix_in_real_window")}
        x = cnt[g]
        cnt[g] = {"real_n_fix": float(x["real_n_fix"].mean()), "gen_n_fix": float(x["gen_n_fix"].mean()), "gen_n_fix_in_real_window": float(x["gen_n_fix_in_real_window"].mean()),
                  "window_test": paired(x["gen_n_fix_in_real_window"], x["real_n_fix"]), "real_total_dur": float(img(f"real_{g}", "total_dur").mean()), "gen_total_dur": float(img(f"gen_{g}", "total_dur").mean()),
                  "real_rate": float(img(f"real_{g}", "fix_rate_per_s").mean()), "gen_rate": float(img(f"gen_{g}", "fix_rate_per_s").mean())}
    res["fixation_count_check"] = cnt
    return res


def nll_uncertainty(v):
    rows = read_csv(f"{dirof(v)}/per_trial_nll.csv")
    out = {}
    def table(unit, sel):
        keys = sorted({r[unit] for r in rows if sel(r)})
        ix = {k: i for i, k in enumerate(keys)}
        A = np.zeros((len(keys), 3))
        for r in rows:
            if sel(r):
                A[ix[r[unit]]] += [float(r["n_events"]), float(r["sum_lp_duration"]), float(r["sum_lp_position"])]
        return A
    for g, sel in (("overall", lambda r: True), ("ASD", lambda r: r["group"] == "ASD"), ("TD", lambda r: r["group"] == "TD")):
        out[g] = {}
        for unit in ("participant", "image_id"):
            A = table(unit, sel)
            s = A[rng.integers(0, len(A), (N, len(A)))].sum(1)
            out[g][unit] = dict(nll_ci=ci(-(s[:, 1] + s[:, 2]) / s[:, 0]), duration_ci=ci(-s[:, 1] / s[:, 0]), position_ci=ci(-s[:, 2] / s[:, 0]))
    return out


def paired_nll(a, b, unit):
    """NLL of b minus NLL of a, bootstrapping the shared units (negative = b fits better)."""
    def tab(v):
        d = {}
        for r in read_csv(f"{dirof(v)}/per_trial_nll.csv"):
            x = d.setdefault(r[unit], np.zeros(3))
            x += [float(r["n_events"]), float(r["sum_lp_duration"]), float(r["sum_lp_position"])]
        return d
    ta, tb = tab(a), tab(b)
    k = sorted(set(ta) & set(tb))
    A, B = np.array([ta[i] for i in k]), np.array([tb[i] for i in k])
    idx = rng.integers(0, len(k), (N, len(k)))
    sa, sb = A[idx].sum(1), B[idx].sum(1)
    pa, pb = -sa[:, 2] / sa[:, 0], -sb[:, 2] / sb[:, 0]
    da, db = -sa[:, 1] / sa[:, 0], -sb[:, 1] / sb[:, 0]
    return dict(unit=unit, n_units=len(k), position_delta=float(-B[:, 2].sum() / B[:, 0].sum() + A[:, 2].sum() / A[:, 0].sum()), position_delta_ci=ci(pb - pa),
                duration_delta=float(-B[:, 1].sum() / B[:, 0].sum() + A[:, 1].sum() / A[:, 0].sum()), duration_delta_ci=ci(db - da),
                n_units_position_improved=int(np.sum(-B[:, 2] / B[:, 0] < -A[:, 2] / A[:, 0])))


def model_vs_centre_bias(v):
    rows = read_csv(f"{dirof(v)}/per_trial_nll.csv")
    ref = json.load(open(f"{dirof(v)}/centre_bias_reference_per_image.json"))
    d = {}
    for r in rows:
        x = d.setdefault(r["image_id"], np.zeros(2))
        x += [float(r["n_events"]), float(r["sum_lp_position"])]
    k = sorted(d)
    M_ = np.array([d[i] for i in k]); R_ = np.array([[ref[i]["n"], ref[i]["sum_log"]] for i in k])
    idx = rng.integers(0, len(k), (N, len(k)))
    sm, sr = M_[idx].sum(1), R_[idx].sum(1)
    return dict(model_position_nll=float(-M_[:, 1].sum() / M_[:, 0].sum()), centre_bias_position_nll=float(-R_[:, 1].sum() / R_[:, 0].sum()),
                model_minus_centre=float(-M_[:, 1].sum() / M_[:, 0].sum() + R_[:, 1].sum() / R_[:, 0].sum()), model_minus_centre_ci=ci(-sm[:, 1] / sm[:, 0] + sr[:, 1] / sr[:, 0]),
                images_model_better=int(np.sum(-M_[:, 1] / M_[:, 0] < -R_[:, 1] / R_[:, 0])), images=len(k))


def multimatch_stats(v):
    rows = read_csv(f"{dirof(v)}/multimatch_per_image.csv")
    T = {}
    for r in rows:
        T.setdefault(r["type"], {})[r["image_id"]] = r
    mean = {t: {m: float(np.mean([float(x[m]) for x in d.values()])) for m in MM} | {"n_images": len(d), "n_pairs": int(sum(int(x["n_pairs"]) for x in d.values()))} for t, d in T.items()}
    tests = []
    for lab, a, b in (("RvG_ASD minus RvR_ASD", "RvG_ASD", "RvR_ASD"), ("RvG_TD minus RvR_TD", "RvG_TD", "RvR_TD"),
                      ("real ASD: gen ASD minus gen TD", "RvG_ASD", "RvG_ASD_mismatch"), ("real TD: gen TD minus gen ASD", "RvG_TD", "RvG_TD_mismatch")):
        k = sorted(set(T[a]) & set(T[b]))
        for m in MM:
            tests.append(dict(test=lab, metric=m, version=v, **paired(np.array([float(T[a][i][m]) for i in k]), np.array([float(T[b][i][m]) for i in k]))))
    return mean, holm(tests)


def duration_table(v):
    z = np.load(f"{dirof(v)}/fixation_durations.npz")
    out = {}
    for p in z.files:
        a = z[p]
        out[p] = dict(n=int(a.size), mean=float(a.mean()), sd=float(a.std()), **{f"p{q}": float(np.percentile(a, q)) for q in (5, 25, 50, 75, 95)}, frac_over_1s=float((a > 1).mean()))
    return out


def screen_scale_term(v, W, H):
    """log|d(norm)/d(screen)| per fixation. Adding it to log p(norm) gives log density per screen px^2."""
    if v.startswith("old"):
        sx = sy = 1.0
    elif v.startswith("stretch"):
        sx, sy = 2560.0 / W, 1440.0 / H
    else:
        sx = sy = 1440.0 / H
    return float(np.log(2.0 / (sx * W)) + np.log(2.0 / (sy * H)))


def screen_position_nll(v, dims):
    """Position NLL on a common screen-pixel scale, so different mappings can be compared."""
    out = {}
    for unit in ("image_id", "participant"):
        d = {}
        for r in read_csv(f"{dirof(v)}/per_trial_nll.csv"):
            W, H = dims[r["image_id"]]
            x = d.setdefault(r[unit], np.zeros(2))
            x += [float(r["n_events"]), float(r["sum_lp_position"]) + float(r["n_events"]) * screen_scale_term(v, W, H)]
        out[unit] = d
    return out


def screen_summary(allv, dims):
    tabs = {v: screen_position_nll(v, dims) for v in allv}
    res = {v: float(-sum(x[1] for x in tabs[v]["image_id"].values()) / sum(x[0] for x in tabs[v]["image_id"].values())) for v in allv}
    pairs = {}
    for a, b in (("old_native_policy_a", "fit_height_policy_a"), ("fit_height_policy_a", "stretch_policy_a"), ("old_native_policy_b", "fit_height_policy_b")):
        pairs[f"{b} minus {a}"] = {}
        for unit in ("image_id", "participant"):
            k = sorted(set(tabs[a][unit]) & set(tabs[b][unit]))
            A, B = np.array([tabs[a][unit][i] for i in k]), np.array([tabs[b][unit][i] for i in k])
            idx = rng.integers(0, len(k), (N, len(k)))
            sa, sb = A[idx].sum(1), B[idx].sum(1)
            pairs[f"{b} minus {a}"][unit] = dict(delta=float(-B[:, 1].sum() / B[:, 0].sum() + A[:, 1].sum() / A[:, 0].sum()), ci=ci(-sb[:, 1] / sb[:, 0] + sa[:, 1] / sa[:, 0]),
                                                 units_favouring_b=int(np.sum(-B[:, 1] / B[:, 0] < -A[:, 1] / A[:, 0])), n_units=len(k))
    return dict(screen_position_nll=res, paired=pairs)


def participant_nll_test(v):
    rows = read_csv(f"{dirof(v)}/per_trial_nll.csv")
    res = {}
    for name, f in (("nll", lambda x: x[1] + x[2]), ("nll_duration", lambda x: x[1]), ("nll_position", lambda x: x[2])):
        d = {"ASD": {}, "TD": {}}
        for r in rows:
            x = d[r["group"]].setdefault(r["participant"], np.zeros(3))
            x += [float(r["n_events"]), float(r["sum_lp_duration"]), float(r["sum_lp_position"])]
        res[name] = independent(np.array([-f(x) / x[0] for x in d["ASD"].values()]), np.array([-f(x) / x[0] for x in d["TD"].values()]))
    return res


def saliency_reference():
    """Centre-bias reference on the Saliency4ASD test split, built as for Gaze4ASD."""
    split = json.load(open("training/runs/baseline/split.json")); test = set(map(int, split["test"]))
    rec = [r for f in ("processed_data/asd_scanpaths.json", "processed_data/td_scanpaths.json") for r in json.load(open(f)) if r["image_id"] in test]
    bins = np.linspace(-1, 1, 61); H = {}
    for r in rec:
        a = np.asarray(r["scanpath"])
        H.setdefault(r["image_id"], np.zeros((60, 60)))
        H[r["image_id"]] += np.histogram2d(a[:, 0], a[:, 1], bins=[bins, bins])[0]
    total = sum(H.values()); nn = ss = 0.0
    for h in H.values():
        d = gaussian_filter(total - h, 1.5, mode="nearest") + 1e-6
        d = d / (d.sum() * (2 / 60) ** 2)
        ss += float((h * np.log(d)).sum()); nn += h.sum()
    return -ss / nn


def csv_write(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r.get(k), (list, dict)) else r.get(k, "")) for k in keys})


# --------------------------------------------------------------------- figures
def save(fig, name):
    fig.savefig(f"{OUT}/figures/{name}.png", bbox_inches="tight")
    plt.close(fig)


def fig_nll(nll, unc, sal):
    order = VERS + REFS
    y = np.arange(len(order))[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, key, cik, title in ((axs[0], "nll_position", "position_ci", "Position NLL (lower is better)"), (axs[1], "nll_duration", "duration_ci", "Duration NLL (lower is better)")):
        for g, col, off in (("ASD", ASD, .17), ("TD", TD, -.17)):
            for i, v in enumerate(order):
                c = unc[v][g]["image_id"][cik]; val = nll[v][g][key]
                ax.plot(c, [y[i] + off] * 2, color=col, lw=1.5, solid_capstyle="round")
                ax.plot(val, y[i] + off, "o", color=col, ms=6, mec=SURF, mew=1)
        s = sal["overall"][key]
        ax.axvline(s, color=INK2, lw=1); ax.text(s, len(order) - .4, " Saliency4ASD test", ha="left", fontsize=7.5, color=INK2)
        ax.set_title(title, loc="left"); ax.grid(axis="y", visible=False)
    axs[0].axvline(np.log(4), color=MUTED, lw=1); axs[0].text(np.log(4), len(order) - .4, " uniform", ha="left", fontsize=7.5, color=INK2)
    axs[0].set_yticks(y); axs[0].set_yticklabels([LABEL[v] for v in order], color=INK2); axs[0].set_ylim(-1, len(order) - .2)
    axs[1].legend(handles=[Line2D([], [], marker="o", color=ASD, label="ASD"), Line2D([], [], marker="o", color=TD, label="TD")], frameon=False, loc="upper right", bbox_to_anchor=(1, -.09), ncol=2)
    fig.suptitle("Frozen Phase-2 model on Gaze4ASD. Intervals: 95% bootstrap over images.", x=.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(); save(fig, "fig1_nll_by_version")


def fig_behaviour(v):
    rows = read_csv(f"{dirof(v)}/per_scanpath_metrics.csv")
    ms = ["dur_mean", "n_fix", "total_dur", "fix_rate_per_s", "sacc_amp_norm_mean", "dispersion", "revisit_rate", "sacc_horizontal_frac"]
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.4))
    for ax, m in zip(axs.ravel(), ms):
        sets = [("real_ASD", "participant", ASD, True), ("real_TD", "participant", TD, True), ("gen_ASD", "image_id", ASD, False), ("gen_TD", "image_id", TD, False)]
        for i, (p, u, col, fill) in enumerate(sets):
            a = np.array(list(means_by(rows, p, u, m).values()))
            x = i + (rng.random(len(a)) - .5) * .3
            ax.scatter(x, a, s=9, color=col if fill else "none", edgecolors=col, linewidths=.6, alpha=.55)
            q = np.percentile(a, [25, 50, 75])
            ax.plot([i + .24, i + .24], [q[0], q[2]], color=INK2, lw=1.5, solid_capstyle="round"); ax.plot([i + .18, i + .30], [q[1], q[1]], color=INK, lw=1.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(["Real\nASD", "Real\nTD", "Gen\nASD", "Gen\nTD"], fontsize=7.5, color=INK2); ax.set_title(METRICS[m], loc="left", fontsize=8.5); ax.grid(axis="x", visible=False)
    fig.suptitle(f"{LABEL[v]}: filled = real, one point per participant; hollow = generated, one point per image. Bars show median and IQR.", x=.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(); save(fig, f"fig2_behaviour_{v}")


def fig_forest(allc):
    ms = ["dur_mean", "n_fix", "total_dur", "fix_rate_per_s", "sacc_amp_norm_mean", "dispersion", "revisit_rate", "sacc_dir_resultant", "sacc_horizontal_frac"]
    y = np.arange(len(ms))[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    marks = {"fit_height_policy_a": ("o", .24), "stretch_policy_a": ("s", 0), "fit_height_policy_b": ("^", -.24), "old_native_policy_a": ("X", -.44)}
    for v, (mk, off) in marks.items():
        col = MUTED if v.startswith("old") else INK
        for i, m in enumerate(ms):
            r = allc[v]["real_ASD_vs_TD_participant"][m]
            axs[0].plot(r["effect_ci"], [y[i] + off] * 2, color=col, lw=1.1, solid_capstyle="round"); axs[0].plot(r["effect"], y[i] + off, mk, color=col, ms=5, mec=SURF, mew=.8)
    for i, m in enumerate(ms):
        r = allc["fit_height_policy_a"]["real_ASD_vs_TD_image_paired"][m]; g = allc["fit_height_policy_a"]["gen_ASD_vs_TD_image_paired"][m]
        axs[1].plot(r["effect_ci"], [y[i] + .14] * 2, color=INK, lw=1.2); axs[1].plot(r["effect"], y[i] + .14, "o", color=INK, ms=6, mec=SURF, mew=.8)
        axs[1].plot(g["effect_ci"], [y[i] - .14] * 2, color=INK2, lw=1.2); axs[1].plot(g["effect"], y[i] - .14, "o", mfc=SURF, mec=INK2, ms=6, mew=1.2)
    for ax in axs:
        ax.axvline(0, color=AXIS, lw=1); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y); axs[0].set_yticklabels([METRICS[m] for m in ms], color=INK2)
    axs[0].set_title("Real ASD minus TD, participants (Hedges' g)", loc="left"); axs[1].set_title("Primary version, per image (Cohen's dz)", loc="left")
    axs[0].legend(handles=[Line2D([], [], marker=mk, color=MUTED if v.startswith("old") else INK, ls="", label=LABEL[v]) for v, (mk, _) in marks.items()], frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(-.02, -.34), ncol=2)
    axs[1].legend(handles=[Line2D([], [], marker="o", color=INK, ls="", label="Real"), Line2D([], [], marker="o", mfc=SURF, mec=INK2, ls="", label="Generated")], frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, -.24), ncol=2)
    fig.tight_layout(); save(fig, "fig3_effect_forest")


def fig_scanpaths(v):
    recs = json.load(open("external_data/Gaze4ASD_processed_v2/fit_height_policy_a/gaze4asd_fit_height_policy_a.json" if v == "fit_height_policy_a" else f"external_data/Gaze4ASD_processed_v2/{v}/gaze4asd_{v}.json"))
    gen = json.load(open(f"{OUT}/generated/generated_scanpaths.json"))
    picks = ["top_image_8.png", "top_image_13.png", "top_image_22.png", "top_image_23.png"]
    fig, axs = plt.subplots(len(picks), 4, figsize=(11, 8.6))
    r2 = np.random.default_rng(5)
    for row, n in enumerate(picks):
        im = Image.open(f"{IMG}/{n}").convert("RGB"); W, H = im.size
        srcs = [("Real ASD", ASD, [np.array(r["scanpath"]) for r in recs if r["image_id"] == n and r["group"] == "ASD"]), ("Real TD", TD, [np.array(r["scanpath"]) for r in recs if r["image_id"] == n and r["group"] == "TD"]),
                ("Generated ASD", ASD, [np.array(r["scanpath"]) for r in gen if r["image_id"] == n and r["condition"] == "ASD"]), ("Generated TD", TD, [np.array(r["scanpath"]) for r in gen if r["image_id"] == n and r["condition"] == "TD"])]
        for c, (lab, col, L) in enumerate(srcs):
            ax = axs[row, c]; ax.imshow(im, alpha=.82); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(H, 0)
            allp = np.vstack(L); h = np.histogram2d((allp[:, 1] + 1) / 2 * H, (allp[:, 0] + 1) / 2 * W, bins=[H // 8, W // 8], range=[[0, H], [0, W]])[0]
            ax.contour(np.linspace(0, W, W // 8), np.linspace(0, H, H // 8), gaussian_filter(h, 3) / (gaussian_filter(h, 3).max() + 1e-9), levels=[.25, .6], colors=[col], linewidths=[.9, 1.6])
            s = L[int(r2.integers(len(L)))]; x, y = (s[:, 0] + 1) / 2 * W, (s[:, 1] + 1) / 2 * H
            ax.plot(x, y, "-", color="white", lw=2.2, alpha=.9); ax.plot(x, y, "-", color=col, lw=1.1); ax.scatter(x, y, s=10 + 80 * s[:, 2], color=col, edgecolors="white", linewidths=.7, zorder=4)
            if row == 0:
                ax.set_title(lab, fontsize=9, color=INK)
    fig.suptitle("Contours: density of all fixations for that population and image. Line: one example scanpath; circle area grows with duration.", x=.01, ha="left", fontsize=9, color=INK2, y=1.0)
    fig.tight_layout(); save(fig, f"fig4_real_vs_generated_{v}")


def fig_durations():
    z = np.load(f"{dirof('fit_height_policy_a')}/fixation_durations.npz")
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3), sharey=True)
    bins = np.linspace(0.06, 1.5, 49)
    for ax, g, col in ((axs[0], "ASD", ASD), (axs[1], "TD", TD)):
        ax.hist(z[f"real_{g}"], bins=bins, density=True, histtype="step", color=col, lw=2, label="Real")
        ax.hist(z[f"gen_{g}"], bins=bins, density=True, histtype="step", color=INK2, lw=1.4, label="Generated")
        ax.set_title(f"{g} fixation duration", loc="left"); ax.set_xlabel("Seconds"); ax.legend(frameon=False)
    axs[0].set_ylabel("Density"); fig.tight_layout(); save(fig, "fig5_duration_distributions")


def fig_counts(cnt):
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.9))
    for ax, keys, title in ((axs[0], [("real_n_fix", "Real"), ("gen_n_fix", "Generated, full 3 s"), ("gen_n_fix_in_real_window", "Generated, real window")], "Fixations per scanpath"),
                            (axs[1], [("real_total_dur", "Real"), ("gen_total_dur", "Generated")], "Total fixation time (s)"), (axs[2], [("real_rate", "Real"), ("gen_rate", "Generated")], "Fixations per second of fixation")):
        y = np.arange(len(keys))[::-1]
        for g, col, off in (("ASD", ASD, .12), ("TD", TD, -.12)):
            ax.plot([cnt[g][k] for k, _ in keys], y + off, "o", color=col, ms=7, mec=SURF, mew=1, ls="")
        ax.set_yticks(y); ax.set_yticklabels([l for _, l in keys], color=INK2); ax.set_title(title, loc="left"); ax.grid(axis="y", visible=False); ax.set_ylim(-.6, len(keys) - .4)
    axs[2].legend(handles=[Line2D([], [], marker="o", color=ASD, ls="", label="ASD"), Line2D([], [], marker="o", color=TD, ls="", label="TD")], frameon=False)
    fig.tight_layout(); save(fig, "fig6_fixation_count_and_rate")


def fig_old_new(allm):
    ms = ["sacc_amp_norm_mean", "dispersion", "revisit_rate"]
    order = ["old_native_policy_a", "fit_height_policy_a", "stretch_policy_a", "fit_height_policy_b"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
    y = np.arange(len(order) + 1)[::-1]
    for ax, m in zip(axs, ms):
        for g, col, off in (("ASD", ASD, .12), ("TD", TD, -.12)):
            ax.plot([allm[v][f"real_{g}"][m] for v in order], y[:-1] + off, "o", color=col, ms=7, mec=SURF, mew=1, ls="")
            ax.plot(allm["fit_height_policy_a"][f"gen_{g}"][m], y[-1] + off, "o", mfc=SURF, mec=col, ms=7, mew=1.3, ls="")
        ax.set_title(METRICS[m], loc="left"); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y); axs[0].set_yticklabels([LABEL[v] for v in order] + ["Generated"], color=INK2); axs[0].set_ylim(-.6, len(order) + .4)
    axs[2].legend(handles=[Line2D([], [], marker="o", color=ASD, ls="", label="ASD"), Line2D([], [], marker="o", color=TD, ls="", label="TD")], frameon=False, loc="lower right")
    fig.tight_layout(); save(fig, "fig7_spatial_old_vs_corrected")


def fig_mm(mm):
    v = "fit_height_policy_a"
    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    y = np.arange(len(MM))[::-1]
    for ax, g, col in ((axs[0], "ASD", ASD), (axs[1], "TD", TD)):
        for i, m in enumerate(MM):
            ax.plot(mm[v][f"RvR_{g}"][m], y[i] + .12, "o", color=col, ms=7, mec=SURF, mew=1)
            ax.plot(mm[v][f"RvG_{g}"][m], y[i] - .12, "o", mfc=SURF, mec=col, ms=7, mew=1.3)
        ax.set_title(f"{g}: similarity (1 = identical)", loc="left"); ax.grid(axis="y", visible=False)
        ax.legend(handles=[Line2D([], [], marker="o", color=col, ls="", label="Real vs real"), Line2D([], [], marker="o", mfc=SURF, mec=col, ls="", label="Real vs generated")], frameon=False, fontsize=7.5, loc="lower left")
    axs[0].set_yticks(y); axs[0].set_yticklabels([m.replace("mm_", "").capitalize() for m in MM], color=INK2)
    fig.tight_layout(); save(fig, "fig8_multimatch_primary")


# ------------------------------------------------------------------------ main
def main():
    allv = VERS + REFS
    nll = json.load(open(f"{OUT}/comparison/nll_all_versions.json"))
    sal = json.load(open(f"{OUT}/comparison/saliency4asd_test_reproduced.json"))
    unc = {v: nll_uncertainty(v) for v in allv}
    allc = {v: contrasts(v) for v in allv}
    mm, mmt = {}, {}
    for v in allv:
        mm[v], mmt[v] = multimatch_stats(v)
    durs = {v: duration_table(v) for v in allv}
    allm = {}
    for v in allv:
        rows = read_csv(f"{dirof(v)}/per_scanpath_metrics.csv")
        allm[v] = {p: {m: float(np.mean([num(r[m]) for r in rows if r["population"] == p and np.isfinite(num(r[m]))])) for m in METRICS} for p in ("real_ASD", "real_TD", "gen_ASD", "gen_TD")}
    delta = {f"{b} minus {a}": {u: paired_nll(a, b, u) for u in ("image_id", "participant")} for a, b in (("old_native_policy_a", "fit_height_policy_a"), ("old_native_policy_b", "fit_height_policy_b"), ("fit_height_policy_a", "stretch_policy_a"), ("fit_height_policy_a", "fit_height_policy_b"))}
    centre = {v: model_vs_centre_bias(v) for v in allv}
    dims = {}
    for f in os.listdir(IMG):
        with Image.open(f"{IMG}/{f}") as im:
            dims[f] = im.size
    screen = screen_summary(allv, dims)
    part_nll = {v: participant_nll_test(v) for v in allv}
    sal_centre = saliency_reference()
    transfer = dict(saliency4asd_test=dict(position_nll=sal["overall"]["nll_position"], duration_nll=sal["overall"]["nll_duration"], centre_bias_position_nll=sal_centre, model_minus_centre=sal["overall"]["nll_position"] - sal_centre, gain_over_uniform=float(np.log(4)) - sal["overall"]["nll_position"]),
                    **{v: dict(position_nll=nll[v]["overall"]["nll_position"], duration_nll=nll[v]["overall"]["nll_duration"], centre_bias_position_nll=centre[v]["centre_bias_position_nll"], model_minus_centre=centre[v]["model_minus_centre"], model_minus_centre_ci=centre[v]["model_minus_centre_ci"], images_model_better=centre[v]["images_model_better"], gain_over_uniform=float(np.log(4)) - nll[v]["overall"]["nll_position"]) for v in allv})
    json.dump(dict(screen_scale=screen, transfer_reference=transfer, participant_nll_ASD_vs_TD=part_nll), open(f"{OUT}/comparison/nll_scale_and_transfer.json", "w"), indent=2)

    json.dump({"nll_ci": unc, "nll_paired_deltas": delta, "model_vs_centre_bias": centre}, open(f"{OUT}/comparison/nll_uncertainty.json", "w"), indent=2)
    for v in allv:
        json.dump({"contrasts": allc[v], "multimatch_means": mm[v], "multimatch_tests": mmt[v], "duration_distribution": durs[v], "group_means": allm[v]}, open(f"{OUT}/statistics/{v}_statistics.json", "w"), indent=2)
    flat = [x for v in allv for n, d in allc[v].items() if n.endswith("participant") or n.endswith("paired") for x in d.values()]
    csv_write(f"{OUT}/statistics/all_contrasts.csv", flat)
    csv_write(f"{OUT}/statistics/effect_reproduction.csv", [x for v in allv for x in allc[v]["effect_reproduction"].values()])
    csv_write(f"{OUT}/statistics/multimatch_tests.csv", [x for v in allv for x in mmt[v]])
    csv_write(f"{OUT}/statistics/participant_level_real.csv", [allc[v]["real_ASD_vs_TD_participant"][m] for v in allv for m in METRICS])
    csv_write(f"{OUT}/statistics/generated_image_level.csv", [allc[v]["gen_ASD_vs_TD_image_paired"][m] for v in VERS for m in METRICS])
    csv_write(f"{OUT}/comparison/nll_summary.csv", [dict(version=v, group=g, **{k: nll[v][g][k] for k in ("n_trials", "n_events", "nll", "nll_duration", "nll_position", "edge_pct", "nll_position_edge", "nll_position_nonedge")}, nll_ci_images=unc[v][g]["image_id"]["nll_ci"], position_ci_images=unc[v][g]["image_id"]["position_ci"], duration_ci_images=unc[v][g]["image_id"]["duration_ci"], nll_ci_participants=unc[v][g]["participant"]["nll_ci"]) for v in allv for g in ("overall", "ASD", "TD")])
    csv_write(f"{OUT}/comparison/behaviour_means.csv", [dict(version=v, population=p, **allm[v][p]) for v in allv for p in allm[v]])
    csv_write(f"{OUT}/comparison/multimatch_means.csv", [dict(version=v, type=t, **mm[v][t]) for v in allv for t in mm[v]])
    csv_write(f"{OUT}/comparison/direction_stability.csv", [dict(metric=m, label=METRICS[m], **{f"real_g_{v}": allc[v]["real_ASD_vs_TD_participant"][m]["effect"] for v in allv}, **{f"real_p_holm_{v}": allc[v]["real_ASD_vs_TD_participant"][m]["p_holm"] for v in allv}, gen_dz=allc["fit_height_policy_a"]["gen_ASD_vs_TD_image_paired"][m]["effect"], real_direction_stable_v2=len({np.sign(allc[v]["real_ASD_vs_TD_participant"][m]["diff"]) for v in VERS}) == 1, gen_matches_real_all_v2=all(allc[v]["effect_reproduction"][m]["same_direction"] for v in VERS)) for m in METRICS])
    csv_write(f"{OUT}/comparison/duration_distributions.csv", [dict(version=v, population=p, **durs[v][p]) for v in allv for p in durs[v]])
    csv_write(f"{OUT}/comparison/fixation_count_check.csv", [dict(version=v, group=g, **{k: x for k, x in allc[v]["fixation_count_check"][g].items() if k != "window_test"}, window_diff=allc[v]["fixation_count_check"][g]["window_test"]["diff"], window_diff_ci=allc[v]["fixation_count_check"][g]["window_test"]["diff_ci"], window_p=allc[v]["fixation_count_check"][g]["window_test"]["p"]) for v in VERS for g in ("ASD", "TD")])

    fig_nll(nll, unc, sal)
    for v in VERS:
        fig_behaviour(v)
        fig_scanpaths(v)
    fig_forest(allc); fig_durations(); fig_counts(allc["fit_height_policy_a"]["fixation_count_check"]); fig_old_new(allm); fig_mm(mm)
    print("STATS_DONE")


if __name__ == "__main__":
    main()
