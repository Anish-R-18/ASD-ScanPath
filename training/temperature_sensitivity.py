"""
Sampling-temperature sensitivity of the Phase-2 model on Saliency4ASD VALIDATION images only.
No training, no test split, no Gaze4ASD. The checkpoint is read-only.

  python training/temperature_sensitivity.py run       checks, generation, metrics, MultiMatch
  python training/temperature_sensitivity.py windowed  post hoc: spatial metrics within the real viewing window
  python training/temperature_sensitivity.py analyze   statistics, tables, figures

Both temperatures use the same seeds, images, conditions and batches. Primary analysis clips coordinates
to the image, as the official generate_predictions does; unclipped results are the sensitivity analysis.
All statistics are exploratory.
"""
from __future__ import annotations
import csv, gzip, json, os, sys, time, zlib
sys.path[:0] = [".", "tppgaze"]
import numpy as np

OUT = "training/runs/temperature_sensitivity"
RAW = OUT + "/raw"
FIG = OUT + "/figures"
PHASE2 = "training/runs/phase2_embedding_only/best_model_conditioned.pth"
PHASE1 = "training/runs/baseline/best_model.pth"
SPLIT = "training/runs/baseline/split.json"
TAUS = (1.0, 2.0)
GROUPS = ("ASD", "TD")
MODES = ("clip", "raw")
BATCH, N_REP, N_SIM, T_END, MAX_PAIRS, N_BOOT = 128, 10, 20, 3.0, 30, 10000
KNOWN_VAL_NLL = -0.5964
METRICS = {"dur_mean": "Fixation duration, mean (s)", "dur_median": "Fixation duration, median (s)", "n_fix": "Fixation count", "n_fix_window": "Fixation count, real time window", "fix_rate_per_s": "Fixation rate (per s)", "total_dur": "Total fixation time (s)",
           "sacc_amp_norm_mean": "Saccade amplitude (/diagonal)", "sacc_horizontal_frac": "Horizontal saccade fraction", "dispersion": "Spatial dispersion (/diagonal)", "revisit_rate": "Revisit rate", "sacc_dir_resultant": "Direction concentration"}
DISTS = {"D_dur": "Fixation durations", "D_nfix": "Fixation counts", "D_amp": "Saccade amplitudes", "D_disp": "Spatial dispersion", "D_rev": "Revisit rates", "D_pos": "Fixation positions", "D_dir": "Saccade directions (total variation)"}
MM = ["mm_shape", "mm_length", "mm_direction", "mm_position", "mm_duration"]
DIR_BINS = np.linspace(-np.pi, np.pi, 13)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(f"{OUT}/run_log.txt", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def write_rows(path, rows):
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


def tau_name(t):
    return f"tau{int(t)}"


# =========================================================================== run
def run():
    import torch
    from omegaconf import OmegaConf
    from scipy.stats import wasserstein_distance as W1
    from training.dataset import load_scanpaths
    from training import metrics as M, model_utils as mu
    from training.conditioned import ConditionedTPPGaze
    from training.generate_cond import sample_conditioned, to_scanpaths
    from training.external_gaze4asd_loader import StimulusBank, collate_records, GROUP_TO_ID
    from training.eval_external_v2 import BASE_CFG, denorm, metric_row, sha256

    if os.path.exists(OUT):
        raise SystemExit(f"Refusing to overwrite: {OUT}")
    os.makedirs(RAW); os.makedirs(FIG)
    for t in TAUS:
        os.makedirs(f"{OUT}/generation_{tau_name(t)}")
    device = torch.device("cuda")
    t0 = time.time()
    h2 = sha256(PHASE2)
    log(f"Phase 2 {PHASE2} sha256={h2}")

    # ---------------- validation split only
    split = json.load(open(SPLIT))
    tr, va, te = [set(map(int, split[k])) for k in ("train", "val", "test")]
    assert len(va) == 45 and not (tr & va) and not (va & te) and not (tr & te)
    recs = [s for s in load_scanpaths("processed_data/asd_scanpaths.json", "processed_data/td_scanpaths.json") if s["image_id"] in va]
    assert {r["image_id"] for r in recs} == va and not ({r["image_id"] for r in recs} & (tr | te))
    images = sorted(va)
    log(f"validation split verified: {len(va)} images; ASD {sum(r['group'] == 'ASD' for r in recs)} TD {sum(r['group'] == 'TD' for r in recs)} scanpaths; no train or test records kept")
    stored = json.load(open("training/runs/phase2_embedding_only/summary_full.json"))
    mean, std = stored["mean_log_inter_time"], stored["std_log_inter_time"]
    log(f"normalization from summary_full.json, not recomputed: mean={mean} std={std}")

    cfg = OmegaConf.load(BASE_CFG)
    base, _ = mu.build_model(BASE_CFG, PHASE1, device, mean, std, used_cfg_out="/tmp/temp_sens.yaml")
    model = ConditionedTPPGaze(base, int(cfg.metadata.size + cfg.context.size)).to(device)
    model.load_state_dict(torch.load(PHASE2, map_location=device), strict=True)
    mu._fix_unregistered_device(model.base, device)
    model.eval()
    for q in model.parameters():
        q.requires_grad = False
    bas = model.base
    log("checkpoint loaded (strict); evaluation mode; all parameters frozen")

    cfg_images = OmegaConf.load("training/config_phase2_embonly.yaml").images_dir
    bank = StimulusBank(cfg_images)
    dims = {i: bank.dims(f"{i}.png") for i in images}
    for r in recs:
        assert (r.get("image_width", dims[r["image_id"]][0]), r.get("image_height", dims[r["image_id"]][1])) == tuple(dims[r["image_id"]])

    # ---------------- what temperature does, measured on the model's own contexts
    meta_cache = {}
    @torch.no_grad()
    def meta_of(ids):
        for i in set(ids):
            if i not in meta_cache:
                meta_cache[i] = bas.get_metadata(bank.tensor(f"{i}.png").unsqueeze(0).to(device))
        return torch.cat([meta_cache[i] for i in ids], 0)

    def spread(pi, loc, sig, tau):
        """Per-axis mixture SD at one context: within-component, and total including between-component."""
        s = sig ** tau
        within = (pi[..., None] * s ** 2).sum(-2)
        mu_ = (pi[..., None] * loc).sum(-2)
        between = (pi[..., None] * (loc - mu_[..., None, :]) ** 2).sum(-2)
        return within.sqrt().mean(-1), (within + between).sqrt().mean(-1)

    checks, sigma_hist = {}, np.zeros(60)
    hist_edges = np.linspace(-5 / np.log(10), 3 / np.log(10), 61)
    tot = {g: dict(n=0.0, dur=0.0, pos_none=0.0, pos_1=0.0, pos_2=0.0) for g in GROUPS}
    acc = {"sig_lt1": 0.0, "mass": 0.0, "clamp_low": 0.0, "clamp_high": 0.0}
    spread_rows = {"history": {t: [[], []] for t in TAUS}, "first_step": {t: [[], []] for t in TAUS}}
    med = []
    with torch.no_grad():
        for st in range(0, len(recs), BATCH):
            b = recs[st:st + BATCH]
            c = collate_records(b, device)
            f = bas.get_features(c["inter_times"], c["marks"])
            ctx = bas.merge.forward([bas.get_context(f, True), meta_of([r["image_id"] for r in b])]) + model.group_emb(c["group_ids"]).unsqueeze(1)
            dn, d1, d2 = (bas.marks_dist.forward(ctx, t) for t in (None, 1.0, 2.0))
            cn, c1, c2 = (d.component_distribution.base_dist for d in (dn, d1, d2))
            if st == 0:
                checks = dict(tau1_scale_identical_to_no_temperature=bool(torch.equal(cn.scale, c1.scale)),
                              tau2_scale_equals_sigma_squared_max_rel_error=float(((c2.scale - cn.scale ** 2).abs() / cn.scale ** 2).max()),
                              tau2_locs_unchanged=bool(torch.equal(cn.loc, c2.loc)), tau2_weights_unchanged=bool(torch.equal(dn.mixture_distribution.logits, d2.mixture_distribution.logits)),
                              duration_mean_ignores_temperature_argument=bool(torch.equal(bas.inter_time_dist.forward(ctx, None).mean, bas.inter_time_dist.forward(ctx, 2.0).mean)))
            m = c["mask"].bool()
            pi = dn.mixture_distribution.probs
            sig, loc = cn.scale, cn.loc
            w = (pi[..., None] * m[..., None, None]).expand_as(sig)
            acc["sig_lt1"] += float((w * (sig < 1)).sum()); acc["mass"] += float(w.sum())
            acc["clamp_low"] += float((w * (sig <= np.exp(-5) * 1.0001)).sum()); acc["clamp_high"] += float((w * (sig >= np.exp(3) * .9999)).sum())
            sigma_hist += np.histogram(torch.log10(sig).cpu().numpy().ravel(), bins=hist_edges, weights=w.cpu().numpy().ravel())[0]
            med.append(torch.sqrt((pi[..., None] * sig ** 2).sum(-2)).mean(-1)[m].cpu().numpy())
            for t in TAUS:
                a, e = spread(pi, loc, sig, t)
                spread_rows["history"][t][0].append(a[m].cpu().numpy()); spread_rows["history"][t][1].append(e[m].cpu().numpy())
            it = torch.where(m, c["inter_times"], torch.ones_like(c["inter_times"]))
            lpt = bas.inter_time_dist.forward(ctx, None).log_prob(it) * c["mask"]
            lp = {k: d.log_prob(c["marks"]) * c["mask"] for k, d in (("pos_none", dn), ("pos_1", d1), ("pos_2", d2))}
            for gid, g in enumerate(GROUPS):
                s = c["group_ids"] == gid
                tot[g]["n"] += float(c["mask"][s].sum()); tot[g]["dur"] += float(lpt[s].sum())
                for k in lp:
                    tot[g][k] += float(lp[k][s].sum())
        for i in images:
            for gid in (0, 1):
                ctx = bas.merge.forward([bas.context.get_context_init()[None, None, :], meta_of([i])]) + model.group_emb(torch.tensor([gid], device=device)).unsqueeze(1)
                d = bas.marks_dist.forward(ctx, None)
                for t in TAUS:
                    a, e = spread(d.mixture_distribution.probs, d.component_distribution.base_dist.loc, d.component_distribution.base_dist.scale, t)
                    spread_rows["first_step"][t][0].append(a.cpu().numpy()); spread_rows["first_step"][t][1].append(e.cpu().numpy())
    n_all = sum(tot[g]["n"] for g in GROUPS)
    likelihood = {g: dict(events=tot[g]["n"], duration_nll=-tot[g]["dur"] / tot[g]["n"], position_nll_model=-tot[g]["pos_none"] / tot[g]["n"], position_nll_tau1=-tot[g]["pos_1"] / tot[g]["n"], position_nll_tempered_tau2=-tot[g]["pos_2"] / tot[g]["n"]) for g in GROUPS}
    likelihood["overall"] = dict(events=n_all, duration_nll=-sum(tot[g]["dur"] for g in GROUPS) / n_all, position_nll_model=-sum(tot[g]["pos_none"] for g in GROUPS) / n_all, position_nll_tau1=-sum(tot[g]["pos_1"] for g in GROUPS) / n_all, position_nll_tempered_tau2=-sum(tot[g]["pos_2"] for g in GROUPS) / n_all)
    likelihood["overall"]["joint_nll_model"] = likelihood["overall"]["duration_nll"] + likelihood["overall"]["position_nll_model"]
    summ = lambda x: float(np.mean(np.concatenate([np.ravel(a) for a in x])))
    spreads = {k: {tau_name(t): dict(within_component_sd=summ(v[t][0]), total_sd=summ(v[t][1])) for t in TAUS} for k, v in spread_rows.items()}
    sigma = dict(mass_below_1=acc["sig_lt1"] / acc["mass"], mass_at_lower_clamp=acc["clamp_low"] / acc["mass"], mass_at_upper_clamp=acc["clamp_high"] / acc["mass"], median_rms_sigma=float(np.median(np.concatenate(med))), hist_log10=sigma_hist.tolist(), hist_edges=hist_edges.tolist())
    log("formula checks: " + json.dumps(checks))
    log("sigma: " + json.dumps({k: v for k, v in sigma.items() if not k.startswith("hist")}))
    log("spreads: " + json.dumps(spreads))
    log("likelihood on validation: " + json.dumps(likelihood))
    if not (checks["tau1_scale_identical_to_no_temperature"] and checks["tau2_locs_unchanged"] and checks["tau2_weights_unchanged"] and checks["duration_mean_ignores_temperature_argument"] and checks["tau2_scale_equals_sigma_squared_max_rel_error"] < 1e-3):
        raise SystemExit("GATE FAILED: temperature formula")
    if abs(likelihood["overall"]["joint_nll_model"] - KNOWN_VAL_NLL) > 0.005:
        raise SystemExit("GATE FAILED: validation NLL")

    # ---------------- real summaries
    def path_stats(xy, dur, Wd, Hd):
        return metric_row("x", 0, "", xy, dur, Wd, Hd)
    def angles(xy):
        v = np.diff(xy, axis=0)
        return np.arctan2(v[:, 1], v[:, 0])
    def amps(xy, diag):
        return np.linalg.norm(np.diff(xy, axis=0), axis=1) / diag
    real = {g: {i: [] for i in images} for g in GROUPS}
    for r in recs:
        Wd, Hd = dims[r["image_id"]]
        sp = np.asarray(r["scanpath"], float)
        real[r["group"]][r["image_id"]].append(dict(xy=denorm(sp, Wd, Hd), dur=sp[:, 2]))
    mkeys = [m for m in METRICS if m != "n_fix_window"]
    arrs = {}

    def samples(paths, iid):
        """Distribution samples for one image from a list of scanpaths."""
        Wd, Hd = dims[iid]
        diag = float(np.hypot(Wd, Hd))
        rows = [path_stats(p["xy"], p["dur"], Wd, Hd) for p in paths]
        two = [p for p in paths if len(p["xy"]) >= 2]
        return dict(rows=rows, dur=np.concatenate([p["dur"] for p in paths]), nfix=np.array([len(p["xy"]) for p in paths], float), amp=np.concatenate([amps(p["xy"], diag) for p in two]) if two else np.zeros(0),
                    disp=np.array([x["dispersion"] for x in rows]), rev=np.array([x["revisit_rate"] for x in rows]), x=np.concatenate([p["xy"][:, 0] for p in paths]) / diag, y=np.concatenate([p["xy"][:, 1] for p in paths]) / diag,
                    hist=np.histogram(np.concatenate([angles(p["xy"]) for p in two]) if two else np.zeros(0), bins=DIR_BINS)[0].astype(float))
    def dist_set(a, b):
        ha, hb = a["hist"], b["hist"]
        return dict(D_dur=W1(a["dur"], b["dur"]), D_nfix=W1(a["nfix"], b["nfix"]), D_amp=W1(a["amp"], b["amp"]), D_disp=W1(a["disp"], b["disp"]), D_rev=W1(a["rev"], b["rev"]), D_pos=0.5 * (W1(a["x"], b["x"]) + W1(a["y"], b["y"])), D_dir=0.5 * np.abs(ha / ha.sum() - hb / hb.sum()).sum())
    def merge(sets):
        return {k: (np.concatenate([s[k] for s in sets]) if k not in ("rows", "hist") else ([r for s in sets for r in s["rows"]] if k == "rows" else sum(s["hist"] for s in sets))) for k in sets[0]}

    real_s = {g: {i: samples(real[g][i], i) for i in images} for g in GROUPS}
    window = {g: {i: float(np.mean([p["dur"].sum() for p in real[g][i]])) for i in images} for g in GROUPS}
    for g in GROUPS:
        for m in METRICS:
            src = "n_fix" if m == "n_fix_window" else m
            arrs[f"real_{g}_{m}"] = np.array([np.nanmean([r[src] for r in real_s[g][i]["rows"]]) for i in images])
        arrs[f"real_{g}_dur_pooled_median"] = np.array([np.median(real_s[g][i]["dur"]) for i in images])
        arrs[f"real_{g}_n_paths"] = np.array([len(real[g][i]) for i in images])

    pre = lambda mode, t, g: f"{mode}_{tau_name(t)}_{g}"
    for mode in MODES:
        for t in TAUS:
            for g in GROUPS:
                for m in list(METRICS) + ["dur_pooled_median"] + list(DISTS):
                    arrs[f"{pre(mode, t, g)}_{m}"] = np.full((N_REP, len(images)), np.nan)
                for d in DISTS:
                    arrs[f"{pre(mode, t, g)}_pooled_{d}"] = np.full(len(images), np.nan)
                    arrs[f"{pre(mode, t, g)}_global_{d}"] = np.full(N_REP, np.nan)
    off = {tau_name(t): {g: np.zeros((N_REP, 2)) for g in GROUPS} for t in TAUS}
    gen_pool = {mode: {t: {g: {i: [] for i in images} for g in GROUPS} for t in TAUS} for mode in MODES}
    mm_rows = []
    first_dur_max_diff, n_empty, max_len, seed_map = 0.0, 0, 0, []

    def mm_mean(tag, la, lb, same, diag):
        pairs = [(i, j) for i in range(len(la)) for j in range(len(lb)) if not (same and j <= i)]
        rng = np.random.default_rng([0, zlib.crc32(tag.encode())])
        v = {k: [] for k in MM}
        for q in rng.permutation(len(pairs))[:MAX_PAIRS]:
            a, b = la[pairs[q][0]], lb[pairs[q][1]]
            r = M.multimatch(a["xy"], a["dur"], b["xy"], b["dur"], diag)
            if r:
                for k in MM:
                    v[k].append(r[k])
        return {k: float(np.mean(v[k])) if v[k] else float("nan") for k in MM}, len(v[MM[0]])

    for i in images:
        diag = float(np.hypot(*dims[i]))
        for g in GROUPS:
            v, n = mm_mean(f"RvR|{g}|{i}", real[g][i], real[g][i], True, diag)
            mm_rows.append(dict(kind="RvR", target=g, tau="", rep=-1, image_id=i, n_pairs=n, **v))
    def gen_of(iid, gid, t, seed):
        torch.manual_seed(seed)
        return sample_conditioned(model, bank.tensor(f"{iid}.png"), gid, t_end=T_END, n_sim=N_SIM, temperature=t)
    a1, a2 = gen_of(images[0], 0, 2.0, 123), gen_of(images[0], 0, 2.0, 123)
    assert all(torch.equal(x, y) for x, y in zip(a1, a2)), "sampler is not deterministic"
    log("same-seed rerun identical")

    for rep in range(N_REP):
        store = {t: [] for t in TAUS}
        sets = {mode: {t: {g: [] for g in GROUPS} for t in TAUS} for mode in MODES}
        for k, iid in enumerate(images):
            Wd, Hd = dims[iid]
            diag = float(np.hypot(Wd, Hd))
            cur = {}
            for g in GROUPS:
                gid = GROUP_TO_ID[g]
                seed = 100000 * (rep + 1) + 1000 * (k + 1) + gid
                first = {}
                for t in TAUS:
                    it, mk, ms = gen_of(iid, gid, t, seed)
                    max_len = max(max_len, it.shape[1])
                    first[t] = it[:, 0].clone()
                    L_raw, L_clip = [], []
                    for sim, spn in enumerate(to_scanpaths(it, mk, ms)):
                        store[t].append(dict(image_id=iid, group=g, sim=sim, seed=seed, scanpath=np.round(spn, 5).tolist()))
                        if len(spn) < 1:
                            n_empty += 1
                            continue
                        off[tau_name(t)][g][rep] += [float((np.abs(spn[:, :2]) > 1).any(1).sum()), len(spn)]
                        xy = denorm(spn, Wd, Hd)
                        L_raw.append(dict(xy=xy, dur=spn[:, 2]))
                        L_clip.append(dict(xy=np.clip(xy, [0, 0], [Wd - 1, Hd - 1]), dur=spn[:, 2]))
                    cur[(t, g)] = L_clip
                    for mode, L in (("clip", L_clip), ("raw", L_raw)):
                        s = samples(L, iid)
                        p = pre(mode, t, g)
                        for m in METRICS:
                            if m == "n_fix_window":
                                arrs[f"{p}_{m}"][rep, k] = np.mean([(np.r_[0.0, np.cumsum(x["dur"])[:-1]] < window[g][iid]).sum() for x in L])
                            else:
                                arrs[f"{p}_{m}"][rep, k] = np.nanmean([r[m] for r in s["rows"]])
                        arrs[f"{p}_dur_pooled_median"][rep, k] = np.median(s["dur"])
                        for d, val in dist_set(s, real_s[g][iid]).items():
                            arrs[f"{p}_{d}"][rep, k] = val
                        gen_pool[mode][t][g][iid].append(s)
                        sets[mode][t][g].append(s)
                first_dur_max_diff = max(first_dur_max_diff, float((first[1.0] - first[2.0]).abs().max()))
            for t in TAUS:
                for g, other in (("ASD", "TD"), ("TD", "ASD")):
                    v, n = mm_mean(f"RvG|{g}|{rep}|{iid}", real[g][iid], cur[(t, g)], False, diag)
                    mm_rows.append(dict(kind="RvG", target=g, tau=tau_name(t), rep=rep, image_id=iid, n_pairs=n, **v))
                    v, n = mm_mean(f"RvG_wrong|{g}|{rep}|{iid}", real[g][iid], cur[(t, other)], False, diag)
                    mm_rows.append(dict(kind="RvG_wrong", target=g, tau=tau_name(t), rep=rep, image_id=iid, n_pairs=n, **v))
        for t in TAUS:
            with gzip.open(f"{OUT}/generation_{tau_name(t)}/seed_{rep + 1:02d}.json.gz", "wt", encoding="utf-8") as f:
                json.dump(dict(temperature=t, replicate=rep + 1, seed_rule="100000*(rep+1)+1000*(image_rank+1)+group_id", scanpaths=store[t]), f)
        for mode in MODES:
            for t in TAUS:
                for g in GROUPS:
                    for d, val in dist_set(merge(sets[mode][t][g]), merge([real_s[g][i] for i in images])).items():
                        arrs[f"{pre(mode, t, g)}_global_{d}"][rep] = val
        log(f"replicate {rep + 1}/{N_REP} done ({time.time() - t0:.0f}s); first-duration max diff across temperatures={first_dur_max_diff:.2e}; longest sequence={max_len}")
    if first_dur_max_diff > 1e-5:
        raise SystemExit("GATE FAILED: first durations differ across temperatures")
    for mode in MODES:
        for t in TAUS:
            for g in GROUPS:
                for k, iid in enumerate(images):
                    for d, val in dist_set(merge(gen_pool[mode][t][g][iid]), real_s[g][iid]).items():
                        arrs[f"{pre(mode, t, g)}_pooled_{d}"][k] = val
    np.savez_compressed(f"{RAW}/summaries.npz", images=np.array(images), **arrs)
    write_rows(f"{RAW}/multimatch_per_image.csv", mm_rows)
    json.dump(dict(formula_checks=checks, sigma=sigma, spreads=spreads, likelihood=likelihood, first_duration_max_diff=first_dur_max_diff, empty_scanpaths=n_empty, longest_sequence=max_len, off_image={t: {g: dict(rate_per_seed=(v[:, 0] / v[:, 1]).tolist(), rate=float(v[:, 0].sum() / v[:, 1].sum())) for g, v in x.items()} for t, x in off.items()}, images=images, normalization=dict(mean=mean, std=std), checkpoint_sha256=h2), open(f"{RAW}/run_checks.json", "w"), indent=2)
    assert sha256(PHASE2) == h2
    log(f"checkpoint unchanged. total {time.time() - t0:.0f}s. RUN_DONE")


# ======================================================================= analyze
def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    rng = np.random.default_rng(0)
    Z = np.load(f"{RAW}/summaries.npz")
    checks = json.load(open(f"{RAW}/run_checks.json"))
    images = list(Z["images"])
    nI = len(images)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    Bidx = rng.integers(0, nI, (N_BOOT, nI))

    def paired(d):
        """Image-level paired test: mean, percentile bootstrap, sign-flip permutation, Cohen's dz."""
        d = np.asarray(d, float)
        s = d.std(ddof=1)
        flips = (rng.choice([-1.0, 1.0], (N_BOOT, nI)) * d).mean(1)
        return dict(mean=float(d.mean()), ci=ci(d[Bidx].mean(1)), p=float((np.sum(np.abs(flips) >= abs(d.mean()) - 1e-15) + 1) / (N_BOOT + 1)), dz=float(d.mean() / s) if s > 0 else 0.0, n_pos=int((d > 0).sum()), n=nI)
    def holm(items, key="p", out="p_holm"):
        o = sorted(range(len(items)), key=lambda i: items[i][key]); r_ = 0.0
        for r, i in enumerate(o):
            r_ = max(r_, min(1.0, (len(items) - r) * items[i][key]))
            items[i][out] = r_
    def seedstat(x, sign=None):
        """x: one value per seed."""
        return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), min=float(x.min()), max=float(x.max()), n_pos=int((x > 0).sum()), n=len(x))
    G = lambda mode, t, g, m: Z[f"{mode}_{tau_name(t)}_{g}_{m}"]

    res = {}
    for mode in MODES:
        direct, realism, dist, gaps = [], [], [], []
        for g in GROUPS:
            for m, lab in {**METRICS, "dur_pooled_median": "Fixation duration, pooled per-image median (s)"}.items():
                a, b, R = G(mode, 1.0, g, m), G(mode, 2.0, g, m), Z[f"real_{g}_{m}"]
                t_eff = b.mean(0) - a.mean(0)
                split = 0.5 * ((a[:5].mean(0) - a[5:].mean(0)) + (b[:5].mean(0) - b[5:].mean(0)))
                seed_sd = 0.5 * (a.mean(1).std(ddof=1) + b.mean(1).std(ddof=1))
                row = dict(metric=m, label=lab, group=g, real=float(R.mean()), tau1=float(a.mean()), tau2=float(b.mean()), tau1_seed_sd=float(a.mean(1).std(ddof=1)), tau2_seed_sd=float(b.mean(1).std(ddof=1)), noise_split_half_mean=float(split.mean()), noise_seed_sd=float(seed_sd))
                row.update(paired(t_eff)); row["seeds"] = seedstat(b.mean(1) - a.mean(1)); row["effect_to_seed_sd"] = float(abs(t_eff.mean()) / seed_sd) if seed_sd > 0 else None
                direct.append(row)
                e1, e2 = np.abs(a.mean(0) - R), np.abs(b.mean(0) - R)
                gain = e2 - e1  # positive favours tau=1
                row2 = dict(metric=m, label=lab, group=g, real=float(R.mean()), tau1=float(a.mean()), tau2=float(b.mean()), bias_tau1=float((a.mean(0) - R).mean()), bias_tau2=float((b.mean(0) - R).mean()), mae_tau1=float(e1.mean()), mae_tau2=float(e2.mean()), rel_pct_of_tau2=float(100 * gain.mean() / e2.mean()))
                row2.update(paired(gain)); row2["seeds"] = seedstat(np.abs(b - R).mean(1) - np.abs(a - R).mean(1))
                realism.append(row2)
            for d, lab in DISTS.items():
                a, b = Z[f"{mode}_tau1_{g}_pooled_{d}"], Z[f"{mode}_tau2_{g}_pooled_{d}"]
                row = dict(metric=d, label=lab, group=g, tau1=float(a.mean()), tau2=float(b.mean()), rel_pct_of_tau2=float(100 * (b - a).mean() / b.mean()))
                row.update(paired(b - a)); row["seeds"] = seedstat(G(mode, 2.0, g, d).mean(1) - G(mode, 1.0, g, d).mean(1))
                row["global_tau1"], row["global_tau2"] = seedstat(Z[f"{mode}_tau1_{g}_global_{d}"]), seedstat(Z[f"{mode}_tau2_{g}_global_{d}"])
                dist.append(row)
        for m, lab in METRICS.items():
            Dr = Z[f"real_ASD_{m}"] - Z[f"real_TD_{m}"]
            D1, D2 = G(mode, 1.0, "ASD", m) - G(mode, 1.0, "TD", m), G(mode, 2.0, "ASD", m) - G(mode, 2.0, "TD", m)
            p1, p2 = D1.mean(0), D2.mean(0)
            gain = np.abs((Dr - p2)[Bidx].mean(1)) - np.abs((Dr - p1)[Bidx].mean(1))  # positive favours tau=1
            r1, r2 = D1.mean(1), D2.mean(1)
            same = lambda x: int((np.sign(x) == np.sign(Dr.mean())).sum())
            gaps.append(dict(metric=m, label=lab, real=float(Dr.mean()), real_ci=ci(Dr[Bidx].mean(1)), tau1=float(p1.mean()), tau1_ci=ci(p1[Bidx].mean(1)), tau2=float(p2.mean()), tau2_ci=ci(p2[Bidx].mean(1)), ratio_tau1=float(p1.mean() / Dr.mean()), ratio_tau2=float(p2.mean() / Dr.mean()),
                             tau1_seeds_real_sign=same(r1), tau2_seeds_real_sign=same(r2), tau1_seeds=seedstat(r1), tau2_seeds=seedstat(r2), gap_tau1=float(abs((Dr - p1).mean())), gap_tau2=float(abs((Dr - p2).mean())), gap_gain_favouring_tau1=float(abs((Dr - p2).mean()) - abs((Dr - p1).mean())), gap_gain_ci=ci(gain), r_tau1=float(np.corrcoef(Dr, p1)[0, 1]), r_tau2=float(np.corrcoef(Dr, p2)[0, 1]), tau2_minus_tau1=paired(p2 - p1), tau2_minus_tau1_seeds=seedstat(r2 - r1)))
        for fam in (direct, realism, dist):
            holm(fam)
        res[mode] = dict(direct=direct, realism=realism, distances=dist, group_differences=gaps)

    # MultiMatch, clipped coordinates
    Mm = list(csv.DictReader(open(f"{RAW}/multimatch_per_image.csv", encoding="utf-8")))
    iix = {int(i): k for k, i in enumerate(images)}
    def cube(kind, g, tau):
        c = np.full((N_REP, nI, len(MM)), np.nan)
        for r in Mm:
            if r["kind"] == kind and r["target"] == g and r["tau"] == tau:
                c[int(r["rep"]), iix[int(r["image_id"])]] = [float(r[k]) if r[k] not in ("", "nan") else np.nan for k in MM]
        return c
    mmres = []
    for g in GROUPS:
        rv = np.full((nI, len(MM)), np.nan)
        for r in Mm:
            if r["kind"] == "RvR" and r["target"] == g:
                rv[iix[int(r["image_id"])]] = [float(r[k]) for k in MM]
        c1, c2, w1, w2 = cube("RvG", g, "tau1"), cube("RvG", g, "tau2"), cube("RvG_wrong", g, "tau1"), cube("RvG_wrong", g, "tau2")
        for q, k in enumerate(MM):
            d = np.nanmean(c1[:, :, q], 0) - np.nanmean(c2[:, :, q], 0)  # positive favours tau=1
            row = dict(metric=k, group=g, real_vs_real=float(np.nanmean(rv[:, q])), tau1=float(np.nanmean(c1[:, :, q])), tau2=float(np.nanmean(c2[:, :, q])))
            row.update(paired(d)); row["seeds"] = seedstat(np.nanmean(c1[:, :, q], 1) - np.nanmean(c2[:, :, q], 1))
            row["own_minus_other_tau1"] = paired(np.nanmean(c1[:, :, q], 0) - np.nanmean(w1[:, :, q], 0))
            row["own_minus_other_tau2"] = paired(np.nanmean(c2[:, :, q], 0) - np.nanmean(w2[:, :, q], 0))
            mmres.append(row)
    holm(mmres)

    # post hoc: grand-mean bias, and spatial metrics truncated to the real viewing window
    for mode in MODES:
        for r in res[mode]["realism"]:
            a, b, R = G(mode, 1.0, r["group"], r["metric"]), G(mode, 2.0, r["group"], r["metric"]), Z[f"real_{r['group']}_{r['metric']}"]
            r["abs_bias_gain_favouring_tau1"] = float(abs((b.mean(0) - R).mean()) - abs((a.mean(0) - R).mean()))
            r["abs_bias_gain_ci"] = ci(np.abs((b.mean(0) - R)[Bidx].mean(1)) - np.abs((a.mean(0) - R)[Bidx].mean(1)))
    W_ = np.load(f"{RAW}/window_matched.npz")
    win_rows = []
    for g in GROUPS:
        for m in ("n_fix", "dispersion", "sacc_amp_norm_mean", "revisit_rate", "sacc_dir_resultant", "sacc_horizontal_frac"):
            a, b = W_[f"tau1_{g}_{m}"], W_[f"tau2_{g}_{m}"]
            R = Z[f"real_{g}_{m}"]
            e1, e2 = np.abs(a.mean(0) - R), np.abs(b.mean(0) - R)
            row = dict(metric=m, label=METRICS[m], group=g, real=float(R.mean()), tau1=float(a.mean()), tau2=float(b.mean()), bias_tau1=float((a.mean(0) - R).mean()), bias_tau2=float((b.mean(0) - R).mean()), mae_tau1=float(e1.mean()), mae_tau2=float(e2.mean()), abs_bias_gain_favouring_tau1=float(abs((b.mean(0) - R).mean()) - abs((a.mean(0) - R).mean())), abs_bias_gain_ci=ci(np.abs((b.mean(0) - R)[Bidx].mean(1)) - np.abs((a.mean(0) - R)[Bidx].mean(1))), direct=paired(b.mean(0) - a.mean(0)))
            row.update(paired(e2 - e1)); row["seeds"] = seedstat(np.abs(b - R).mean(1) - np.abs(a - R).mean(1))
            win_rows.append(row)
    holm(win_rows)
    win_gaps = []
    for m in ("dispersion", "sacc_amp_norm_mean", "revisit_rate", "sacc_dir_resultant", "n_fix"):
        Dr = Z[f"real_ASD_{m}"] - Z[f"real_TD_{m}"]
        p1, p2 = (W_[f"tau1_ASD_{m}"] - W_[f"tau1_TD_{m}"]), (W_[f"tau2_ASD_{m}"] - W_[f"tau2_TD_{m}"])
        win_gaps.append(dict(metric=m, real=float(Dr.mean()), tau1=float(p1.mean()), tau2=float(p2.mean()), tau1_ci=ci(p1.mean(0)[Bidx].mean(1)), tau2_ci=ci(p2.mean(0)[Bidx].mean(1)), tau1_seeds_real_sign=int((np.sign(p1.mean(1)) == np.sign(Dr.mean())).sum()), tau2_seeds_real_sign=int((np.sign(p2.mean(1)) == np.sign(Dr.mean())).sum())))
    res["window_matched"] = dict(realism=win_rows, group_differences=win_gaps, note="Post hoc. Generated scanpaths truncated to fixations with onset before the mean real fixation time of that group and image. Uses real timing information, so it is a diagnostic, not a model property.")

    # outputs
    rows = []
    for mode in MODES:
        for fam in ("direct", "realism", "distances"):
            for r in res[mode][fam]:
                rows.append(dict(analysis="clipped_primary" if mode == "clip" else "unclipped_sensitivity", family=fam, **{k: (v if not isinstance(v, dict) else json.dumps(v)) for k, v in r.items()}))
    for r in win_rows:
        rows.append(dict(analysis="window_matched_post_hoc", family="realism", **{k: (v if not isinstance(v, dict) else json.dumps(v)) for k, v in r.items()}))
    write_rows(f"{OUT}/temperature_comparison.csv", rows)
    json.dump(dict(notes="tau2_minus_tau1 is direct. In realism, distances and MultiMatch, positive p-tested mean favours tau=1. Seeds are shared across temperatures.", real_means={g: {m: float(Z[f"real_{g}_{m}"].mean()) for m in METRICS} for g in GROUPS}, per_seed={mode: {tau_name(t): {g: {m: G(mode, t, g, m).mean(1).tolist() for m in METRICS} for g in GROUPS} for t in TAUS} for mode in MODES}, group_differences={m: res[m]["group_differences"] for m in MODES}, distances={m: res[m]["distances"] for m in MODES}, multimatch=mmres, diagnostics=checks), open(f"{OUT}/behavioural_comparison.json", "w"), indent=2)
    json.dump({m: res[m] for m in list(MODES) + ["window_matched"]} | dict(multimatch=mmres, scheme="One value per image, pooled over 10 seeds. Percentile bootstrap and sign-flip permutation, 10,000 each. Holm within each family and analysis."), open(f"{OUT}/statistical_comparison.json", "w"), indent=2)

    # figures
    ASD, TD, INK, INK2, MUTED, GRID, AXIS, SURF = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "savefig.dpi": 200, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True})
    # 1. metrics
    ms = ["dispersion", "sacc_amp_norm_mean", "revisit_rate", "dur_mean", "n_fix", "n_fix_window", "fix_rate_per_s", "total_dur", "sacc_horizontal_frac", "sacc_dir_resultant"]
    fig, axs = plt.subplots(2, 5, figsize=(12, 4.8))
    for ax, m in zip(axs.ravel(), ms):
        for j, (g, c) in enumerate((("ASD", ASD), ("TD", TD))):
            ax.plot([j - .3, j + .3], [Z[f"real_{g}_{m}"].mean()] * 2, color=c, lw=2.2, solid_capstyle="round")
            for off, t, face in ((-.15, 1.0, c), (.15, 2.0, SURF)):
                v = G("clip", t, g, m).mean(1)
                ax.plot([j + off] * 2, [v.min(), v.max()], color=c, lw=1); ax.plot(j + off, v.mean(), "o", mfc=face, mec=c, ms=6, mew=1.3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(GROUPS, color=INK2); ax.set_xlim(-.6, 1.6); ax.set_title(METRICS[m], loc="left", fontsize=8.2); ax.grid(axis="x", visible=False)
    fig.legend(handles=[Line2D([], [], color=INK2, lw=2.2, label="Real validation mean (bar)"), Line2D([], [], marker="o", color=INK2, ls="", label="Generated, tau = 1"), Line2D([], [], marker="o", mfc=SURF, mec=INK2, ls="", label="Generated, tau = 2"), Line2D([], [], color=INK2, lw=1, label="Range across 10 seeds")], frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(.05, -.05))
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_metrics_by_temperature.png", bbox_inches="tight"); plt.close(fig)
    # 2. mechanism
    s = checks["sigma"]; e = np.array(s["hist_edges"]); h = np.array(s["hist_log10"]); h = h / h.sum()
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 2.9))
    ax = axs[0]
    ax.stairs(h, 10 ** e, color=INK2, lw=1.5, label="tau = 1 (model)"); ax.stairs(h, 10 ** (2 * e), color=MUTED, lw=1.5, ls="--", label="tau = 2 (sigma squared)")
    ax.axvline(1, color=AXIS, lw=1); ax.set_xscale("log"); ax.set_xlabel("Component sigma in normalized image coordinates"); ax.set_ylabel("Share of mixture weight"); ax.set_title("Temperature squares sigma; below 1 it shrinks", loc="left"); ax.set_xlim(5e-4, 3); ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.24))
    ax = axs[1]
    labs, y = [], 0
    for key, name in (("first_step", "First fixation"), ("history", "Later fixations")):
        for sd, sn, mk in (("within_component_sd", "within component", "o"), ("total_sd", "total", "s")):
            a, b = checks["spreads"][key]["tau1"][sd], checks["spreads"][key]["tau2"][sd]
            ax.plot([a, b], [y, y], color=AXIS, lw=1.2); ax.plot(a, y, "o", color=INK2, ms=6); ax.plot(b, y, "o", mfc=SURF, mec=INK2, ms=6, mew=1.3)
            labs.append(f"{name}, {sn}"); y += 1
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, color=INK2); ax.set_xlabel("Per-axis SD of next-fixation density"); ax.grid(axis="y", visible=False); ax.set_title("Predicted spread", loc="left")
    ax.legend(handles=[Line2D([], [], marker="o", color=INK2, ls="", label="tau = 1"), Line2D([], [], marker="o", mfc=SURF, mec=INK2, ls="", label="tau = 2")], frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_mechanism.png", bbox_inches="tight"); plt.close(fig)
    # 3. group differences
    fig, axs = plt.subplots(2, 5, figsize=(12, 3.8))
    gd = {d["metric"]: d for d in res["clip"]["group_differences"]}
    for ax, m in zip(axs.ravel(), ms):
        d = gd[m]
        ax.plot(d["real_ci"], [2, 2], color=INK, lw=1.4); ax.plot(d["real"], 2, "o", color=INK, ms=6, mec=SURF)
        ax.plot(d["tau1_ci"], [1, 1], color=INK2, lw=1.4); ax.plot(d["tau1"], 1, "o", color=INK2, ms=6, mec=SURF)
        ax.plot(d["tau2_ci"], [0, 0], color=INK2, lw=1.4); ax.plot(d["tau2"], 0, "o", mfc=SURF, mec=INK2, ms=6, mew=1.2)
        ax.axvline(0, color=AXIS, lw=1); ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["Real", "tau = 1", "tau = 2"], color=INK2, fontsize=7.5); ax.set_ylim(-.6, 2.6); ax.grid(axis="y", visible=False); ax.set_title(METRICS[m], loc="left", fontsize=8); ax.tick_params(axis="x", labelsize=7); ax.xaxis.set_major_locator(MaxNLocator(4))
    fig.suptitle("ASD minus TD, with 95% image bootstrap", x=.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_group_differences.png", bbox_inches="tight"); plt.close(fig)
    # 4. MultiMatch
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True)
    y = np.arange(len(MM))[::-1]
    for ax, g, c in ((axs[0], "ASD", ASD), (axs[1], "TD", TD)):
        for i, r in enumerate([r for r in mmres if r["group"] == g]):
            ax.plot(r["real_vs_real"], y[i] + .2, "D", color=MUTED, ms=5); ax.plot(r["tau1"], y[i], "o", color=c, ms=6, mec=SURF); ax.plot(r["tau2"], y[i] - .2, "o", mfc=SURF, mec=c, ms=6, mew=1.3)
        ax.set_title(f"Similarity to real {g} (1 = identical)", loc="left"); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y); axs[0].set_yticklabels([m.replace("mm_", "").capitalize() for m in MM], color=INK2)
    fig.legend(handles=[Line2D([], [], marker="D", color=MUTED, ls="", label="Real vs real"), Line2D([], [], marker="o", color=INK2, ls="", label="tau = 1"), Line2D([], [], marker="o", mfc=SURF, mec=INK2, ls="", label="tau = 2")], frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(.1, -.1))
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_multimatch.png", bbox_inches="tight"); plt.close(fig)
    # 5. seeds: paired temperature effect per seed against the split-half noise
    fig, axs = plt.subplots(2, 4, figsize=(11, 4.2))
    for ax, m in zip(axs.ravel(), ["dispersion", "sacc_amp_norm_mean", "revisit_rate", "dur_mean", "n_fix", "fix_rate_per_s", "sacc_dir_resultant", "sacc_horizontal_frac"]):
        for j, (g, c) in enumerate((("ASD", ASD), ("TD", TD))):
            d = G("clip", 2.0, g, m).mean(1) - G("clip", 1.0, g, m).mean(1)
            ax.plot(np.arange(1, 11) + (j - .5) * .25, d, "o", color=c, ms=4.5, mec=SURF, mew=.6)
        ax.axhline(0, color=AXIS, lw=1); ax.set_title(METRICS[m], loc="left", fontsize=8.2); ax.set_xticks([1, 5, 10]); ax.grid(axis="x", visible=False); ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.legend(handles=[Line2D([], [], marker="o", color=ASD, ls="", label="ASD"), Line2D([], [], marker="o", color=TD, ls="", label="TD")], frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(.05, -.05))
    fig.suptitle("tau = 2 minus tau = 1 for each of the 10 seeds", x=.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_seed_stability.png", bbox_inches="tight"); plt.close(fig)

    # digest
    c = checks
    print("=== checks ===", json.dumps(c["formula_checks"]), "first-dur", c["first_duration_max_diff"], "empty", c["empty_scanpaths"], "longest", c["longest_sequence"])
    print("sigma", json.dumps({k: v for k, v in c["sigma"].items() if not k.startswith("hist")}))
    print("spreads", json.dumps(c["spreads"])); print("likelihood", json.dumps(c["likelihood"]))
    print("off image", json.dumps({t: {g: round(v["rate"], 5) for g, v in x.items()} for t, x in c["off_image"].items()}))
    for mode in MODES:
        print(f"\n########## {mode}")
        print("--- direct effect, tau2 - tau1")
        for r in res[mode]["direct"]:
            print(f"{r['group']:4s}{r['metric']:22s} real={r['real']:.4f} t1={r['tau1']:.4f}(sd{r['tau1_seed_sd']:.4f}) t2={r['tau2']:.4f}(sd{r['tau2_seed_sd']:.4f}) d={r['mean']:+.4f}{[round(x, 4) for x in r['ci']]} dz={r['dz']:+.2f} holm={r['p_holm']:.4f} I+{r['n_pos']}/45 seeds+{r['seeds']['n_pos']}/10 [{r['seeds']['min']:+.4f},{r['seeds']['max']:+.4f}] noise={r['noise_split_half_mean']:+.4f} eff/sd={r['effect_to_seed_sd']:.1f}")
        print("--- realism (MAE; + favours tau1)")
        for r in res[mode]["realism"]:
            print(f"{r['group']:4s}{r['metric']:22s} bias {r['bias_tau1']:+.4f}/{r['bias_tau2']:+.4f} MAE {r['mae_tau1']:.4f}/{r['mae_tau2']:.4f} gain={r['mean']:+.4f}{[round(x, 4) for x in r['ci']]} ({r['rel_pct_of_tau2']:+.1f}%) holm={r['p_holm']:.4f} I+{r['n_pos']}/45 seeds+{r['seeds']['n_pos']}/10")
        print("--- distances (+ favours tau1)")
        for r in res[mode]["distances"]:
            print(f"{r['group']:4s}{r['metric']:8s} t1={r['tau1']:.4f} t2={r['tau2']:.4f} gain={r['mean']:+.4f}{[round(x, 4) for x in r['ci']]} ({r['rel_pct_of_tau2']:+.1f}%) holm={r['p_holm']:.4f} I+{r['n_pos']}/45 seeds+{r['seeds']['n_pos']}/10 global {r['global_tau1']['mean']:.4f}/{r['global_tau2']['mean']:.4f}")
        print("--- group differences")
        for d in res[mode]["group_differences"]:
            print(f"{d['metric']:22s} real={d['real']:+.4f}{[round(x, 4) for x in d['real_ci']]} t1={d['tau1']:+.4f}{[round(x, 4) for x in d['tau1_ci']]} t2={d['tau2']:+.4f}{[round(x, 4) for x in d['tau2_ci']]} ratio {d['ratio_tau1']:+.2f}/{d['ratio_tau2']:+.2f} sign {d['tau1_seeds_real_sign']}/{d['tau2_seeds_real_sign']} gap {d['gap_tau1']:.4f}/{d['gap_tau2']:.4f} gain={d['gap_gain_favouring_tau1']:+.4f}{[round(x, 4) for x in d['gap_gain_ci']]} r {d['r_tau1']:.2f}/{d['r_tau2']:.2f}")
    print("\n--- multimatch (+ favours tau1)")
    for r in mmres:
        print(f"{r['group']:4s}{r['metric'][3:]:10s} RvR={r['real_vs_real']:.4f} t1={r['tau1']:.4f} t2={r['tau2']:.4f} gain={r['mean']:+.4f}{[round(x, 4) for x in r['ci']]} holm={r['p_holm']:.4f} I+{r['n_pos']}/45 seeds+{r['seeds']['n_pos']}/10 | own-other t1 {r['own_minus_other_tau1']['mean']:+.4f}{[round(x, 4) for x in r['own_minus_other_tau1']['ci']]} t2 {r['own_minus_other_tau2']['mean']:+.4f}{[round(x, 4) for x in r['own_minus_other_tau2']['ci']]}")
    print("\n--- grand-mean absolute bias (+ favours tau1)")
    for r in res["clip"]["realism"]:
        if r["metric"] in ("dispersion", "sacc_amp_norm_mean", "revisit_rate", "sacc_dir_resultant", "sacc_horizontal_frac"):
            print(f"{r['group']:4s}{r['metric']:22s} {r['abs_bias_gain_favouring_tau1']:+.4f}{[round(x, 4) for x in r['abs_bias_gain_ci']]}")
    print("\n--- window matched (post hoc; + favours tau1)")
    for r in win_rows:
        print(f"{r['group']:4s}{r['metric']:22s} real={r['real']:.4f} t1={r['tau1']:.4f} t2={r['tau2']:.4f} bias {r['bias_tau1']:+.4f}/{r['bias_tau2']:+.4f} absbias gain {r['abs_bias_gain_favouring_tau1']:+.4f}{[round(x, 4) for x in r['abs_bias_gain_ci']]} MAE {r['mae_tau1']:.4f}/{r['mae_tau2']:.4f} gain={r['mean']:+.4f}{[round(x, 4) for x in r['ci']]} holm={r['p_holm']:.4f} seeds+{r['seeds']['n_pos']}/10 direct {r['direct']['mean']:+.4f}")
    for d in win_gaps:
        print(f"gap {d['metric']:22s} real={d['real']:+.4f} t1={d['tau1']:+.4f}{[round(x, 4) for x in d['tau1_ci']]} t2={d['tau2']:+.4f}{[round(x, 4) for x in d['tau2_ci']]} sign {d['tau1_seeds_real_sign']}/{d['tau2_seeds_real_sign']}")
    print("\n--- group difference, tau2 - tau1")
    for d in res["clip"]["group_differences"]:
        t = d["tau2_minus_tau1"]
        print(f"{d['metric']:22s} {t['mean']:+.4f}{[round(x, 4) for x in t['ci']]} p={t['p']:.4f} seeds+{d['tau2_minus_tau1_seeds']['n_pos']}/10")
    print("ANALYZE_DONE")


# ======================================================================= windowed
def windowed():
    """Post hoc: recompute metrics on saved scanpaths, truncated to each group's real viewing window on each image."""
    from training.eval_external_v2 import denorm, metric_row
    split = json.load(open(SPLIT))
    va = set(map(int, split["val"]))
    images = sorted(va)
    rank = {i: k for k, i in enumerate(images)}
    window, dims = {g: {} for g in GROUPS}, {}
    for g, f in (("ASD", "processed_data/asd_scanpaths.json"), ("TD", "processed_data/td_scanpaths.json")):
        tot = {}
        for r in json.load(open(f)):
            if int(r["image_id"]) in va:
                i = int(r["image_id"])
                dims[i] = (r["image_width"], r["image_height"])
                tot.setdefault(i, []).append(sum(x[2] for x in r["scanpath"]))
        window[g] = {i: float(np.mean(v)) for i, v in tot.items()}
    keys = ("n_fix", "dispersion", "sacc_amp_norm_mean", "revisit_rate", "sacc_dir_resultant", "sacc_horizontal_frac")
    arrs = {f"{tau_name(t)}_{g}_{m}": np.full((N_REP, len(images)), np.nan) for t in TAUS for g in GROUPS for m in keys}
    for t in TAUS:
        for rep in range(N_REP):
            with gzip.open(f"{OUT}/generation_{tau_name(t)}/seed_{rep + 1:02d}.json.gz", "rt", encoding="utf-8") as f:
                data = json.load(f)["scanpaths"]
            acc = {}
            for s in data:
                sp = np.asarray(s["scanpath"], float)
                if len(sp) < 1:
                    continue
                i, g = s["image_id"], s["group"]
                Wd, Hd = dims[i]
                on = np.r_[0.0, np.cumsum(sp[:, 2])[:-1]]
                sp = sp[on < window[g][i]]
                xy = np.clip(denorm(sp, Wd, Hd), [0, 0], [Wd - 1, Hd - 1])
                row = metric_row("g", i, "", xy, sp[:, 2], Wd, Hd)
                acc.setdefault((g, i), []).append([row[m] for m in keys])
            for (g, i), v in acc.items():
                for q, m in enumerate(keys):
                    arrs[f"{tau_name(t)}_{g}_{m}"][rep, rank[i]] = np.nanmean(np.array(v, float)[:, q])
    np.savez_compressed(f"{RAW}/window_matched.npz", **arrs)
    print("WINDOWED_DONE")


if __name__ == "__main__":
    {"run": run, "analyze": analyze, "windowed": windowed}[sys.argv[1]]()
