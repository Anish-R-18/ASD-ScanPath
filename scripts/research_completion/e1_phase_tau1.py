"""
E1 (research completion, v2): Phase 1 vs Phase 2 generated behaviour at temperature 1.0.
No training; checkpoints read-only; runs on CPU. See ../EXPERIMENT_PLAN.md.
v1 (single process, all CPU threads) was stopped for thread oversubscription; v2 shards replicates over processes.

  python scripts/research_completion/e1_phase_tau1.py run <shard> <n_shards>   (one process per shard; E1_THREADS, default 2)
  python scripts/research_completion/e1_phase_tau1.py merge
  python scripts/research_completion/e1_phase_tau1.py analyze

Sets per image, seed and group-seed g: P2_ASD (Phase 2, ASD, seed g=0), P2_TD (Phase 2, TD, seed g=1),
P1_s0 and P1_s1 (Phase 1 with an in-memory zero embedding, seeds g=0 and g=1). Seed = 100000*(rep+1)+1000*(rank+1)+g.
Phase 1 and Phase 2 share bit-identical base weights, so same-seed sets differ only by the embedding.
"""
from __future__ import annotations
import csv, gzip, json, os, sys, time, zlib
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")  # repository root
os.chdir(ROOT); sys.path[:0] = [ROOT, ROOT + "/tppgaze"]
import numpy as np

OUT = "training/runs/research_completion/e1_phase_tau1_v2"
LOG = {"name": "run_log.txt"}
PHASE1 = "training/runs/baseline/best_model.pth"
PHASE2 = "training/runs/phase2_embedding_only/best_model_conditioned.pth"
G4A = "external_data/Gaze4ASD_processed_v2/fit_height_policy_a/gaze4asd_fit_height_policy_a.json"
GPU_TAU1 = "training/runs/external_gaze4asd_v2/temperature_tau1/raw/summaries.npz"
N_REP, N_SIM, TAU, T_END, MAX_PAIRS, N_BOOT = 10, 20, 1.0, 3.0, 30, 10000
METRICS = {"dur_mean": "Fixation duration (s)", "n_fix": "Fixation count", "fix_rate_per_s": "Fixation rate (per s)", "total_dur": "Total fixation time (s)", "sacc_amp_norm_mean": "Saccade amplitude", "sacc_horizontal_frac": "Horizontal saccade fraction", "dispersion": "Spatial dispersion", "revisit_rate": "Revisit rate", "sacc_dir_resultant": "Direction concentration"}
PRIMARY = ("dispersion", "sacc_amp_norm_mean", "revisit_rate")
SETS = {"P2_ASD": ("p2", 0, 0), "P2_TD": ("p2", 1, 1), "P1_s0": ("p1", 0, 0), "P1_s1": ("p1", 0, 1)}  # model, group id, seed g
MM = ["mm_shape", "mm_length", "mm_direction", "mm_position", "mm_duration"]


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(f"{OUT}/{LOG['name']}", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def run(shard, n_shards):
    import torch
    from omegaconf import OmegaConf
    from training import model_utils as mu, metrics as M
    from training.conditioned import ConditionedTPPGaze
    from training.generate_cond import sample_conditioned, to_scanpaths
    from training.external_gaze4asd_loader import StimulusBank, load_external_scanpaths
    from training.eval_external_v2 import IMG_DIR, BASE_CFG, denorm, metric_row, sha256
    from training.dataset import load_scanpaths
    SH = f"{OUT}/shards/shard{shard}"
    if os.path.exists(SH + ".npz"):
        raise SystemExit("refusing to overwrite " + SH)
    os.makedirs(OUT + "/generated", exist_ok=True); os.makedirs(OUT + "/shards", exist_ok=True)
    LOG["name"] = f"run_log_shard{shard}.txt"
    torch.set_num_threads(int(os.environ.get("E1_THREADS", 2)))
    dev = torch.device("cpu")
    t0 = time.time()
    reps = list(range(shard, N_REP, n_shards))
    h1, h2 = sha256(PHASE1), sha256(PHASE2)
    log(f"shard {shard}/{n_shards} replicates {[r + 1 for r in reps]}; torch {torch.__version__} cpu threads {torch.get_num_threads()}; phase1 {h1}; phase2 {h2}")
    st = json.load(open("training/runs/phase2_embedding_only/summary_full.json"))
    mean, std = st["mean_log_inter_time"], st["std_log_inter_time"]
    cfg = OmegaConf.load(BASE_CFG); cdim = int(cfg.metadata.size + cfg.context.size)
    b1, _ = mu.build_model(BASE_CFG, PHASE1, dev, mean, std, used_cfg_out=f"{OUT}/shards/cfg1_shard{shard}.yaml")
    b2, _ = mu.build_model(BASE_CFG, PHASE1, dev, mean, std, used_cfg_out=f"{OUT}/shards/cfg2_shard{shard}.yaml")
    p2 = ConditionedTPPGaze(b2, cdim).to(dev)
    p2.load_state_dict(torch.load(PHASE2, map_location=dev), strict=True)
    mu._fix_unregistered_device(p2.base, dev); p2.eval()
    assert all(torch.equal(a, b) for a, b in zip(b1.state_dict().values(), p2.base.state_dict().values())), "base weights differ"
    del b1  # Phase 1 = the same (bit-identical) base with a zero embedding; one base per process saves memory
    p1 = ConditionedTPPGaze(p2.base, cdim).to(dev).eval()
    for m in (p1, p2):
        for q in m.parameters():
            q.requires_grad = False
    assert float(p1.group_emb.weight.abs().sum()) == 0.0
    models = {"p1": p1, "p2": p2}

    # image features computed once per image (shared base, so one cache); the sampler code is unchanged
    current = {"key": None}; caches = {"shared": {}}
    def make_cached(name, orig):
        def cached(imgs):
            c = caches[name]
            if current["key"] not in c:
                c[current["key"]] = orig(imgs[:1])
            v = c[current["key"]]
            return v.expand(imgs.shape[0], *v.shape[1:])
        return cached
    gbank = StimulusBank(IMG_DIR)
    gate_img = None
    recs = load_external_scanpaths(G4A, gbank)
    gimgs = sorted({r["image_id"] for r in recs}, key=lambda s: int(s.replace("top_image_", "").replace(".png", "")))
    gate_img = gbank.tensor(gimgs[0])
    torch.manual_seed(123)
    ref = sample_conditioned(p2, gate_img, 0, t_end=T_END, n_sim=N_SIM, temperature=TAU)
    p2.base.get_metadata = make_cached("shared", p2.base.get_metadata)
    assert p1.base is p2.base
    current["key"] = ("gate", gimgs[0])
    torch.manual_seed(123)
    new = sample_conditioned(p2, gate_img, 0, t_end=T_END, n_sim=N_SIM, temperature=TAU)
    md = max(float((a - b).abs().max()) for a, b in zip(ref, new)) if all(a.shape == b.shape for a, b in zip(ref, new)) else float("inf")
    log(f"cache gate: max abs diff cached vs uncached sampling = {md:.2e}; masks equal = {bool(ref[0].shape == new[0].shape and torch.equal(ref[2], new[2]))}")
    if not md < 1e-4:
        raise SystemExit("GATE FAILED: cached features change sampling")

    # data
    sbank = StimulusBank(OmegaConf.load("configs/phase2_embedding_only.yaml").images_dir.split("#")[0].strip())
    test_ids = sorted(map(int, json.load(open("training/runs/baseline/split.json"))["test"]))
    sal = [s for s in load_scanpaths("processed_data/asd_scanpaths.json", "processed_data/td_scanpaths.json") if s["image_id"] in set(test_ids)]
    DS = {"gaze4asd": dict(images=gimgs, bank=gbank, key=lambda i: i, recs=recs), "saliency4asd_test": dict(images=test_ids, bank=sbank, key=lambda i: f"{i}.png", recs=sal)}
    arrs, mm_rows, meta = {}, [], {}
    def mm_mean(tag, la, lb, diag):
        pairs = [(i, j) for i in range(len(la)) for j in range(len(lb))]
        rng = np.random.default_rng([0, zlib.crc32(tag.encode())])
        v = {k: [] for k in MM}
        for q in rng.permutation(len(pairs))[:MAX_PAIRS]:
            a, b = la[pairs[q][0]], lb[pairs[q][1]]
            r = M.multimatch(a["xy"], a["dur"], b["xy"], b["dur"], diag)
            if r:
                for k in MM:
                    v[k].append(r[k])
        return {k: float(np.mean(v[k])) if v[k] else float("nan") for k in MM}
    for ds, D in DS.items():
        imgs, bank, key = D["images"], D["bank"], D["key"]
        real = {g: {i: [] for i in imgs} for g in ("ASD", "TD")}
        for r in D["recs"]:
            W, H = bank.dims(key(r["image_id"])); sp = np.asarray(r["scanpath"], float)
            real[r["group"]][r["image_id"]].append(dict(xy=denorm(sp, W, H), dur=sp[:, 2]))
        for g in real:
            for m in METRICS:
                arrs[f"{ds}_real_{g}_{m}"] = np.array([np.nanmean([metric_row("r", i, "", x["xy"], x["dur"], *bank.dims(key(i)))[m] for x in real[g][i]]) for i in imgs])
            arrs[f"{ds}_real_{g}_n"] = np.array([len(real[g][i]) for i in imgs])
        for s in SETS:
            for m in METRICS:
                arrs[f"{ds}_{s}_{m}"] = np.full((N_REP, len(imgs)), np.nan)
        off = {s: [0, 0] for s in SETS}
        for rep in reps:
            store = {s: [] for s in SETS}
            for k, iid in enumerate(imgs):
                W, H = bank.dims(key(iid)); diag = float(np.hypot(W, H))
                img = bank.tensor(key(iid)); current["key"] = (ds, iid)
                cur = {}
                for s, (mname, gid, sg) in SETS.items():
                    seed = 100000 * (rep + 1) + 1000 * (k + 1) + sg
                    torch.manual_seed(seed)
                    it, mk, ms = sample_conditioned(models[mname], img, gid, t_end=T_END, n_sim=N_SIM, temperature=TAU)
                    L = []
                    for sim, sp in enumerate(to_scanpaths(it, mk, ms)):
                        if not len(sp):
                            continue
                        sp = np.round(sp, 5)
                        store[s].append(dict(image_id=iid, sim=sim, seed=seed, scanpath=sp.tolist()))
                        off[s][0] += int((np.abs(sp[:, :2]) > 1).any(1).sum()); off[s][1] += len(sp)
                        L.append(dict(xy=np.clip(denorm(sp, W, H), [0, 0], [W - 1, H - 1]), dur=sp[:, 2]))
                    cur[s] = L
                    rows = [metric_row("g", iid, "", x["xy"], x["dur"], W, H) for x in L]
                    for m in METRICS:
                        arrs[f"{ds}_{s}_{m}"][rep, k] = np.nanmean([r[m] for r in rows])
                if ds == "gaze4asd":
                    for tgt in ("ASD", "TD"):
                        for s in SETS:
                            sg = SETS[s][2]
                            v = mm_mean(f"{tgt}|{rep}|{iid}|seed{sg}", real[tgt][iid], cur[s], diag)
                            mm_rows.append(dict(target=tgt, set=s, rep=rep, image_id=iid, **v))
            for s in SETS:
                with gzip.open(f"{OUT}/generated/{ds}_{s}_seed{rep + 1:02d}.json.gz", "wt", encoding="utf-8") as f:
                    json.dump(dict(dataset=ds, set=s, temperature=TAU, replicate=rep + 1, scanpaths=store[s]), f)
            log(f"{ds} replicate {rep + 1}/{N_REP} done ({time.time() - t0:.0f}s)")
        meta[ds] = dict(images=[str(i) for i in imgs], off_image_counts=off)
    assert sha256(PHASE1) == h1 and sha256(PHASE2) == h2
    with open(SH + "_multimatch.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mm_rows[0])); w.writeheader(); w.writerows(mm_rows)
    json.dump(dict(meta, replicates=[r + 1 for r in reps], seconds=time.time() - t0), open(SH + "_meta.json", "w"), indent=1)
    np.savez_compressed(SH + ".npz", **arrs)  # written last: its presence marks the shard complete
    log(f"shard {shard} done; checkpoints unchanged; {time.time() - t0:.0f}s; SHARD_DONE")


def merge():
    shards = sorted(f for f in os.listdir(f"{OUT}/shards") if f.endswith(".npz"))
    metas = [json.load(open(f"{OUT}/shards/{f[:-4]}_meta.json")) for f in shards]
    reps = sorted(r for m in metas for r in m["replicates"])
    assert reps == list(range(1, N_REP + 1)), f"replicates present: {reps}"
    if os.path.exists(f"{OUT}/summaries.npz"):
        raise SystemExit("refusing to overwrite merged outputs")
    parts = [dict(np.load(f"{OUT}/shards/{f}")) for f in shards]
    arrs = {}
    for k in parts[0]:
        if "_real_" in k:
            assert all(np.array_equal(p[k], parts[0][k], equal_nan=True) for p in parts); arrs[k] = parts[0][k]
        else:
            a = np.full_like(parts[0][k], np.nan)
            for p in parts:
                ok = ~np.isnan(p[k]); assert not np.any(ok & ~np.isnan(a)); a[ok] = p[k][ok]
            arrs[k] = a
    mm_rows = [r for f in shards for r in csv.DictReader(open(f"{OUT}/shards/{f[:-4]}_multimatch.csv", encoding="utf-8"))]
    meta = {}
    for ds in ("gaze4asd", "saliency4asd_test"):
        cnt = {s: [sum(m[ds]["off_image_counts"][s][0] for m in metas), sum(m[ds]["off_image_counts"][s][1] for m in metas)] for s in SETS}
        meta[ds] = dict(images=metas[0][ds]["images"], off_image={s: v[0] / max(v[1], 1) for s, v in cnt.items()})
    meta["shard_seconds"] = [m["seconds"] for m in metas]
    np.savez_compressed(f"{OUT}/summaries.npz", **arrs)
    with open(f"{OUT}/multimatch_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mm_rows[0])); w.writeheader(); w.writerows(mm_rows)
    # distributional agreement of CPU Phase 2 with the saved GPU tau=1 run (same seeds, different random streams)
    Z = np.load(GPU_TAU1); agree = {}
    for g, s in (("ASD", "P2_ASD"), ("TD", "P2_TD")):
        for m in METRICS:
            a, b = arrs[f"gaze4asd_{s}_{m}"].mean(1), Z[f"clip_tau1_{g}_{m}"].mean(1)
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            agree[f"{g}_{m}"] = dict(cpu=float(a.mean()), gpu=float(b.mean()), z=float((a.mean() - b.mean()) / se) if se > 0 else 0.0)
    meta["cpu_vs_gpu_phase2_tau1"] = agree
    meta["max_abs_z"] = max(abs(v["z"]) for v in agree.values())
    meta["checkpoints_unchanged"] = True  # asserted by every shard before and after sampling
    json.dump(meta, open(f"{OUT}/run_checks.json", "w"), indent=1)
    log(f"merged {len(shards)} shards; CPU vs GPU Phase 2 tau=1 agreement: max |z| = {meta['max_abs_z']:.2f}; MERGE_DONE")


def analyze():
    rng = np.random.default_rng(0)
    Z = np.load(f"{OUT}/summaries.npz")
    meta = json.load(open(f"{OUT}/run_checks.json"))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    def paired(d):
        d = np.asarray(d, float); n = len(d)
        b = d[rng.integers(0, n, (N_BOOT, n))].mean(1)
        f = (rng.choice([-1.0, 1.0], (N_BOOT, n)) * d).mean(1)
        sd = d.std(ddof=1)  # d_z: paired standardized effect over images (not comparable with the participant-level Hedges' g)
        return dict(mean=float(d.mean()), ci=ci(b), p=float((np.sum(np.abs(f) >= abs(d.mean()) - 1e-15) + 1) / (N_BOOT + 1)), n_pos=int((d > 0).sum()), n=n, dz=float(d.mean() / sd) if sd > 0 else 0.0)
    def holm(rows, key="p", out="p_holm"):
        o = sorted(range(len(rows)), key=lambda i: rows[i][key]); run_ = 0.0
        for r, i in enumerate(o):
            run_ = max(run_, min(1.0, (len(rows) - r) * rows[i][key])); rows[i][out] = run_
    res = {}
    prim = []
    for ds in ("gaze4asd", "saliency4asd_test"):
        G = lambda s, m: Z[f"{ds}_{s}_{m}"]
        contrasts, match = [], []
        for m, lab in METRICS.items():
            Dr = Z[f"{ds}_real_ASD_{m}"] - Z[f"{ds}_real_TD_{m}"]
            P2r, P1r = G("P2_ASD", m) - G("P2_TD", m), G("P1_s0", m) - G("P1_s1", m)
            P2c, P1c = P2r.mean(0), P1r.mean(0)
            eff = paired(P2c - P1c)
            Bi = rng.integers(0, len(Dr), (N_BOOT, len(Dr)))
            gg = np.abs((Dr - P1c)[Bi].mean(1)) - np.abs((Dr - P2c)[Bi].mean(1))
            row = dict(dataset=ds, metric=m, label=lab, primary=m in PRIMARY, real=float(Dr.mean()), real_ci=ci(Dr[Bi].mean(1)), phase2=float(P2c.mean()), phase1_null=float(P1c.mean()),
                       conditioning_effect=eff, seeds_effect_pos=int(((P2r - P1r).mean(1) > 0).sum()), seeds_phase2_real_sign=int((np.sign(P2r.mean(1)) == np.sign(Dr.mean())).sum()),
                       phase2_share_of_real=float(P2c.mean() / Dr.mean()) if Dr.mean() else None, same_direction=bool(np.sign(P2c.mean()) == np.sign(Dr.mean())),
                       gap_phase1=float(abs((Dr - P1c).mean())), gap_phase2=float(abs((Dr - P2c).mean())), gap_gain=float(abs((Dr - P1c).mean()) - abs((Dr - P2c).mean())), gap_gain_ci=ci(gg),
                       r_phase2=float(np.corrcoef(Dr, P2c)[0, 1]), r_phase1=float(np.corrcoef(Dr, P1c)[0, 1]))
            row["p"] = eff["p"]; contrasts.append(row)
            if m in PRIMARY:
                prim.append(row)
            for g, s2, s1 in (("ASD", "P2_ASD", "P1_s0"), ("TD", "P2_TD", "P1_s1")):
                R = Z[f"{ds}_real_{g}_{m}"]
                e1, e2 = np.abs(G(s1, m).mean(0) - R), np.abs(G(s2, m).mean(0) - R)
                t = paired(e1 - e2)
                match.append(dict(dataset=ds, metric=m, label=lab, group=g, real=float(R.mean()), phase1=float(G(s1, m).mean()), phase2=float(G(s2, m).mean()), mae_phase1=float(e1.mean()), mae_phase2=float(e2.mean()),
                                  gain=t, rel_pct=float(100 * (e1 - e2).mean() / e1.mean()), seeds_favour_phase2=int((np.abs(G(s1, m) - R).mean(1) - np.abs(G(s2, m) - R).mean(1) > 0).sum()), p=t["p"]))
        holm(contrasts); holm(match)
        res[ds] = dict(contrasts=contrasts, match_to_real=match)
    holm(prim, out="p_holm_primary")
    # MultiMatch: does real ASD (TD) resemble ASD- (TD-) conditioned samples more than the other condition, beyond seed noise?
    mmr = list(csv.DictReader(open(f"{OUT}/multimatch_per_image.csv", encoding="utf-8")))
    imgs = sorted({r["image_id"] for r in mmr}, key=lambda s: int(s.replace("top_image_", "").replace(".png", "")))
    def cube(t, s):
        c = np.full((N_REP, len(imgs), len(MM)), np.nan)
        for r in mmr:
            if r["target"] == t and r["set"] == s:
                c[int(r["rep"]), imgs.index(r["image_id"])] = [float(r[k]) for k in MM]
        return np.nanmean(c, 0)
    mmres = []
    for tgt, own2, oth2, own1, oth1 in (("ASD", "P2_ASD", "P2_TD", "P1_s0", "P1_s1"), ("TD", "P2_TD", "P2_ASD", "P1_s1", "P1_s0")):
        d2, d1 = cube(tgt, own2) - cube(tgt, oth2), cube(tgt, own1) - cube(tgt, oth1)
        for q, k in enumerate(MM):
            mmres.append(dict(target=tgt, metric=k, phase2_own_minus_other=paired(d2[:, q]), phase1_seed_null=paired(d1[:, q]), conditioning_effect=paired(d2[:, q] - d1[:, q]), p=0.0))
            mmres[-1]["p"] = mmres[-1]["conditioning_effect"]["p"]
    holm(mmres)
    # tau=2 reference (E1b): same metric code on the saved tau=2 scanpaths; Phase 1 there is one independent-seed set
    T2 = np.load("training/runs/research_completion/e1b_tau2_reference_v1/summaries_tau2_clipped.npz")
    t2c, t2m = [], []
    for m, lab in METRICS.items():
        Dr = Z[f"gaze4asd_real_ASD_{m}"] - Z[f"gaze4asd_real_TD_{m}"]
        assert np.allclose(Dr, T2[f"real_ASD_{m}"] - T2[f"real_TD_{m}"], equal_nan=True)
        c2 = (T2[f"clip_P2_ASD_{m}"] - T2[f"clip_P2_TD_{m}"]).mean(0)
        t2c.append(dict(metric=m, real=float(Dr.mean()), phase2_contrast=paired(c2), share=float(c2.mean() / Dr.mean()) if Dr.mean() else None))
        for g, s2 in (("ASD", "P2_ASD"), ("TD", "P2_TD")):
            R = Z[f"gaze4asd_real_{g}_{m}"]
            e1, e2 = np.abs(T2[f"clip_P1_{m}"].mean(0) - R), np.abs(T2[f"clip_{s2}_{m}"].mean(0) - R)
            t = paired(e1 - e2)
            t2m.append(dict(metric=m, group=g, mae_phase1=float(e1.mean()), mae_phase2=float(e2.mean()), gain=t, rel_pct=float(100 * (e1 - e2).mean() / e1.mean()), p=t["p"]))
    holm(t2m)
    for r in res["gaze4asd"]["contrasts"]:
        r["phase2_contrast_test"] = paired((Z[f"gaze4asd_P2_ASD_{r['metric']}"] - Z[f"gaze4asd_P2_TD_{r['metric']}"]).mean(0))
    out = dict(design="Common random numbers; see EXPERIMENT_PLAN.md E1", primary=prim, results=res, multimatch=mmres, tau2_reference=dict(contrasts=t2c, match_to_real=t2m), checks=meta)
    json.dump(out, open(f"{OUT}/e1_results.json", "w"), indent=1)
    print("checks:", json.dumps({k: v for k, v in meta.items() if k in ("max_abs_z", "checkpoints_unchanged")}), "off-image:", json.dumps(meta["gaze4asd"]["off_image"]))
    for ds in res:
        print(f"\n=== {ds} contrasts (ASD-TD); conditioning effect = Phase2 contrast - Phase1 seed-null contrast")
        for r in res[ds]["contrasts"]:
            e = r["conditioning_effect"]
            print(f"{'*' if r['primary'] else ' '}{r['metric']:22s} real={r['real']:+.4f}{[round(x, 4) for x in r['real_ci']]} P2={r['phase2']:+.4f} P1null={r['phase1_null']:+.4f} eff={e['mean']:+.4f}{[round(x, 4) for x in e['ci']]} dz={e['dz']:+.2f} p={e['p']:.4f} holm={r['p_holm']:.4f} {'holmP=' + str(round(r.get('p_holm_primary', 9), 4)) if r['primary'] else ''} seeds+{r['seeds_effect_pos']}/10 share={r['phase2_share_of_real']} gap {r['gap_phase1']:.4f}->{r['gap_phase2']:.4f} gain={r['gap_gain']:+.4f}{[round(x, 4) for x in r['gap_gain_ci']]} r={r['r_phase2']:.2f}/{r['r_phase1']:.2f}")
        print(f"--- {ds} match to real (MAE; + favours Phase 2)")
        for r in res[ds]["match_to_real"]:
            print(f"  {r['group']:3s} {r['metric']:22s} real={r['real']:.4f} P1={r['phase1']:.4f} P2={r['phase2']:.4f} MAE {r['mae_phase1']:.4f}->{r['mae_phase2']:.4f} gain={r['gain']['mean']:+.4f}{[round(x, 4) for x in r['gain']['ci']]} dz={r['gain']['dz']:+.2f} ({r['rel_pct']:+.1f}%) holm={r['p_holm']:.4f} seeds {r['seeds_favour_phase2']}/10")
    print("\n--- MultiMatch own-other (Gaze4ASD)")
    for r in mmres:
        print(f"  {r['target']:3s} {r['metric'][3:]:9s} P2 {r['phase2_own_minus_other']['mean']:+.4f} P1null {r['phase1_seed_null']['mean']:+.4f} effect {r['conditioning_effect']['mean']:+.4f}{[round(x, 4) for x in r['conditioning_effect']['ci']]} dz={r['conditioning_effect']['dz']:+.2f} holm={r['p_holm']:.4f}")
    print("\n--- tau=2 reference (Gaze4ASD, clipped; Phase 1 = one independent-seed set)")
    for r in t2c:
        e = r["phase2_contrast"]; e1r = [x for x in res["gaze4asd"]["contrasts"] if x["metric"] == r["metric"]][0]
        print(f"  {r['metric']:22s} real={r['real']:+.4f} P2 contrast tau2={e['mean']:+.4f}{[round(x, 4) for x in e['ci']]} share={r['share']} | tau1={e1r['phase2']:+.4f} share={e1r['phase2_share_of_real']}")
    for r in t2m:
        e1r = [x for x in res["gaze4asd"]["match_to_real"] if x["metric"] == r["metric"] and x["group"] == r["group"]][0]
        print(f"  {r['group']:3s} {r['metric']:22s} tau2 MAE {r['mae_phase1']:.4f}->{r['mae_phase2']:.4f} gain={r['gain']['mean']:+.4f}{[round(x, 4) for x in r['gain']['ci']]} ({r['rel_pct']:+.1f}%) holm={r['p_holm']:.4f} | tau1 gain={e1r['gain']['mean']:+.4f} ({e1r['rel_pct']:+.1f}%) holm={e1r['p_holm']:.4f}")
    print("ANALYZE_DONE")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    else:
        {"merge": merge, "analyze": analyze}[sys.argv[1]]()
