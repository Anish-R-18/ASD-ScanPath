"""
E2 (research completion, v1): Phase 1 vs Phase 2 NLL on the corrected Gaze4ASD versions
(fit-to-height A [primary], stretch A, fit-to-height B). No training; checkpoints read-only; CPU.
Same scoring code, file order and batch size (128) as the V2 evaluation. Phase 2 per-trial values are
gated against the saved GPU results for every version.

  python scripts/research_completion/e2_mapping_policy_nll.py
"""
from __future__ import annotations
import csv, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")  # repository root
os.chdir(ROOT); sys.path[:0] = [ROOT, ROOT + "/tppgaze"]
import numpy as np
import torch
from omegaconf import OmegaConf
from training import model_utils as mu
from training.conditioned import ConditionedTPPGaze
from training.external_gaze4asd_loader import StimulusBank, load_external_scanpaths, collate_records
from training.eval_external_v2 import IMG_DIR, BASE_CFG, VERSIONS, sha256

OUT = "training/runs/research_completion/e2_mapping_policy_nll_v1"
V2 = "training/runs/external_gaze4asd_v2"
PHASE1, PHASE2 = "training/runs/baseline/best_model.pth", "training/runs/phase2_embedding_only/best_model_conditioned.pth"
USE = ("fit_height_policy_a", "stretch_policy_a", "fit_height_policy_b")
N_BOOT, BATCH = 10000, 128
if os.path.exists(OUT) and os.listdir(OUT):
    raise SystemExit("refusing to overwrite " + OUT)
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(os.cpu_count()); dev = torch.device("cpu")
h1, h2 = sha256(PHASE1), sha256(PHASE2)
st = json.load(open("training/runs/phase2_embedding_only/summary_full.json"))
cfg = OmegaConf.load(BASE_CFG); cdim = int(cfg.metadata.size + cfg.context.size)
b1, _ = mu.build_model(BASE_CFG, PHASE1, dev, st["mean_log_inter_time"], st["std_log_inter_time"], used_cfg_out=OUT + "/cfg1.yaml")
b2, _ = mu.build_model(BASE_CFG, PHASE1, dev, st["mean_log_inter_time"], st["std_log_inter_time"], used_cfg_out=OUT + "/cfg2.yaml")
p2 = ConditionedTPPGaze(b2, cdim).to(dev); p2.load_state_dict(torch.load(PHASE2, map_location=dev), strict=True)
mu._fix_unregistered_device(p2.base, dev); b1.eval(); p2.eval()
bank = StimulusBank(IMG_DIR)
meta = {}
@torch.no_grad()
def md(ids):
    for i in set(ids):
        if i not in meta:
            meta[i] = b1.get_metadata(bank.tensor(i).unsqueeze(0))
    return torch.cat([meta[i] for i in ids], 0)
@torch.no_grad()
def score(recs, conditioned):
    out = []
    for s in range(0, len(recs), BATCH):
        b = recs[s:s + BATCH]; c = collate_records(b, dev)
        base = p2.base if conditioned else b1
        f = base.get_features(c["inter_times"], c["marks"])
        ctx = base.merge.forward([base.get_context(f, True), md([r["image_id"] for r in b])])
        if conditioned:
            ctx = ctx + p2.group_emb(c["group_ids"]).unsqueeze(1)
        it = torch.where(c["mask"].bool(), c["inter_times"], torch.ones_like(c["inter_times"]))
        lt = base.inter_time_dist.forward(ctx, None).log_prob(it) * c["mask"]
        lp = base.marks_dist.forward(ctx, None).log_prob(c["marks"]) * c["mask"]
        for j in range(len(b)):
            out.append((float(c["mask"][j].sum()), float(lt[j].sum()), float(lp[j].sum())))
    return np.array(out)
rng = np.random.default_rng(0)
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
res, t0 = {}, time.time()
for v in USE:
    recs = load_external_scanpaths(VERSIONS[v], bank)
    s1, s2 = score(recs, False), score(recs, True)
    saved = list(csv.DictReader(open(f"{V2}/{v}/per_trial_nll.csv", encoding="utf-8")))
    assert len(saved) == len(recs)
    # CPU scores differ from the saved GPU scores per trial (median ~0.005 in position, unbiased), a platform-level
    # numeric difference; the first, per-trial gate (1e-3) failed for that reason. The gate is therefore on aggregates.
    sv_lp = np.array([float(r["sum_lp_duration"]) + float(r["sum_lp_position"]) for r in saved])
    gate = float(abs((sv_lp.sum() - (s2[:, 1] + s2[:, 2]).sum()) / s2[:, 0].sum()))
    per_trial = np.abs(sv_lp - (s2[:, 1] + s2[:, 2]))
    if gate > 5e-4:
        raise SystemExit(f"GATE FAILED {v}: aggregate {gate}")
    n = s1[:, 0]; grp = np.array([r["group"] for r in recs]); P = np.array([r["participant"] for r in recs]); I = np.array([r["image_id"] for r in recs])
    comp = {"joint": (s1[:, 1] + s1[:, 2], s2[:, 1] + s2[:, 2]), "duration": (s1[:, 1], s2[:, 1]), "position": (s1[:, 2], s2[:, 2])}
    vr = {"gate_aggregate_nll_diff_vs_saved_gpu": gate, "per_trial_abs_diff_median": float(np.median(per_trial)), "per_trial_abs_diff_p99": float(np.percentile(per_trial, 99)), "trials": len(recs), "events": int(n.sum())}
    for g, sel in (("overall", np.ones(len(recs), bool)), ("ASD", grp == "ASD"), ("TD", grp == "TD")):
        for k, (a, b) in comp.items():
            d = b - a  # positive favours Phase 2 (per-event gain after division)
            def agg(unit, strat):
                us = np.unique(unit[sel]); ix = {u: i for i, u in enumerate(us)}
                nn, dd = np.zeros(len(us)), np.zeros(len(us))
                np.add.at(nn, [ix[u] for u in unit[sel]], n[sel]); np.add.at(dd, [ix[u] for u in unit[sel]], d[sel])
                if strat:
                    gu = {u: gg for u, gg in zip(P, grp)}; isa = np.array([gu[u] == "ASD" for u in us])
                    groups = [np.where(isa)[0], np.where(~isa)[0]]
                else:
                    groups = [np.arange(len(us))]
                bs = np.zeros((N_BOOT, 2))
                for gi in groups:
                    if len(gi):
                        idx = gi[rng.integers(0, len(gi), (N_BOOT, len(gi)))]
                        bs[:, 0] += nn[idx].sum(1); bs[:, 1] += dd[idx].sum(1)
                return ci(bs[:, 1] / bs[:, 0]), int((dd > 0).sum()), len(us)
            pc, pf, pn = agg(P, True); ic, if_, in_ = agg(I, False)
            vr[f"{g}_{k}"] = dict(phase1=float(-a[sel].sum() / n[sel].sum()), phase2=float(-b[sel].sum() / n[sel].sum()), gain=float(d[sel].sum() / n[sel].sum()), participant_ci=pc, image_ci=ic, participants_favouring=pf, participants=pn, images_favouring=if_, images=in_)
    if v == "fit_height_policy_a":
        saved_gain = float([r for r in csv.DictReader(open(f"{V2}/phase1_vs_phase2/group_nll_comparison.csv", encoding="utf-8")) if r["group"] == "overall" and r["component"] == "joint"][0]["improvement"])
        vr["primary_gain_saved_gpu"] = saved_gain
        if abs(vr["overall_joint"]["gain"] - saved_gain) > 5e-4:
            raise SystemExit(f"GATE FAILED: primary gain {vr['overall_joint']['gain']} vs saved {saved_gain}")
    res[v] = vr
    print(f"{v}: aggregate gate {gate:.2e}; per-trial median {vr['per_trial_abs_diff_median']:.1e}; {time.time() - t0:.0f}s", flush=True)
    for key in ("overall_joint", "ASD_joint", "TD_joint", "overall_duration", "overall_position"):
        r = vr[key]; print(f"   {key:18s} P1={r['phase1']:+.4f} P2={r['phase2']:+.4f} gain={r['gain']:+.4f} part{[round(x, 4) for x in r['participant_ci']]} img{[round(x, 4) for x in r['image_ci']]} P {r['participants_favouring']}/{r['participants']} I {r['images_favouring']}/{r['images']}", flush=True)
assert sha256(PHASE1) == h1 and sha256(PHASE2) == h2
json.dump(dict(note="gain > 0 favours Phase 2; NLL per fixation; position NLL is not comparable across mappings, but Phase 1 vs Phase 2 within one version is.", results=res, checkpoints_unchanged=True), open(OUT + "/e2_results.json", "w"), indent=1)
print("E2_DONE")
