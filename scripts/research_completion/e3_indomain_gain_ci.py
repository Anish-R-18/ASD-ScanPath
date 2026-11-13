"""
E3 (research completion, v1): image-level uncertainty for the Saliency4ASD held-out Phase 2 gain.
No training; checkpoints read-only; CPU; evaluation only (no selection).

  python scripts/research_completion/e3_indomain_gain_ci.py
"""
from __future__ import annotations
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")  # repository root
os.chdir(ROOT); sys.path[:0] = [ROOT, ROOT + "/tppgaze"]
import numpy as np
import torch
from omegaconf import OmegaConf
from training import model_utils as mu
from training.conditioned import ConditionedTPPGaze
from training.dataset import load_scanpaths
from training.external_gaze4asd_loader import StimulusBank, collate_records
from training.eval_external_v2 import BASE_CFG, sha256

OUT = "training/runs/research_completion/e3_indomain_gain_ci_v1"
PHASE1, PHASE2 = "training/runs/baseline/best_model.pth", "training/runs/phase2_embedding_only/best_model_conditioned.pth"
if os.path.exists(OUT):
    raise SystemExit("refusing to overwrite " + OUT)
os.makedirs(OUT)
torch.set_num_threads(os.cpu_count()); dev = torch.device("cpu")
h1, h2 = sha256(PHASE1), sha256(PHASE2)
st = json.load(open("training/runs/phase2_embedding_only/summary_full.json"))
cfg = OmegaConf.load(BASE_CFG); cdim = int(cfg.metadata.size + cfg.context.size)
b1, _ = mu.build_model(BASE_CFG, PHASE1, dev, st["mean_log_inter_time"], st["std_log_inter_time"], used_cfg_out=OUT + "/cfg1.yaml")
b2, _ = mu.build_model(BASE_CFG, PHASE1, dev, st["mean_log_inter_time"], st["std_log_inter_time"], used_cfg_out=OUT + "/cfg2.yaml")
p2 = ConditionedTPPGaze(b2, cdim).to(dev); p2.load_state_dict(torch.load(PHASE2, map_location=dev), strict=True)
mu._fix_unregistered_device(p2.base, dev); b1.eval(); p2.eval()
bank = StimulusBank(OmegaConf.load("configs/phase2_embedding_only.yaml").images_dir)
test = set(map(int, json.load(open("training/runs/baseline/split.json"))["test"]))
recs = [s for s in load_scanpaths("processed_data/asd_scanpaths.json", "processed_data/td_scanpaths.json") if s["image_id"] in test]
meta = {}
@torch.no_grad()
def score(cond):
    out = []
    for s in range(0, len(recs), 128):
        b = recs[s:s + 128]; c = collate_records(b, dev)
        for i in {r["image_id"] for r in b}:
            if i not in meta:
                meta[i] = b1.get_metadata(bank.tensor(f"{i}.png").unsqueeze(0))
        md = torch.cat([meta[r["image_id"]] for r in b], 0)
        base = p2.base if cond else b1
        f = base.get_features(c["inter_times"], c["marks"])
        ctx = base.merge.forward([base.get_context(f, True), md])
        if cond:
            ctx = ctx + p2.group_emb(c["group_ids"]).unsqueeze(1)
        it = torch.where(c["mask"].bool(), c["inter_times"], torch.ones_like(c["inter_times"]))
        lt = base.inter_time_dist.forward(ctx, None).log_prob(it) * c["mask"]; lp = base.marks_dist.forward(ctx, None).log_prob(c["marks"]) * c["mask"]
        out += [(float(c["mask"][j].sum()), float(lt[j].sum()), float(lp[j].sum())) for j in range(len(b))]
    return np.array(out)
s1, s2 = score(False), score(True)
T = json.load(open("training/runs/phase2_embedding_only/test_eval_phase2.json"))
n = s1[:, 0]; nll = lambda s, sel: float(-(s[sel, 1] + s[sel, 2]).sum() / n[sel].sum())
allsel = np.ones(len(recs), bool)
gate = dict(phase1=nll(s1, allsel), phase2=nll(s2, allsel), saved_phase1=T["phase1"]["overall"]["nll"], saved_phase2=T["phase2"]["overall"]["nll"])
gate["max_abs_diff"] = max(abs(gate["phase1"] - gate["saved_phase1"]), abs(gate["phase2"] - gate["saved_phase2"]))
print("gate", json.dumps(gate))
if gate["max_abs_diff"] > 1e-4:
    raise SystemExit("GATE FAILED")
rng = np.random.default_rng(0)
grp = np.array([r["group"] for r in recs]); img = np.array([r["image_id"] for r in recs])
res = dict(gate=gate)
for g, sel in (("overall", allsel), ("ASD", grp == "ASD"), ("TD", grp == "TD")):
    for k, (a, b) in {"joint": (s1[:, 1] + s1[:, 2], s2[:, 1] + s2[:, 2]), "duration": (s1[:, 1], s2[:, 1]), "position": (s1[:, 2], s2[:, 2])}.items():
        d = b - a
        us = np.unique(img[sel]); ix = {u: i for i, u in enumerate(us)}
        nn, dd = np.zeros(len(us)), np.zeros(len(us))
        np.add.at(nn, [ix[u] for u in img[sel]], n[sel]); np.add.at(dd, [ix[u] for u in img[sel]], d[sel])
        bi = rng.integers(0, len(us), (10000, len(us)))
        bs = dd[bi].sum(1) / nn[bi].sum(1)
        fl = (rng.choice([-1.0, 1.0], (10000, len(us))) * dd).sum(1) / nn.sum()
        obs = dd.sum() / nn.sum()
        res[f"{g}_{k}"] = dict(phase1=float(-a[sel].sum() / n[sel].sum()), phase2=float(-b[sel].sum() / n[sel].sum()), gain=float(obs), image_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                               images_favouring_phase2=int((dd > 0).sum()), images=len(us), signflip_p=float((np.sum(np.abs(fl) >= abs(obs) - 1e-15) + 1) / 10001))
        print(f"{g:8s}{k:9s} P1={res[f'{g}_{k}']['phase1']:+.4f} P2={res[f'{g}_{k}']['phase2']:+.4f} gain={obs:+.4f} img{[round(x, 4) for x in res[f'{g}_{k}']['image_ci']]} {res[f'{g}_{k}']['images_favouring_phase2']}/{len(us)} p={res[f'{g}_{k}']['signflip_p']:.4f}")
assert sha256(PHASE1) == h1 and sha256(PHASE2) == h2
res["checkpoints_unchanged"] = True
json.dump(res, open(OUT + "/e3_results.json", "w"), indent=1)
print("E3_DONE")
