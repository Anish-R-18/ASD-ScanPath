"""
External Gaze4ASD evaluation of the FROZEN Phase-2 ASD/TD-conditioned model.
NO retraining/tuning. Policy A is primary; Policy B is a sensitivity check.
Outputs under training/runs/external_gaze4asd/.
"""
from __future__ import annotations
import sys, os, json, csv, random
sys.path.insert(0, "."); sys.path.insert(0, "tppgaze")
import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import wasserstein_distance
from tppgaze.utils import get_preprocess, pad_sequence
from training import model_utils as mu
from training.conditioned import ConditionedTPPGaze
from training.generate_cond import sample_conditioned, to_scanpaths
from training import metrics as M
from training.dataset import load_scanpaths

BASE_CFG = "tppgaze/data/config.yaml"
P2 = "training/runs/phase2_embedding_only/best_model_conditioned.pth"
IMG_DIR = "external_data/Gaze4ASD/Image Stimuli"
EXT = {"A": "external_data/Gaze4ASD_processed/gaze4asd_policy_a.json",
       "B": "external_data/Gaze4ASD_processed/gaze4asd_policy_b.json"}
SAL_TEST = "training/runs/phase2_embedding_only/test_eval_phase2.json"
OUT = "training/runs/external_gaze4asd"
N_SIM, TEMP, SEED, MAX_PAIRS = 20, 2.0, 0, 30
GID = {"ASD": 0, "TD": 1}
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"

# ---- frozen-model normalization = Saliency4ASD train-split stats ----
split = json.load(open("training/runs/baseline/split.json")); train_ids = set(map(int, split["train"]))
sal = load_scanpaths("processed_data/asd_scanpaths.json", "processed_data/td_scanpaths.json")
mean, std = mu.compute_log_inter_time_stats([s for s in sal if s["image_id"] in train_ids])

base, _ = mu.build_model(BASE_CFG, "training/runs/baseline/best_model.pth", device, mean, std, used_cfg_out="/tmp/x.yaml")
cond_dim = int(OmegaConf.load(BASE_CFG).metadata.size + OmegaConf.load(BASE_CFG).context.size)
cmodel = ConditionedTPPGaze(base, cond_dim).to(device)
cmodel.load_state_dict(torch.load(P2, map_location=device), strict=True)
mu._fix_unregistered_device(cmodel.base, device); cmodel.eval()

prep = get_preprocess()
img_cache, dim_cache, meta_cache = {}, {}, {}
def get_img(iid):
    if iid not in img_cache:
        im = Image.open(os.path.join(IMG_DIR, iid)).convert("RGB")
        dim_cache[iid] = im.size; img_cache[iid] = prep(im)
    return img_cache[iid]
@torch.no_grad()
def meta_for(iids):
    out = []
    for iid in iids:
        if iid not in meta_cache:
            meta_cache[iid] = cmodel.base.get_metadata(get_img(iid).unsqueeze(0).to(device))
        out.append(meta_cache[iid])
    return torch.cat(out, 0)
def dims(iid):
    if iid not in dim_cache:
        get_img(iid)
    return dim_cache[iid]
def denorm(spn, W, H):
    return np.stack([(spn[:, 0] + 1) / 2 * (W - 1), (spn[:, 1] + 1) / 2 * (H - 1)], axis=1)


@torch.no_grad()
def nll_eval(records, bs=128):
    cmodel.eval(); rows = []
    for st in range(0, len(records), bs):
        b = records[st:st + bs]
        marks = pad_sequence([torch.tensor([f[:2] for f in r["scanpath"]], dtype=torch.float32) for r in b])
        inter = pad_sequence([torch.tensor([f[2] for f in r["scanpath"]], dtype=torch.float32) for r in b])
        lens = torch.tensor([len(r["scanpath"]) for r in b]); L = marks.shape[1]
        mask = (torch.arange(L)[None, :] < lens[:, None]).float()
        gids = torch.tensor([GID[r["group"]] for r in b])
        marks, inter, mask, gids = marks.to(device), inter.to(device), mask.to(device), gids.to(device)
        meta = meta_for([r["image_id"] for r in b])
        feats = cmodel.base.get_features(inter, marks)
        ctx = cmodel.base.merge.forward([cmodel.base.get_context(feats, True), meta]) + cmodel.group_emb(gids).unsqueeze(1)
        it = torch.where(mask.bool(), inter, torch.ones_like(inter))
        lpt = cmodel.base.inter_time_dist.forward(ctx, None).log_prob(it) * mask
        lpp = cmodel.base.marks_dist.forward(ctx, None).log_prob(marks) * mask
        s1, s2, ne = lpt.sum(1), lpp.sum(1), mask.sum(1)
        for i, r in enumerate(b):
            rows.append((r["group"], r["participant"], float(s1[i]), float(s2[i]), float(ne[i])))
    return rows

def agg_nll(rows, grp=None):
    sel = [r for r in rows if grp is None or r[0] == grp]
    s1 = sum(r[2] for r in sel); s2 = sum(r[3] for r in sel); ne = max(sum(r[4] for r in sel), 1.0)
    return {"nll": -(s1 + s2) / ne, "nll_duration": -s1 / ne, "nll_position": -s2 / ne,
            "n_trials": len(sel), "n_events": int(ne)}

def participant_macro_nll(rows, grp):
    pids = sorted({r[1] for r in rows if r[0] == grp})
    per = []
    for pid in pids:
        sel = [r for r in rows if r[1] == pid]
        ne = sum(r[4] for r in sel)
        if ne > 0:
            per.append(-(sum(r[2] for r in sel) + sum(r[3] for r in sel)) / ne)
    return {"n_participants": len(pids), "macro_nll_mean": float(np.mean(per)),
            "macro_nll_std": float(np.std(per))}


# ============ NLL EVAL: Policy A (primary) + Policy B (sensitivity) ============
nll_results = {}
rowsA = None
for pol in ("A", "B"):
    recs = json.load(open(EXT[pol]))
    rows = nll_eval(recs)
    if pol == "A":
        rowsA = rows
    nll_results[pol] = {
        "overall": agg_nll(rows), "ASD": agg_nll(rows, "ASD"), "TD": agg_nll(rows, "TD"),
        "participant_macro_ASD": participant_macro_nll(rows, "ASD"),
        "participant_macro_TD": participant_macro_nll(rows, "TD"),
    }
sal_test = json.load(open(SAL_TEST))
nll_results["saliency4asd_phase2_test"] = {"overall": sal_test["phase2"]["overall"],
                                           "ASD": sal_test["phase2"]["ASD"], "TD": sal_test["phase2"]["TD"]}
json.dump(nll_results, open(os.path.join(OUT, "external_nll.json"), "w"), indent=2)


# ================= BEHAVIORAL (Policy A primary) =================
recsA = json.load(open(EXT["A"]))
img_ids = sorted({r["image_id"] for r in recsA}, key=lambda s: int(s.replace("top_image_", "").replace(".png", "")))
pops = {"real_ASD": [], "real_TD": [], "gen_ASD": [], "gen_TD": []}
by_img = {k: {} for k in pops}
# real
for r in recsA:
    W, H = dims(r["image_id"]); sp = np.asarray(r["scanpath"], float)
    rec = {"image_id": r["image_id"], "participant": r["participant"], "xy": denorm(sp, W, H), "dur": sp[:, 2]}
    key = "real_ASD" if r["group"] == "ASD" else "real_TD"
    pops[key].append(rec); by_img[key].setdefault(r["image_id"], []).append(rec)
# generated
for iid in img_ids:
    W, H = dims(iid); img = get_img(iid)
    for gid, name in [(0, "gen_ASD"), (1, "gen_TD")]:
        it, mk, ms = sample_conditioned(cmodel, img, gid, t_end=3.0, n_sim=N_SIM, temperature=TEMP)
        for spn in to_scanpaths(it, mk, ms):
            if len(spn) < 1:
                continue
            rec = {"image_id": iid, "participant": None, "xy": denorm(spn, W, H), "dur": spn[:, 2]}
            pops[name].append(rec); by_img[name].setdefault(iid, []).append(rec)
print("population sizes:", {k: len(v) for k, v in pops.items()})

# per-scanpath metrics (no AOI for Gaze4ASD)
rows = []
for pop, recs in pops.items():
    for r in recs:
        W, H = dims(r["image_id"]); diag = float(np.hypot(W, H))
        m = M.scanpath_metrics(r["xy"], r["dur"], diag)
        rows.append({"population": pop, "image_id": r["image_id"], "participant": r["participant"], **m})
mkeys = [k for k in rows[0] if k not in ("population", "image_id", "participant")]
with open(os.path.join(OUT, "external_per_scanpath_metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["population", "image_id", "participant"] + mkeys); w.writeheader(); w.writerows(rows)

def arr(pop, k):
    return np.array([r[k] for r in rows if r["population"] == pop and np.isfinite(r[k])], float)

agg = {}
for pop in pops:
    agg[pop] = {"n_scanpaths": len(pops[pop])}
    for k in mkeys:
        a = arr(pop, k)
        agg[pop][k] = {"mean": float(np.mean(a)) if a.size else None, "median": float(np.median(a)) if a.size else None,
                       "std": float(np.std(a)) if a.size else None}
json.dump(agg, open(os.path.join(OUT, "external_aggregate_summary.json"), "w"), indent=2)

# distribution distances
dkeys = ["n_fix", "dur_mean", "total_dur", "sacc_amp_norm_mean", "dispersion", "revisit_rate"]
comps = [("real_ASD", "real_TD"), ("gen_ASD", "gen_TD"), ("real_ASD", "gen_ASD"), ("real_TD", "gen_TD")]
wd = {}
for a, b in comps:
    wd[f"{a}__vs__{b}"] = {k: (float(wasserstein_distance(arr(a, k), arr(b, k))) if arr(a, k).size and arr(b, k).size else None) for k in dkeys}
json.dump(wd, open(os.path.join(OUT, "external_distribution_distances.json"), "w"), indent=2)

# MultiMatch
def sample_pairs(la, lb, same=False, cap=MAX_PAIRS):
    pairs = [(la[i], lb[j]) for i in range(len(la)) for j in range(len(lb)) if not (same and j <= i)]
    random.shuffle(pairs); return pairs[:cap]
mm_types = {"RvR_ASD": ("real_ASD", "real_ASD", True), "RvR_TD": ("real_TD", "real_TD", True),
            "RvG_ASD": ("real_ASD", "gen_ASD", False), "RvG_TD": ("real_TD", "gen_TD", False),
            "GenASD_vs_GenTD": ("gen_ASD", "gen_TD", False)}
mm_res = {}
for name, (pa, pb, same) in mm_types.items():
    acc = {k: [] for k in ["mm_shape", "mm_length", "mm_direction", "mm_position", "mm_duration"]}
    for iid in img_ids:
        la, lb = by_img[pa].get(iid, []), by_img[pb].get(iid, [])
        if not la or not lb:
            continue
        W, H = dims(iid); diag = float(np.hypot(W, H))
        for ra, rb in sample_pairs(la, lb, same=same):
            mm = M.multimatch(ra["xy"], ra["dur"], rb["xy"], rb["dur"], diag)
            if mm:
                for k in acc:
                    acc[k].append(mm[k])
    mm_res[name] = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}; mm_res[name]["n_pairs"] = len(acc["mm_shape"])
json.dump(mm_res, open(os.path.join(OUT, "external_similarity_multimatch.json"), "w"), indent=2)

# participant-level real summary (mean per participant, then group mean/std)
def participant_summary(pop):
    byp = {}
    for r in rows:
        if r["population"] != pop or r["participant"] is None:
            continue
        byp.setdefault(r["participant"], []).append(r)
    out = {}
    for k in ["n_fix", "dur_mean", "sacc_amp_norm_mean", "dispersion", "revisit_rate"]:
        pm = [np.nanmean([rr[k] for rr in rs]) for rs in byp.values()]
        pm = [v for v in pm if np.isfinite(v)]
        out[k] = {"participant_mean": float(np.mean(pm)), "participant_std": float(np.std(pm))}
    out["n_participants"] = len(byp)
    return out
part = {"real_ASD": participant_summary("real_ASD"), "real_TD": participant_summary("real_TD")}
json.dump(part, open(os.path.join(OUT, "external_participant_summary.json"), "w"), indent=2)

# ---- text summary ----
def f(pop, k):
    v = agg[pop][k]["mean"]; return f"{v:.3f}" if v is not None else "n/a"
T = ["GAZE4ASD EXTERNAL EVALUATION (frozen Phase-2 model)", "",
     "== NLL (per event) ==",
     f"{'policy/group':22}{'NLL':>10}{'dur':>10}{'pos':>10}{'trials':>9}"]
for pol in ("A", "B"):
    for g in ("overall", "ASD", "TD"):
        d = nll_results[pol][g]
        T.append(f"{'Policy'+pol+' '+g:22}{d['nll']:>10.4f}{d['nll_duration']:>10.4f}{d['nll_position']:>10.4f}{d['n_trials']:>9}")
s = nll_results["saliency4asd_phase2_test"]
for g in ("overall", "ASD", "TD"):
    T.append(f"{'Sal4ASD test '+g:22}{s[g]['nll']:>10.4f}{s[g]['nll_duration']:>10.4f}{s[g]['nll_position']:>10.4f}{s[g]['n_scanpaths']:>9}")
T += ["", "participant-macro NLL (Policy A):",
      f"  ASD: {nll_results['A']['participant_macro_ASD']}",
      f"  TD : {nll_results['A']['participant_macro_TD']}",
      "", "== Behavioral aggregate means (Policy A) ==",
      f"populations: " + ", ".join(f"{k}={len(v)}" for k, v in pops.items()),
      f"{'metric':22}{'real_ASD':>12}{'real_TD':>12}{'gen_ASD':>12}{'gen_TD':>12}"]
for k in mkeys:
    T.append(f"{k:22}{f('real_ASD',k):>12}{f('real_TD',k):>12}{f('gen_ASD',k):>12}{f('gen_TD',k):>12}")
T += ["", "MultiMatch (mean; 1=identical):",
      f"{'type':18}{'shape':>9}{'length':>9}{'direct.':>9}{'position':>9}{'duration':>9}{'pairs':>8}"]
for name, d in mm_res.items():
    T.append(f"{name:18}{d['mm_shape']:>9.3f}{d['mm_length']:>9.3f}{d['mm_direction']:>9.3f}{d['mm_position']:>9.3f}{d['mm_duration']:>9.3f}{d['n_pairs']:>8}")
open(os.path.join(OUT, "external_summary.txt"), "w").write("\n".join(T) + "\n")
print("\n".join(T))

# ---- plots ----
def draw(ax, xy, dur, c):
    if len(xy):
        ax.plot(xy[:, 0], xy[:, 1], "-", color=c, alpha=0.6, lw=1.5)
        ax.scatter(xy[:, 0], xy[:, 1], s=20 + 600 * dur, c=c, alpha=0.5, edgecolors="k", linewidths=0.5)
rep = img_ids[:4]
fig, axs = plt.subplots(len(rep), 2, figsize=(10, 4 * len(rep)))
for r, iid in enumerate(rep):
    im = Image.open(os.path.join(IMG_DIR, iid)).convert("RGB")
    for c, (pop, col, t) in enumerate([("gen_ASD", "red", "ASD"), ("gen_TD", "blue", "TD")]):
        ax = axs[r, c]; ax.imshow(im); ax.set_title(f"{iid} gen {t}"); ax.axis("off")
        if by_img[pop].get(iid):
            draw(ax, by_img[pop][iid][0]["xy"], by_img[pop][iid][0]["dur"], col)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_representative_scanpaths.png"), dpi=110); plt.close(fig)

def panel(fname, groups, labels, colors, title):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for ax, k in zip(axs.ravel(), ["n_fix", "dur_mean", "sacc_amp_norm_mean", "dispersion"]):
        for g, lab, col in zip(groups, labels, colors):
            a = arr(g, k)
            if a.size:
                ax.hist(a, bins=25, alpha=0.5, label=lab, color=col, density=True)
        ax.set_title(k); ax.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=110); plt.close(fig)
panel("fig_asd_vs_td_generated.png", ["gen_ASD", "gen_TD"], ["gen ASD", "gen TD"], ["red", "blue"], "Gaze4ASD Generated ASD vs TD")
panel("fig_real_vs_generated_ASD.png", ["real_ASD", "gen_ASD"], ["real ASD", "gen ASD"], ["darkred", "orange"], "Gaze4ASD Real vs Gen (ASD)")
panel("fig_real_vs_generated_TD.png", ["real_TD", "gen_TD"], ["real TD", "gen TD"], ["darkblue", "cyan"], "Gaze4ASD Real vs Gen (TD)")

print("\nARTIFACTS:", sorted(os.listdir(OUT)))
print("EXTERNAL_DONE")
