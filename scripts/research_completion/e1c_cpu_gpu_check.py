"""
E1c (research completion, v1): agreement of E1's CPU Phase 2 tau=1 samples with the saved GPU tau=1 run.
Same seeds, but the CPU and GPU random streams differ, so agreement can only be distributional.
Per-image means over 10 seeds are paired by image (seeds differ per image, so images are independent);
sign-flip test over 30 images. Also compares the Phase 2 ASD-TD contrasts reported in the paper (GPU) with E1 (CPU).
v2 adds the direct test of the contrast difference (image-paired sign-flip, and seed-level z with both runs' seed noise);
v1 reported only the GPU seed SD, which understates the uncertainty of the difference.

  python scripts/research_completion/e1c_cpu_gpu_check.py
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")  # repository root
os.chdir(ROOT)
import numpy as np

OUT = "training/runs/research_completion/e1c_cpu_gpu_check_v2"
METRICS = ["dur_mean", "n_fix", "fix_rate_per_s", "total_dur", "sacc_amp_norm_mean", "sacc_horizontal_frac", "dispersion", "revisit_rate", "sacc_dir_resultant"]
if os.path.exists(OUT + "/cpu_gpu_check.json"):
    raise SystemExit("refusing to overwrite " + OUT)
os.makedirs(OUT, exist_ok=True)
A = np.load("training/runs/research_completion/e1_phase_tau1_v2/summaries.npz")
G = np.load("training/runs/external_gaze4asd_v2/temperature_tau1/raw/summaries.npz")
rng = np.random.default_rng(1)
rows = []
for g, s in (("ASD", "P2_ASD"), ("TD", "P2_TD")):
    for m in METRICS:
        d = A[f"gaze4asd_{s}_{m}"].mean(0) - G[f"clip_tau1_{g}_{m}"].mean(0)
        f = (rng.choice([-1.0, 1.0], (20000, len(d))) * d).mean(1)
        rows.append(dict(group=g, metric=m, cpu=float(A[f"gaze4asd_{s}_{m}"].mean()), gpu=float(G[f"clip_tau1_{g}_{m}"].mean()), diff=float(d.mean()),
                         rel_pct=float(100 * d.mean() / G[f"clip_tau1_{g}_{m}"].mean()), z_image=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))),
                         signflip_p=float((np.sum(np.abs(f) >= abs(d.mean())) + 1) / 20001), images_cpu_higher=int((d > 0).sum())))
contr = {}
for m in METRICS:
    ca, cg = A[f"gaze4asd_P2_ASD_{m}"] - A[f"gaze4asd_P2_TD_{m}"], G[f"clip_tau1_ASD_{m}"] - G[f"clip_tau1_TD_{m}"]
    d = ca.mean(0) - cg.mean(0)
    f = (rng.choice([-1.0, 1.0], (20000, len(d))) * d).mean(1)
    sa, sg = ca.mean(1), cg.mean(1)
    contr[m] = dict(cpu=float(ca.mean()), gpu=float(cg.mean()), diff=float(d.mean()), image_signflip_p=float((np.sum(np.abs(f) >= abs(d.mean())) + 1) / 20001),
                    seed_level_z=float((sa.mean() - sg.mean()) / np.sqrt(sa.var(ddof=1) / len(sa) + sg.var(ddof=1) / len(sg))),
                    cpu_seed_range=[float(sa.min()), float(sa.max())], gpu_seed_range=[float(sg.min()), float(sg.max())])
json.dump(dict(note="Unadjusted; 18 tests. Distributional agreement only (different random streams).", per_metric=rows, phase2_contrast_cpu_vs_gpu=contr,
               min_p=min(r["signflip_p"] for r in rows)), open(OUT + "/cpu_gpu_check.json", "w"), indent=1)
for r in sorted(rows, key=lambda r: r["signflip_p"])[:5]:
    print(f"{r['group']} {r['metric']:22s} {r['rel_pct']:+.2f}% p={r['signflip_p']:.4f}")
for m, c in contr.items():
    print(f"contrast {m:22s}: CPU {c['cpu']:+.4f} GPU {c['gpu']:+.4f} diff {c['diff']:+.4f} image p={c['image_signflip_p']:.4f} seed z={c['seed_level_z']:+.2f} CPU seeds {np.round(c['cpu_seed_range'], 4)} GPU seeds {np.round(c['gpu_seed_range'], 4)}")
