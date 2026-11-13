"""
Curated, reproducible demonstration of ASD/TD-conditioned TPP-Gaze scanpath generation.

Research demonstration only: not a diagnostic or clinical tool. It does not retrain or
modify any model, checkpoint, dataset or experiment output. Sampling goes through the same
functions as inference/inference.py (training/generate_cond.sample_conditioned).

  python inference/demo/run_demo.py                 # generate every example (about 2 minutes on CPU)
  python inference/demo/run_demo.py --example s4a_random_60
  python inference/demo/run_demo.py --check         # regenerate and compare with the saved results

Where outputs go
  inference/demo/local_only/<example>/   figures drawn over the stimulus photograph and real-vs-generated
                               fixation plots. Dataset material: kept on this machine, git-ignored
                               (inference/demo/local_only/.gitignore), never to be redistributed.
  inference/demo/demo_results/<example>/ the same figures on a blank canvas, generated scanpaths as JSON,
                               and numeric summaries. No stimulus pixels and no real gaze records,
                               except for the self-made synthetic image, whose overlays are kept here.

Stimulus selection (fixed before any demo output was generated, from saved E1 results only):
  two Saliency4ASD held-out test images drawn at random (numpy default_rng(2026)); the test image
  whose generated ASD-TD dispersion contrast is closest to the real contrast (best agreement), and
  the one where it is furthest (worst agreement); and a synthetic image drawn by this script.
"""
from __future__ import annotations
import argparse, gzip, json, os, platform, sys, time
import numpy as np

# Repository root (this file lives in <root>/inference/demo/).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "inference")):  # inference/ first, so `import inference` finds inference.py
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
os.chdir(_ROOT)

import inference as I                                   # also puts tppgaze/ on sys.path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
from tppgaze.utils import get_preprocess
from training.eval_external_v2 import metric_row

DEMO = os.path.join(_ROOT, "inference", "demo")
LOCAL = os.path.join(DEMO, "local_only")
RESULTS = os.path.join(DEMO, "demo_results")
SYNTH = os.path.join(DEMO, "inputs", "synthetic_scene.png")
S4A_IMAGES = "TrainingData/TrainingData/Images"
SPLIT = "training/runs/baseline/split.json"
E1 = "training/runs/research_completion/e1_phase_tau1_v2"
REAL = ("processed_data/asd_scanpaths.json", "processed_data/td_scanpaths.json")
# recorded in training/runs/research_completion/e1_phase_tau1_v2/environment.json
EXPECTED_SHA = {"phase1": "b613413027e775b98e449cbdbc9afa0253d6bd40f0cb2502230ed5bf1000b0bb",
                "phase2": "35319673382378487d23c1871cab0cf05b71b1e7963e180ec024e5d80901a9a4"}
SEED, SEEDS_GRID, TAUS = 0, [0, 1, 2, 3], (1.0, 2.0)
METRICS = ["n_fix", "dur_mean", "dispersion", "sacc_amp_norm_mean", "revisit_rate"]
LABEL = {I.PHASE1: "Phase 1 (unconditioned)", "ASD": "Phase 2, ASD condition", "TD": "Phase 2, TD condition"}


def test_ids():
    return sorted(map(int, json.load(open(SPLIT))["test"]))


def select_examples():
    """The prespecified selection rule; uses only saved E1 per-image summaries."""
    z = np.load(f"{E1}/summaries.npz")
    ids, p = test_ids(), "saliency4asd_test_"
    real = z[p + "real_ASD_dispersion"] - z[p + "real_TD_dispersion"]
    gen = (z[p + "P2_ASD_dispersion"] - z[p + "P2_TD_dispersion"]).mean(0)
    err = np.abs(gen - real)
    best, worst = ids[int(err.argmin())], ids[int(err.argmax())]
    rest = [i for i in ids if i not in (best, worst)]
    rnd = sorted(np.random.default_rng(2026).choice(rest, 2, replace=False).tolist())
    why = lambda i: dict(real_asd_minus_td_dispersion=round(float(real[ids.index(i)]), 4),
                         e1_generated_asd_minus_td_dispersion=round(float(gen[ids.index(i)]), 4))
    ex = [dict(name=f"s4a_random_{i}", image_id=i, role="random held-out test image (typical case)", **why(i)) for i in rnd]
    ex.append(dict(name=f"s4a_best_{best}", image_id=best, role="best agreement of generated and real ASD-TD dispersion contrast", **why(best)))
    ex.append(dict(name=f"s4a_worst_{worst}", image_id=worst, role="worst agreement of generated and real ASD-TD dispersion contrast (limitation)", **why(worst)))
    for e in ex:
        e["image"] = f"{S4A_IMAGES}/{e['image_id']}.png"
        e["dataset"] = "Saliency4ASD (held-out test split; not used for training or model selection)"
    ex.append(dict(name="synthetic_scene", image_id=None, image=os.path.relpath(SYNTH, _ROOT).replace("\\", "/"),
                   role="self-made synthetic image, outside the training distribution (limitation)",
                   dataset="none: drawn by demo/run_demo.py, no third-party content"))
    summary = dict(generated_contrast_range=[round(float(gen.min()), 4), round(float(gen.max()), 4)],
                   real_contrast_range=[round(float(real.min()), 4), round(float(real.max()), 4)],
                   images_with_positive_generated_contrast=int((gen > 0).sum()), images=len(ids))
    return ex, summary


def make_synthetic():
    """A simple drawn scene: shapes, a schematic face and text. Created here, so it can be shared."""
    if os.path.isfile(SYNTH):
        return
    os.makedirs(os.path.dirname(SYNTH), exist_ok=True)
    W, H = 1024, 768
    im = Image.new("RGB", (W, H), (205, 225, 240))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 520, W, H], fill=(120, 170, 90))                       # ground
    d.ellipse([820, 60, 940, 180], fill=(250, 210, 60))                     # sun
    d.rectangle([110, 330, 390, 560], fill=(200, 120, 90), outline=(60, 40, 30), width=4)   # house
    d.polygon([(90, 335), (250, 200), (410, 335)], fill=(150, 60, 50), outline=(60, 40, 30))
    d.rectangle([215, 450, 285, 560], fill=(90, 60, 40))
    d.rectangle([140, 370, 200, 420], fill=(230, 240, 255), outline=(60, 40, 30), width=3)
    d.rectangle([780, 380, 810, 580], fill=(110, 80, 50))                   # tree
    d.ellipse([700, 250, 890, 420], fill=(60, 130, 60))
    d.ellipse([480, 300, 640, 460], fill=(240, 200, 170), outline=(80, 60, 50), width=4)   # schematic face
    d.ellipse([520, 350, 545, 375], fill=(40, 40, 40)); d.ellipse([575, 350, 600, 375], fill=(40, 40, 40))
    d.arc([520, 370, 600, 430], 20, 160, fill=(120, 40, 40), width=5)
    d.rectangle([440, 600, 700, 680], fill=(250, 250, 250), outline=(40, 40, 40), width=3)  # sign
    d.text((470, 625), "RESEARCH DEMO", fill=(20, 20, 20), font_size=28)
    im.save(SYNTH)


def gen(models, tensor, cond, seed, tau):
    model = models["phase1" if cond == I.PHASE1 else "phase2"]
    return I.sample(model, tensor, cond, seed, 1, tau, 3.0)[0]


def record(fig, cond, seed, tau, spn, W, H):
    ev = I.events(spn, W, H)
    return dict(figure=fig, condition=cond, model="phase1_unconditioned" if cond == I.PHASE1 else "phase2_conditioned",
                seed=seed, temperature=tau, n_fixations=len(spn), n_clipped=int(sum(e["clipped"] for e in ev)), events=ev)


def figures(ex, im, panels_by_fig, footer, shareable):
    """Each figure twice: over the stimulus (local only, unless shareable) and on a blank canvas."""
    for fig, (title, panels, ncols) in panels_by_fig.items():
        over = os.path.join(RESULTS if shareable else LOCAL, ex["name"], f"{fig}.png")
        blank = os.path.join(RESULTS, ex["name"], f"{fig}_no_image.png")
        for path, show in ((over, True), (blank, False)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            I.save_panels(panels, im, path, title, footer(fig), show, ncols)


def real_scanpaths(image_id):
    out = {"ASD": [], "TD": []}
    for f in REAL:
        for s in json.load(open(f)):
            if s["image_id"] == image_id:
                out[s["group"]].append(np.asarray(s["scanpath"], float))
    return out


def e1_scanpaths(image_id, replicate=1):
    out = {}
    for s, g in (("P2_ASD", "ASD"), ("P2_TD", "TD")):
        d = json.load(gzip.open(f"{E1}/generated/saliency4asd_test_{s}_seed{replicate:02d}.json.gz"))
        out[g] = [np.asarray(r["scanpath"], float) for r in d["scanpaths"] if r["image_id"] == image_id]
    return out


def summarize(sps, W, H):
    rows = [metric_row("x", 0, "", I.to_pixels(sp, W, H)[:, :2], sp[:, 2], W, H) for sp in sps if len(sp)]
    return dict(n_scanpaths=len(rows), **{m: round(float(np.nanmean([r[m] for r in rows])), 4) for m in METRICS})


def real_vs_generated(ex, im, W, H):
    """Fixation clouds: all real scanpaths for the image vs the saved E1 samples (replicate 1).
    Real gaze is dataset material, so this figure is written to local_only only."""
    real, g = real_scanpaths(ex["image_id"]), e1_scanpaths(ex["image_id"])
    stats = {f"{k}_{c}": summarize(v[c], W, H) for k, v in (("real", real), ("generated", g)) for c in ("ASD", "TD")}
    pw, title_h, gap, top, bottom = 6.0, 0.6, 0.25, 0.55, 0.6
    img_h = pw * H / W
    fig_w, fig_h = 2 * pw + gap, 2 * (title_h + img_h) + top + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    for r, (kind, src) in enumerate((("Real", real), ("Generated (E1, replicate 1)", g))):
        for c, grp in enumerate(("ASD", "TD")):
            y0 = fig_h - top - r * (title_h + img_h) - title_h - img_h
            ax = fig.add_axes([c * (pw + gap) / fig_w, y0 / fig_h, pw / fig_w, img_h / fig_h])
            ax.imshow(im); ax.axis("off")
            for sp in src[grp]:
                px = I.to_pixels(sp, W, H)
                ax.scatter(px[:, 0], px[:, 1], s=12 + 150 * px[:, 2], c="#f6a623",
                           edgecolors="black", linewidths=0.7, alpha=0.75)
            st = stats[f"{'real' if r == 0 else 'generated'}_{grp}"]
            ax.set_title(f"{kind}: {grp}\n{st['n_scanpaths']} scanpaths, {st['n_fix']:.1f} fixations each, "
                         f"dispersion {st['dispersion']:.3f}", fontsize=10.5)
    fig.suptitle(f"Real vs generated fixations, Saliency4ASD test image {ex['image_id']} (tau = 1)", fontsize=13,
                 y=1 - 0.12 / fig_h, va="top")
    fig.text(0.5, 0.08 / fig_h, "Each dot is one fixation (area grows with duration). Real: all recorded scanpaths for this "
             "image. Generated: the saved E1 samples (replicate 1), 20 per group;\nfigures from one replicate are noisy, so "
             "use the paper's tables for group statistics. " + I.NOTE, ha="center", va="bottom", fontsize=8.5)
    path = os.path.join(LOCAL, ex["name"], "real_vs_generated.png")
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return stats


def build(models, ex, full):
    """Generate one example. full=True adds the seed grid and the temperature comparison."""
    im = Image.open(ex["image"]).convert("RGB")
    W, H = im.size
    tensor = get_preprocess()(im)
    recs, P = [], {}
    # 1. Phase 1 vs Phase 2 ASD vs Phase 2 TD, same seed (common random numbers), tau = 1
    panels = []
    for cond in (I.PHASE1, "ASD", "TD"):
        spn = gen(models, tensor, cond, SEED, 1.0)
        recs.append(record("phase_comparison", cond, SEED, 1.0, spn, W, H))
        panels.append((LABEL[cond], I.to_pixels(spn, W, H), None))
    P["phase_comparison"] = ("Phase 1 vs Phase 2 (ASD, TD): same image, seed and settings", panels, 3)
    if full:
        # 2. four seeds per condition: variation within a condition vs between conditions
        panels = []
        for cond in ("ASD", "TD"):
            for s in SEEDS_GRID:
                spn = gen(models, tensor, cond, s, 1.0)
                recs.append(record("seeds_grid", cond, s, 1.0, spn, W, H))
                panels.append((LABEL[cond], I.to_pixels(spn, W, H), f"seed {s}"))
        P["seeds_grid"] = ("Four samples per condition (top: ASD, bottom: TD), tau = 1", panels, len(SEEDS_GRID))
        # 3. temperature 1 vs 2, same seed
        panels = []
        for tau in TAUS:
            for cond in ("ASD", "TD"):
                spn = gen(models, tensor, cond, SEED, tau)        # tau = 1 repeats the phase_comparison sample
                if tau != 1.0:
                    recs.append(record("temperature", cond, SEED, tau, spn, W, H))
                panels.append((LABEL[cond], I.to_pixels(spn, W, H), f"tau = {tau:g}"))
        P["temperature"] = ("Sampling temperature: tau = 1 (top, default) vs tau = 2 (bottom); same seed", panels, 2)
    footer = lambda fig: (f"Generated samples, viewing window 3 s. Panels with the same seed share their random numbers. {I.NOTE}")
    shareable = ex["image_id"] is None
    figures(ex, im, P, footer, shareable)
    out = dict(example=ex["name"], image_width=W, image_height=H, image_sha256=I.sha256(ex["image"]), records=recs)
    if ex["image_id"] is not None:
        out["real_vs_generated_summary"] = real_vs_generated(ex, im, W, H)
    os.makedirs(os.path.join(RESULTS, ex["name"]), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, ex["name"], "samples.json"), "w"), indent=1)
    return out


def validate(out):
    """Event-level sanity checks on every generated scanpath of one example."""
    W, H, problems = out["image_width"], out["image_height"], []
    for r in out["records"]:
        ev, tag = r["events"], f"{r['figure']}/{r['condition']}/seed{r['seed']}/tau{r['temperature']:g}"
        if len(ev) != r["n_fixations"] or [e["index"] for e in ev] != list(range(1, len(ev) + 1)):
            problems.append(f"{tag}: event count or order")
        if len(ev) == 0:
            problems.append(f"{tag}: empty scanpath")
        for e in ev:
            if not (0 <= e["x_px"] <= W - 1 and 0 <= e["y_px"] <= H - 1):
                problems.append(f"{tag}: fixation {e['index']} outside the image after clipping")
            if not e["duration_s"] > 0:
                problems.append(f"{tag}: non-positive duration at fixation {e['index']}")
        on = [e["onset_s"] for e in ev]
        if any(b <= a for a, b in zip(on, on[1:])):
            problems.append(f"{tag}: onsets not increasing")
        if ev and on[-1] + ev[-1]["duration_s"] > 3.0 + 1e-3:
            problems.append(f"{tag}: total time exceeds the 3 s window")
    durs = [e["duration_s"] for r in out["records"] for e in r["events"]]
    stats = dict(scanpaths=len(out["records"]), fixations=len(durs), clipped=sum(r["n_clipped"] for r in out["records"]),
                 duration_min_s=min(durs), duration_median_s=float(np.median(durs)), duration_max_s=max(durs),
                 fixations_per_scanpath=[min(r["n_fixations"] for r in out["records"]), max(r["n_fixations"] for r in out["records"])])
    return problems, stats


def e1_crosscheck(models):
    """Regenerate two saved E1 sampling calls (replicate 1, first test image, n_sim = 20) with the
    demo's loading code and compare: shows the demo samples from the same model as the paper."""
    ids = test_ids()
    im = Image.open(f"{S4A_IMAGES}/{ids[0]}.png").convert("RGB")
    tensor = get_preprocess()(im)
    res = {}
    for s, mname, gid in (("P2_ASD", "phase2", "ASD"), ("P1_s0", "phase1", I.PHASE1)):
        seed = 100000 * 1 + 1000 * 1 + 0
        new = I.sample(models[mname], tensor, gid, seed, 20, 1.0, 3.0)
        d = json.load(gzip.open(f"{E1}/generated/saliency4asd_test_{s}_seed01.json.gz"))
        old = [np.asarray(r["scanpath"]) for r in d["scanpaths"] if r["image_id"] == ids[0]]
        new = [np.round(x, 5) for x in new if len(x)]
        same_len = [len(a) for a in old] == [len(b) for b in new]
        diff = max(float(np.abs(a - b).max()) for a, b in zip(old, new)) if same_len else float("inf")
        res[s] = dict(image_id=ids[0], seed=seed, scanpaths=len(old), lengths_equal=same_len, max_abs_diff=diff,
                      passed=bool(same_len and diff <= 2e-5))
    return res


def main():
    ap = argparse.ArgumentParser(description="Curated ASD/TD scanpath demonstration (research only)")
    ap.add_argument("--example", default=None, help="generate one example by name (default: all)")
    ap.add_argument("--check", action="store_true", help="regenerate and compare with saved demo_results")
    args = ap.parse_args()
    t0 = time.time()
    make_synthetic()
    examples, sel_summary = select_examples()
    if args.example:
        examples = [e for e in examples if e["name"] == args.example] or sys.exit(f"unknown example {args.example}")
    dev = torch.device("cpu")
    mean, std = I.load_stats("training/runs/phase2_embedding_only/summary_full.json")
    models = {"phase2": I.load_model(I.PHASE2_CKPT, "tppgaze/data/config.yaml", mean, std, dev),
              "phase1": I.load_model(I.PHASE1_CKPT, "tppgaze/data/config.yaml", mean, std, dev, phase1=True)}
    sha = {"phase1": I.sha256(I.PHASE1_CKPT), "phase2": I.sha256(I.PHASE2_CKPT)}
    assert sha == EXPECTED_SHA, f"checkpoint hash mismatch: {sha}"
    assert I.same_base(models["phase1"], models["phase2"]), "Phase 1 and Phase 2 base weights differ"
    print(f"[demo] checkpoints verified; Phase 1 base == Phase 2 base; {time.time() - t0:.0f}s")

    if args.check:
        report = dict(e1_crosscheck=e1_crosscheck(models), examples={})
        for ex in examples:
            saved = json.load(open(os.path.join(RESULTS, ex["name"], "samples.json")))
            im = Image.open(ex["image"]).convert("RGB"); W, H = im.size
            tensor = get_preprocess()(im)
            mism = [f"{r['figure']}/{r['condition']}/seed{r['seed']}/tau{r['temperature']:g}" for r in saved["records"]
                    if record(r["figure"], r["condition"], r["seed"], r["temperature"],
                              gen(models, tensor, r["condition"], r["seed"], r["temperature"]), W, H) != r]
            report["examples"][ex["name"]] = dict(records=len(saved["records"]), identical=len(saved["records"]) - len(mism),
                                                  mismatches=mism, image_sha256_unchanged=I.sha256(ex["image"]) == saved["image_sha256"])
            print(f"[demo] check {ex['name']}: {len(saved['records']) - len(mism)}/{len(saved['records'])} identical")
        for k, v in report["e1_crosscheck"].items():
            print(f"[demo] E1 cross-check {k}: lengths equal {v['lengths_equal']}, max diff {v['max_abs_diff']:.1e}, passed {v['passed']}")
        report["passed"] = all(v["passed"] for v in report["e1_crosscheck"].values()) and \
            all(not v["mismatches"] and v["image_sha256_unchanged"] for v in report["examples"].values())
        report["environment"] = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                     platform=platform.platform(), threads=torch.get_num_threads())
        json.dump(report, open(os.path.join(RESULTS, "reproducibility_check.json"), "w"), indent=1)
        print(f"[demo] reproducibility check passed: {report['passed']} ({time.time() - t0:.0f}s)")
        return

    # seed grid and temperature: the first random image (lowest ID) and the synthetic image
    full = {f"s4a_random_{select_examples()[0][0]['image_id']}", "synthetic_scene"}
    valid = {}
    for ex in examples:
        out = build(models, ex, full=ex["name"] in full)
        problems, stats = validate(out)
        valid[ex["name"]] = dict(problems=problems, **stats)
        print(f"[demo] {ex['name']}: {stats['scanpaths']} scanpaths, problems {len(problems)} ({time.time() - t0:.0f}s)")
    if not args.example:
        json.dump(dict(note=I.NOTE, selection_rule=__doc__.split("Stimulus selection")[1].strip(), selection_summary=sel_summary,
                       examples=examples, checkpoints={k: dict(path=v, sha256=sha[k]) for k, v in
                                                       (("phase1", I.PHASE1_CKPT), ("phase2", I.PHASE2_CKPT))},
                       settings=dict(seed=SEED, seeds_grid=SEEDS_GRID, temperatures=list(TAUS), t_end_s=3.0, n_sim=1,
                                     device="cpu"),
                       environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                        platform=platform.platform()),
                       validation=valid), open(os.path.join(RESULTS, "manifest.json"), "w"), indent=1)
    print(f"[demo] done in {time.time() - t0:.0f}s; problems: {sum(len(v['problems']) for v in valid.values())}")


if __name__ == "__main__":
    main()
