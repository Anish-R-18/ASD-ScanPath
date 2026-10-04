"""
Phase 2 trainer: ASD/TD-conditioned TPP-Gaze (one shared model + group embedding).

Reuses the Phase-1 data/model glue unchanged. The only new learnable parameters
are the 2 x context_dim group embedding (see training/conditioned.py). Starts
from the Phase-1 best checkpoint, keeps the same image-level split, same joint
NLL loss (Eq. 10), AdamW, early stopping, best-checkpoint saving, logging.

Subcommands: smoke | short | train   (same semantics as train_tpp.py)
  python training/train_tpp_cond.py smoke --config training/config_phase2.yaml
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
import torch
from omegaconf import OmegaConf

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for p in (_ROOT, os.path.join(_ROOT, "tppgaze")):
    if p not in sys.path:
        sys.path.insert(0, p)

from training.dataset import load_scanpaths, ImageBank  # noqa: E402
from training.splits import make_image_split, assign_samples, summarise, save_split  # noqa: E402
from training import model_utils as mu  # noqa: E402
from training.conditioned import ConditionedTPPGaze  # noqa: E402
from training.train_tpp import set_global_seed, pick_device, resolve, make_loader  # noqa: E402


def load_cfg(config_path):
    cfg = OmegaConf.load(config_path)
    for k in ("model_cfg_path", "checkpoint_path", "asd_json", "td_json", "images_dir", "out_dir"):
        cfg[k] = resolve(_ROOT, cfg[k])
    if cfg.get("split_json"):
        cfg["split_json"] = resolve(_ROOT, cfg["split_json"])
    return cfg


def cond_dim_from_cfg(model_cfg_path):
    mc = OmegaConf.load(model_cfg_path)
    return int(mc.metadata.size + mc.context.size)  # merge == cat -> 512


def run_batch(cmodel, batch, image_bank, device, backbone_cache):
    inter = batch["inter_times"].to(device)
    marks = batch["marks"].to(device)
    mask = batch["mask"].to(device)
    gids = batch["group_ids"].to(device)
    metadata = mu.compute_metadata(cmodel.base, batch["image_ids"], image_bank, device,
                                   backbone_cache=backbone_cache)
    return cmodel.nll(inter, marks, mask, metadata, gids)


@torch.no_grad()
def evaluate(cmodel, loader, image_bank, device, backbone_cache, limit_batches=None):
    cmodel.eval()
    tot = {"nll": 0.0, "nll_duration": 0.0, "nll_position": 0.0}
    tot_ev = 0.0
    for bi, batch in enumerate(loader):
        if limit_batches is not None and bi >= limit_batches:
            break
        out = run_batch(cmodel, batch, image_bank, device, backbone_cache)
        ev = float(out["n_events"])
        for k in tot:
            tot[k] += float(out[k]) * ev
        tot_ev += ev
    tot_ev = max(tot_ev, 1.0)
    return {k: v / tot_ev for k, v in tot.items()}


def prepare(cfg, device, limit_images=None):
    samples = load_scanpaths(cfg.asd_json, cfg.td_json)
    # reuse the EXACT Phase-1 split if available
    if cfg.get("split_json") and os.path.exists(cfg.split_json):
        split = json.load(open(cfg.split_json))
        split = {k: list(map(int, v)) for k, v in split.items()}
        split_source = cfg.split_json
    else:
        split = make_image_split(samples, frac=tuple(cfg.split), seed=cfg.seed)
        split_source = f"seeded(make_image_split, seed={cfg.seed})"
    assigned = assign_samples(samples, split)

    if limit_images is not None:
        keep_tr = set(split["train"][:limit_images])
        keep_va = set(split["val"][:max(1, limit_images // 4)])
        assigned["train"] = [s for s in assigned["train"] if s["image_id"] in keep_tr]
        assigned["val"] = [s for s in assigned["val"] if s["image_id"] in keep_va]

    mean_lit, std_lit = mu.compute_log_inter_time_stats(assigned["train"])
    os.makedirs(cfg.out_dir, exist_ok=True)
    base, _ = mu.build_model(cfg.model_cfg_path, cfg.checkpoint_path, device,
                             mean_log_inter_time=mean_lit, std_log_inter_time=std_lit,
                             used_cfg_out=os.path.join(cfg.out_dir, "model_config_used.yaml"))
    ckpt_loaded = True  # build_model uses strict=True; would have raised otherwise

    n_base = sum(p.numel() for p in base.parameters())
    n_base_train = sum(p.numel() for p in base.parameters() if p.requires_grad)
    if cfg.freeze_backbone:
        mu.freeze_backbone(base)
    if cfg.get("train_embedding_only"):
        for p in base.parameters():
            p.requires_grad = False  # freeze the ENTIRE base; only group_emb trains
    n_base_train_frozen = sum(p.numel() for p in base.parameters() if p.requires_grad)

    cdim = cond_dim_from_cfg(cfg.model_cfg_path)
    cmodel = ConditionedTPPGaze(base, cond_dim=cdim).to(device)

    n_all = sum(p.numel() for p in cmodel.parameters())
    n_all_train = sum(p.numel() for p in cmodel.parameters() if p.requires_grad)
    n_emb = sum(p.numel() for p in cmodel.group_emb.parameters())

    image_bank = ImageBank(cfg.images_dir, ext=cfg.image_ext)
    backbone_cache = None
    if cfg.get("cache_backbone", True):
        used_ids = sorted({s["image_id"] for s in assigned["train"] + assigned["val"]})
        backbone_cache = mu.build_backbone_cache(base, image_bank, used_ids, device)

    info = {"split": split, "split_source": split_source, "assigned": assigned,
            "mean_log_inter_time": mean_lit, "std_log_inter_time": std_lit,
            "cond_dim": cdim, "ckpt_loaded": ckpt_loaded,
            "params": {"base_total": n_base, "base_trainable_prefreeze": n_base_train,
                       "base_trainable_postfreeze": n_base_train_frozen,
                       "group_emb": n_emb, "conditioned_total": n_all,
                       "conditioned_trainable": n_all_train}}
    return cmodel, image_bank, backbone_cache, info


def set_train_mode(cmodel, cfg):
    if cfg.get("train_embedding_only"):
        cmodel.eval()  # base frozen & deterministic (no dropout); only group_emb (a param) learns
    else:
        cmodel.train()
        if cfg.get("freeze_backbone", True):
            cmodel.base.metadata.backbone.eval()


# --------------------------------------------------------------------------- #
def cmd_smoke(cfg, args):
    device = pick_device(cfg.device if args.device is None else args.device)
    set_global_seed(cfg.seed)
    print(f"[smoke-cond] device = {device}")
    cmodel, image_bank, backbone_cache, info = prepare(cfg, device, limit_images=6)

    pc = info["params"]
    print(f"[smoke-cond] Phase-1 checkpoint loaded into base: {info['ckpt_loaded']}  "
          f"(from {os.path.basename(cfg.checkpoint_path)})")
    print(f"[smoke-cond] split source: {info['split_source']}")
    print(f"[smoke-cond] cond_dim (context injected): {info['cond_dim']}")
    print(f"[smoke-cond] PARAM COUNTS:")
    print(f"   base total                    = {pc['base_total']:,}")
    print(f"   base trainable (backbone frozen) = {pc['base_trainable_postfreeze']:,}")
    print(f"   + group_emb (NEW)             = {pc['group_emb']:,}  (2 x {info['cond_dim']})")
    print(f"   conditioned total             = {pc['conditioned_total']:,}  "
          f"(delta vs base = +{pc['conditioned_total']-pc['base_total']})")
    print(f"   conditioned trainable         = {pc['conditioned_trainable']:,}")

    # a batch containing BOTH groups
    loader = make_loader(info["assigned"]["train"], batch_size=8, shuffle=True, seed=cfg.seed)
    batch = next(iter(loader))
    print(f"[smoke-cond] batch groups={batch['groups']} group_ids={batch['group_ids'].tolist()} "
          f"marks={tuple(batch['marks'].shape)}")

    inter = batch["inter_times"].to(device); marks = batch["marks"].to(device)
    mask = batch["mask"].to(device)
    metadata = mu.compute_metadata(cmodel.base, batch["image_ids"], image_bank, device, backbone_cache)

    # --- forward for BOTH conditions on the SAME inputs, in EVAL mode so dropout
    #     noise does not confound the comparison ---
    cmodel.eval()
    B = inter.shape[0]
    g_asd = torch.zeros(B, dtype=torch.long, device=device)   # all ASD
    g_td = torch.ones(B, dtype=torch.long, device=device)     # all TD
    with torch.no_grad():
        nll_asd0 = float(cmodel.nll(inter, marks, mask, metadata, g_asd)["nll"])
        nll_td0 = float(cmodel.nll(inter, marks, mask, metadata, g_td)["nll"])
    print(f"[smoke-cond] zero-init embedding -> ASD nll={nll_asd0:.5f}  TD nll={nll_td0:.5f}  "
          f"identical={abs(nll_asd0-nll_td0)<1e-6} (expected: identical == Phase 1 preserved)")

    # temporarily set a non-zero embedding to prove the conditioning pathway diverges
    with torch.no_grad():
        saved = cmodel.group_emb.weight.detach().clone()
        cmodel.group_emb.weight.copy_(torch.randn_like(saved) * 0.1)
        nll_asd1 = float(cmodel.nll(inter, marks, mask, metadata, g_asd)["nll"])
        nll_td1 = float(cmodel.nll(inter, marks, mask, metadata, g_td)["nll"])
        cmodel.group_emb.weight.copy_(saved)  # restore zero init
    print(f"[smoke-cond] non-zero embedding   -> ASD nll={nll_asd1:.5f}  TD nll={nll_td1:.5f}  "
          f"differ={abs(nll_asd1-nll_td1)>1e-6} (expected: differ -> condition is wired in)")

    # --- forward + backward with the REAL labels (train mode); check grads ---
    set_train_mode(cmodel, cfg)
    out = cmodel.nll(inter, marks, mask, metadata, batch["group_ids"].to(device))
    print(f"[smoke-cond] loss (real labels): nll={float(out['nll']):.4f} "
          f"dur={float(out['nll_duration']):.4f} pos={float(out['nll_position']):.4f} "
          f"finite={bool(torch.isfinite(out['nll']))}")
    out["nll"].backward()
    emb_grad = cmodel.group_emb.weight.grad
    emb_ok = emb_grad is not None and torch.isfinite(emb_grad).all().item() and float(emb_grad.abs().sum()) > 0
    # per-row (ASD row 0, TD row 1) gradient norms
    g0 = float(emb_grad[0].norm()) if emb_grad is not None else 0.0
    g1 = float(emb_grad[1].norm()) if emb_grad is not None else 0.0
    base_grads = [(n, p) for n, p in cmodel.base.named_parameters() if p.requires_grad and p.grad is not None]
    base_nonzero = any(float(p.grad.abs().sum()) > 0 for _, p in base_grads)
    base_finite = all(torch.isfinite(p.grad).all().item() for _, p in base_grads)
    bb_grad = any(p.grad is not None for p in cmodel.base.metadata.backbone.parameters())
    print(f"[smoke-cond] group_emb grad: finite&nonzero={emb_ok}  |ASD-row|={g0:.4e}  |TD-row|={g1:.4e}")
    print(f"[smoke-cond] base trainable grads nonzero={base_nonzero} finite={base_finite}  "
          f"frozen-backbone got grad={bb_grad} (expected False)")

    passed = (bool(torch.isfinite(out['nll'])) and abs(nll_asd0 - nll_td0) < 1e-6
              and abs(nll_asd1 - nll_td1) > 1e-6 and emb_ok and base_nonzero
              and base_finite and not bb_grad)
    print(f"[smoke-cond] RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def train_loop(cfg, args, device, cmodel, image_bank, backbone_cache, info, epochs, tag):
    os.makedirs(cfg.out_dir, exist_ok=True)
    save_split(info["split"], os.path.join(cfg.out_dir, "split.json"))
    tr = make_loader(info["assigned"]["train"], cfg.batch_size, True, cfg.seed, cfg.num_workers)
    va = make_loader(info["assigned"]["val"], cfg.batch_size, False, cfg.seed, cfg.num_workers)
    opt = torch.optim.AdamW([p for p in cmodel.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    log_path = os.path.join(cfg.out_dir, f"log_{tag}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_nll", "val_nll", "val_dur", "val_pos", "secs", "best"])
    best, best_ep, noimp = float("inf"), -1, 0
    best_path = os.path.join(cfg.out_dir, "best_model_conditioned.pth")
    init_val = evaluate(cmodel, va, image_bank, device, backbone_cache, cfg.get("limit_val_batches"))
    print(f"[{tag}] initial val: nll={init_val['nll']:.4f}")
    hist = []
    for epoch in range(1, epochs + 1):
        t0 = time.time(); set_train_mode(cmodel, cfg)
        tn = ev = 0.0
        for bi, batch in enumerate(tr):
            if cfg.get("limit_train_batches") is not None and bi >= cfg.limit_train_batches:
                break
            opt.zero_grad(set_to_none=True)
            out = run_batch(cmodel, batch, image_bank, device, backbone_cache)
            out["nll"].backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_([p for p in cmodel.parameters() if p.requires_grad], cfg.grad_clip)
            opt.step()
            n = float(out["n_events"]); tn += float(out["nll"]) * n; ev += n
        tr_nll = tn / max(ev, 1.0)
        v = evaluate(cmodel, va, image_bank, device, backbone_cache, cfg.get("limit_val_batches"))
        secs = time.time() - t0
        improved = v["nll"] < best - 1e-6
        if improved:
            best, best_ep, noimp = v["nll"], epoch, 0
            torch.save(cmodel.state_dict(), best_path)
        else:
            noimp += 1
        print(f"[{tag}] epoch {epoch:3d}/{epochs} train nll={tr_nll:.4f} val nll={v['nll']:.4f} "
              f"(dur={v['nll_duration']:.4f} pos={v['nll_position']:.4f}) {secs:.1f}s{'  *best' if improved else ''}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_nll, v["nll"], v["nll_duration"], v["nll_position"], round(secs, 2), int(improved)])
        hist.append({"epoch": epoch, "train_nll": tr_nll, "val": v})
        if noimp >= cfg.patience:
            print(f"[{tag}] early stopping at epoch {epoch}")
            break
    summary = {"tag": tag, "epochs_run": len(hist), "initial_val": init_val,
               "best_val_nll": best, "best_epoch": best_ep, "best_checkpoint": best_path,
               "params": info["params"], "mean_log_inter_time": info["mean_log_inter_time"],
               "std_log_inter_time": info["std_log_inter_time"]}
    json.dump(summary, open(os.path.join(cfg.out_dir, f"summary_{tag}.json"), "w"), indent=2)
    return summary


def cmd_short(cfg, args):
    device = pick_device(cfg.device if args.device is None else args.device)
    set_global_seed(cfg.seed)
    if args.batch_size: cfg.batch_size = args.batch_size
    li = args.limit_images if args.limit_images is not None else 12
    ep = args.epochs or 3
    cmodel, bank, cache, info = prepare(cfg, device, limit_images=li)
    pc = info["params"]
    print(f"[short] lr={cfg.lr}  train_embedding_only={bool(cfg.get('train_embedding_only'))}  "
          f"trainable params={pc['conditioned_trainable']:,} (base_total={pc['base_total']:,})")
    print("[short] " + summarise(info["assigned"], info["split"]))

    # embedding before training (should be zero-init)
    W0 = cmodel.group_emb.weight.detach().clone()
    print(f"[short] embedding BEFORE: ASD-norm={float(W0[0].norm()):.4f} TD-norm={float(W0[1].norm()):.4f}")

    summary = train_loop(cfg, args, device, cmodel, bank, cache, info, ep, "short")

    # --- post-training embedding inspection ---
    W = cmodel.group_emb.weight.detach()
    asd, td = W[0], W[1]
    moved = float((W - W0).norm())
    print(f"[short] embedding AFTER:  ASD-norm={float(asd.norm()):.4f}  TD-norm={float(td.norm()):.4f}  "
          f"L2(ASD-TD)={float((asd-td).norm()):.4f}  moved_from_zero={moved:.4f}")
    if float(asd.norm()) > 0 and float(td.norm()) > 0:
        cos = float(torch.dot(asd, td) / (asd.norm() * td.norm()))
        print(f"[short] cosine(ASD,TD)={cos:.4f}  (grads flowed: {moved > 0})")

    # same image, ASD vs TD -> different outputs? (eval, deterministic)
    cmodel.eval()
    vb = next(iter(make_loader(info["assigned"]["val"], cfg.batch_size, False, cfg.seed)))
    inter = vb["inter_times"].to(device); marks = vb["marks"].to(device); mask = vb["mask"].to(device)
    meta = mu.compute_metadata(cmodel.base, vb["image_ids"], bank, device, cache)
    B = inter.shape[0]
    with torch.no_grad():
        na = float(cmodel.nll(inter, marks, mask, meta, torch.zeros(B, dtype=torch.long, device=device))["nll"])
        nt = float(cmodel.nll(inter, marks, mask, meta, torch.ones(B, dtype=torch.long, device=device))["nll"])
    print(f"[short] SAME val batch -> all-ASD nll={na:.5f}  all-TD nll={nt:.5f}  "
          f"different_outputs={abs(na - nt) > 1e-6}")
    return 0


def cmd_train(cfg, args):
    device = pick_device(cfg.device if args.device is None else args.device)
    set_global_seed(cfg.seed)
    if args.batch_size: cfg.batch_size = args.batch_size
    ep = args.epochs or cfg.max_epochs
    cmodel, bank, cache, info = prepare(cfg, device, limit_images=None)
    print("[train] " + summarise(info["assigned"], info["split"]))
    s = train_loop(cfg, args, device, cmodel, bank, cache, info, ep, "full")
    print(f"[train] done. best val nll={s['best_val_nll']:.4f} @ epoch {s['best_epoch']} -> {s['best_checkpoint']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "short", "train"])
    ap.add_argument("--config", default=os.path.join(_THIS_DIR, "config_phase2.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    return {"smoke": cmd_smoke, "short": cmd_short, "train": cmd_train}[args.cmd](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
