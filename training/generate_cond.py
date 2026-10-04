"""
Conditioned autoregressive sampler for Phase-2 ASD/TD TPP-Gaze.

Mirrors the official TPPGaze.sample() exactly, but adds the learned group
embedding to the merged context at every step (same injection point as training).
The original tppgaze/ code is NOT modified. Because the entire base is frozen in
the embedding-only model, the image metadata is constant across steps, so we
compute it once per image and reuse it (numerically identical, much faster).
"""
from __future__ import annotations
import torch
import numpy as np
from tppgaze.utils import diff


@torch.no_grad()
def sample_conditioned(cmodel, image_tensor, group_id, t_end=3.0, n_sim=20,
                       temperature=2.0, max_steps=80):
    """image_tensor: (3,512,512) preprocessed. Returns (inter_times, marks, mask)
    on CPU: inter_times (n_sim,L) seconds, marks (n_sim,L,2) in [-1,1], mask (n_sim,L)."""
    base = cmodel.base
    base.eval()
    device = next(base.parameters()).device
    cond = cmodel.group_emb(torch.tensor([group_id], device=device))  # (1, C)

    def merge_cond(history, metadata):
        return base.merge.forward([history, metadata]) + cond.unsqueeze(1)

    imgs = image_tensor.to(device).unsqueeze(0).repeat(n_sim, 1, 1, 1)  # (n_sim,3,512,512)
    metadata = base.get_metadata(imgs)  # constant (base frozen) -> compute once

    context_init = base.context.get_context_init()
    history = context_init[None, None, :].expand(n_sim, 1, -1)
    next_context = merge_cond(history, metadata)

    inter_times = torch.empty(n_sim, 0, device=device)
    marks = torch.empty(n_sim, 0, 2, device=device)
    generated = False
    step = 0
    while not generated and step < max_steps:
        nit = base.get_inter_time_dist(next_context).sample()          # (n_sim,1)
        inter_times = torch.cat([inter_times, nit], dim=1)
        cm = base.get_marks_dist(next_context, temperature).sample()    # (n_sim,1,2)
        marks = torch.cat([marks, cm], dim=1)
        generated = bool(inter_times.sum(-1).min() >= t_end)
        feats = base.get_features(inter_times, marks)
        history = base.get_context(feats, remove_last=False)
        next_context = merge_cond(history, metadata)[:, [-1], :]
        step += 1

    arrival = inter_times.cumsum(-1)
    inter_times = diff(arrival.clamp(max=t_end), dim=-1)
    mask = (arrival <= t_end).float()
    marks = marks * torch.stack([mask, mask], dim=2)
    return inter_times.cpu(), marks.cpu(), mask.cpu()


def to_scanpaths(inter_times, marks, mask):
    """List of (L,3) arrays [x_norm, y_norm, dur_s], padding removed."""
    out = []
    for i in range(inter_times.shape[0]):
        n = int(mask[i].sum().item())
        if n < 1:
            out.append(np.zeros((0, 3), dtype=float)); continue
        x = marks[i, :n, 0].numpy(); y = marks[i, :n, 1].numpy(); d = inter_times[i, :n].numpy()
        out.append(np.stack([x, y, d], axis=1))
    return out
