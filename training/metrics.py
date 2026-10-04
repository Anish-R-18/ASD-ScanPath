"""
Gaze-dynamics metrics for the Phase-2 behavioral analysis (numpy/scipy only).
All geometric metrics operate in PIXEL coordinates; durations in seconds.
"""
from __future__ import annotations
import csv
import os
import numpy as np


# --------------------------------------------------------------------------- #
# per-scanpath metrics
# --------------------------------------------------------------------------- #
def scanpath_metrics(xy, dur, diag):
    """xy: (L,2) pixels; dur: (L,) seconds; diag: image diagonal in px.
    Returns a flat dict of robust per-scanpath gaze measures."""
    L = len(xy)
    out = {
        "n_fix": L,
        "dur_mean": float(np.mean(dur)) if L else np.nan,
        "dur_median": float(np.median(dur)) if L else np.nan,
        "total_dur": float(np.sum(dur)) if L else 0.0,
    }
    if L >= 2:
        sacc = np.diff(xy, axis=0)
        amp = np.linalg.norm(sacc, axis=1)              # pixels
        ang = np.arctan2(sacc[:, 1], sacc[:, 0])        # radians (-pi,pi]
        out["sacc_amp_mean"] = float(np.mean(amp))
        out["sacc_amp_median"] = float(np.median(amp))
        out["sacc_amp_norm_mean"] = float(np.mean(amp) / diag)
        out["sacc_dir_mean"] = float(np.mean(ang))
        out["sacc_dir_std"] = float(np.std(ang))
    else:
        for k in ("sacc_amp_mean", "sacc_amp_median", "sacc_amp_norm_mean",
                  "sacc_dir_mean", "sacc_dir_std"):
            out[k] = np.nan
    # spatial dispersion
    if L >= 1:
        c = xy.mean(axis=0)
        out["dispersion"] = float(np.mean(np.linalg.norm(xy - c, axis=1)) / diag)
        out["bbox_area_norm"] = float(((xy[:, 0].max() - xy[:, 0].min()) *
                                       (xy[:, 1].max() - xy[:, 1].min())) / (diag * diag)) if L >= 2 else 0.0
    else:
        out["dispersion"] = np.nan; out["bbox_area_norm"] = np.nan
    # revisit / return rate: a fixation that lands within `r` of an earlier (>=2 back) fixation
    r = 0.05 * diag
    returns, offsets = 0, []
    for i in range(L):
        for j in range(i - 2, -1, -1):
            if np.linalg.norm(xy[i] - xy[j]) <= r:
                returns += 1; offsets.append(i - j); break
    out["revisit_rate"] = float(returns / L) if L else 0.0
    out["return_offset_mean"] = float(np.mean(offsets)) if offsets else np.nan
    return out


# --------------------------------------------------------------------------- #
# AOI (ImageAnnotations.csv): social-attention analysis
# --------------------------------------------------------------------------- #
SOCIAL = {"face", "people", "people background", "hand"}


def load_aoi(path):
    """image_id -> list of (x1,y1,x2,y2,label)."""
    aoi = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            fn = os.path.basename(row[0])
            try:
                iid = int(os.path.splitext(fn)[0])
                x1, y1, x2, y2 = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
            except ValueError:
                continue
            label = row[5].strip().lower()
            aoi.setdefault(iid, []).append((min(x1, x2), min(y1, y2),
                                            max(x1, x2), max(y1, y2), label))
    return aoi


def _in(box, x, y):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def aoi_metrics(xy, dur, boxes):
    """Fraction of fixations and dwell-time on face / any-social AOIs."""
    total = float(np.sum(dur)) if len(dur) else 0.0
    L = len(xy)
    face_b = [b for b in boxes if b[4] == "face"]
    soc_b = [b for b in boxes if b[4] in SOCIAL]
    fcount = scount = 0
    fdwell = sdwell = 0.0
    for i in range(L):
        x, y = xy[i]
        if any(_in(b, x, y) for b in face_b):
            fcount += 1; fdwell += dur[i]
        if any(_in(b, x, y) for b in soc_b):
            scount += 1; sdwell += dur[i]
    return {
        "face_fix_frac": fcount / L if L else np.nan,
        "face_dwell_frac": fdwell / total if total else np.nan,
        "social_fix_frac": scount / L if L else np.nan,
        "social_dwell_frac": sdwell / total if total else np.nan,
    }


# --------------------------------------------------------------------------- #
# MultiMatch (5-dim), self-contained. Dijkstra/DP alignment of saccade vectors.
# Similarities in [0,1] (1 = identical). Normalisation constants documented inline.
# --------------------------------------------------------------------------- #
def multimatch(xy1, dur1, xy2, dur2, diag):
    if len(xy1) < 2 or len(xy2) < 2:
        return None
    v1 = np.diff(xy1, axis=0); v2 = np.diff(xy2, axis=0)
    n, m = len(v1), len(v2)
    C = np.linalg.norm(v1[:, None, :] - v2[None, :, :], axis=2)  # (n,m) vector-diff cost
    INF = 1e18
    D = np.full((n, m), INF); D[0, 0] = C[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best = INF
            if i > 0:
                best = min(best, D[i - 1, j])
            if j > 0:
                best = min(best, D[i, j - 1])
            if i > 0 and j > 0:
                best = min(best, D[i - 1, j - 1])
            D[i, j] = best + C[i, j]
    i, j = n - 1, m - 1
    idx1, idx2 = [], []
    while not (i == 0 and j == 0):
        idx1.append(i); idx2.append(j)
        opts = []
        if i > 0:
            opts.append((D[i - 1, j], i - 1, j))
        if j > 0:
            opts.append((D[i, j - 1], i, j - 1))
        if i > 0 and j > 0:
            opts.append((D[i - 1, j - 1], i - 1, j - 1))
        _, i, j = min(opts)
    idx1.append(0); idx2.append(0)
    idx1 = idx1[::-1]; idx2 = idx2[::-1]
    V1, V2 = v1[idx1], v2[idx2]
    l1 = np.linalg.norm(V1, axis=1); l2 = np.linalg.norm(V2, axis=1)
    f1, f2 = xy1[idx1], xy2[idx2]
    d1, d2 = dur1[idx1], dur2[idx2]
    ang = np.arccos(np.clip(np.sum(V1 * V2, axis=1) / (l1 * l2 + 1e-9), -1, 1))
    return {
        "mm_shape": float(1 - np.mean(np.linalg.norm(V1 - V2, axis=1)) / (2 * diag)),
        "mm_length": float(1 - np.mean(np.abs(l1 - l2)) / diag),
        "mm_direction": float(1 - np.mean(ang) / np.pi),
        "mm_position": float(1 - np.mean(np.linalg.norm(f1 - f2, axis=1)) / diag),
        "mm_duration": float(1 - np.mean(np.abs(d1 - d2) / (np.maximum(d1, d2) + 1e-9))),
    }
