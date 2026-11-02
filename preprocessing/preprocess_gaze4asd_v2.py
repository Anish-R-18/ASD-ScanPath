#!/usr/bin/env python
"""
Gaze4ASD -> TPP-Gaze preprocessing, version 2 (corrected coordinate mapping).

Supersedes the native-size centred mapping in preprocess_gaze4asd.py, which the
coordinate audit found to be contradicted by the data. That script and its output
(external_data/Gaze4ASD_processed/) are left untouched.

Versions written under external_data/Gaze4ASD_processed_v2/:
  fit_height_policy_a  PRIMARY      fit-to-height, out-of-image coordinates clipped
  stretch_policy_a     SENSITIVITY  full-screen stretch, same cleaning as primary
  fit_height_policy_b  SENSITIVITY  fit-to-height, out-of-image fixations removed

Cleaning order, applied to each original scanpath:
  1. Remove the first fixation.
  2. Remove fixations shorter than 60 ms.
  3. Apply the 3 s cutoff on a timeline starting at zero: drop fixations starting at
     or after 3 s, clip a straddling fixation, drop it if under 60 ms remain.
  4. Map screen coordinates to normalized image coordinates.
  5. Policy A clips to [-1, 1]. Policy B removes fixations outside the image.
  6. Keep scanpaths with at least 3 fixations.

Fit-to-height is strongly supported but partly inferred: the original draw rectangle
is not recorded in the dataset. See reports/mapping_methodology.txt.

Raw inputs are read-only. The script refuses to write into an existing output folder.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "external_data" / "Gaze4ASD"
IMG_DIR = RAW_DIR / "Image Stimuli"
OLD_DIR = ROOT / "external_data" / "Gaze4ASD_processed"
OUT_DIR = ROOT / "external_data" / "Gaze4ASD_processed_v2"

SCREEN_W, SCREEN_H = 2560.0, 1440.0
MIN_FIX_S, T_END_S, MIN_LEN = 0.060, 3.0, 3
GROUPS = ("ASD", "TD")
N_STIMULI = 30
EPS = 1e-6

VERSIONS = {
    "fit_height_policy_a": {"mapping": "fit_height", "policy": "A", "role": "PRIMARY"},
    "stretch_policy_a": {"mapping": "stretch", "policy": "A", "role": "SENSITIVITY"},
    "fit_height_policy_b": {"mapping": "fit_height", "policy": "B", "role": "SENSITIVITY"},
}


# --------------------------------------------------------------------------- mapping
def geometry(mapping, W, H):
    """Return (offset_x, offset_y, scale_x, scale_y) from image pixels to screen pixels."""
    if mapping == "fit_height":
        scale = SCREEN_H / H
        return (SCREEN_W - scale * W) / 2.0, 0.0, scale, scale
    if mapping == "stretch":
        return 0.0, 0.0, SCREEN_W / W, SCREEN_H / H
    if mapping == "native_centred":  # legacy, used only to explain old counts
        return (SCREEN_W - W) / 2.0, (SCREEN_H - H) / 2.0, 1.0, 1.0
    raise ValueError(mapping)


def to_normalized(X, Y, W, H, mapping):
    """Screen pixels to continuous normalized image coordinates (edges at exactly -1 and +1)."""
    ox, oy, sx, sy = geometry(mapping, W, H)
    image_x = (X - ox) / sx
    image_y = (Y - oy) / sy
    return 2.0 * image_x / W - 1.0, 2.0 * image_y / H - 1.0


# --------------------------------------------------------------------------- io
def stimulus_dimensions():
    dims = {p.name: Image.open(p).size for p in IMG_DIR.glob("*.png")}
    if len(dims) != N_STIMULI:
        raise SystemExit(f"expected {N_STIMULI} stimuli, found {len(dims)}")
    return dims


def scanpath_files():
    """(tsv_path, group, participant, stimulus filename), sorted for reproducibility."""
    out = []
    for g in GROUPS:
        for p in sorted((RAW_DIR / f"{g}_ScanPath").glob("*/*.tsv")):
            out.append((p, g, p.parent.name, p.name.replace(".tsv", ".png")))
    return out


def read_tsv(path):
    """Columns: screen X (px), screen Y (px), duration (ms). Verified against raw Tobii gaze."""
    return np.loadtxt(path, delimiter="\t", ndmin=2)


# --------------------------------------------------------------------------- cleaning
def apply_cutoff(xs, ys, ds, c):
    """3 s cutoff with the clock starting at zero on the sequence retained so far."""
    if len(ds) == 0:
        return xs, ys, ds
    starts = np.r_[0.0, np.cumsum(ds)[:-1]]
    keep = starts < T_END_S
    c["fix_removed_after_cutoff"] += int((~keep).sum())
    xs, ys, ds, starts = xs[keep], ys[keep], ds[keep], starts[keep]
    if len(ds) and starts[-1] + ds[-1] > T_END_S:
        remainder = T_END_S - starts[-1]
        if remainder < MIN_FIX_S:
            xs, ys, ds = xs[:-1], ys[:-1], ds[:-1]
            c["fix_removed_cutoff_too_short"] += 1
        else:
            ds = ds.copy()
            ds[-1] = remainder
            c["fix_truncated_at_cutoff"] += 1
    return xs, ys, ds


def process(rows, W, H, mapping, policy, c):
    """Return (xn, yn, dur_s) arrays for one scanpath, or None if it is dropped."""
    c["raw_scanpaths"] += 1
    c["raw_fixations"] += len(rows)
    X, Y, d = rows[1:, 0], rows[1:, 1], rows[1:, 2] / 1000.0
    c["fix_removed_first"] += min(1, len(rows))

    keep = d >= MIN_FIX_S
    c["fix_removed_short"] += int((~keep).sum())
    X, Y, d = apply_cutoff(X[keep], Y[keep], d[keep], c)

    xn, yn = to_normalized(X, Y, W, H, mapping)
    c["fix_after_cleaning"] += len(d)
    out = (np.abs(xn) > 1.0) | (np.abs(yn) > 1.0)
    c["fix_out_of_image"] += int(out.sum())
    if policy == "A":
        c["coord_x_clipped"] += int((np.abs(xn) > 1.0).sum())
        c["coord_y_clipped"] += int((np.abs(yn) > 1.0).sum())
        xn, yn = np.clip(xn, -1.0, 1.0), np.clip(yn, -1.0, 1.0)
    else:
        c["fix_removed_out_of_image"] += int(out.sum())
        xn, yn, d = xn[~out], yn[~out], d[~out]

    if len(d) < MIN_LEN:
        c["scanpaths_dropped_too_short"] += 1
        c["fix_lost_in_dropped_scanpaths"] += len(d)
        return None
    c["scanpaths_kept"] += 1
    return xn, yn, d


def build_version(mapping, policy, dims, files):
    records, c = [], Counter()
    for path, group, participant, stim in files:
        W, H = dims[stim]  # no fallback: a missing stimulus must fail loudly
        res = process(read_tsv(path), W, H, mapping, policy, c)
        if res is None:
            continue
        xn, yn, d = res
        records.append({
            "image_id": stim,
            "group": group,
            "participant": participant,
            "scanpath": [[round(float(a), 6), round(float(b), 6), round(float(e), 6)] for a, b, e in zip(xn, yn, d)],
            "stimulus_id": int(stim.replace("top_image_", "").replace(".png", "")),
            "source_file": path.relative_to(RAW_DIR).as_posix(),
            "image_width": W,
            "image_height": H,
            "n_fixations": len(d),
            "mapping": mapping,
            "policy": policy,
        })
    return records, c


# --------------------------------------------------------------------------- validation
def describe(a):
    a = np.asarray(a, float)
    return {"n": int(a.size), "min": float(a.min()), "p5": float(np.percentile(a, 5)), "median": float(np.median(a)),
            "mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "std": float(a.std())}


def check_against_raw(records, mapping, policy, dims):
    """Re-derive every stored fixation from the raw TSV, separately from process()."""
    worst_coord = worst_dur = 0.0
    clip_bad = order_bad = 0
    for r in records:
        rows = read_tsv(RAW_DIR / r["source_file"])[1:]
        rows = rows[rows[:, 2] >= 60.0]
        W, H = dims[r["image_id"]]
        ox, oy, sx, sy = geometry(mapping, W, H)
        rx0, rx1, ry0, ry1 = ox, ox + sx * W, oy, oy + sy * H
        inside = (rows[:, 0] >= rx0) & (rows[:, 0] <= rx1) & (rows[:, 1] >= ry0) & (rows[:, 1] <= ry1)
        if policy == "B":
            rows, inside = rows[inside], inside[inside]
        sp = np.array(r["scanpath"])
        if len(sp) != len(rows):
            order_bad += 1
            continue
        px = rx0 + (sp[:, 0] + 1.0) / 2.0 * (rx1 - rx0)  # stored value projected back to the screen
        py = ry0 + (sp[:, 1] + 1.0) / 2.0 * (ry1 - ry0)
        if inside.any():
            worst_coord = max(worst_coord, float(np.abs(px[inside] - rows[inside, 0]).max()),
                              float(np.abs(py[inside] - rows[inside, 1]).max()))
        expx = np.clip(rows[:, 0], rx0, rx1)
        expy = np.clip(rows[:, 1], ry0, ry1)
        if (~inside).any():
            clip_bad += int(((np.abs(px - expx) > 0.01) | (np.abs(py - expy) > 0.01))[~inside].sum())
        worst_dur = max(worst_dur, float(np.abs(sp[:, 2] - rows[:, 2] / 1000.0).max()))
    return {"max_backprojection_error_px": worst_coord, "max_duration_error_s": worst_dur,
            "clipped_values_wrong": clip_bad, "scanpaths_misaligned_with_raw": order_bad}


def loader_compatibility(path):
    """Exercise the real loaders. Old and new schemas are compared in the report."""
    sys.path[:0] = [str(ROOT), str(ROOT / "tppgaze")]
    from training.dataset import ScanpathDataset, collate_fn, load_scanpaths
    data = json.load(open(path))
    batch = collate_fn([ScanpathDataset(data)[i] for i in range(8)])
    ok_batch = (batch["marks"].ndim == 3 and batch["inter_times"].ndim == 2
                and bool((batch["mask"].sum(1).long() == batch["lengths"]).all()))
    eval_keys = all(k in r for r in data for k in ("image_id", "group", "participant", "scanpath"))
    files_ok = all((IMG_DIR / r["image_id"]).exists() for r in data)
    groups_ok = all(r["group"] in GROUPS for r in data)
    try:
        load_scanpaths(str(path), str(path))
        ls = "ok"
    except ValueError:
        ls = "ValueError: int(image_id) cannot parse a filename"
    return {"ScanpathDataset_collate_fn": bool(ok_batch), "eval_external_required_keys": bool(eval_keys),
            "image_id_resolves_to_stimulus_file": bool(files_ok), "group_labels_valid": bool(groups_ok),
            "training_dataset_load_scanpaths": ls}


def validate(records, c, mapping, policy, dims, files, path):
    arr = [np.array(r["scanpath"]) for r in records]
    durs = np.concatenate([a[:, 2] for a in arr])
    xs = np.concatenate([a[:, 0] for a in arr])
    ys = np.concatenate([a[:, 1] for a in arr])
    lens = np.array([len(a) for a in arr])
    cum = np.array([a[:, 2].sum() for a in arr])

    raw_group = {}
    for _, g, part, _s in files:
        raw_group.setdefault(part, set()).add(g)
    combos = Counter((r["participant"], r["image_id"]) for r in records)
    all_parts = sorted(raw_group)
    expected = {(p, s) for p in all_parts for s in dims}
    raw_combos = {(p, s) for _, _, p, s in files}
    v = {
        "participants_ASD": len({r["participant"] for r in records if r["group"] == "ASD"}),
        "participants_TD": len({r["participant"] for r in records if r["group"] == "TD"}),
        "trials_ASD": sum(r["group"] == "ASD" for r in records),
        "trials_TD": sum(r["group"] == "TD" for r in records),
        "fixations_ASD": int(sum(len(a) for a, r in zip(arr, records) if r["group"] == "ASD")),
        "fixations_TD": int(sum(len(a) for a, r in zip(arr, records) if r["group"] == "TD")),
        "fixations_total": int(lens.sum()),
        "unique_stimuli": len({r["image_id"] for r in records}),
        "duration_s": describe(durs),
        "fixations_per_trial": describe(lens),
        "cumulative_duration_s": describe(cum),
        "norm_x_min": float(xs.min()), "norm_x_max": float(xs.max()),
        "norm_y_min": float(ys.min()), "norm_y_max": float(ys.max()),
        "sequences_below_3_fixations": int((lens < MIN_LEN).sum()),
        "scanpaths_dropped_too_short": c["scanpaths_dropped_too_short"],
        "fixations_out_of_image_pct": 100.0 * c["fix_out_of_image"] / c["fix_after_cleaning"],
        "coord_x_clipped_pct": 100.0 * c["coord_x_clipped"] / c["fix_after_cleaning"],
        "coord_y_clipped_pct": 100.0 * c["coord_y_clipped"] / c["fix_after_cleaning"],
        "fixations_removed_out_of_image": c["fix_removed_out_of_image"],
        "duplicate_participant_image": sum(1 for n in combos.values() if n > 1),
        "participant_image_possible": len(expected),
        "participant_image_no_raw_file": len(expected - raw_combos),
        "participant_image_raw_but_dropped": len(raw_combos - set(combos)),
        "participant_image_present": len(combos),
    }
    v["raw_check"] = check_against_raw(records, mapping, policy, dims)
    v["loader"] = loader_compatibility(path)
    rc = v["raw_check"]
    v["checks"] = {
        "all_coordinates_within_bounds": bool(np.abs(xs).max() <= 1.0 and np.abs(ys).max() <= 1.0),
        "all_durations_valid": bool(np.isfinite(durs).all() and durs.min() >= MIN_FIX_S - EPS),
        "cumulative_within_3s": bool(cum.max() <= T_END_S + EPS),
        "all_sequences_min_3": bool(lens.min() >= MIN_LEN),
        "group_labels_preserved": all(raw_group[r["participant"]] == {r["group"]} for r in records),
        "source_file_matches_labels": all(r["source_file"].startswith(f"{r['group']}_ScanPath/{r['participant']}/")
                                          and r["source_file"].endswith(r["image_id"].replace(".png", ".tsv")) for r in records),
        "image_dimensions_correct": all(tuple(dims[r["image_id"]]) == (r["image_width"], r["image_height"]) for r in records),
        "no_duplicates": v["duplicate_participant_image"] == 0,
        "matches_raw_within_0.01px": bool(rc["max_backprojection_error_px"] < 0.01 and rc["clipped_values_wrong"] == 0
                                          and rc["scanpaths_misaligned_with_raw"] == 0 and rc["max_duration_error_s"] < 1e-5),
        "loader_compatible": bool(v["loader"]["ScanpathDataset_collate_fn"] and v["loader"]["eval_external_required_keys"]
                                  and v["loader"]["image_id_resolves_to_stimulus_file"] and v["loader"]["group_labels_valid"]),
    }
    return v


# --------------------------------------------------------------------------- comparison
def legacy_policy_b_counts(dims, files, mapping):
    """Old cleaning order (out-of-image filter BEFORE removing the first fixation), for explaining old counts."""
    trials = fixations = 0
    for path, _g, _p, stim in files:
        W, H = dims[stim]
        rows = read_tsv(path)
        ox, oy, sx, sy = geometry(mapping, W, H)
        ix, iy = (rows[:, 0] - ox) / sx, (rows[:, 1] - oy) / sy
        rows = rows[(ix >= 0) & (ix <= W) & (iy >= 0) & (iy <= H)][1:]
        rows = rows[rows[:, 2] >= 60.0]
        if len(rows) >= MIN_LEN:
            trials += 1
            fixations += len(rows)
    return trials, fixations


def compare_with_old(new, dims):
    old = json.load(open(OLD_DIR / "gaze4asd_policy_a.json"))
    ok = {(r["group"], r["participant"], r["image_id"]): np.array(r["scanpath"]) for r in old}
    nk = {(r["group"], r["participant"], r["image_id"]): np.array(r["scanpath"]) for r in new}
    common = set(ok) & set(nk)
    same_len = sum(len(ok[k]) == len(nk[k]) for k in common)
    dur_diff = max(float(np.abs(ok[k][:, 2] - nk[k][:, 2]).max()) for k in common if len(ok[k]) == len(nk[k]))
    moved = np.concatenate([np.hypot(*(ok[k][:, :2] - nk[k][:, :2]).T) for k in common])
    per_size = {}
    for k in common:
        d = dims[k[2]]
        per_size.setdefault(d, []).append(np.hypot(*(ok[k][:, :2] - nk[k][:, :2]).T))
    return {
        "old_trials": len(ok), "new_trials": len(nk), "only_in_old": len(set(ok) - set(nk)), "only_in_new": len(set(nk) - set(ok)),
        "old_fixations": int(sum(len(a) for a in ok.values())), "new_fixations": int(sum(len(a) for a in nk.values())),
        "same_sequence_length": same_len, "max_duration_difference_s": dur_diff,
        "coord_shift_median": float(np.median(moved)), "coord_shift_p90": float(np.percentile(moved, 90)),
        "coord_shift_gt_0.05_pct": 100.0 * float((moved > 0.05).mean()), "coord_shift_gt_0.25_pct": 100.0 * float((moved > 0.25).mean()),
        "coord_shift_median_by_size": {f"{a}x{b}": float(np.median(np.concatenate(v))) for (a, b), v in sorted(per_size.items())},
        "old_clipped_pct": 100.0 * float(np.mean(np.concatenate([(np.abs(a[:, :2]) >= 1.0).any(axis=1) for a in ok.values()]))),
    }


# --------------------------------------------------------------------------- reports
METHODOLOGY = """GAZE4ASD COORDINATE MAPPING - METHODOLOGY (version 2)

STATUS
Fit-to-height is STRONGLY SUPPORTED BUT PARTLY INFERRED. It is NOT experimentally
confirmed: the original draw rectangle and presentation script are not in the dataset.
The mapping was fixed from the audit evidence below, before any model was evaluated on
this dataset, and must not be changed according to which mapping scores best.

DIRECT EVIDENCE
- Screen 2560x1440: README line 17 and expt.winRect = [0,0,2560,1440] in all 171 .mat files.
- TSV X,Y are screen pixels: regression on raw Tobii onDisplayArea gives
  X = 2559.84*gx + 0.01 and Y = 1439.51*gy + 0.40 (33,238 fixation matches).
- Origin is top-left with y pointing down, per the Tobii Active Display Coordinate System.

INFERRED EVIDENCE FOR FIT-TO-HEIGHT
- The authors' own saliency maps (dataset git history, commit ea333cf) match this geometry
  with 4:3 padding at median r = 0.9999 over 4,533 maps, with no free parameters. Native
  centred placement gives r = 0.23. The maps show the authors' processing, not the display.
- Fixations inside the image: 96.5% here versus 81.4% for native centred placement.
- Eye landmarks from a face detector, using no gaze: estimated scale 1.71 (CI 1.58-1.75) for
  1200-wide images and 1.21 (CI 1.00-1.29) for 1650x1100, against 1.80 and 1.31 predicted.
  On Saliency4ASD, where the true scale is 1.00, the same estimator reads 0.77, so these
  estimators read low.

EQUATIONS
  X = 2560 * gx,  Y = 1440 * gy        (already applied in the dataset's TSV files)
  scale = 1440 / H
  displayed_width = scale * W
  offset_x = (2560 - displayed_width) / 2
  image_x = (X - offset_x) / scale
  image_y = Y / scale
  norm_x = 2 * image_x / W - 1
  norm_y = 2 * image_y / H - 1

WHY CONTINUOUS COORDINATES
Gaze is a continuous screen position, and after resizing it does not fall on the source
image's pixel grid. The image therefore occupies the continuous interval [0, W] x [0, H],
so its outer edges sit at exactly -1 and +1. Saliency4ASD instead stores 1-indexed pixel
indices and uses 2*(x-1)/(W-1)-1. The two conventions differ by at most about 2/W (0.0017
at W = 1200), about one image pixel. That is well below the roughly 21 screen pixel median
offset measured on the central fixation icon, so it is not material.

STRETCH SENSITIVITY
  norm_x = 2 * X / 2560 - 1,  norm_y = 2 * Y / 1440 - 1
This maps X and Y independently and does not preserve aspect ratio. It is included because
gaze alone does not exclude it. The case against it rests mainly on the authors' maps
(median r = 0.91 versus 0.9999), plus a likelihood comparison that is itself biased toward
smaller scales.

REMAINING UNCERTAINTY
- No file records the draw rectangle. Confirmation requires the authors or their script.
- The 1650x1100 group has the weakest landmark evidence (5 images; interval touches 1.00).
- Cropping cannot be tested directly.
- Indexing and pixel-centre conventions are undeterminable, and matter by at most about 1 px.
"""


def fmt(d, nd=3):
    return f"min={d['min']:.{nd}f} p5={d['p5']:.{nd}f} median={d['median']:.{nd}f} mean={d['mean']:.{nd}f} p95={d['p95']:.{nd}f} max={d['max']:.{nd}f} sd={d['std']:.{nd}f}"


def build_report(results, cmp_a, b_path, dims):
    L = ["GAZE4ASD PREPROCESSING REPORT - VERSION 2", "=" * 72,
         "Fit-to-height is strongly supported but partly inferred; the original draw rectangle",
         "is unavailable and the mapping is not experimentally confirmed.",
         "No model has been evaluated on these datasets.", "",
         "Cleaning: remove first fixation -> remove <60 ms -> 3 s cutoff from zero -> map ->",
         "clip (Policy A) or remove out-of-image (Policy B) -> keep >=3 fixations.", ""]
    L += ["Image rectangle on screen by stimulus size (x0-x1, y0-y1):"]
    for (W, H), n in sorted(Counter(dims.values()).items()):
        ox, oy, sx, sy = geometry("fit_height", W, H)
        L.append(f"  {W}x{H} ({n} stimuli): fit-height scale={sx:.4f} rect x {ox:.1f}-{ox + sx * W:.1f}, y 0-1440")
    L.append("")
    for name, (cfg, c, v) in results.items():
        L += [f"[{name}]  {cfg['role']}  mapping={cfg['mapping']}  policy={cfg['policy']}", "-" * 72,
              f"participants ASD/TD      : {v['participants_ASD']} / {v['participants_TD']}",
              f"trials ASD/TD            : {v['trials_ASD']} / {v['trials_TD']}  (total {v['trials_ASD'] + v['trials_TD']})",
              f"fixations ASD/TD         : {v['fixations_ASD']} / {v['fixations_TD']}  (total {v['fixations_total']})",
              f"unique stimuli           : {v['unique_stimuli']}",
              f"duration (s)             : {fmt(v['duration_s'])}",
              f"fixations per trial      : {fmt(v['fixations_per_trial'], 2)}",
              f"cumulative duration (s)  : min={v['cumulative_duration_s']['min']:.3f} max={v['cumulative_duration_s']['max']:.3f}",
              f"normalized x range       : {v['norm_x_min']:.6f} to {v['norm_x_max']:.6f}",
              f"normalized y range       : {v['norm_y_min']:.6f} to {v['norm_y_max']:.6f}",
              f"out of image             : {v['fixations_out_of_image_pct']:.2f}% of fixations after cleaning",
              f"clipped (Policy A)       : x {v['coord_x_clipped_pct']:.2f}%, y {v['coord_y_clipped_pct']:.2f}% of fixations",
              f"removed out-of-image     : {v['fixations_removed_out_of_image']} fixations",
              f"sequences below 3 kept   : {v['sequences_below_3_fixations']} (dropped by the filter: {v['scanpaths_dropped_too_short']})",
              f"participant-image pairs  : possible {v['participant_image_possible']}, no raw file {v['participant_image_no_raw_file']}, "
              f"raw but dropped {v['participant_image_raw_but_dropped']}, present {v['participant_image_present']}, duplicates {v['duplicate_participant_image']}",
              "accounting               : " + ", ".join(f"{k}={c[k]}" for k in (
                  "raw_scanpaths", "raw_fixations", "fix_removed_first", "fix_removed_short", "fix_removed_after_cutoff",
                  "fix_truncated_at_cutoff", "fix_removed_cutoff_too_short", "fix_lost_in_dropped_scanpaths")),
              f"raw back-projection      : {v['raw_check']}",
              f"loader                   : {v['loader']}",
              "checks                   : " + ("ALL PASS" if all(v["checks"].values()) else f"FAILED {[k for k, x in v['checks'].items() if not x]}"), ""]
    L += ["COMPARISON WITH THE OLD POLICY A DATASET", "-" * 72,
          f"trials    old {cmp_a['old_trials']} -> new {cmp_a['new_trials']} (only in old {cmp_a['only_in_old']}, only in new {cmp_a['only_in_new']})",
          f"fixations old {cmp_a['old_fixations']} -> new {cmp_a['new_fixations']}",
          f"same sequence length in {cmp_a['same_sequence_length']} trials; largest duration difference {cmp_a['max_duration_difference_s']:.2e} s",
          "Cleaning decisions in Policy A do not depend on the mapping, so identical counts are",
          "expected. Only coordinates change.",
          f"coordinate shift: median {cmp_a['coord_shift_median']:.3f}, p90 {cmp_a['coord_shift_p90']:.3f}, "
          f">0.05 {cmp_a['coord_shift_gt_0.05_pct']:.1f}%, >0.25 {cmp_a['coord_shift_gt_0.25_pct']:.1f}%",
          f"median shift by size: {cmp_a['coord_shift_median_by_size']}",
          f"fixations at the clip boundary: old {cmp_a['old_clipped_pct']:.2f}%", "",
          "COMPARISON WITH THE OLD POLICY B DATASET (trials, fixations)", "-" * 72]
    L += [f"  {k:62s} {a}, {b}" for k, (a, b) in b_path.items()]
    L += ["The mapping change keeps more fixations inside the image. The corrected order then",
          "changes which fixation counts as first: the original first fixation is always removed,",
          "instead of the first one that happened to be in-image.", ""]
    return "\n".join(L) + "\n"


def write_summary_csv(path, results):
    fields = ["version", "role", "mapping", "policy", "participants_ASD", "participants_TD", "trials_ASD", "trials_TD", "fixations_ASD",
              "fixations_TD", "unique_stimuli", "duration_median_s", "duration_mean_s", "duration_min_s", "duration_max_s",
              "fixations_per_trial_mean", "fixations_per_trial_median", "out_of_image_pct", "x_clipped_pct", "y_clipped_pct",
              "fixations_removed_out_of_image", "norm_x_min", "norm_x_max", "norm_y_min", "norm_y_max", "cumulative_min_s", "cumulative_max_s",
              "sequences_below_3", "scanpaths_dropped_too_short", "duplicates", "pairs_no_raw_file", "pairs_raw_but_dropped", "all_checks_pass"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for name, (cfg, _c, v) in results.items():
            w.writerow([name, cfg["role"], cfg["mapping"], cfg["policy"], v["participants_ASD"], v["participants_TD"], v["trials_ASD"], v["trials_TD"],
                        v["fixations_ASD"], v["fixations_TD"], v["unique_stimuli"], round(v["duration_s"]["median"], 4), round(v["duration_s"]["mean"], 4),
                        round(v["duration_s"]["min"], 4), round(v["duration_s"]["max"], 4), round(v["fixations_per_trial"]["mean"], 3),
                        v["fixations_per_trial"]["median"], round(v["fixations_out_of_image_pct"], 3), round(v["coord_x_clipped_pct"], 3),
                        round(v["coord_y_clipped_pct"], 3), v["fixations_removed_out_of_image"], v["norm_x_min"], v["norm_x_max"], v["norm_y_min"],
                        v["norm_y_max"], round(v["cumulative_duration_s"]["min"], 4), round(v["cumulative_duration_s"]["max"], 4),
                        v["sequences_below_3_fixations"], v["scanpaths_dropped_too_short"], v["duplicate_participant_image"],
                        v["participant_image_no_raw_file"], v["participant_image_raw_but_dropped"], all(v["checks"].values())])


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {out}")

    dims = stimulus_dimensions()
    files = scanpath_files()
    missing = {s for _, _, _, s in files} - set(dims)
    if missing:
        raise SystemExit(f"Scanpaths reference stimuli with no image: {sorted(missing)}")

    (out / "reports").mkdir(parents=True)
    results = {}
    for name, cfg in VERSIONS.items():
        records, c = build_version(cfg["mapping"], cfg["policy"], dims, files)
        (out / name).mkdir()
        path = out / name / f"gaze4asd_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f)
        results[name] = (cfg, c, validate(records, c, cfg["mapping"], cfg["policy"], dims, files, path))
        print(f"{name}: trials={len(records)} fixations={results[name][2]['fixations_total']} "
              f"checks={'PASS' if all(results[name][2]['checks'].values()) else 'FAIL'}", flush=True)

    new_a = json.load(open(out / "fit_height_policy_a" / "gaze4asd_fit_height_policy_a.json"))
    new_b = json.load(open(out / "fit_height_policy_b" / "gaze4asd_fit_height_policy_b.json"))
    old_b = json.load(open(OLD_DIR / "gaze4asd_policy_b.json"))
    cmp_a = compare_with_old(new_a, dims)
    b_path = {
        "old file (native centred; filter, then remove first)": (len(old_b), sum(len(r["scanpath"]) for r in old_b)),
        "recomputed: native centred; filter, then remove first": legacy_policy_b_counts(dims, files, "native_centred"),
        "fit-to-height; filter, then remove first": legacy_policy_b_counts(dims, files, "fit_height"),
        "new file (fit-to-height; remove first, then filter)": (len(new_b), sum(len(r["scanpath"]) for r in new_b)),
    }
    (out / "reports" / "preprocessing_report.txt").write_text(build_report(results, cmp_a, b_path, dims), encoding="utf-8")
    (out / "reports" / "mapping_methodology.txt").write_text(METHODOLOGY, encoding="utf-8")
    write_summary_csv(out / "reports" / "dataset_summary.csv", results)
    json.dump({"versions": {k: {"config": a, "counters": dict(b), "validation": v} for k, (a, b, v) in results.items()},
               "comparison_old_policy_a": cmp_a, "policy_b_path": {k: list(map(int, v)) for k, v in b_path.items()}},
              open(out / "reports" / "validation_results.json", "w", encoding="utf-8"), indent=2)
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
