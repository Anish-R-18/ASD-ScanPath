#!/usr/bin/env python
"""
Saliency4ASD -> TPP-Gaze preprocessing pipeline.

Reads the raw scanpath files (read-only) and writes a cleaned, TPP-Gaze-ready
dataset to processed_data/. The original dataset is never modified: SHA-256
hashes of every raw scanpath file are compared before and after the run.

Pipeline, applied to every scanpath segment in this order:
  0. Split each file into segments at every Idx reset (Idx == 0).
  1. Drop the whole segment if any fixation exceeds --max-plausible-ms
     (default 3000 ms, which can't happen inside a 3 s stimulus window).
  2. Remove the first fixation (TPP-Gaze paper, Sec. 4.1).
  3. Compute each remaining fixation's onset-anchored start time on a 0-BASED
     timeline (cumulative sum of the RETAINED durations; saccade time is not
     recorded, so this is an approximation). The removed first fixation does NOT
     consume the t_end budget, and each retained duration is left unchanged.
  4. Remove fixations shorter than --min-fix-ms (default 60 ms).
  5. Truncate at --t-end (default 3.0 s) on this 0-based timeline: drop fixations
     starting at/after t_end and clip the duration of a fixation that straddles
     t_end. If the clipped remainder is < --min-fix-ms, that fixation is dropped too.
  6. Optionally clamp remaining durations to --clamp-max-ms (off by default).
  7. Drop the segment if fewer than --min-len fixations remain (default 4).
  8. Convert ms -> s and normalise 1-indexed pixel coords to [-1, 1]:
        x_norm = 2 * (x - 1) / (W - 1) - 1      (same for y with H)
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "TrainingData" / "TrainingData"
IMG_DIR = RAW_DIR / "Images"
GROUPS = {"ASD": RAW_DIR / "ASD", "TD": RAW_DIR / "TD"}
N_IMAGES = 300
README_SUBJECTS_PER_GROUP = 14  # "for each image, we selected 14 subjects" (README.txt)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(ROOT / "processed_data"))
    p.add_argument("--min-fix-ms", type=float, default=60.0)
    p.add_argument("--t-end", type=float, default=3.0, help="seconds")
    p.add_argument("--max-plausible-ms", type=float, default=3000.0)
    p.add_argument("--clamp-max-ms", type=float, default=None)
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--keep-first", action="store_true",
                   help="do NOT remove the first fixation")
    return p.parse_args()


# --------------------------------------------------------------------------- io
def sha256_files(paths):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def parse_scanpath_file(path):
    """Return (segments, idx_anomalies). Each segment is a list of (idx, x, y, dur_ms)."""
    segments, cur, anomalies = [], [], 0
    with open(path, "r") as f:
        f.readline()  # header: "Idx, x, y, duration"
        for line_no, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = [s.strip() for s in line.split(",")]
            if len(parts) != 4:
                raise ValueError(f"{path.name}:{line_no}: expected 4 columns, got {len(parts)}")
            idx, x, y, d = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            if idx == 0:
                if cur:
                    segments.append(cur)
                cur = []
            elif not cur or idx != cur[-1][0] + 1:
                anomalies += 1  # non-contiguous Idx inside a segment
            cur.append((idx, x, y, d))
    if cur:
        segments.append(cur)
    return segments, anomalies


# ---------------------------------------------------------------------- process
def process_segment(rows, W, H, cfg, c):
    """Apply the pipeline to one segment. Mutates counter dict `c`.
    Returns (scanpath or None, info)."""
    n = len(rows)
    c["raw_segments"] += 1
    c["raw_fixations"] += n
    durs_all = np.array([r[3] for r in rows], dtype=float)
    info = {"raw_n": n, "truncated": False, "clipped": False,
            "n_short_removed": 0, "reason": None}

    # 1. physically impossible duration -> drop whole segment (checked on the raw recording)
    if durs_all.max() > cfg.max_plausible_ms:
        c["seg_drop_impossible"] += 1
        c["fix_lost_impossible_seg"] += n
        info["reason"] = "impossible_duration"
        return None, info

    # 2. remove the first fixation FIRST, so its duration does not consume the t_end budget
    rows_kept = rows
    if not cfg.keep_first:
        rows_kept = rows[1:]
        c["fix_removed_first"] += 1

    # 3. onset-anchored start times on a 0-BASED timeline of the remaining fixations
    #    (durations are kept unchanged; only the timeline origin is reset to 0)
    durs = np.array([r[3] for r in rows_kept], dtype=float)
    if durs.size:
        starts = np.concatenate([[0.0], np.cumsum(durs)[:-1]])
    else:
        starts = np.empty(0, dtype=float)
    fix = [(r[1], r[2], r[3], float(s)) for r, s in zip(rows_kept, starts)]

    # 4. remove short fixations
    before = len(fix)
    fix = [f for f in fix if f[2] >= cfg.min_fix_ms]
    k = before - len(fix)
    c["fix_removed_short"] += k
    info["n_short_removed"] = k

    # 5. truncate at t_end
    t_end_ms = cfg.t_end * 1000.0
    before = len(fix)
    fix = [f for f in fix if f[3] < t_end_ms]
    k = before - len(fix)
    c["fix_removed_after_tend"] += k
    if k:
        info["truncated"] = True
    if fix and fix[-1][3] + fix[-1][2] > t_end_ms:
        x, y, d, s = fix[-1]
        new_d = t_end_ms - s
        info["truncated"] = True
        if new_d < cfg.min_fix_ms:
            fix = fix[:-1]
            c["fix_removed_clip_too_short"] += 1
        else:
            fix[-1] = (x, y, new_d, s)
            c["fix_clipped_at_tend"] += 1
            info["clipped"] = True
    if info["truncated"]:
        c["seg_truncated"] += 1

    # 6. optional clamp
    if cfg.clamp_max_ms is not None:
        c["fix_clamped_max"] += sum(1 for f in fix if f[2] > cfg.clamp_max_ms)
        fix = [(x, y, min(d, cfg.clamp_max_ms), s) for x, y, d, s in fix]

    # 7. minimum length
    if len(fix) < cfg.min_len:
        c["seg_drop_too_short"] += 1
        c["fix_lost_too_short_seg"] += len(fix)
        info["reason"] = "too_short"
        return None, info

    # 8. units + normalisation
    sp = []
    for x, y, d, _ in fix:
        xn = 2.0 * (x - 1.0) / (W - 1.0) - 1.0
        yn = 2.0 * (y - 1.0) / (H - 1.0) - 1.0
        if not (-1.0 <= xn <= 1.0 and -1.0 <= yn <= 1.0):
            c["coord_clamped"] += 1
        sp.append([round(float(np.clip(xn, -1, 1)), 6),
                   round(float(np.clip(yn, -1, 1)), 6),
                   round(d / 1000.0, 6)])
    c["valid_segments"] += 1
    c["valid_fixations"] += len(sp)
    if info["truncated"]:
        c["valid_truncated"] += 1
    return sp, info


# ------------------------------------------------------------------------ stats
def describe(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {"count": 0}
    p5, p25, p50, p75, p95 = np.percentile(a, [5, 25, 50, 75, 95])
    return {"count": int(a.size), "min": float(a.min()), "p5": float(p5),
            "p25": float(p25), "median": float(p50), "mean": float(a.mean()),
            "p75": float(p75), "p95": float(p95), "max": float(a.max()),
            "std": float(a.std())}


def fmt_desc(d, unit="", scale=1.0, nd=1):
    if d.get("count", 0) == 0:
        return "n=0"
    f = lambda v: f"{v * scale:.{nd}f}{unit}"
    return (f"n={d['count']:,}  min={f(d['min'])}  p25={f(d['p25'])}  "
            f"median={f(d['median'])}  mean={f(d['mean'])}  p75={f(d['p75'])}  "
            f"p95={f(d['p95'])}  max={f(d['max'])}")


# ------------------------------------------------------------------------- main
def main():
    cfg = parse_args()
    out_dir = Path(cfg.out_dir)
    if not RAW_DIR.exists():
        raise SystemExit(f"Raw dataset not found at {RAW_DIR}")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(p for g in GROUPS.values() for p in g.glob("*_scanpath_*.txt"))
    hashes_before = sha256_files(raw_files)

    sizes = {}
    for i in range(1, N_IMAGES + 1):
        with Image.open(IMG_DIR / f"{i}.png") as im:
            sizes[i] = im.size  # (W, H)

    # ---- pass 1: parse + duplicate detection
    raw = {}                       # (group, img) -> list of segments
    idx_anomalies = defaultdict(int)
    dup_index = defaultdict(set)   # rows-tuple -> {(group, img)}
    for g, gdir in GROUPS.items():
        for i in range(1, N_IMAGES + 1):
            segs, an = parse_scanpath_file(gdir / f"{g}_scanpath_{i}.txt")
            raw[(g, i)] = segs
            idx_anomalies[g] += an
            for s in segs:
                if len(s) >= 2:
                    dup_index[tuple(s)].add((g, i))
    dup_keys = {k for k, v in dup_index.items() if len(v) > 1}

    # ---- pass 2: process
    counters = {g: defaultdict(int) for g in GROUPS}
    records = {g: [] for g in GROUPS}
    removed = []
    raw_store = {}
    before_stats = {g: {"dur_ms": [], "seg_sum_ms": [], "fix_per_seg": [], "seg_per_img": []}
                    for g in GROUPS}
    after_stats = {g: {"dur_ms": [], "seg_sum_ms": [], "fix_per_seg": [], "seg_per_img": []}
                   for g in GROUPS}
    dup_flag_counts = {g: {"raw": 0, "valid": 0} for g in GROUPS}

    for g in GROUPS:
        c = counters[g]
        for i in range(1, N_IMAGES + 1):
            W, H = sizes[i]
            segs = raw[(g, i)]
            before_stats[g]["seg_per_img"].append(len(segs))
            n_valid_img = 0
            for si, rows in enumerate(segs, start=1):
                d = [r[3] for r in rows]
                before_stats[g]["dur_ms"].extend(d)
                before_stats[g]["seg_sum_ms"].append(sum(d))
                before_stats[g]["fix_per_seg"].append(len(rows))
                is_dup = tuple(rows) in dup_keys
                if is_dup:
                    dup_flag_counts[g]["raw"] += 1

                sp, info = process_segment(rows, W, H, cfg, c)
                if sp is None:
                    removed.append({"group": g, "image_id": i, "segment_index": si,
                                    "raw_n_fixations": len(rows),
                                    "raw_total_ms": sum(d),
                                    "raw_max_ms": max(d),
                                    "reason": info["reason"],
                                    "duplicate_flag": is_dup})
                    continue
                n_valid_img += 1
                if is_dup:
                    dup_flag_counts[g]["valid"] += 1
                durs_s = [p[2] for p in sp]
                after_stats[g]["dur_ms"].extend(v * 1000 for v in durs_s)
                after_stats[g]["seg_sum_ms"].append(sum(durs_s) * 1000)
                after_stats[g]["fix_per_seg"].append(len(sp))
                rec = {"image_id": i, "group": g, "segment_index": si,
                       "image_width": W, "image_height": H,
                       "n_fixations": len(sp),
                       "total_duration_s": round(sum(durs_s), 6),
                       "raw_n_fixations": len(rows),
                       "truncated": info["truncated"],
                       "short_fixations_removed": info["n_short_removed"],
                       "duplicate_flag": is_dup,
                       "scanpath": sp}
                records[g].append(rec)
                raw_store[(g, i, si)] = (rows, info)
            after_stats[g]["seg_per_img"].append(n_valid_img)

    # ---- integrity checks
    checks = []
    for g in GROUPS:
        c = counters[g]
        lost = (c["fix_lost_impossible_seg"] + c["fix_removed_first"] + c["fix_removed_short"]
                + c["fix_removed_after_tend"] + c["fix_removed_clip_too_short"]
                + c["fix_lost_too_short_seg"] + c["valid_fixations"])
        assert lost == c["raw_fixations"], f"{g}: fixation accounting mismatch"
        assert c["raw_segments"] == c["valid_segments"] + c["seg_drop_impossible"] + c["seg_drop_too_short"]
        for r in records[g]:
            sp = np.array(r["scanpath"])
            assert len(sp) >= cfg.min_len
            assert np.all(np.abs(sp[:, :2]) <= 1.0)
            assert np.all(sp[:, 2] >= cfg.min_fix_ms / 1000 - 1e-9)
            assert sp[:, 2].sum() <= cfg.t_end + 1e-6
    checks += ["fixation accounting identity holds for ASD and TD",
               "segment accounting identity holds for ASD and TD",
               f"every scanpath has >= {cfg.min_len} fixations",
               "every x,y lies in [-1, 1]",
               f"every duration >= {cfg.min_fix_ms / 1000:.3f} s",
               f"every scanpath total duration <= {cfg.t_end:.1f} s"]

    hashes_after = sha256_files(raw_files)
    unchanged = hashes_before == hashes_after
    assert unchanged, "Raw dataset files changed during preprocessing!"
    checks.append(f"all {len(raw_files)} raw scanpath files unchanged (SHA-256 before == after)")

    # ---- aggregate stats
    stats = {"generated": datetime.now().isoformat(timespec="seconds"),
             "config": vars(cfg), "groups": {}}
    for g in GROUPS:
        b, a, c = before_stats[g], after_stats[g], counters[g]
        log_d = np.log(np.array(a["dur_ms"]) / 1000.0)
        stats["groups"][g] = {
            "counters": dict(c),
            "idx_anomalies": idx_anomalies[g],
            "duplicate_segments": dup_flag_counts[g],
            "before": {"duration_ms": describe(b["dur_ms"]),
                       "segment_total_ms": describe(b["seg_sum_ms"]),
                       "fixations_per_segment": describe(b["fix_per_seg"]),
                       "segments_per_image": describe(b["seg_per_img"])},
            "after": {"duration_ms": describe(a["dur_ms"]),
                      "segment_total_ms": describe(a["seg_sum_ms"]),
                      "fixations_per_segment": describe(a["fix_per_seg"]),
                      "segments_per_image": describe(a["seg_per_img"]),
                      "images_with_zero_valid": int(sum(1 for v in a["seg_per_img"] if v == 0))},
            "log_duration_seconds": {"mean": float(log_d.mean()), "std": float(log_d.std())},
        }
    all_log = np.log(np.concatenate([np.array(after_stats[g]["dur_ms"]) for g in GROUPS]) / 1000.0)
    stats["log_duration_seconds_combined"] = {"mean": float(all_log.mean()), "std": float(all_log.std())}
    stats["integrity_checks"] = checks

    # ---- write outputs
    for g in GROUPS:
        with open(out_dir / f"{g.lower()}_scanpaths.json", "w", encoding="utf-8") as f:
            json.dump(records[g], f)
    with open(out_dir / "removed_segments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(removed[0].keys()))
        w.writeheader()
        w.writerows(removed)
    with open(out_dir / "preprocessing_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    examples = pick_examples(records, raw_store)
    report = build_report(cfg, stats, examples, raw_store, sizes)
    (out_dir / "preprocessing_report.txt").write_text(report, encoding="utf-8")

    print(f"ASD valid scanpaths: {counters['ASD']['valid_segments']}  "
          f"TD valid scanpaths: {counters['TD']['valid_segments']}")
    print(f"Outputs written to {out_dir}")


# --------------------------------------------------------------------- examples
def pick_examples(records, raw_store):
    chosen, keys = [], set()

    def take(group, pred, label):
        for r in records[group]:
            k = (r["group"], r["image_id"], r["segment_index"])
            if k not in keys and pred(r, raw_store[k][1]):
                keys.add(k)
                chosen.append((label, r))
                return

    take("ASD", lambda r, i: r["image_id"] == 1, "ASD, image 1, first valid scanpath")
    take("TD", lambda r, i: r["image_id"] == 1, "TD, image 1, first valid scanpath")
    take("ASD", lambda r, i: i["clipped"], "ASD, last fixation clipped at t_end")
    take("TD", lambda r, i: i["truncated"] and i["n_short_removed"] > 0,
         "TD, short fixations removed + truncated at t_end")
    take("ASD", lambda r, i: i["n_short_removed"] >= 3 and not i["truncated"],
         "ASD, several <60 ms fixations removed, no truncation")
    return chosen


# ----------------------------------------------------------------------- report
def build_report(cfg, stats, examples, raw_store, sizes):
    L = []
    p = L.append
    bar = "=" * 88
    p(bar)
    p("  SALIENCY4ASD -> TPP-GAZE PREPROCESSING REPORT")
    p(bar)
    p(f"Generated: {stats['generated']}")
    p("Config:    " + ", ".join(f"{k}={v}" for k, v in stats["config"].items() if k != "out_dir"))
    p("")
    p("Pipeline order (per segment):")
    p(f"  1. drop segment if any fixation > {cfg.max_plausible_ms:.0f} ms")
    p(f"  2. {'keep' if cfg.keep_first else 'remove'} first fixation")
    p("  3. onset-anchored start times on a 0-based timeline of the remaining fixations")
    p(f"  4. remove fixations < {cfg.min_fix_ms:.0f} ms")
    p(f"  5. truncate at {cfg.t_end:.1f} s on the 0-based timeline (drop after, clip straddling; drop clip < {cfg.min_fix_ms:.0f} ms)")
    p(f"  6. clamp durations: {cfg.clamp_max_ms if cfg.clamp_max_ms is not None else 'off'}")
    p(f"  7. drop segment if < {cfg.min_len} fixations remain")
    p("  8. ms -> s; x,y -> [-1,1] via 2*(v-1)/(dim-1)-1")
    p("")

    p("1. VALID SCANPATHS")
    p("-" * 88)
    p(f"{'':38}{'ASD':>14}{'TD':>14}")
    rows = [("raw segments", "raw_segments"),
            ("dropped: impossible duration (>max)", "seg_drop_impossible"),
            ("dropped: < min_len after cleaning", "seg_drop_too_short"),
            ("VALID scanpaths", "valid_segments"),
            ("  of which truncated at t_end", "valid_truncated")]
    for label, key in rows:
        a = stats["groups"]["ASD"]["counters"].get(key, 0)
        t = stats["groups"]["TD"]["counters"].get(key, 0)
        p(f"{label:38}{a:>14,}{t:>14,}")
    for g in ("ASD", "TD"):
        rs = stats["groups"][g]["counters"]["raw_segments"]
        vs = stats["groups"][g]["counters"]["valid_segments"]
        p(f"  {g} retention: {vs / rs * 100:.1f}% of raw segments")
    p("")

    p("2. FIXATION COUNTS (accounting: raw = sum of all removals + valid)")
    p("-" * 88)
    p(f"{'':38}{'ASD':>14}{'TD':>14}")
    rows = [("raw fixations", "raw_fixations"),
            ("lost in impossible-duration segments", "fix_lost_impossible_seg"),
            ("removed: first fixation", "fix_removed_first"),
            (f"removed: < {cfg.min_fix_ms:.0f} ms", "fix_removed_short"),
            ("removed: start >= t_end", "fix_removed_after_tend"),
            ("removed: clipped remainder too short", "fix_removed_clip_too_short"),
            ("lost in too-short segments", "fix_lost_too_short_seg"),
            ("VALID fixations", "valid_fixations"),
            ("  clipped at t_end (kept)", "fix_clipped_at_tend"),
            ("  coordinates clamped to [-1,1]", "coord_clamped")]
    for label, key in rows:
        a = stats["groups"]["ASD"]["counters"].get(key, 0)
        t = stats["groups"]["TD"]["counters"].get(key, 0)
        p(f"{label:38}{a:>14,}{t:>14,}")
    p("")

    p("3. DURATION STATISTICS (ms)")
    p("-" * 88)
    for g in ("ASD", "TD"):
        s = stats["groups"][g]
        p(f"[{g}] per-fixation duration")
        p("  before: " + fmt_desc(s["before"]["duration_ms"], nd=0))
        p("  after:  " + fmt_desc(s["after"]["duration_ms"], nd=0))
        p(f"[{g}] per-scanpath total duration")
        p("  before: " + fmt_desc(s["before"]["segment_total_ms"], nd=0))
        p("  after:  " + fmt_desc(s["after"]["segment_total_ms"], nd=0))
        p(f"[{g}] fixations per scanpath")
        p("  before: " + fmt_desc(s["before"]["fixations_per_segment"], nd=1))
        p("  after:  " + fmt_desc(s["after"]["fixations_per_segment"], nd=1))
        p(f"[{g}] scanpaths per image")
        p("  before: " + fmt_desc(s["before"]["segments_per_image"], nd=1))
        p("  after:  " + fmt_desc(s["after"]["segments_per_image"], nd=1))
        p(f"  images with zero valid scanpaths: {s['after']['images_with_zero_valid']}")
        p("")

    p("4. LOG-DURATION STATISTICS (seconds) - candidates for config.yaml")
    p("-" * 88)
    p("  Pretrained config.yaml: mean_log_inter_times=-1.5126, std_log_inter_times=0.5892")
    for g in ("ASD", "TD"):
        ld = stats["groups"][g]["log_duration_seconds"]
        p(f"  {g:4}: mean={ld['mean']:.4f}  std={ld['std']:.4f}")
    ld = stats["log_duration_seconds_combined"]
    p(f"  ALL : mean={ld['mean']:.4f}  std={ld['std']:.4f}")
    p("  NOTE: computed on ALL processed data. For training, recompute on the train split only.")
    p("")

    p("5. DATA-INTEGRITY FLAGS")
    p("-" * 88)
    for g in ("ASD", "TD"):
        s = stats["groups"][g]
        dup = s["duplicate_segments"]
        p(f"  {g}: non-contiguous Idx inside segments: {s['idx_anomalies']}")
        p(f"  {g}: raw segments identical to a segment in another image file: "
          f"{dup['raw']} (still present after cleaning: {dup['valid']})")
        mx = s["before"]["segments_per_image"]["max"]
        n_over = None
        p(f"  {g}: max raw segments per image = {mx:.0f} "
          f"(README states {README_SUBJECTS_PER_GROUP} subjects per group per image)")
    p("  Duplicated segments are flagged ('duplicate_flag') but NOT removed.")
    p("")

    p("6. INTEGRITY CHECKS (all asserted at runtime)")
    p("-" * 88)
    for c in stats["integrity_checks"]:
        p(f"  [PASS] {c}")
    p("")

    p("7. EXAMPLE PROCESSED SCANPATHS (raw -> processed)")
    p("-" * 88)
    for n, (label, r) in enumerate(examples, start=1):
        rows, info = raw_store[(r["group"], r["image_id"], r["segment_index"])]
        p(f"Example {n}: {label}")
        p(f"  image_id={r['image_id']}  group={r['group']}  segment={r['segment_index']}  "
          f"image={r['image_width']}x{r['image_height']}  raw_n={r['raw_n_fixations']}  "
          f"processed_n={r['n_fixations']}  total={r['total_duration_s']:.3f}s  "
          f"truncated={r['truncated']}")
        p("  raw (Idx, x_px, y_px, dur_ms): " +
          "  ".join(f"({a},{int(b)},{int(c)},{int(d)})" for a, b, c, d in rows))
        p("  processed (x, y, dur_s):")
        for x, y, d in r["scanpath"]:
            p(f"      ({x:+.4f}, {y:+.4f}, {d:.3f})")
        p("")

    p("8. OUTPUT FILES")
    p("-" * 88)
    p("  asd_scanpaths.json / td_scanpaths.json : list of records")
    p("     {image_id, group, segment_index, image_width, image_height, n_fixations,")
    p("      total_duration_s, raw_n_fixations, truncated, short_fixations_removed,")
    p("      duplicate_flag, scanpath: [[x_norm, y_norm, duration_s], ...]}")
    p("  removed_segments.csv     : every dropped segment with reason")
    p("  preprocessing_stats.json : all counters and statistics (machine-readable)")
    p("  preprocessing_report.txt : this report")
    p(bar)
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
