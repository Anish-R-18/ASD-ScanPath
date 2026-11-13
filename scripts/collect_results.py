"""
Copy the final, publishable result summaries from local Git-ignored outputs into results/.

No model is run and nothing is recomputed. Every file is copied from an explicit list below (byte-for-byte,
except that absolute machine paths in text files are replaced by "<project_root>" and the raw example
scanpaths in the Saliency4ASD preprocessing report are removed), and each destination folder gets a PROVENANCE.csv with the source path and SHA-256.

Only group-level summaries are copied. Excluded on purpose: datasets, processed scanpaths,
per-scanpath / per-trial / per-participant records, generated scanpaths, checkpoints, logs,
and any figure that shows a dataset stimulus photograph.

The external Gaze4ASD package (results/external_evaluation/) is built separately by
training/build_results_v2.py.

  python scripts/collect_results.py          # copy (overwrites the listed files)
  python scripts/collect_results.py --check  # only verify that results/ matches the local sources
"""
from __future__ import annotations
import argparse, csv, hashlib, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = "training/runs"
RC = f"{R}/research_completion"

# destination folder -> [(source, destination file name)]
PLAN = {
    "results/in_domain": [
        (f"{R}/baseline/test_eval.json", "phase1_test_eval.json"),
        (f"{R}/phase2_embedding_only/test_eval_phase2.json", "phase2_test_eval.json"),
        (f"{R}/baseline/summary_full.json", "phase1_training_summary.json"),
        (f"{R}/phase2_embedding_only/summary_full.json", "phase2_training_summary.json"),
        (f"{R}/baseline/model_config_used.yaml", "tppgaze_model_config_used.yaml"),
        (f"{R}/baseline/split.json", "saliency4asd_image_split_seed0.json"),
        (f"{RC}/e3_indomain_gain_ci_v1/e3_results.json", "e3_indomain_gain_ci.json"),
        ("processed_data/preprocessing_report.txt", "saliency4asd_preprocessing_report.txt"),
        ("processed_data/preprocessing_stats.json", "saliency4asd_preprocessing_stats.json"),
    ],
    "results/behavioural_analysis": [
        (f"{RC}/e1_phase_tau1_v2/e1_analysis_report.txt", "e1_analysis_report.txt"),
        (f"{RC}/e1_phase_tau1_v2/e1_results.json", "e1_results.json"),
        (f"{RC}/e1_phase_tau1_v2/run_checks.json", "e1_run_checks.json"),
        (f"{RC}/e1_phase_tau1_v2/environment.json", "e1_environment.json"),
        (f"{RC}/e1b_tau2_reference_v1/checks.json", "e1b_tau2_reference_checks.json"),
        (f"{RC}/e1d_multimatch_reference_v1/e1d_results.json", "e1d_multimatch_reference_results.json"),
        (f"{RC}/a5_multimatch_validation_v1/a5_results.json", "a5_multimatch_validation_results.json"),
    ],
    "results/sensitivity_analysis": [
        (f"{RC}/e2_mapping_policy_nll_v1/e2_results.json", "e2_mapping_policy_nll_results.json"),
        (f"{RC}/a346_sensitivity_stats_v1/a346_results.json", "a346_real_effects_policy_accounting_results.json"),
        (f"{RC}/e1c_cpu_gpu_check_v2/cpu_gpu_check.json", "e1c_cpu_gpu_check.json"),
    ],
    "results/temperature_analysis": [
        (f"{R}/temperature_sensitivity/final_report.txt", "validation_temperature_report.txt"),
        (f"{R}/temperature_sensitivity/temperature_methodology.txt", "validation_temperature_methodology.txt"),
        (f"{R}/temperature_sensitivity/temperature_comparison.csv", "validation_temperature_comparison_all.csv"),
        (f"{R}/temperature_sensitivity/statistical_comparison.json", "validation_statistical_comparison.json"),
        (f"{R}/temperature_sensitivity/behavioural_comparison.json", "validation_behavioural_comparison.json"),
        (f"{R}/temperature_sensitivity/figures/fig3_group_differences.png", "figures/validation_group_differences_by_temperature.png"),
        (f"{R}/temperature_sensitivity/figures/fig4_multimatch.png", "figures/validation_multimatch_by_temperature.png"),
        (f"{R}/temperature_sensitivity/figures/fig5_seed_stability.png", "figures/validation_seed_stability.png"),
    ],
    "results/figures": [
        ("paper/final/figures/fig1_architecture.png", "fig1_architecture.png"),
        ("paper/final/figures/fig2_pipeline.png", "fig2_pipeline.png"),
        ("paper/final/figures/fig3_likelihood_gains.png", "fig3_likelihood_gains.png"),
        ("paper/final/figures/fig4_real_effects_forest.png", "fig4_real_effects_forest.png"),
        ("paper/final/figures/fig5_e1_contrast_shares.png", "fig5_e1_contrast_shares.png"),
        ("paper/final/figures/fig6_temperature_tau1_vs_tau2.png", "fig6_temperature_tau1_vs_tau2.png"),
    ],
    "archive/results/in_domain_behaviour_tau2_initial": [
        (f"{R}/phase2_embedding_only/analysis/summary_tables.txt", "summary_tables.txt"),
        (f"{R}/phase2_embedding_only/analysis/aggregate_summary.json", "aggregate_summary.json"),
        (f"{R}/phase2_embedding_only/analysis/distribution_distances.json", "distribution_distances.json"),
        (f"{R}/phase2_embedding_only/analysis/similarity_multimatch.json", "similarity_multimatch.json"),
    ],
}


# Absolute machine paths recorded inside some summaries (training server and local PC) are replaced by
# "<project_root>" so that no account or folder names are published. Only text files are touched; the
# PROVENANCE.csv records the SHA-256 of the original source and whether the copy was modified for publication
# (machine paths replaced, or record-level examples removed by redact_examples).
TEXT = (".json", ".txt", ".csv", ".yaml", ".md")
SEP = r"(?:\\\\|\\|/)"  # JSON-escaped backslash pair, single backslash, or forward slash
ABS = re.compile(r'(/home/[^/"]+/gaze-ag|[A-Za-z]:' + SEP + r'Users' + SEP + r'[^"]*?gaze-ag)')


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def redact_examples(text: str) -> str:
    """The Saliency4ASD preprocessing report prints raw example scanpaths (individual participant records)
    in its section 7. Those rows are dataset content and are replaced by a note in the public copy."""
    start = text.find("7. EXAMPLE PROCESSED SCANPATHS")
    end = text.find("8. OUTPUT FILES")
    if start < 0 or end < 0:
        return text
    note = ("7. EXAMPLE PROCESSED SCANPATHS\n" + "-" * 88 + "\n[Removed from the public copy: this section lists raw "
            "individual scanpaths from the Saliency4ASD release, which may not be redistributed.]\n\n")
    return text[:start] + note + text[end:]


def content(src: str) -> tuple[bytes, bool]:
    with open(src, "rb") as f:
        data = f.read()
    if not src.endswith(TEXT):
        return data, False
    text = data.decode("utf-8")
    clean = ABS.sub("<project_root>", text)
    if src.endswith("preprocessing_report.txt"):
        clean = redact_examples(clean)
    return clean.encode("utf-8"), clean != text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare results/ with the local sources; copy nothing")
    args = ap.parse_args()
    os.chdir(ROOT)
    missing, differ = [], []
    for dest, items in PLAN.items():
        rows = []
        for src, name in items:
            out = f"{dest}/{name}"
            if not os.path.exists(src):
                missing.append(src)
                continue
            data, sanitized = content(src)
            rows.append(dict(file=name, source=src, source_sha256=sha256(src), modified_for_publication=sanitized))
            if args.check:
                if not os.path.exists(out) or open(out, "rb").read() != data:
                    differ.append(out)
            else:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "wb") as f:
                    f.write(data)
        if not args.check and rows:
            with open(f"{dest}/PROVENANCE.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["file", "source", "source_sha256", "modified_for_publication"])
                w.writeheader(); w.writerows(rows)
    for m in missing:
        print("missing local source (not copied):", m)
    for d in differ:
        print("differs from its source:", d)
    n = sum(len(v) for v in PLAN.values())
    print(f"{'checked' if args.check else 'copied'} {n - len(missing)} of {n} files")
    return 1 if (missing or differ) else 0


if __name__ == "__main__":
    sys.exit(main())
