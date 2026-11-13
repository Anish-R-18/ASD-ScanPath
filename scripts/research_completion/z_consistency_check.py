"""
Research completion: traceability check. Every key number cited in the research_completion documents is recomputed
from the saved outputs and compared at the precision it is cited; then each cited string is searched for in the documents.
Read-only.

  python scripts/research_completion/z_consistency_check.py
"""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")  # repository root
os.chdir(ROOT)
import numpy as np

O = "training/runs/research_completion"
DOCS = {d: open(f"docs/research_audit/{d}", encoding="utf-8").read() for d in os.listdir("docs/research_audit") if d.endswith(".md")}
# The manuscript changelog is kept with the unpublished manuscript (paper/, not in the public repository).
CHANGELOG = "paper/MANUSCRIPT_CHANGELOG_research_completion.md"
if os.path.exists(CHANGELOG):
    DOCS["MANUSCRIPT_CHANGELOG.md"] = open(CHANGELOG, encoding="utf-8").read()
e1 = json.load(open(f"{O}/e1_phase_tau1_v2/e1_results.json"))
e2 = json.load(open(f"{O}/e2_mapping_policy_nll_v1/e2_results.json"))["results"]
e3 = json.load(open(f"{O}/e3_indomain_gain_ci_v1/e3_results.json"))
a346 = json.load(open(f"{O}/a346_sensitivity_stats_v1/a346_results.json"))
e1c = json.load(open(f"{O}/e1c_cpu_gpu_check_v2/cpu_gpu_check.json"))
chk = json.load(open(f"{O}/e1_phase_tau1_v2/run_checks.json"))
C = lambda ds, m: next(r for r in e1["results"][ds]["contrasts"] if r["metric"] == m)
MT = lambda ds, m, g: next(r for r in e1["results"][ds]["match_to_real"] if r["metric"] == m and r["group"] == g)
f4 = lambda x: f"{x:+.4f}".replace("-", "−")
pct = lambda x: f"{100 * x:.0f}%"
claims = []  # (description, cited string, recomputed string, document that must contain it)
def add(desc, cited, value, doc):
    claims.append((desc, cited, value, doc))

# E1 conditioning effects and shares (Gaze4ASD)
for m, eff, share in (("dispersion", "+0.0107", "69%"), ("sacc_amp_norm_mean", "+0.0078", "64%"), ("revisit_rate", "−0.0227", "32%")):
    r = C("gaze4asd", m)
    add(f"E1 G4A {m} effect", eff, f4(r["conditioning_effect"]["mean"]), "ADDITIONAL_RESULTS.md")
    add(f"E1 G4A {m} share", share, pct(r["phase2_share_of_real"]), "ADDITIONAL_RESULTS.md")
add("E1 primary Holm max", "0.0006", f"{max(r['p_holm_primary'] for r in e1['primary']):.4f}", "ADDITIONAL_RESULTS.md")
for m, g, cited in (("revisit_rate", "ASD", "−11.8%"), ("dur_mean", "ASD", "−32.5%"), ("dur_mean", "TD", "+28.2%"), ("n_fix", "TD", "+10.0%"), ("fix_rate_per_s", "TD", "+22.6%"), ("sacc_amp_norm_mean", "TD", "−9.1%"), ("dispersion", "TD", "−6.5%"), ("revisit_rate", "TD", "+13.0%")):
    add(f"E1 match {g} {m}", cited, f"{MT('gaze4asd', m, g)['rel_pct']:+.1f}%".replace("-", "−"), "ADDITIONAL_RESULTS.md")
t2 = {(r["metric"], r["group"]): r for r in e1["tau2_reference"]["match_to_real"]}
for m, g, cited in (("sacc_amp_norm_mean", "TD", "−25.5%"), ("dispersion", "TD", "−32.5%"), ("revisit_rate", "TD", "−21.4%")):
    add(f"tau2 match {g} {m}", cited, f"{t2[(m, g)]['rel_pct']:+.1f}%".replace("-", "−"), "ADDITIONAL_RESULTS.md")
t2c = {r["metric"]: r for r in e1["tau2_reference"]["contrasts"]}
for m, cited in (("dispersion", "46%"), ("sacc_amp_norm_mean", "33%"), ("revisit_rate", "15%")):
    add(f"tau2 share {m}", cited, pct(t2c[m]["share"]), "ADDITIONAL_RESULTS.md")
mm = {(r["target"], r["metric"]): r for r in e1["multimatch"]}
add("E1 MM ASD position", "−0.0062", f4(mm[("ASD", "mm_position")]["conditioning_effect"]["mean"]), "ADDITIONAL_RESULTS.md")
add("E1 MM TD position", "+0.0074", f4(mm[("TD", "mm_position")]["conditioning_effect"]["mean"]), "ADDITIONAL_RESULTS.md")
add("E1 off-image P2_ASD", "0.42%", f"{100 * chk['gaze4asd']['off_image']['P2_ASD']:.2f}%", "ADDITIONAL_RESULTS.md")
add("E1 off-image P2_TD", "0.25%", f"{100 * chk['gaze4asd']['off_image']['P2_TD']:.2f}%", "ADDITIONAL_RESULTS.md")
add("E1 CPU-GPU max |z|", "2.39", f"{chk['max_abs_z']:.2f}", "STATISTICAL_VALIDATION.md")
add("E1c min p", "0.061", f"{e1c['min_p']:.3f}", "ADDITIONAL_RESULTS.md")
add("E1c amp contrast p", "0.050", f"{e1c['phase2_contrast_cpu_vs_gpu']['sacc_amp_norm_mean']['image_signflip_p']:.3f}", "ADDITIONAL_RESULTS.md")
# E2
for v, g, cited in (("fit_height_policy_a", "overall", "−0.0172"), ("stretch_policy_a", "overall", "−0.0274"), ("fit_height_policy_b", "overall", "−0.0087"),
                    ("fit_height_policy_a", "ASD", "+0.0104"), ("stretch_policy_a", "ASD", "+0.0027"), ("fit_height_policy_b", "ASD", "+0.0152"), ("fit_height_policy_a", "TD", "−0.0230")):
    add(f"E2 {v} {g} joint", cited, f4(e2[v][f"{g}_joint"]["gain"]), "ADDITIONAL_RESULTS.md")
for v, cited in (("fit_height_policy_a", "+0.0075"), ("stretch_policy_a", "−0.0003"), ("fit_height_policy_b", "+0.0122")):
    add(f"E2 {v} ASD position", cited, f4(e2[v]["ASD_position"]["gain"]), "MANUSCRIPT_CHANGELOG.md")
# E3
E3 = e3["results"] if "results" in e3 else e3
def find_e3(obj, group, comp):
    for k, v in (obj.items() if isinstance(obj, dict) else []):
        if isinstance(v, dict) and k.lower().startswith(group.lower()) and comp in k.lower():
            return v
    return None
r = find_e3(E3, "overall", "joint")
if r is not None:
    add("E3 overall gain", "+0.0073", f4(r.get("gain", r.get("mean", np.nan))), "MANUSCRIPT_CHANGELOG.md")
    ci_ = r.get("image_ci", r.get("ci"))
    add("E3 overall CI low", "−0.0012", f4(ci_[0]), "MANUSCRIPT_CHANGELOG.md")
    add("E3 overall CI high", "+0.0163", f4(ci_[1]), "MANUSCRIPT_CHANGELOG.md")
else:
    print("E3 structure not recognised; keys:", list(E3)[:10])
# A4
A4 = a346["A4_policy_accounting"]
add("A4 ASD fixations removed", "8.07%", f"{A4['pct_fixations_removed']['ASD']:.2f}%", "ADDITIONAL_RESULTS.md")
add("A4 edge ASD", "7.35%", f"{A4['pct_fixations_on_image_edge_policy_a']['ASD']:.2f}%", "ADDITIONAL_RESULTS.md")
add("A4 edge diff CI", "+0.020", f"{A4['participant_edge_fraction']['diff_ci'][0]:+.3f}", "ADDITIONAL_RESULTS.md")

bad = 0
for desc, cited, val, doc in claims:
    ok_val = cited.replace("+", "") == val.replace("+", "") or cited == val
    if doc not in DOCS:  # document not available in this checkout: check the value only
        if not ok_val:
            bad += 1
        print(f"{'OK ' if ok_val else 'BAD'} {desc:32s} cited {cited:9s} recomputed {val:9s} ({doc} not available; value checked only)")
        continue
    ok_doc = cited in DOCS[doc]
    if not (ok_val and ok_doc):
        bad += 1
    print(f"{'OK ' if ok_val and ok_doc else 'BAD'} {desc:32s} cited {cited:9s} recomputed {val:9s} in {doc}: {ok_doc}")
# no superseded native-coordinate numbers (old Gaze4ASD v1 results) in the new documents
for doc, txt in DOCS.items():
    if re.search(r"native[- ]coordinate (results|NLL) (show|give|are) ", txt):
        print("WARNING: possible superseded result cited in", doc); bad += 1
print(f"{len(claims)} claims checked; {bad} problems")
