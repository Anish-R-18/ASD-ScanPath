"""Headline numbers: the curated result files reproduce the verified values (final manuscript and
research-completion audit), and the README and key-results page state them the same way."""
from __future__ import annotations
import csv
import json
import unittest

from tests._common import ROOT

R = ROOT / "results"


def load(path):
    return json.loads((R / path).read_text(encoding="utf-8"))


def fmt(x, nd=4):
    """Signed value as written in the documents, with a true minus sign."""
    return f"{x:+.{nd}f}".replace("-", "−")


P2EVAL = load("in_domain/phase2_test_eval.json")
P1EVAL = load("in_domain/phase1_test_eval.json")
E3 = load("in_domain/e3_indomain_gain_ci.json")["overall_joint"]
E2 = load("sensitivity_analysis/e2_mapping_policy_nll_results.json")["results"]
E1 = {c["metric"]: c for c in load("behavioural_analysis/e1_results.json")["results"]["gaze4asd"]["contrasts"]}
with open(R / "external_evaluation" / "tables" / "external_nll_phase1_vs_phase2.csv", encoding="utf-8") as f:
    EXT = {(r["group"], r["component"]): r for r in csv.DictReader(f)}

# value in the result files -> the verified, rounded value used in every document
EXPECTED = {
    "pretrained test NLL": (P1EVAL["pretrained_same_stats"]["nll"], "−0.1665"),
    "Phase 1 test NLL": (P2EVAL["phase1"]["overall"]["nll"], "−0.4957"),
    "Phase 2 test NLL": (P2EVAL["phase2"]["overall"]["nll"], "−0.5030"),
    "in-domain gain": (E3["gain"], "+0.0073"),
    "in-domain CI low": (E3["image_ci"][0], "−0.0012"),
    "in-domain CI high": (E3["image_ci"][1], "+0.0163"),
    "external Phase 1 NLL": (float(EXT[("overall", "joint")]["phase1_nll"]), "−0.5568"),
    "external Phase 2 NLL": (float(EXT[("overall", "joint")]["phase2_nll"]), "−0.5396"),
    "external gain": (float(EXT[("overall", "joint")]["improvement"]), "−0.0172"),
    "external TD gain": (float(EXT[("TD", "joint")]["improvement"]), "−0.0229"),
    "external ASD gain": (float(EXT[("ASD", "joint")]["improvement"]), "+0.0104"),
    "stretch gain": (E2["stretch_policy_a"]["overall_joint"]["gain"], "−0.0274"),
    "Policy B gain": (E2["fit_height_policy_b"]["overall_joint"]["gain"], "−0.0087"),
}
SHARES = {"dispersion": 69, "sacc_amp_norm_mean": 64, "revisit_rate": 32}
WRONG_DIRECTION = ("total_dur", "sacc_dir_resultant", "n_fix")


class ResultFiles(unittest.TestCase):
    def test_likelihood_values(self):
        for name, (value, written) in EXPECTED.items():
            with self.subTest(name=name):
                self.assertEqual(fmt(value), written)

    def test_in_domain_gain_is_inconclusive(self):
        self.assertLess(E3["image_ci"][0], 0)
        self.assertGreater(E3["image_ci"][1], 0)
        self.assertEqual((E3["images_favouring_phase2"], E3["images"]), (27, 45))

    def test_external_phase1_better_in_all_three_versions(self):
        for version in ("fit_height_policy_a", "stretch_policy_a", "fit_height_policy_b"):
            with self.subTest(version=version):
                r = E2[version]["overall_joint"]
                self.assertLess(r["gain"], 0)
                self.assertLess(r["participant_ci"][1], 0)

    def test_generated_contrast_shares_and_directions(self):
        for metric, pct in SHARES.items():
            with self.subTest(metric=metric):
                self.assertTrue(E1[metric]["same_direction"])
                self.assertEqual(round(100 * E1[metric]["phase2_share_of_real"]), pct)
        for metric in WRONG_DIRECTION:
            with self.subTest(metric=metric):
                self.assertFalse(E1[metric]["same_direction"])


class Documents(unittest.TestCase):
    def check(self, rel):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for name, (_, written) in EXPECTED.items():
            if name in ("external TD gain", "pretrained test NLL") and rel != "README.md":
                continue
            with self.subTest(doc=rel, value=name):
                self.assertIn(written, text)
        for pct in SHARES.values():
            with self.subTest(doc=rel, share=pct):
                self.assertIn(f"{pct}%", text)

    def test_readme(self):
        self.check("README.md")

    def test_key_results_page(self):
        self.check("results/final/KEY_RESULTS.md")


if __name__ == "__main__":
    unittest.main()
