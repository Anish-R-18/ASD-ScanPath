"""Every Python file compiles, the core modules import, and the command-line entry points start."""
from __future__ import annotations
import importlib
import py_compile
import unittest

from tests._common import ROOT, publishable_files, run_python

CORE_MODULES = [
    "training.conditioned", "training.dataset", "training.splits", "training.model_utils",
    "training.metrics", "training.generate_cond", "training.external_gaze4asd_loader",
    "tppgaze.tppgaze", "tppgaze.utils",
]


class Imports(unittest.TestCase):
    def test_all_python_files_compile(self):
        # Maintained code only: tppgaze/ is unmodified upstream code, archive/ is kept for provenance.
        published = publishable_files()
        if published is not None:
            files = [ROOT / f for f in published if f.endswith(".py")]
        else:
            files = [p for d in ("training", "preprocessing", "inference", "scripts", "tests", "presentation")
                     for p in (ROOT / d).rglob("*.py")]
        files = [p for p in files if p.relative_to(ROOT).parts[0] not in ("tppgaze", "archive")]
        self.assertGreater(len(files), 30)
        for p in files:
            with self.subTest(file=str(p.relative_to(ROOT))):
                py_compile.compile(str(p), doraise=True)

    def test_core_modules_import(self):
        for name in CORE_MODULES:
            with self.subTest(module=name):
                importlib.import_module(name)


class CommandLine(unittest.TestCase):
    def test_help_messages(self):
        for script in ("inference/inference.py", "training/train_tpp.py", "training/train_tpp_cond.py",
                       "preprocessing/preprocess_dataset.py", "preprocessing/preprocess_gaze4asd_v2.py",
                       "scripts/collect_results.py"):
            with self.subTest(script=script):
                r = run_python(script, "--help")
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                self.assertIn("usage", r.stdout.lower())

    def test_trainers_default_to_the_reported_configs(self):
        for script, cfg in (("training/train_tpp.py", "phase1_baseline.yaml"),
                            ("training/train_tpp_cond.py", "phase2_embedding_only.yaml")):
            with self.subTest(script=script):
                text = (ROOT / script).read_text(encoding="utf-8")
                self.assertIn(f'"configs", "{cfg}"', text)
                self.assertTrue((ROOT / "configs" / cfg).exists())


if __name__ == "__main__":
    unittest.main()
