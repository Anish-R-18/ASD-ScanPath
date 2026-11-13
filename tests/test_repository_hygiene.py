"""The public repository must never contain datasets, participant records, stimulus photographs,
checkpoints, raw outputs or machine-specific paths, and its documentation links must resolve."""
from __future__ import annotations
import hashlib
import re
import unittest
from pathlib import PurePosixPath

from tests._common import ROOT, publishable_files

FILES = publishable_files()

FORBIDDEN_PREFIXES = (
    "TrainingData/", "AdditionalData/", "external_data/", "processed_data/", "training/runs/",
    "paper/", "local_only/", "inference/inputs/", "inference/outputs/", "outputs/",
)
FORBIDDEN_PARTS = ("/local_only/", "/local_only_restricted/", "meeting_walkthrough/01_example_image/")
FORBIDDEN_SUFFIXES = (".pth", ".pt", ".ckpt", ".safetensors", ".mat", ".npz", ".edf", ".asc", ".gz")
# Figures known to show dataset stimulus photographs (removed from the tree in the cleanup).
FORBIDDEN_NAMES = re.compile(r"(representative_scanpaths|^photo_\d+|^overlay_\d+|^rvg_\d+|beach|^1009$)", re.I)
TSV_ALLOWED = "scripts/research_completion/external_reference/"
MAX_BYTES = 10 * 1024 * 1024
ABS_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+|/home/user\d+|OneDrive[\\/]+Desktop")
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".yaml", ".yml", ".js", ".ps1", ".cff", ".sh"}


@unittest.skipIf(FILES is None, "not a Git checkout")
class RepositoryHygiene(unittest.TestCase):
    def test_no_dataset_checkpoint_or_raw_output_paths(self):
        bad = [f for f in FILES if f.startswith(FORBIDDEN_PREFIXES) or any(p in "/" + f for p in FORBIDDEN_PARTS)]
        self.assertEqual(bad, [], "dataset, raw-output or local-only files would be published")

    def test_no_forbidden_file_types(self):
        bad = [f for f in FILES if f.lower().endswith(FORBIDDEN_SUFFIXES)
               or (f.lower().endswith(".tsv") and not f.startswith(TSV_ALLOWED))]
        self.assertEqual(bad, [], "checkpoints, raw eye-tracking files or archives would be published")

    def test_no_known_stimulus_figures(self):
        bad = [f for f in FILES if FORBIDDEN_NAMES.search(PurePosixPath(f).stem)]
        self.assertEqual(bad, [], "figures that show dataset stimuli would be published")

    def test_file_sizes(self):
        big = [f for f in FILES if (ROOT / f).stat().st_size > MAX_BYTES]
        self.assertEqual(big, [], "files larger than 10 MB would be published")

    def test_no_machine_specific_absolute_paths(self):
        bad = []
        for f in FILES:
            if PurePosixPath(f).suffix.lower() in TEXT_SUFFIXES and not f.startswith("tests/"):
                text = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
                if ABS_PATH.search(text):
                    bad.append(f)
        self.assertEqual(bad, [], "files contain absolute paths of a specific machine or account")

    def test_markdown_relative_links_resolve(self):
        link = re.compile(r"\]\(([^)\s]+)\)")
        broken = []
        for f in FILES:
            if not f.endswith(".md"):
                continue
            text = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
            for target in link.findall(text):
                if re.match(r"^(https?:|mailto:|#)", target):
                    continue
                path = target.split("#", 1)[0]
                if path and not (ROOT / f).parent.joinpath(path).resolve().exists():
                    broken.append(f"{f} -> {target}")
        self.assertEqual(broken, [], "broken relative links in Markdown files")


class UpstreamCode(unittest.TestCase):
    """Runs without Git: the vendored TPP-Gaze files must match upstream commit 59cf572.

    The manifest hashes the files as stored in the upstream Git repository (LF line endings). Text files are
    compared after converting CRLF to LF, so the check gives the same answer on Windows checkouts.
    NOT_DISTRIBUTED lists upstream files deliberately left out of this repository; if present locally
    (for example after downloading them from upstream) they must still match."""

    NOT_DISTRIBUTED = {"data/1009.jpg"}  # upstream sample photo: image-specific redistribution rights unclear

    def test_tppgaze_is_unmodified_upstream_code(self):
        manifest = (ROOT / "tests" / "data" / "tppgaze_upstream_59cf572.sha256").read_text().splitlines()
        self.assertEqual(len(manifest), 22)
        for line in manifest:
            digest, name = line.split("  ", 1)
            with self.subTest(file=name):
                path = ROOT / "tppgaze" / name
                if name in self.NOT_DISTRIBUTED and not path.exists():
                    continue
                data = path.read_bytes()
                if b"\0" not in data:  # text file: undo Windows line-ending conversion
                    data = data.replace(b"\r\n", b"\n")
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
