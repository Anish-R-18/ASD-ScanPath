"""Shared helpers for the lightweight test suite (no dataset or checkpoint required)."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tppgaze")):
    if p not in sys.path:
        sys.path.insert(0, p)


def publishable_files() -> list[str] | None:
    """Files Git would publish: tracked files plus untracked files that are not ignored.
    Returns None when this is not a Git checkout (the hygiene tests are then skipped)."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                             cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    files = [f for f in out.decode("utf-8").split("\0") if f]
    return [f for f in files if (ROOT / f).exists()]  # skip deletions staged in the index


def run_python(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, env=env, encoding="utf-8")
