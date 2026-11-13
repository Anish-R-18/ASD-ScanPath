# Tests

Lightweight tests written with the standard-library `unittest`. They also run under `pytest`.

```bash
python -m unittest discover -s tests -t .
```

| File | Checks | Needs |
|---|---|---|
| `test_repository_hygiene.py` | No dataset files, checkpoints, raw outputs, local-only folders, stimulus figures, files over 10 MB, or machine-specific absolute paths would be published. `tppgaze/` matches upstream commit `59cf572` (the sample photo `data/1009.jpg` is not distributed; it is checked only if present). Markdown links resolve. | Git |
| `test_imports_and_cli.py` | Every maintained Python file compiles. The core modules import. The command-line tools show `--help`. The trainers default to the reported configs. | — |
| `test_data_logic.py` | Saliency4ASD preprocessing rules on synthetic sequences. The fit-to-height and stretch mappings. The split (70/15/15, seeded) and that it reproduces the published split. Behavioural metrics on a hand-checked path. | — |
| `test_conditioning.py` | The embedding is 2 × 512 and zero-initialised. A zero embedding gives exactly the Phase 1 NLL. A non-zero embedding separates ASD and TD. Only 1,024 parameters are trainable in embedding-only mode. | DenseNet201 via `torch.hub` (internet on first run); skipped otherwise |
| `test_checkpoints.py` | Checkpoint SHA-256 hashes. Phase 2 = the 1,290 unchanged Phase 1 tensors + `group_emb.weight` (2 × 512). | The trained checkpoints; skipped otherwise |
| `test_results_consistency.py` | Headline numbers in `results/` match the verified values, and the README and `results/final/KEY_RESULTS.md` state them | — |

`data/tppgaze_upstream_59cf572.sha256` lists the SHA-256 of every file in the official TPP-Gaze repository at commit `59cf572`.
