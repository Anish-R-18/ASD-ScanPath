# Git History Cleanup (2026-11-13)

**Status: published.** The rewritten history is live on GitHub.

- **Rewrite.** The history of `main` was rewritten locally in two passes:
  - **Pass 1:** restricted dataset material and the paper PDF.
  - **Pass 2:** the upstream TPP-Gaze sample photo `tppgaze/data/1009.jpg`.
- **Commit.** The handover reorganisation was committed on top of the rewritten history as `c2345136e1551d16976bfe67f53ed86448ee2cea` ("Reorganise repository for research handover"; parent `3707633`).
- **Publication.** On 2026-11-13 it was published to https://github.com/Anish-R-18/ASD-ScanPath with a guarded force-push (Section 7).
  - **Guard.** The push used `--force-with-lease` against `bca2af03…`, which a fresh fetch confirmed was still the remote `main`.
  - **Result.** `+ bca2af0...c234513 main -> main (forced update)`.
- **Published branch.** The previous restricted history is no longer part of the active published branch.
  - **Remote refs.** GitHub's `main` and `HEAD` point to `c234513`, and there are no other branches, tags or pull-request refs.
  - **Verified on a fresh clone.** A fresh clone from GitHub has 8 commits and 278 files. None of the removed paths or blobs appears in any commit.
- **Still pending: GitHub cache removal.** GitHub still serves the old commits when they are requested directly by their SHA (Section 5).
- **Not recallable.** Copies downloaded or cloned before the force-push cannot be recalled automatically, so collaborators must make a fresh clone (Section 6).

## 1. Why the history had to be rewritten

The first public commit (`d48d165`, 2026-10-04) added material that may not be redistributed. Every later commit carried it forward:
- dataset stimulus photographs, including a photograph of two children;
- the Saliency4ASD usage agreement;
- audit files that list individual fixation records.

Removing these files from the current tree, as done in [`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md), does not remove them from the repository: every old commit still contains them, and anyone can check those commits out. Only a history rewrite removes them from what the repository publishes.

## 2. Audit of the complete history (before the rewrite)

**Scope.**
- **Commits:** all 7 commits on every ref (`main`, `origin/main`). There were no tags, stashes or other branches.
- **Paths and versions:** 125 distinct paths and 152 distinct blobs, including every earlier version and renamed path.
- **Images:** every image blob was inspected visually.
- **Text:** every text blob was scanned for record-level tables, raw example scanpaths, credentials, e-mail addresses and absolute machine paths.

| Historical material | Path(s) | Blob | Commits | Classification | Action |
|---|---|---|---|---|---|
| Saliency4ASD stimulus photographs with generated scanpaths, including **two children** | `results/final_results/figures/fig4a_representative_scanpaths_saliency4asd.png` | `4a45e9cd` | all 7 | **Restricted** (dataset stimuli; images of children) | Removed from every commit |
| Gaze4ASD stimulus photographs (identifiable faces) | `results/final_results/figures/fig4b_representative_scanpaths_gaze4asd.png` and `results/external_gaze4asd/fig_representative_scanpaths.png` (same blob) | `4ffb20b1` | all 7 | **Restricted / uncertain** (stimulus source and rights not stated) | Removed from every commit |
| Saliency4ASD usage agreement | `docs/UsageAgreement.txt` | `e82a5657` | all 7 | **Restricted / uncertain** (part of the dataset distribution) | Removed from every commit; key clauses are quoted in `docs/dataset_documentation.md` |
| TPP-Gaze paper PDF | `docs/TPP-Gaze_Modelling_..._Processes (2).pdf` | `356f3859` | all 7 | **Uncertain** (third-party publication; redistribution rights not established) | Removed from every commit; linked on arXiv instead |
| Saliency4ASD audit with per-participant segment tables | `docs/dataset_audit.txt` (2 versions) | `66c8ed77`, `d81e91d9` | all 7 | **Restricted** (record-level dataset content) | Each version **redacted**: the six per-segment tables and "Rows per Segment" lines were removed (113 lines); everything else kept |
| Duration audit with individual fixation records (file, segment, duration, screen position) | `docs/duration_audit.txt` (2 versions) | `60e7c489`, `919332f4` | all 7 | **Restricted** (record-level dataset content) | Each version **redacted**: the two "Top 10" record tables were removed (25 lines); the aggregate tables kept |
| Upstream TPP-Gaze sample photograph (a puppy) | `tppgaze/data/1009.jpg` | `498031cf` | all 7 | **Uncertain** (shipped by the TPP-Gaze authors, but the photo's own redistribution rights are not stated) | Removed from every commit (pass 2); Git-ignored; available from the upstream repository |

**Raw example scanpaths.**
- **Never committed.** The Saliency4ASD preprocessing report, which prints raw example scanpaths, was never committed. Its public copy in the staged tree is redacted by `scripts/collect_results.py`.
- **A false positive.** The only historical text match for "raw (Idx, x_px" is the `print` format string inside `preprocessing/preprocess_dataset.py`, which is code, not data.

### Material reviewed and kept

| Material | Reason it stays public |
|---|---|
| All other historical figures (20 blobs: plots, forest plots, histograms, the architecture diagram) | Visually checked: group-level plots, with no stimulus image |
| `results/**` tables and text reports, including `real_asd_vs_td_participant_level.csv` | Group-level statistics; one row per measure, not per participant |
| `tppgaze/figure.jpg` | The TPP-Gaze paper's overview figure, from the original repository (MIT). It is the authors' own illustration of their method, not a stimulus from either of our datasets. |
| Absolute paths in old script versions (the original workstation's home-folder path in 11 historical scripts; the training server's home-folder path in two historical configs) | Account and folder names only, with no data or credentials. Kept, to avoid altering historical code. The current tree contains none (checked by `tests/test_repository_hygiene.py`). |
| Author name and e-mail in commit metadata | Already public and chosen by the author; preserved unchanged |
| Credentials, tokens, keys | None found in any blob |

## 3. Method

1. **Backup.** Everything was backed up first to a local folder outside the repository and outside OneDrive (`gaze-ag_backups/2026-11-13_pre_history_rewrite/` in the owner's home folder):
   - `STATE.txt`: branch, HEAD, `origin/main`, the index tree, refs, remote and status;
   - `staged_changes.patch`, `staged_name_status.txt`, `staged_diff_stat.txt`;
   - `dot_git/`, a full copy of `.git`, including the index with the staged changes; it reproduces index tree `5c831101` and HEAD `bca2af03`;
   - `history_before_rewrite.bundle` (`git bundle --all`; verified);
   - `staged_tree_snapshot/`: all 278 staged files as plain files;
   - `local_untracked/`: the Git-ignored local material (`training/runs/` with the checkpoints, `paper/`, `local_only/`, the demo and presentation local-only folders, inference inputs and outputs).
2. **Rewrite in an isolated bare clone.** The rewrite ran in a bare clone of the committed history, so the working tree and index were never touched. The tool was `git filter-branch --index-filter` (Git 2.52.0; `git filter-repo` was not installed, and with only 7 commits `filter-branch` is reliable). For each commit, the index filter:
   - ran `git rm --cached` on the 5 restricted paths;
   - replaced each historical audit blob with its redacted version via `git update-index --cacheinfo`. The mapping covers all 4 historical blobs, and the filter stops if it meets an unmapped audit blob.
   - **Same rules as the current files.** The redaction rules are the ones applied to the current files in `docs/audits/`. Redacting the last historical versions reproduces the staged redacted files byte for byte (blobs `03a750d8` and `6e8d115b`), so history and the current tree are continuous.
3. **Per-commit verification in the clone.** Each rewritten commit differs from its original only in the 5 removed paths and the 2 redacted audits. The redaction diffs contain only removed lines plus the inserted notes. Author, committer, dates and messages are identical.
4. **Moving the branch.**
   - **Fetch.** The rewritten `main` was fetched into the working repository.
   - **Move.** `refs/heads/main` was moved with `git update-ref`, guarded by the old value `bca2af03`. This changes neither the index nor the working tree; there are 0 unstaged changes.
   - **Temporary ref.** The temporary ref was deleted.
5. **Pass 2 (`1009.jpg`).**
   - **Rewrite.** The same procedure was repeated in a new bare clone, with a single index filter: `git rm --cached --ignore-unmatch -- tppgaze/data/1009.jpg`.
   - **Per-commit check.** Each commit lost exactly that file, and its metadata is identical.
   - **Move.** `main` was moved from `388bf16` to `3707633`, guarded by `update-ref` with the old value.
   - **Staged tree.** The file was then removed from the staged tree with `git rm --cached`, keeping the local copy, which is now Git-ignored.
   - **Backup.** A second backup of the state between the passes is in `gaze-ag_backups/2026-11-13_before_1009_removal/`.

### Commit mapping

| Original (public) | After pass 1 | **Final (after pass 2)** | Message |
|---|---|---|---|
| `d48d165` | `875e9de` | `8999e2b` | Initial commit for ASD-ScanPath project |
| `d28e69b` | `4658b37` | `e4fdb64` | Update dataset note in README.md |
| `3de7383` | `a548afc` | `d31304e` | Add ASD TD scanpath inference demo |
| `be5989e` | `62f94f6` | `4d27d77` | Remove unused CILIA dataset references |
| `a60926f` | `4b4ea11` | `99f7f07` | Organize inference outputs by image |
| `654d11a` | `37759e3` | `08a9317` | Ignore local training artifacts |
| `bca2af0` | `388bf16` | `3707633` | Finalize corrected research results and documentation |

## 4. Verification commands and results

```bash
# objects that would be published: all of main's history plus the staged tree
{ git rev-list --objects main; git ls-tree -r $(git write-tree) | awk '{print $3}'; } | awk '{print $1}' | sort -u > reachable.txt

# none of the removed or unredacted blobs is reachable
for b in 4a45e9cd 4ffb20b1 356f3859 e82a5657 66c8ed77 d81e91d9 60e7c489 919332f4 498031cf; do
  grep -q "^$(git rev-parse $b)" reachable.txt && echo "REACHABLE $b" || echo "absent $b"; done

# no restricted path in any reachable commit
for c in $(git rev-list main); do git ls-tree -r --name-only $c; done | sort -u \
  | grep -iE "representative_scanpaths|UsageAgreement|\.pdf$|\.pth$|\.mat$|\.npz$|TrainingData/|external_data/|processed_data/"

# content scan of every reachable blob (record tables, raw scanpaths, secrets, e-mails, paths)
#   (same patterns as above; see the audit table in Section 2)
python -m unittest discover -s tests -t .
```

| Check | Result |
|---|---|
| 9 removed or unredacted blobs (`4a45e9cd`, `4ffb20b1`, `356f3859`, `e82a5657`, `66c8ed77`, `d81e91d9`, `60e7c489`, `919332f4`, `498031cf`) reachable from `main` or the staged tree | **None** |
| Restricted paths in any reachable commit | **None** |
| Record-level patterns in reachable blobs (per-segment tables, "Top 10" fixation-record tables, rows of file / segment / duration / screen position, raw example scanpaths) | **None.** Only the code format string in `preprocess_dataset.py`, as expected. |
| Credentials, tokens, keys, e-mail addresses in reachable blobs | **None** |
| Images reachable from `main` or staged, including media inside both PPTX decks (46 new images and 21 historical) | **No stimulus photographs.** Blank canvases, the synthetic scene, plots and diagrams only. |
| Index tree before vs after each rewrite pass | Pass 1: `5c831101` = `5c831101`; pass 2: `6f05a8ef` = `6f05a8ef` (**unchanged**) |
| Unstaged changes after the rewrite | 0 |
| Branch and remote | `main`; `origin` = `https://github.com/Anish-R-18/ASD-ScanPath.git` (unchanged) |

The full test, demo, consistency and link checks after the rewrite are recorded in [`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md) (Section 6).

## 5. Remaining risks

**What a rewrite cannot undo.**
- **Existing copies.** Rewriting history does not remove copies that already exist elsewhere. The old history, with the restricted files, has been public since 2026-10-04, so it may exist in:
  - clones and downloaded ZIP archives on other machines;
  - forks of the GitHub repository (forks keep their own history);
  - GitHub's caches. Old commits stay viewable by their SHA until GitHub garbage-collects them; there were no forks or pull requests at publication.
  - third-party archives and mirrors, for example Software Heritage, search-engine caches and package or code-search indexes.

**GitHub cache removal: still pending.**
- **Checked after publication.** The old commit page and the old raw file addresses still respond when requested directly by SHA, for example `…/commit/bca2af03a91209a305eafa1c3dc2197fe42a41c2`.
- **What is needed.** Only GitHub Support can purge these cached views and garbage-collect the unreachable objects.
- **The request** (at https://support.github.com/contact) should name:
  - the repository, `Anish-R-18/ASD-ScanPath`;
  - the old commits: `d48d165`, `d28e69b`, `3de7383`, `be5989e`, `a60926f`, `654d11a`, `bca2af03a91209a305eafa1c3dc2197fe42a41c2`;
  - the reason: the removed files include dataset stimulus photographs of children.
- **Afterwards,** check that the old commit URL returns "not found".

**Local copies of the old history.** These are kept deliberately until GitHub Support confirms the cleanup:
- the reflog and the old objects in the original working copy's `.git`;
- the private backups outside the repository.

None of this is pushed by `git push origin main`.

**When GitHub Support confirms the cleanup**, the local copy can be purged:
```bash
git reflog expire --expire=now --all
git gc --prune=now
```

**The backup keeps everything.** It contains the restricted material, so keep it private and local.

## 6. Instructions for collaborators

The force-push was made on 2026-11-13. Anyone who cloned or downloaded the repository before then must make a fresh clone. Existing downloaded copies and old clones cannot be updated or recalled automatically. They should:

1. **Make a fresh clone instead of pulling.** Pulling or merging would bring the old history back:
   ```bash
   git clone https://github.com/Anish-R-18/ASD-ScanPath.git ASD-ScanPath-clean
   ```
2. **Move any unpushed work to the new history.** For example:
   ```bash
   # in the OLD clone: export your own commits as patches
   git format-patch <old-base-commit>..<your-branch> -o ../my-patches
   # in the NEW clone: apply them
   git am ../my-patches/*.patch
   ```
3. **Delete the old clone and any downloaded archives** of the old repository.
4. **Never push from an old clone.** A push from an old clone would re-publish the removed material.

## 7. Publication (done on 2026-11-13, with the owner's approval)

```bash
git commit -m "Reorganise repository for research handover"   # -> c2345136e1551d16976bfe67f53ed86448ee2cea
git fetch origin
git rev-parse origin/main        # printed bca2af03a91209a305eafa1c3dc2197fe42a41c2 (remote unchanged)
git push --force-with-lease=main:bca2af03a91209a305eafa1c3dc2197fe42a41c2 origin main
#   + bca2af0...c234513 main -> main (forced update)
```

**Verification after the push:**

| Check | Result |
|---|---|
| `git ls-remote origin` | `HEAD` and `refs/heads/main` point to `c234513`; no other refs |
| Fresh clone from GitHub | 8 commits and 278 files; the tree is identical to the local commit |
| Removed paths and blobs in the fresh clone | None present in any commit or in the object store |
| Tests in the fresh clone | 33 run: 30 pass, 3 checkpoint tests skipped because the checkpoints are not distributed |
| Forks of the repository | 0 |

**Follow-up, in this order:**
1. Ask GitHub Support to remove the cached views (Section 5). **Pending.**
2. Tell collaborators to re-clone (Section 6).
3. After GitHub confirms, purge the old objects locally (Section 5) and decide when to delete the private backups.

All later updates to `main` are normal pushes, without force.
