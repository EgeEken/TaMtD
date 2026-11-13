# Repository storage audit

This audit measured the repository before deleting local regenerable caches. Traversal did not follow reparse points. Allocated bytes came from Windows file allocation metadata; logical bytes are file lengths. Sizes use decimal GB unless GiB is shown. External Codex worktrees were measured separately and left untouched.

## Before cleanup

| Path/category | Allocated size | Logical size | Files | Git status | Purpose / action |
|---|---:|---:|---:|---|---|
| Entire TaMtD checkout | 25.844 GB (24.07 GiB) | 25.206 GB (23.47 GiB) | 337,278 | tracked plus ignored/untracked | Baseline; no reparse points followed. |
| `data/cache/` | 24.31 GB | — | about 319,927 | ignored | Dominated by generated per-sample features. |
| `data/cache/neural_accessibility/imagenette_320/` | 22.947 GB | — | — | ignored | Main benchmark cache; streams/manifests/IR/checkpoints retained, derived features removed. |
| Imagenette `features/` | 12.281 GB | — | 34 | ignored | Dense JPEG DCT and PBC feature arrays; deleted as regenerable. |
| Imagenette `phase47/features/` | 5.102 GB | — | 12 | ignored | AVIF/WebP stage features, including two pre-fix WebP maps; deleted as regenerable from retained streams and pinned instrumentation. |
| Imagenette `phase48/features/` | 2.637 GB | — | 5 | ignored | JPEG q1 feature arrays; deleted as regenerable. |
| All three feature-map trees | 20.020 GB combined | — | 51 | ignored | Dense `.npy` memmaps/feature arrays; largest original artifact type and safely regenerable from retained inputs. |
| Imagenette exact streams | 2.638 GB combined across `streams/`, `phase4_streams/`, `phase48_streams/` | — | — | ignored | Retained as exact experiment inputs. |
| Imagenette checkpoints | 206.0 MB | — | 123 | ignored | All retained. |
| Other CIFAR cache trees | 1.339 GB across eight principal trees | — | about 317,943 | ignored | Retained for provenance because some PBC streams depend on an external dirty PBC checkout. |
| `data/raw/` | 1.246 GB | — | 13,416 | ignored | Imagenette and CIFAR archives/extractions; retained. |
| `results/` | 139.7 MB | — | 1,277 | tracked plus untracked local logs/diagnostics | Findings, figures, CSV/JSON, early checkpoints and user logs; retained. |
| `.git/` | 94.2 MB | — | 1,246 | repository metadata | Retained; no history cleanup. `git count-objects` reported about 82 MiB loose objects and two old temporary garbage objects; no `gc`/prune performed. |
| `.uv-cache/` | 45.5 MB | — | 990 | untracked/ignored | Deleted as regenerable. |
| `.venv/` + `.venv313/` | about 1.9 MB | — | — | local environments | Retained. |
| Checkpoints overall | about 247 MB | about 247 MB | 186 `.pt` | ignored/untracked | All retained, including early results and Imagenette checkpoints. |

The latest pre-cleanup C: reading was 82.94 GiB free; an earlier inventory reading was 84.26 GiB. These are point-in-time measurements and varied during the audit.

## External worktrees

These are registered PBC repository worktrees outside TaMtD. Sizes below are allocated/logical bytes from the prior no-follow scan; they were not changed during this finalization. Git worktree records showed the branch and commit ancestry below. Each worktree references the PBC checkout's common Git object database, so Git objects are not a second copy inside these directory totals; checked-out files and local build/toolchain data are separate files.

| Worktree location | Allocated / logical | Branch at commit | Main ancestry / unique commits | Session, status, and disposition |
|---|---:|---|---|---|
| `%USERPROFILE%\.codex\worktrees\pbc-demo-performance\PBC` | 1,275,888 / 1,124,949 bytes | detached `98ade0c8072a4571e9674eaac3d91f861e3b9867` | Not reachable from PBC `main`; 6 unique commits | `git status` is clean, with no untracked or ignored entries. A Codex chat titled “Optimize portfolio performance” points to this unloaded worktree. Preserve its unmerged commits and session association. |
| `%USERPROFILE%\.codex\worktrees\pbc-research-experiments\PBC` | 6,846,304,288 / 6,713,881,731 bytes | `research/experiments` at `4cb3b83319111639ff6028fc1a1dcc218b9468db` | Not reachable from PBC `main`; 118 unique commits | No tracked modifications or untracked files. `git status --ignored` lists extensive ignored research outputs, including `hpt_data/`, `research_pass*` directories spanning pass6–pass47, `release_prep_v31/`, and `dist/`. A Codex chat titled “PBC” points to this unloaded worktree. Preserve the ignored outputs, unique commits, and session association. |
| `%USERPROFILE%\.codex\worktrees\pbc-rust-audit` | 3,206,284,224 / 3,074,070,652 bytes | detached `93e9c0591a710397bb3f09d2d51473313ce23db6` | Not reachable from PBC `main`; 64 unique commits | Tracked edits: `src/palette.rs`, `src/pbc3.rs`. Untracked: `.cargo/`, `.rustup/`, `rustup-init.exe`, `src/bin/`; ignored: `target/` and Cargo registry/build files. No matching Codex chat was found. Preserve all changes and toolchain/build files. |

**Total external worktree allocation:** 10,053,864,400 bytes (10.054 GB); logical size 9,789,077,332 bytes. No worktree was removed. The status and unique-file inventories remain for manual review; old/detached status alone was not treated as evidence of disposability. No branches were deleted.

All three directories under `%USERPROFILE%\.codex\worktrees` are registered in the PBC repository's `git worktree list --porcelain`; no unregistered workspace directory was found there.

## Cleanup performed

| Deleted path | Allocated before | Rationale |
|---|---:|---|
| `data/cache/neural_accessibility/imagenette_320/features/` | 12.281 GB | Derived dense JPEG DCT and PBC feature arrays; retained streams/manifests, source and build scripts allow regeneration. |
| `data/cache/neural_accessibility/imagenette_320/phase47/features/` | 5.102 GB | Derived AVIF/WebP arrays; exact streams, patches, versions, manifests, extraction and rebuild scripts remain. |
| `data/cache/neural_accessibility/imagenette_320/phase48/features/` | 2.637 GB | Derived JPEG q1 arrays; exact streams and build script remain. |
| `.uv-cache/`, Python `__pycache__/`, root `.pytest_cache/`, `tmp/`, Phase 4.8 `test_tmp/` | about 48 MB combined | Regenerable dependency, bytecode, and test scratch. |

No checkpoint, source dataset, exact Imagenette stream, result, log, extraction diagnostic, CIFAR cache, external worktree, or Git object was deleted. Cache-consuming model runs must rebuild removed features first; stored result rows and checkpoints are unchanged.

## After cleanup

| Path/category | Allocated size | Logical size | Files | Git status | Purpose / action |
|---|---:|---:|---:|---|---|
| Entire TaMtD checkout | 5.776 GB (5.38 GiB) | 5.141 GB (4.79 GiB) | 336,042 | tracked plus ignored/untracked | Measured with the same no-follow Windows allocation scan. 681 child directories remain (682 including root). |
| `data/` | 5.536 GB | 4.905 GB | 333,292 | ignored | Includes retained source data, exact codec streams, manifests, IR, checkpoints, and old CIFAR exploratory caches. |
| Imagenette cache | 2.928 GB | 2.927 GB | 315 | ignored | 2.638 GB exact streams, 206.0 MB checkpoints, 29.0 MB IR, manifests, and smaller analysis data. Feature-map directories are absent. |
| `results/` | 139.7 MB | 137.8 MB | 1,274 | tracked plus untracked local artifacts | Findings, figures, tables, logs, raw early checkpoints, and diagnostics retained. |
| `.git/` | 94.3 MB | 92.3 MB | 1,265 | repository metadata | Retained without pruning or history rewrite. |
| `.venv/` + `.venv313/` | 1.94 MB | 1.86 MB | 48 | local environments | Retained. |
| All checkpoints | 247.4 MB allocated | 247.1 MB logical | 186 `.pt` | ignored/untracked | All retained. |

Largest remaining extensions by allocated size are `.bin` 3.570 GB (318,248 files, mostly retained encoded streams and exploratory cache payloads), `.jsonl` 424.4 MB (31 files), `.jpeg` 375.6 MB (13,394 files), `.tgz` 341.7 MB, `.pt` 247.4 MB (186 checkpoints), and `.gz` 217.1 MB. The remaining `.bin` files are the largest avoidable category only if the project later accepts losing cached CIFAR/PBC inputs; they were retained for provenance.

**Measured repository reduction:** allocated size fell by 20,067,704,368 bytes (20.068 GB; 18.69 GiB), or 77.66% of baseline. Logical size fell by 20,065,551,409 bytes (20.066 GB). File count fell by 1,236 after adding four archive documents; child-directory count fell by 173. No reparse points or scan errors occurred.

**Free-space observation:** C: showed 82.94 GiB free immediately before cleanup, 100.70 GiB just after deletion, 99.34 GiB after the first final audit scan, 99.12 GiB before the last validation check, 95.59 GiB during a pagefile expansion, and 99.11 GiB after it shrank. The final net increase from pre-cleanup is 16.17 GiB, versus 18.69 GiB released from repository allocation. No pagefile setting was modified.

`C:\pagefile.sys` grew from 7.516 GB (7.00 GiB) to 11.291 GB (10.52 GiB) at 18:30 on 2026-11-13, then returned to 7.516 GB at 18:32. The temporary increase was 3.775 GB (3.516 GiB), closely matching the 3.53 GiB temporary drop in free space. `C:\hiberfil.sys` remained 13.605 GB (12.67 GiB). These system files are outside the repository inventory and were not modified by this cleanup. Windows memory commit counters were unavailable from the current shell, so the observed pagefile expansion cannot be tied to a specific process.

`git fsck --full --no-reflogs` exited 0 with no integrity errors. It reported 328 dangling objects and two old temporary object files (5.89 MiB); `git count-objects -vH` showed 1,231 loose objects (~82.03 MiB), 3 packed objects in one ~2.09 KiB pack, and the same two garbage files. No garbage collection, pruning, or ref changes were performed.

### Remaining `data/cache/` roots

| Cache path | Allocated | Logical | Files | Purpose / action |
|---|---:|---:|---:|---|
| `neural_accessibility/` | 2,927.53 MB | 2,926.80 MB | 315 | Retained source streams, manifests, IR, checkpoints, and phase metadata; large feature maps removed. |
| `jpeg_quality_ablation/` | 253.93 MB | 103.87 MB | 55,002 | Retained CIFAR exploratory cache. |
| `pbc_high_quality_ablation/` | 186.33 MB | 109.58 MB | 33,002 | Retained due external dirty PBC provenance. |
| `cifar10_smoke/` | 184.17 MB | 112.35 MB | 44,002 | Retained exploratory cache. |
| `pbc_quality_ablation/` | 180.03 MB | 106.81 MB | 33,002 | Retained due external dirty PBC provenance. |
| `pbc_quality_diagnostics_full/` | 173.21 MB | 82.79 MB | 44,003 | Retained due external dirty PBC provenance. |
| `pbc_balanced_ablation/` | 135.96 MB | 99.07 MB | 33,002 | Retained due external dirty PBC provenance. |
| `pbc_quality_diagnostics/` | 118.65 MB | 30.44 MB | 42,928 | Retained due external dirty PBC provenance. |
| `pbc_compression_ablation/` | 106.66 MB | 94.06 MB | 33,002 | Retained due external dirty PBC provenance. |
| `staged_decode/` | 21.22 MB | 18.73 MB | 1,269 | Retained small staged-decode cache. |
| `cifar10_overfit/` | 1.18 MB | 0.91 MB | 258 | Retained small test cache. |
| `staged-decode-deps/` | 0.86 MB | 0.82 MB | 31 | Retained native dependency/runtime files. |
| `tamtd-numba-cache/` | 0.35 MB | 0.32 MB | 14 | Retained small runtime cache. |
| `ablation_probe/` | 0.10 MB | 0.04 MB | 26 | Retained exploratory cache. |
| `cifar10_pbc_probe/` | 0.08 MB | 0.04 MB | 18 | Retained exploratory cache. |
| `test-temp/` | 0.02 MB | 0.02 MB | 2 | Small, unclear test artifact; retained because it is negligible and provenance is unresolved. |

`codec_benchmark_temp/` was present as an empty directory and contributes no file allocation. The cache root totals 4.290 GB allocated and 3.687 GB logical after cleanup.

### Results storage by phase

| Results path | Allocated | Files | Status / action |
|---|---:|---:|---|
| `Codec Benchmark/` | 52.22 MB | 415 | Retain. |
| `Phase 4.5B Analysis/` | 34.23 MB | 35 | Retain. |
| `raw/` | 29.36 MB | 451 | Retain early result/checkpoint artifacts. |
| `Phase 4.8 Extreme Compression/` | 16.36 MB | 94 | Retain; local logs remain ignored and unmodified. |
| `Phase 4.7 Standard Codec Stages/` | 2.77 MB | 28 | Retain findings and local untracked diagnostics. |
| `Staged Decoding/` | 1.61 MB | 8 | Retain. |
| `Neural Accessibility/` | 1.23 MB | 20 | Retain. |
| `Phase 4.6 Mechanistic Controls/` | 0.85 MB | 85 | Retain. |
| `Phase 4 Confirmation/` | 0.84 MB | 110 | Retain; local logs remain ignored and unmodified. |
| `Early Testing/` | 0.17 MB | 17 | Retain. |
| `Phase 4.5A Diagnostics/` | 0.04 MB | 10 | Retain; local logs remain ignored and unmodified. |

**Validation:** all 66 tests passed and 3 were skipped using the available Python 3.13 environment with pytest cache disabled; 11 Pillow deprecation warnings were emitted. The project specifies Python 3.12, whose local environment is missing pytest/PyTorch. Local Markdown links checked: 77, with none missing. FFmpeg/ffprobe 6.1.1 and the `libx264` H.264 encoder are available. Python syntax checks passed for `src/`, `scripts/`, and `tests/`. Tests did not regenerate any removed feature directories.

## Finalization audit — 2026-11-13

The documentation and `.gitignore` finalization removed no files and reclaimed no additional storage. The three external worktrees remain at 10.054 GB allocated; none met all removal criteria because each contains commits not reachable from PBC `main`, and tracked/untracked status could not be established from this shell. Two are linked to unloaded Codex chats. No external worktree, branch, or object was changed.

The pre-existing untracked `results/Phase 4.7 Standard Codec Stages/extraction_resource_diagnostics.json` is a valid, 2,893-byte run diagnostic. It records Windows platform, disk free space, process/system memory and GPU availability at extraction time. Those machine-specific measurements make it unsuitable for the shared research archive as-is. It was preserved unchanged and excluded from the documentation commit for manual review.

`git fsck --full --no-reflogs` completed without integrity errors. The cleanup-time snapshot reported 328 dangling objects; the final audit reported 329. Current `git count-objects -vH` reports 1,241 loose objects (82.05 MiB), three packed objects, and the same two old temporary object files totaling 5.89 MiB. These objects may contain recoverable work; neither garbage collection nor pruning was run. The Python 3.12 `.venv` has no pytest, PyTorch, NumPy, or Pillow. PDM 2.26.9 and `pdm.lock` are available; a locked `pdm sync -G test` is documented in [Reproducibility and Retention](Reproducibility%20and%20Retention.md) but was not run because it would install the large PyTorch dependency set. Windows pagefile and hibernation settings were not changed.

The checkout measured 5,775,813,168 bytes allocated immediately before this documentation finalization and 5,775,851,744 bytes allocated after the documentation was assembled, before commit (5.776 GB; 5,140,646,348 logical bytes; 336,053 files). The 38,576-byte increase is the documentation/repository metadata footprint; this pass deleted nothing and reclaimed 0 bytes. C: had 112,316,313,600 bytes free (112.316 GB) at that scan. Relative to the initial 25,843,517,536-byte allocation, the net reduction at that point is 20,067,665,792 bytes (20.068 GB). External worktree allocation remained 10,053,864,400 bytes before and after. No multi-gigabyte feature maps were regenerated.

Finalization checks: **67 tests passed, 2 skipped** on the available Python 3.13.5 environment (11 Pillow deprecation warnings); syntax compilation passed for 112 Python files; all 67 checked local Markdown links resolved across 23 Markdown files; `pdm lock --check`, `git diff --check`, and `git fsck --full --no-reflogs` passed. A PDM dry run confirmed the documented locked sync command and showed the large Torch 2.14.1 install it would require; no packages were installed. The base, Phase 4.7, and Phase 4.8 Imagenette feature directories remain absent. FFmpeg/ffprobe 6.1.1 and `libx264` are present, so video V0/V1 feasibility does not depend on the deleted image feature maps. Video dataset access and licensing remain unverified.
