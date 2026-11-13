# Reproducibility and retention

This document records what survives local cache cleanup and what is needed to reproduce or extend the image track. Detailed numerical findings remain in the [research archive](Research%20Archive.md), [consolidated report](ta-mtd-research-report.md), and original phase reports.

## Reproduction inputs

| Area | Retained source of truth | Notes |
|---|---|---|
| Dataset and split | `data/raw/`, `data/cache/neural_accessibility/imagenette_320/manifest.jsonl`, `codec_manifest.jsonl`, phase result metadata | The retained Imagenette2-320 source archive is [the fast.ai ImageNette archive](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz). The prepared 256×256 view has 13,394 samples, split 8,522/947/3,925; manifests retain IDs and source/prepared RGB SHA-256 values. Seed 42 is used for the fixed split. Exact codec streams remain local. |
| Codec benchmark | `results/Codec Benchmark/`, `scripts/run_codec_benchmark.py`, codec modules under `src/tamtd/` | Per-image measurements, settings, summary tables, and selected points remain versioned. |
| Training and probes | `src/tamtd/models/`, `src/tamtd/training/`, `scripts/run_*.py`, `configs/` | Preserve architecture, preprocessing, optimizer/runtime, seed, checkpoint-selection, and evaluation code. |
| Decoder instrumentation | `patches/phase47/versions.json`, the two Phase 4.7 patches, `scripts/build_phase47_codecs.*`, extraction scripts and manifests | Rebuilding patched native decoders is required to regenerate AVIF/WebP pre-reconstruction features. Decoder versions are pinned in the retained manifest. |
| Results and uncertainty | `results/` CSV/JSON/Markdown, `docs/research/figures/`, `docs/research/data/` | Compact per-seed and per-sample numerical summaries, figures, and analysis scripts are retained. Use detailed CSVs rather than rounded report tables to regenerate headline numbers. |
| Checkpoints | Imagenette cache checkpoint directories and `results/raw/` | All 186 discovered `.pt` checkpoints are retained. They are useful for inference rechecks and avoid expensive retraining; no checkpoint directory was removed. |
| Environment | `pyproject.toml`, `pdm.lock`, `.venv/`, `.venv313/` | The project requires Python 3.12 and `pdm.lock` records exact resolved versions and hashes. The Python 3.12 `.venv` lacks pytest, PyTorch, NumPy, and Pillow; the retained Python 3.13 environment has them through system site packages but is outside the project specification. |

## Cache retention manifest

| Artifact | Decision | Rebuild/reproduction impact |
|---|---|---|
| Imagenette exact streams (`streams`, `phase4_streams`, `phase48_streams`) | Retain | Preserves tested codec inputs and sample-to-stream provenance. |
| Imagenette manifests, source/prepared images, IR bundles, Phase 4.5B artifacts | Retain | Preserves sample identity, metadata, and compact canonical artifacts. |
| Three Imagenette `features` directories (base, `phase47/features`, `phase48/features`) | Delete | Derived model-ready arrays. Rebuild from retained streams, manifests, source, scripts, and Phase 4.7 pinned decoder sources/patches. Training/inference consuming these arrays must rebuild them first; multi-GB feature generation is not part of routine tests. |
| Checkpoints and result outputs | Retain | No training is needed to revisit stored scores or evaluate retained models. |
| CIFAR exploratory cache directories | Retain | Some early PBC streams came from an external dirty PBC checkout whose complete source changes are not guaranteed to be versioned. Retaining caches preserves a future audit option. |
| `data/raw/` source archives/extractions | Retain | Avoids redownload/re-extraction and preserves benchmark inputs. |
| Root `.uv-cache`, Python bytecode, pytest scratch | Delete | Regenerable package/download or test artifacts; no research result depends on them. |
| Local logs, extraction diagnostics, other untracked user files | Retain | Uncommitted user artifacts are not treated as disposable. |
| Codex worktrees outside this repository | Retain | Outside the authorized cleanup boundary. |

## Reproduction procedure and limits

1. Install Python 3.12 and PDM, then run `pdm sync -d -G test` from the repository root. This syncs the exact locked project and test dependencies. For the optional JPEG coefficient binding, use `pdm sync -d -G test -G staged-jpeg`. This audit did not run the sync because the locked default environment includes PyTorch and would require a large package installation; the local Python 3.12 environment remains incomplete.
2. Obtain the fast.ai ImageNette2-320 archive from the URL above if setting up another machine; retain its original archive and the prepared source views.
3. Use retained manifests and result configs to preserve sample IDs, splits, crop/resize behavior, encoder settings, and seeds.
4. Rebuild deleted model-ready Imagenette features from retained exact streams using the corresponding cache-build scripts. For Phase 4.7, rebuild the pinned patched codecs using `scripts/build_phase47_codecs.ps1` or `.sh` before feature extraction.
5. Reuse existing checkpoints for validation/test rechecks; preserve validation-selected checkpoints and do not retrain to reproduce an already recorded metric unless a new seed is required.
6. Interpret timing as host/library specific. Phase 4.7 instrumentation timings include adapter costs where documented; they are not algorithm-only latency measurements.

The early CIFAR findings remain exploratory and some external PBC provenance was dirty/unversioned. Cross-codec byte model seed counts are unequal; JPEG q1 and other Phase 4.8 low-rate points are single-seed. The Phase 4.7 AUC computed from inconsistent endpoints is withdrawn; use the corrected same-endpoint analysis recorded in Phase 4.8. No stored result establishes an intrinsic codec ranking independent of architecture or preprocessing.
