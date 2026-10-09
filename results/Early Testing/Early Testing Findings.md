# Early Testing Findings

This folder records the TaMtD experiments completed before any ByteFormer implementation. The experiments use real CIFAR-10 data and the local PBC checkout specified below. The original generated cache remains ignored under `data/cache/`; the tracked JSON files here are the reproducibility record and the curated numerical results.

## Scope and provenance

- Dataset: CIFAR-10, 32x32 RGB, one deterministic source view per image.
- Cached source rows per representation: 10,000 train images and 1,000 test images.
- Corrected classification split: 8,000 train / 2,000 validation / 1,000 test.
- Reconstruction split: 8,000 train / 2,000 validation / 1,000 test.
- Checkpoints were selected on validation only; the test set was evaluated once for the selected checkpoint.
- Classification chance baseline: empirical majority accuracy, 11.2%.
- Byte padding/truncation limit: 3,072 bytes. The measured classification streams and reconstruction streams had 0% truncation.
- Main experiment seed: 42. Repeated classification conditions used seeds 42, 123, and 777 where stated.
- TaMtD repository commit recorded in the run metadata: `a573812edb8c0162e6e3b03ca54c12c5c67bb35a`.
- PBC source used by the cache: local `PBC_ROOT` at `C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC`.
- PBC source state recorded in every PBC cache row: branch `multistage_pbc31`, commit `3e9e8c94b7497ecefc5e482a66fde697dfad6e19`, worktree dirty at encoding time. TaMtD did not modify that checkout.

The exact copied numerical artifacts are:

- [`attribution_summary.json`](attribution_summary.json)
- [`plain_ablation_summary.json`](plain_ablation_summary.json)
- [`pbc_preset_stats.json`](pbc_preset_stats.json)
- [`jpeg_quality_stats.json`](jpeg_quality_stats.json)
- [`stream_stats_max3072.json`](stream_stats_max3072.json)
- [`raw_rgb_overfit_metrics.json`](raw_rgb_overfit_metrics.json)
- [`reconstruction_summary.json`](reconstruction_summary.json)

## Initial pipeline sanity check

The raw RGB ByteCNN substantially overfit a fixed 128-example set:

| item | result |
| --- | ---: |
| model parameters | 135,178 |
| epochs / batch size | 200 / 32 |
| best validation accuracy | 100.0% |
| final train accuracy | 96.1% |
| held-out evaluation accuracy | 100.0% |
| wall time | 37.3 s |
| peak CUDA VRAM | 131.5 MB |

This ruled out a basic label, data, or optimization failure before codec comparisons.

## PBC preset ablation

The plain PBC results use `ENTROPY_STORE`. PBC quality metrics are computed from the decoded image. ByteCNN and conventional decoded-RGB CNN results use the corrected validation-selected protocol.

| preset | mean bytes | p95 bytes | decoded MSE | PSNR (dB) | mean ratio | decoded RGB acc. | length | histogram | ByteCNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compression | 298.5 | 517.0 | 257.60 | 24.35 | 11.97 | 41.5% | 16.8% | 19.0% | 22.5% |
| balanced | 543.0 | 854.0 | 121.56 | 27.62 | 6.49 | 42.6% | 15.2% | 19.5% | 22.9% |
| quality | 939.1 | 1372.05 | 51.94 | 31.34 | 3.57 | 43.0% | 15.0% | 17.3% | 24.0% |
| high_quality | 1034.1 | 1573.0 | 25.10 | 34.61 | 3.18 | 42.8% | 13.0% | 19.6% | 25.9% |

PBC `quality` and `high_quality` were the most informative matched points for comparison with JPEG. The higher ByteCNN accuracy of `high_quality` is not separable from its higher decoded quality by this table alone.

### PBC configuration

The cache records the full encoder configuration. Common values were:

```text
anchor_block_size=8
auto_downsample_init=true
auto_downsample_max_pixels=250000
cell_selection_mode=gradient
cell_sizes_per_candidate=3
channel_cycle=Sum
color_space=YCbCr
compute_final_mse=true
debug_mode=false, debug_print=false, debug_path=null
downsample_init_cell_size=12
downsample_palette_bitcount=6
downsample_rate=-1
exact_depth=10
init_search_depth=3
learned_filler_candidates=1
learned_filler_enabled=true
learned_filler_model_path=<PBC_ROOT>\patch_policy.npz
learned_filler_top_k=1
mask_size=4
max_cell_size=64
max_patch_size=400
min_cell_size=1
min_patch_size=16
patch_palette_bitcount=2
positive_bias=true
proposal_depth=50
q_end=0.8, q_init=0.7, q_start=0.8
quality_target_mae=0.0
random_seed=2003
representability_threshold=0.0
residual_projection_mode=bicubic
reuse_selected_delta=true
search_depth=200
search_q_end=0.2
top_k=20
use_lzma=false before TaMtD's controlled wrapper split
warm_downsample_max_pixels=750000
warmup_ratio=-1
```

Preset-specific values were:

| preset | patch count | learned filler q | search q start |
| --- | ---: | ---: | ---: |
| compression | 50 | 0.40 | 0.5 |
| balanced | 50 | 0.60 | 0.5 |
| quality | 50 | 0.80 | 0.5 |
| high_quality | 20 | 0.95 | 0.7 |

## JPEG quality sweep

All JPEG files were Pillow JPEGs with `optimize=false`, `progressive=false`, and 4:2:0 subsampling (`subsampling=2`). All 11,000 q1 files decoded as valid JPEGs.

| quality | mean bytes | p95 bytes | MSE | PSNR (dB) | mean ratio | decoded RGB acc. | length | histogram | ByteCNN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 654.6 | 666.0 | 922.73 | 18.48 | 4.69 | 34.4% | 14.6% | 11.2% | 21.5% |
| 60 | 853.2 | 918.0 | 90.68 | 28.56 | 3.61 | 42.2% | 14.8% | 13.5% | 13.1% |
| 75 | 920.1 | 1002.0 | 65.26 | 29.98 | 3.35 | 41.7% | 15.6% | 14.0% | 13.6% |
| 80 | 959.8 | 1051.0 | 54.96 | 30.73 | 3.21 | 42.1% | 16.0% | 14.3% | 12.4% |
| 95 | 1298.8 | 1461.0 | 19.49 | 35.23 | 2.38 | 42.3% | 15.2% | 15.1% | 16.7% |

JPEG q1 is a real outlier: it has very poor decoded quality but substantially higher ByteCNN accuracy than q60–q80. It was preserved as a result for later investigation, not treated as evidence of good reconstruction.

## Attribution and order controls

The invariant models were intended to test whether the classification effect was mainly aggregate byte statistics rather than learned sequence structure.

| representation | ordered ByteCNN | shuffled ByteCNN seed 42 | histogram MLP | DeepSets seed 42 |
| --- | ---: | ---: | ---: | ---: |
| PBC quality | 24.03% ± 0.75% | 23.7% | 25.93% ± 0.49% | 22.7% |
| PBC high_quality | 25.33% ± 0.74% | 23.0% | 24.40% ± 0.66% | 21.4% |
| JPEG q80 | 13.53% ± 0.99% | 14.2% | 15.23% ± 0.47% | 14.1% |
| JPEG q95 | 16.77% ± 0.70% | 14.3% | 14.37% ± 0.29% | 14.7% |
| JPEG q1 | 21.5% | 10.6% | 13.43% ± 2.11% | 12.9% |
| PBC quality forced-LZMA | 14.63% ± 0.97% | 13.9% | 15.27% ± 0.35% | 16.5% |

The main classification finding is that PBC exposes more class-correlated information through aggregate statistics than JPEG at similar rates. The PBC quality histogram MLP matched or exceeded the ByteCNN, so classification alone did not establish decoder-like sequence learning.

Additional PBC diagnostics showed:

- PBC initialization-only decoded RGB accuracy: 38.0%; full PBC quality: 43.0%.
- Residual-only ByteCNN: 23.8%; initialization-only ByteCNN: 24.5%.
- Zeroing only initialization grid indices: 26.3% ByteCNN, so those indices were not the sole source of the effect.
- Complete residual-patch reordering: 24.2% versus 24.0% ordered, with pixel-identical decoded images on 32 checked samples.
- Forced LZMA reduced classification accuracy sharply despite preserving the decoded image.

## Coarse spatial reconstruction probe

The reconstruction target is `conventional codec decode -> grayscale -> deterministic 8x8 BOX downsample`. The sequence model receives only byte values and the padding mask. It has 92,225 parameters and does not use parsed codec fields. The histogram MLP receives the normalized 256-byte histogram and encoded length.

The mean-image baseline is computed from the training targets. Relative reduction is `1 - model_MSE / mean_baseline_MSE`.

| condition | model | test MSE | MAE | PSNR (dB) | mean-baseline MSE | relative MSE reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PBC quality ordered | sequence | 0.03413 | 0.14869 | 14.67 | 0.04496 | 24.1% |
| PBC quality shuffled | sequence | 0.04112 | 0.16403 | 13.86 | 0.04496 | 8.5% |
| PBC quality | histogram MLP | 0.03927 | 0.16030 | 14.06 | 0.04496 | 12.6% |
| JPEG q80 ordered | sequence | 0.03285 | 0.14349 | 14.83 | 0.04492 | 26.9% |
| JPEG q80 shuffled | sequence | 0.04495 | 0.17355 | 13.47 | 0.04492 | -0.1% |
| JPEG q80 | histogram MLP | 0.04490 | 0.17290 | 13.48 | 0.04492 | 0.0% |
| PBC quality forced-LZMA ordered | sequence | 0.03691 | 0.15362 | 14.33 | 0.04496 | 17.9% |
| PBC quality forced-LZMA shuffled | sequence | 0.04483 | 0.17315 | 13.48 | 0.04496 | 0.3% |
| PBC quality forced-LZMA | histogram MLP | 0.04491 | 0.17279 | 13.48 | 0.04496 | 0.1% |

PBC STORE and forced-LZMA targets were exactly identical for 32 held-out samples. Deterministic shuffling preserved the exact byte sequence length and byte histogram for checked PBC and JPEG samples and shuffled only the actual encoded bytes before padding.

Representative held-out grids are in [`previews/`](previews/). Each grid contains the target on the left and the model prediction on the right for 16 fixed test examples. The ordered models produce sample-dependent coarse structure; the shuffled and histogram models are much closer to average-looking outputs.

The reconstruction result is the strongest early evidence for spatial byte-structure learning:

1. Ordered PBC beats both shuffled PBC and the histogram baseline.
2. Ordered JPEG also beats its shuffled and histogram controls, and is slightly better than PBC on this particular 8x8 grayscale MSE.
3. Forced LZMA preserves useful ordered structure but degrades it substantially; its shuffled and histogram controls are at the mean-image baseline.
4. The result justifies a stronger sequence encoder as a future experiment, but does not yet establish that PBC is easier than JPEG for spatial reconstruction.

## Implementation and verification

The committed implementation includes:

- local dataset/cache ingestion and deterministic byte shuffling;
- PBC STORE/forced-LZMA pairing and invariant checks;
- histogram and DeepSets invariant classification models;
- 8x8 reconstruction dataset, sequence model, histogram baseline, validation-selected trainer, metrics, and preview generation;
- [`scripts/run_reconstruction_probe.py`](../../scripts/run_reconstruction_probe.py);
- unit tests for the reconstruction models and byte-shuffle behavior.

Verification at this checkpoint:

```text
python -m unittest discover -q
10 tests passed, 1 skipped
python -m compileall -q src scripts
```

No ByteFormer, QOI, PNG, WebP, AVIF, reconstruction-at-32x32, or PBC source changes were made. Dataset caches and model checkpoints remain untracked by design; they can be regenerated from the manifests and the commands used in the run metadata.
