# Phase 4 Findings

## Scope and protocol

- Phase 3 seed 42 results were reused where present. The longer PBC byte conditions use one reduced batch size per condition across all three seeds after GPU runs approached the system commit limit; all bytes and the model architecture were retained. Balanced LZMA/STORE and both quality byte streams ran on CPU to keep system commit below the safety stop. Each training ran in a fresh Python process with workers=0 and pin_memory=False.
- Phase 4A includes PBC compression/balanced/quality byte, STORE, canonical IR, and common RGB references; JPEG q10/q60 file bytes, DCT IR, and decoded RGB references; and the original RGB reference.
- Phase 4B adds raw-byte-only WebP q30/q75, AVIF q40/q60, and JXL distance 6/3 at seed 42. The selections follow the Phase 1 measured low/mid rate matches.
- Checkpoints, logs, and codec streams remain in ignored cache/result storage and are not part of the compact report artifacts.

## Multi-seed confirmation

Test and validation columns show mean ± sample standard deviation across seeds 42, 123, and 777. For decoded RGB references, validation accuracy is measured with the common RGB model selected on original validation data; it is not used for checkpoint selection.

| Condition | Representation | Batch/device | Test accuracy | Validation accuracy | Semantic accessibility | Mean train time (s) | Mean examples/s | Mean peak VRAM (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pbc_compression | pbc_lzma_bytes | 32/cuda | 19.67 ± 0.69% | 21.12 ± 0.82% | 0.152 ± 0.011 (from means 0.152) | 213.8 | 327.5 | 390.1 |
| pbc_compression | pbc_store_bytes | 32/cuda | 35.76 ± 0.45% | 37.66 ± 0.65% | 0.405 ± 0.004 (from means 0.405) | 236.3 | 390.1 | 465.2 |
| pbc_compression | pbc_canonical_ir | 64/cuda | 57.55 ± 0.46% | 61.11 ± 0.67% | 0.748 ± 0.012 (from means 0.748) | 37.3 | 4873.1 | 81.1 |
| pbc_compression | decoded_rgb_reference | n/a | 73.58 ± 0.42% | 75.96 ± 0.76% | 1.000 ± 0.000 (from means 1.000) | n/a | n/a | n/a |
| pbc_balanced | pbc_lzma_bytes | 8/cpu | 18.46 ± 0.05% | 20.56 ± 0.30% | 0.121 ± 0.001 (from means 0.121) | 1177.1 | 53.9 | n/a |
| pbc_balanced | pbc_store_bytes | 8/cpu | 39.07 ± 1.72% | 40.94 ± 2.65% | 0.415 ± 0.025 (from means 0.415) | 2383.8 | 34.7 | n/a |
| pbc_balanced | pbc_canonical_ir | 64/cuda | 57.07 ± 0.34% | 61.14 ± 1.12% | 0.673 ± 0.009 (from means 0.673) | 36.0 | 4860.8 | 81.1 |
| pbc_balanced | decoded_rgb_reference | n/a | 79.97 ± 0.47% | 81.70 ± 0.65% | 1.000 ± 0.000 (from means 1.000) | n/a | n/a | n/a |
| pbc_quality | pbc_lzma_bytes | 8/cpu | 16.63 ± 1.09% | 18.90 ± 0.18% | 0.093 ± 0.015 (from means 0.093) | 2077.9 | 30.6 | n/a |
| pbc_quality | pbc_store_bytes | 8/cpu | 39.41 ± 2.04% | 42.24 ± 0.97% | 0.412 ± 0.029 (from means 0.412) | 4662.8 | 19.9 | n/a |
| pbc_quality | pbc_canonical_ir | 64/cuda | 56.70 ± 0.17% | 59.91 ± 1.04% | 0.654 ± 0.002 (from means 0.654) | 41.8 | 4356.4 | 81.1 |
| pbc_quality | decoded_rgb_reference | n/a | 81.43 ± 0.34% | 83.77 ± 1.10% | 1.000 ± 0.000 (from means 1.000) | n/a | n/a | n/a |
| jpeg_q10 | jpeg_file_bytes | 32/cuda | 36.64 ± 1.12% | 37.45 ± 0.27% | 0.377 ± 0.018 (from means 0.377) | 75.8 | 955.5 | 121.6 |
| jpeg_q10 | jpeg_dct_ir | 32/cuda | 76.21 ± 0.32% | 78.85 ± 0.96% | 0.937 ± 0.009 (from means 0.937) | 108.4 | 1580.2 | 93.2 |
| jpeg_q10 | decoded_rgb_reference | n/a | 80.68 ± 0.37% | 83.70 ± 1.08% | 1.000 ± 0.000 (from means 1.000) | n/a | n/a | n/a |
| jpeg_q60 | jpeg_file_bytes | 32/cuda | 32.01 ± 0.54% | 34.18 ± 0.37% | 0.306 ± 0.007 (from means 0.306) | 176.8 | 499.2 | 306.4 |
| jpeg_q60 | jpeg_dct_ir | 32/cuda | 76.44 ± 0.19% | 78.25 ± 1.28% | 0.923 ± 0.004 (from means 0.923) | 106.7 | 1606.2 | 92.9 |
| jpeg_q60 | decoded_rgb_reference | n/a | 81.96 ± 0.29% | 84.58 ± 1.02% | 1.000 ± 0.000 (from means 1.000) | n/a | n/a | n/a |
| original_rgb | decoded_rgb_reference | 32/cuda | 82.13 ± 0.10% | 84.97 ± 0.96% | 1.000 ± 0.000 (from means 1.000) | 1192.5 | 215.4 | 331.6 |

Semantic accessibility uses `(model accuracy − 0.1) / (matched decoded-RGB accuracy − 0.1)`. The `from means` value applies that formula to the three-seed mean accuracies, as specified.

## Direct checks of the Phase 3 claims

| Comparison | Seed-wise result |
| --- | --- |
| pbc_compression: STORE − LZMA | 16.09 ± 0.69 percentage points; positive in 3/3 seeds |
| pbc_balanced: STORE − LZMA | 20.60 ± 1.67 percentage points; positive in 3/3 seeds |
| pbc_quality: STORE − LZMA | 22.79 ± 3.13 percentage points; positive in 3/3 seeds |
| jpeg_q10: DCT IR − file bytes | 39.58 ± 0.81 percentage points; positive in 3/3 seeds |
| jpeg_q10: decoded RGB − DCT IR | 4.47 ± 0.68 percentage points |
| jpeg_q60: DCT IR − file bytes | 44.43 ± 0.67 percentage points; positive in 3/3 seeds |
| jpeg_q60: decoded RGB − DCT IR | 5.52 ± 0.33 percentage points |

### PBC LZMA to STORE

The paired changes above measure the same PBC body with and without the outer LZMA wrapper. Compression improves by 16.09 ± 0.69 percentage points on average. This controlled result supports a conclusion about PBC's wrapper and this byte CNN, not all entropy-coded formats.

### PBC canonical IR

Canonical IR semantic accessibility calculated from multi-seed means: compression: 0.748 semantic accessibility at 0.576 mean test accuracy; balanced: 0.673 semantic accessibility at 0.571 mean test accuracy; quality: 0.654 semantic accessibility at 0.567 mean test accuracy.

### Intermediate preprocessing trade-off

At PBC compression, canonical IR reaches 0.748 semantic accessibility after 45.6% of that preset's full decode time. JPEG q10 DCT IR reaches 0.937 after 84.4% of its full RGB decode time. The absolute timings are in the refreshed table; PBC has a lower relative-to-full-decode cost at this point, while JPEG exposes more class information.

### JPEG DCT IR

The paired q10/q60 DCT-minus-file changes and DCT-to-decoded-RGB gaps are in the table above. The refreshed JPEG timing shows DCT access and full RGB decode in the same pass and environment; it does not include JPEG encode time.

## Cross-codec raw-byte extension

The low-rate points were selected near PBC compression and the mid-rate points near PBC balanced/JPEG q60 using Phase 1's 95-image rate measurements. Model comparisons use seed 42 for all codecs. Phase 4 mean bpp is measured on the Imagenette cache; Phase 1 rate/PSNR are shown as calibration context.

| Rate band | Codec | Setting | Phase 1 bpp | Phase 1 PSNR (dB) | Imagenette bpp | Raw-byte test accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| low-rate | PBC LZMA | pbc_compression | 0.420 | 27.86 | 0.617 | 0.189 |
| low-rate | PBC STORE | pbc_compression | n/a | 27.86 | 0.828 | 0.355 |
| low-rate | JPEG | jpeg_q10 | 0.370 | 27.67 | 0.446 | 0.356 |
| low-rate | WebP | webp_q30 | 0.426 | 31.98 | 0.593 | 0.244 |
| low-rate | AVIF | avif_q40 | 0.407 | 32.83 | 0.585 | 0.184 |
| low-rate | JXL | jxl_d6 | 0.457 | 31.84 | 0.566 | 0.204 |
| mid-rate | PBC LZMA | pbc_balanced | 0.844 | 30.25 | 1.388 | 0.184 |
| mid-rate | PBC STORE | pbc_balanced | n/a | 30.25 | 2.166 | 0.380 |
| mid-rate | JPEG | jpeg_q60 | 0.982 | 33.80 | 1.233 | 0.323 |
| mid-rate | WebP | webp_q75 | 0.801 | 35.24 | 1.091 | 0.225 |
| mid-rate | AVIF | avif_q60 | 0.847 | 36.65 | 1.171 | 0.168 |
| mid-rate | JXL | jxl_d3 | 0.774 | 34.52 | 0.961 | 0.206 |

The one-seed cross-codec results are exploratory. JPEG q10 and q60, and the selected WebP/AVIF/JXL operating points, have different codec syntax and rate-distortion behavior; quality numbers are not treated as equivalent. The controlled PBC LZMA-to-STORE comparison supports an effect of outer wrapping within the same PBC body. The other codecs do not have an equivalent paired raw-versus-unwrapped condition here, so these results do not establish that entropy coding is generally harmful.

## Refreshed JPEG timing

The pass used 95 fixed prepared 256×256 images, 1 warmup and 5 timed repetitions per image. Environment: Python 3.13.5, Pillow 11.3.0, libjpeg-turbo 3.1.1, pyjpegio 0.3.0, NumPy 2.2.6; Windows-11-10.0.26200-SP0.

| JPEG | File → DCT IR median (ms) | p95 (ms) | Full RGB median (ms) | p95 (ms) | Median ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| jpeg_q10 | 0.5213 | 0.6145 | 0.6175 | 0.7181 | 0.844× |
| jpeg_q20 | 0.5222 | 0.6549 | 0.4694 | 0.6859 | 1.112× |
| jpeg_q60 | 0.5165 | 0.5975 | 0.4546 | 0.6178 | 1.136× |
| jpeg_q80 | 0.5936 | 1.0397 | 0.5329 | 0.8971 | 1.114× |

The earlier q20 extension reported a 2.20× ratio. The refreshed q20 ratio is 1.112×; q10/q60/q80 ratios span 0.844×–1.136×. The q20 anomaly does not remain an outlier relative to q10, q60, and q80 in this pass.

## Interpretation and limitations

The three-seed results answer whether the Phase 3 PBC/JPEG differences hold across these model seeds. The cross-codec raw-byte extension uses one seed and one classifier family, so it describes this Imagenette setup rather than a general codec ranking. Absolute preprocessing times remain implementation-level measurements. No raw-byte model is claimed to decode images, and these results do not demonstrate state-of-the-art accuracy.

The strongest current claim is that, for this fixed Imagenette benchmark and byte CNN, removing PBC's outer LZMA wrapper changes classification accessibility substantially while preserving the PBC body, and JPEG's quantized DCT representation exposes more class information than JPEG file bytes. Whether those effects generalize to other datasets and architectures remains open.
The evidence supports continuing TaMtD as a measured compressed-domain vision systems study: it yields reproducible questions about representation structure, task accessibility, and preprocessing cost without requiring a decoder-replacement claim.

## Artifacts

- `confirmation_summary.csv`, `confirmation_per_seed.csv`, `multi_seed_summary.json`
- `byte_level_result_table.csv`, `jpeg_timing_refresh.csv`
- `multi_seed_accessibility_frontier.png`, `cross_codec_raw_bytes.png`, `jpeg_timing_refresh.png`
- `resource_diagnostics.json` and `config.json`
