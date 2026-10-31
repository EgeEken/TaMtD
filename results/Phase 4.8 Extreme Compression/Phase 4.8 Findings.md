# Phase 4.8 Findings

## Scope and reproducibility

This pass added JPEG q1/q5, WebP q5/q10/q20 and AVIF q5/q10/q20 to the exact prepared Imagenette 256×256 split. The raw-byte primary experiment used ByteStreamCNN, seed 42, the established length-bucket token-budget batching and CUDA. Each condition was encoded into its own resumable, hash-verified stream cache. No existing condition was retrained. Raw-byte test evaluation followed validation checkpoint selection; all eight new conditions had zero truncation.

Calibration used 200 stratified images (20 per class), three timed encodes and decodes per image after warmup. All 1600 decodes succeeded. Encoder versions/settings and stream identities are in `calibration_metadata.json` and `calibration_per_sample.csv`. Historical q10+ measurements remain identified as historical, with their original unequal seed counts; new raw results are one-seed observations.

## New operating points

| Condition | bpp | PSNR dB | SSIM | Raw accuracy, seed 42 | U-probe bits, seed 42 | RGB accuracy, 3 seeds mean ± SD |
|---|---:|---:|---:|---:|---:|---:|
| jpeg_q1 | 0.254 | 21.17 | 0.569 | 0.332 | 0.618 | 0.673 ± 0.020 |
| jpeg_q5 | 0.320 | 23.46 | 0.671 | 0.344 | 0.648 | 0.778 ± 0.004 |
| webp_q5 | 0.299 | 27.60 | 0.805 | 0.258 | 0.289 | 0.783 ± 0.007 |
| webp_q10 | 0.367 | 28.52 | 0.836 | 0.251 | 0.281 | 0.796 ± 0.002 |
| webp_q20 | 0.484 | 29.86 | 0.873 | 0.228 | 0.281 | 0.808 ± 0.002 |
| avif_q5 | 0.125 | 24.37 | 0.693 | 0.190 | 0.199 | 0.725 ± 0.014 |
| avif_q10 | 0.163 | 25.57 | 0.739 | 0.193 | 0.191 | 0.759 ± 0.008 |
| avif_q20 | 0.269 | 27.88 | 0.818 | 0.197 | 0.231 | 0.797 ± 0.003 |

The low-quality settings occupy distinct encoded streams and operating points; calibration exact-stream checks did not identify redundant adjacent settings. JPEG q1→q5 increased mean calibration bpp while improving PSNR/SSIM. WebP q5→q20 and AVIF q5→q20 also increased rate and reconstruction quality. See the calibration tables for per-image variation and median/p95 timing.

## Compression level and raw-byte accessibility

| Condition | bpp | PSNR dB | Raw accuracy | U-probe bits | Raw seeds | Source |
|---|---:|---:|---:|---:|---:|---|
| jpeg_q10 | 0.448 | 26.25 | 0.366 | 0.760 | 3 | Phase 4.5B mean |
| jpeg_q20 | 0.652 | 28.75 | 0.367 | 0.708 | 1 | Phase 4.5B mean |
| jpeg_q60 | 1.238 | 32.81 | 0.320 | 0.419 | 3 | Phase 4.5B mean |
| jpeg_q80 | 1.808 | 35.67 | 0.300 | 0.509 | 1 | Phase 4.5B mean |
| webp_q30 | 0.598 | 31.01 | 0.244 | 0.316 | 1 | Phase 4.5B mean |
| webp_q75 | 1.099 | 34.96 | 0.225 | 0.308 | 1 | Phase 4.5B mean |
| avif_q40 | 0.590 | 31.95 | 0.184 | 0.187 | 1 | Phase 4.5B mean |
| avif_q60 | 1.181 | 36.55 | 0.168 | 0.206 | 1 | Phase 4.5B mean |
| jpeg_q1 | 0.254 | 21.17 | 0.332 | 0.618 | 1 | new seed 42 |
| jpeg_q5 | 0.320 | 23.46 | 0.344 | 0.648 | 1 | new seed 42 |
| webp_q5 | 0.299 | 27.60 | 0.258 | 0.289 | 1 | new seed 42 |
| webp_q10 | 0.367 | 28.52 | 0.251 | 0.281 | 1 | new seed 42 |
| webp_q20 | 0.484 | 29.86 | 0.228 | 0.281 | 1 | new seed 42 |
| avif_q5 | 0.125 | 24.37 | 0.190 | 0.199 | 1 | new seed 42 |
| avif_q10 | 0.163 | 25.57 | 0.193 | 0.191 | 1 | new seed 42 |
| avif_q20 | 0.269 | 27.88 | 0.197 | 0.231 | 1 | new seed 42 |

The observed pattern is codec-specific rather than a universal inverse rate–accessibility law. JPEG's prior q10/q20 seed-42 accuracy was about 0.36–0.37; the new q1/q5 seed-42 results are lower (see table), so the low-quality extension does not show accessibility rising monotonically as rate falls. WebP and AVIF remain low in raw accuracy across the added and historical settings, with small, nonmonotonic U-probe changes. These are per-setting trained probes, not a controlled causal estimate of the effect of quality alone.

The within-codec Pearson/Spearman table reports all measured points, and the paired table compares adjacent settings on the same 3,925 images. Within-image-centered correlations separately remove each image's average level and bootstrap image identities. These summaries reduce image-difficulty confounding but cannot separate quality from all other setting-dependent bitstream changes. Small setting counts, different historic seed counts, and the single-seed new sweep make correlations exploratory; no general law or causal mechanism is established.

Numerical maxima among the common seed-42 operating points are:

| Codec | Highest raw accuracy | Highest U-probe |
|---|---|---|
| jpeg | jpeg_q20 (0.367) | jpeg_q10 (0.734) |
| webp | webp_q5 (0.258) | webp_q30 (0.316) |
| avif | avif_q20 (0.197) | avif_q20 (0.231) |

For JPEG, the accuracy maximum is q20 and the U-probe maximum is q10, both away from the extreme low and high ends. WebP accuracy is highest at q5 while U is highest at q30. AVIF's seed-42 maxima occur at q20 for both. These peak locations are exploratory because historical seed counts differ and the new points are single-seed.

## Rate–distortion and cross-codec matching

Selected-point correlations use the common seed-42 raw results and are descriptive:

| Codec | n settings | bpp vs raw accuracy r / ρ | bpp vs raw U r / ρ | bpp vs RGB accuracy r / ρ |
|---|---:|---:|---:|---:|
| jpeg | 6 | -0.75 / -0.43 | -0.77 / -0.49 | 0.60 / 1.00 |
| webp | 5 | -0.75 / -0.90 | 0.65 / 0.60 | 0.81 / 1.00 |
| avif | 5 | -0.94 / -0.60 | -0.00 / 0.10 | 0.76 / 1.00 |

The rate–accessibility association is not shared by all responses: WebP's bpp–accuracy relationship is negative while its bpp–U relationship is positive; AVIF's bpp–accuracy relationship is negative while bpp–U is approximately zero. Each coefficient uses only five or six selected settings, so it is not a general or causal relationship.

After centering each image across its codec's quality settings, bpp–correctness correlations are much smaller than the between-setting summaries. The paired-bootstrap intervals condition on the fitted checkpoints and resample images; they do not capture training-seed variation. Per-image rate-complexity variation therefore explains part of the pooled pattern, but does not remove the codec/setting association.

| Codec | Within-image r: bpp vs correct (95% image-bootstrap CI) | Within-image r: bpp vs per-image CE |
|---|---:|---:|
| jpeg | -0.038 [-0.054, -0.022] | 0.045 |
| webp | -0.043 [-0.059, -0.026] | -0.003 |
| avif | -0.024 [-0.037, -0.011] | 0.006 |

| Codec pair | Match basis | Observed pair | Match error | Δ raw accuracy (A−B) | Δ raw U (A−B) bits |
|---|---|---|---:|---:|---:|
| jpeg vs webp | rate | jpeg_q5 vs webp_q5 | 6.4% | +0.086 | +0.359 |
| jpeg vs webp | distortion | jpeg_q20 vs webp_q10 | 0.24 dB | +0.116 | +0.427 |
| jpeg vs avif | rate | jpeg_q60 vs avif_q60 | 4.7% | +0.155 | +0.362 |
| webp vs avif | rate | webp_q30 vs avif_q40 | 1.2% | +0.060 | +0.129 |
| webp vs avif | distortion | webp_q5 vs avif_q20 | 0.28 dB | +0.061 | +0.058 |


`matched_operating_point_candidates.csv` records the closest measured rate and distortion pairs plus errors. `paired_operating_point_comparisons.csv` includes only pairs meeting the predeclared thresholds: at most 10% relative bpp difference or at most 0.5 dB absolute PSNR difference. Accessibility is compared only at observed settings; no neural accessibility values are interpolated. Inadequate matches are excluded from the paired file and remain visible in the candidate file.

Across the observed points, raw accessibility is associated with both codec family and compression setting. A pooled correlation can therefore reflect family clustering. The per-codec correlations and matched-point comparisons are the relevant descriptive views, but sample counts remain too small for confirmatory inference.

## Extreme-setting structured comparison

The existing JPEG q1 stream was evaluated with the validated JPEGDCTNet using the entropy-decoded quantized coefficients, sampling layout and quantization tables. The q1 DCT cache stores one model-ready array representation; it does not also store a second dense training grid or per-sample canonical NPZ bundle.

| JPEG q1 stage | Accuracy | U-probe bits | Extraction/decode median, ms | Probe forward, ms/sample |
|---|---:|---:|---:|---:|
| Raw bytes, seed 42 | 0.332 | 0.618 | 0.000 | 0.419 |
| Quantized DCT, seed 42 | 0.717 | 2.033 | 1.208 | 0.133 |
| Decoded RGB, seed 42 | 0.688 | 1.398 | 0.264 | 1.438 |

DCT extraction includes entropy decoding, canonical coefficient-array construction, and model-array adaptation. RGB decode latency is from the deterministic calibration subset; model-forward timing is from the matched seed-42 test evaluation. These costs are reported separately, and their sum is only a descriptive per-sample estimate because the measurements use different timing batches.

The entries are representation-plus-probe measurements: ByteStreamCNN was trained for raw bytes, JPEGDCTNet was trained on the DCT grid, and the RGBResNet18 endpoint is a previously trained frozen reference. The 0.717 DCT accuracy versus 0.688 RGB accuracy must not be interpreted as a representation-only advantage or a matched-probe comparison.

The matching low-quality WebP q5 and AVIF q5 structured probes were not run: the Phase 4.7 instrumentation source is present, but the compiled libwebp/libaom/libavif binaries were absent from the expected build location. Rebuilding that native stack was outside this focused pass. The availability table records this limitation; no missing stage is assigned an accessibility score.

## Phase 4.7 correction

The historical Phase 4.7 AUC files and report were preserved unchanged. Corrected Phase 4.8 AUC requires the same `pre_reconstruction_coefficients → decoded_rgb` endpoints for every seed, with linear trapezoidal area between those two measured endpoints only. Raw bytes are excluded because that measurement exists only for seed 42 and uses a different probe family. Missing endpoints and nonmonotonic cost order are reported as unavailable. JXL has no measured pre-reconstruction stage; WebP's early decoder-plus-adapter cost is not ordered before RGB for all applicable paths. Neither is assigned a zero score.

The corrected Pareto table retains every measured raw, structured and RGB point and labels nondominated points per seed and metric. The corrected AUC summary reports only available seeds, so no across-seed SD is calculated from different endpoint sets. The old q40/q60 values are superseded for interpretation by the Phase 4.8 corrected table.

| Condition | Historical Phase 4.7 U AUC mean | Corrected U AUC mean ± SD | Seeds |
|---|---:|---:|---:|
| avif_q40 | 0.449 | 0.377 ± 0.004 | 3/3 |
| avif_q60 | 0.377 | 0.290 ± 0.002 | 3/3 |
| jxl_d3 | 0.183 | unavailable | 0/3 |
| jxl_d6 | 0.179 | unavailable | 0/3 |
| webp_q30 | 0.190 | unavailable | 0/3 |
| webp_q75 | 0.189 | unavailable | 0/3 |

The historical JXL U AUC means (0.179 and 0.183) included a single seed with a raw-byte endpoint while other seeds had RGB only; those corrected conditions are unavailable because no pre-reconstruction stage was emitted. AVIF's same-endpoint means fall to 0.377 and 0.290. WebP is unavailable because its measured pre-reconstruction decoder-plus-adapter cost is not strictly before the RGB endpoint. The old files remain as historical records.

## AVIF early-exit and YUV reference

| Condition | Pre-reconstruction + adapters | Full YUV | Full RGB | Savings vs YUV | Savings vs RGB |
|---|---:|---:|---:|---:|---:|
| avif_q40 | 1.376 ms | 1.434 ms | 2.471 ms | 4.0% | 44.3% |
| avif_q60 | 2.090 ms | 2.204 ms | 3.154 ms | 5.2% | 33.7% |

The pre-reconstruction path does expose an actual syntax/coefficient representation before inverse transform, prediction and reconstruction. However, its capture plus model-adapter cost is close to full YUV decode. Existing code and artifacts do not provide a matched classifier over native YUV; converting YUV to RGB and then using the RGB model would not be a YUV-inference control. Accordingly, savings relative to YUV and end-to-end latency benefit remain unestablished; the RGB comparison includes avoided color conversion. A future systems frontier must show decoder, decoder-plus-adapter, and neural inference costs separately.

## Implications for video and remaining limits

The low-quality sweep does not establish a stable monotonic relationship between compression efficiency and byte-model utility. It supports carrying codec-specific, rate-matched operating points and explicit byte/structured/decoded stages into video planning, while avoiding assumptions that a better rate–distortion point exposes more or less semantic information. The image work is sufficient for the narrow quality sweep; further broad image sweeps or architecture searches are not justified before the video track. A later targeted multi-seed confirmation is warranted only for the largest observed paired changes, not for the full matrix.

### Scientific assessment

**Supported:** all eight new streams were distinct on the deterministic calibration subset; all full test raw-byte runs completed without truncation; raw-byte accessibility differs by codec family and is not monotonic across the observed settings; corrected Phase 4.7 AUC requires matched endpoints and excludes unavailable/misordered paths; AVIF pre-reconstruction capture is close in cost to full YUV decode.

**Suggestive:** JPEG's historical mid-low settings outperformed the new q1/q5 seed-42 probe, and per-image paired changes may identify quality transitions worth confirming. Codec-specific correlations may help select video conditions.

**Unsupported:** a universal inverse compression-efficiency/accessibility law; a causal explanation attributing accessibility changes to entropy coding alone; practical end-to-end AVIF speedup before reconstruction; any claim about native-YUV neural inference; or a general trend inferred from selected settings and unequal seeds.

## Artifact index

Calibration: `calibration_per_sample.csv`, `calibration_summary.csv`, `calibration_metadata.json`. Full-set results: `operating_point_summary.csv`, `raw_byte_accessibility_per_condition.csv`, `paired_quality_changes.csv`, `within_image_centered_correlations.csv`, `descriptive_correlations.csv`, `matched_operating_point_candidates.csv`, and `paired_operating_point_comparisons.csv`. Structured evidence: `structured_representation_comparison.csv`, `structured_representation_availability.csv`, `jpeg_q1_dct_feature_cache.json`, and `jpeg_q1_dct_extraction_timing_per_sample.csv`. Phase 4.7 corrections: `phase47_corrected_auc_per_seed.csv`, `phase47_corrected_auc_summary.csv`, `phase47_corrected_pareto_points.csv`, `phase47_auc_supersession.csv`, and `phase47_avif_yuv_reassessment.csv`. Plots are PNG files in this directory. Training/resource diagnostics remain alongside the per-seed JSON and `training_resource_parent.json`.
