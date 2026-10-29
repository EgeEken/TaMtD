# Phase 4.7 Findings

## Why this phase exists

Earlier staged evidence centered on project-owned PBC and JPEG's late, relatively simple DCT representation. Phase 4.7 measured exact cached JPEG XL, lossy WebP and AVIF streams to find where standard-codec representations expose Imagenette semantics as decoding proceeds.

## Decoder provenance

| Library | Tag / commit | Build and instrumentation |
|---|---|---|
| libjxl | `v0.12.0` / `a7a9c787341cf703dede03c2009fa460cae5e5df` | Release shared decoder; tools/tests/examples off; pinned Brotli/Highway/skcms submodules; runtime uses the resizable runner at one thread. |
| libwebp | `v1.6.0` / `4fa21912338357f89e4fd51cf2368325b59e9bd9` | Release shared decoder; animation, mux, tools and tests off; patch `libwebp-pre-reconstruction.patch`. |
| libaom | `v3.15.1` / `44d0a57786f432d933ff64b653347c66f4d0fa1d` | Release decoder-only shared build; encoder, multithreading, tests/examples/docs off; patch `libaom-pre-reconstruction.patch`. |
| libavif | `v1.4.2` / `c5240fc79fe5c2407e10afd35f5505ef6333ea49` | Release shared libavif with system libaom decoder only; encoder and libyuv off. |

Windows x64 build used CMake 3.29.2, Ninja 1.12.0 and GCC/G++ 13.2.0. Builds and source checkouts are external to the repository. Rebuild with `scripts/build_phase47_codecs.ps1` or `scripts/build_phase47_codecs.sh`; source/commit pins are in `patches/phase47/versions.json`.

## Exact stage definitions

The manifest records the operation state and boundary for every condition. A pre-reconstruction feature grid contains no RGB/YUV pixels. The WebP callback stops after the last VP8 residual row; the AV1 hook returns after the tile parse-only pass and before frame reconstruction/filter wrapup.

| Condition | Stage | Available | Boundary | Entropy / coeffs | Inverse transform / prediction | Pixels / filters / RGB |
|---|---|---:|---|---|---|---|
| jxl_d6 | jxl_dc_progressive | no | JXL_DEC_FRAME_PROGRESSION / JxlDecoderFlushImage | syntax no; coeffs no | no | pixels no; filters no; RGB no |
| jxl_d6 | decoded_rgb | yes | JXL_DEC_SUCCESS with full RGB output | syntax yes; coeffs yes | yes | pixels yes; filters no; RGB yes |
| jxl_d3 | jxl_dc_progressive | no | JXL_DEC_FRAME_PROGRESSION / JxlDecoderFlushImage | syntax no; coeffs no | no | pixels no; filters no; RGB no |
| jxl_d3 | decoded_rgb | yes | JXL_DEC_SUCCESS with full RGB output | syntax yes; coeffs yes | yes | pixels yes; filters no; RGB yes |
| webp_q30 | pre_reconstruction_coefficients | yes | VP8ProcessRow after final ParseResiduals, before ReconstructRow/FinishRow | syntax yes; coeffs yes | no | pixels no; filters no; RGB no |
| webp_q30 | decoded_yuv | yes | full native YUV output returned | syntax yes; coeffs yes | yes | pixels yes; filters yes; RGB no |
| webp_q30 | decoded_rgb | yes | full RGB output returned | syntax yes; coeffs yes | yes | pixels yes; filters yes; RGB yes |
| webp_q75 | pre_reconstruction_coefficients | yes | VP8ProcessRow after final ParseResiduals, before ReconstructRow/FinishRow | syntax yes; coeffs yes | no | pixels no; filters no; RGB no |
| webp_q75 | decoded_yuv | yes | full native YUV output returned | syntax yes; coeffs yes | yes | pixels yes; filters yes; RGB no |
| webp_q75 | decoded_rgb | yes | full RGB output returned | syntax yes; coeffs yes | yes | pixels yes; filters yes; RGB yes |
| avif_q40 | pre_reconstruction_coefficients | yes | libaom parse-only tile visitor before reconstruction and frame wrapup | syntax yes; coeffs yes | no | pixels no; filters no; RGB no |
| avif_q40 | decoded_yuv | yes | full native YUV output returned | syntax yes; coeffs yes | yes | pixels yes; filters yes; RGB no |
| avif_q40 | decoded_rgb | yes | full RGB output returned | syntax yes; coeffs yes | yes | pixels yes; filters yes; RGB yes |
| avif_q60 | pre_reconstruction_coefficients | yes | libaom parse-only tile visitor before reconstruction and frame wrapup | syntax yes; coeffs yes | no | pixels no; filters no; RGB no |
| avif_q60 | decoded_yuv | yes | full native YUV output returned | syntax yes; coeffs yes | yes | pixels yes; filters yes; RGB no |
| avif_q60 | decoded_rgb | yes | full RGB output returned | syntax yes; coeffs yes | yes | pixels yes; filters yes; RGB yes |

## Timing methodology

All six operating points use the exact Phase 4.5B in-memory streams and the fixed 8,522/947/3,925 train/validation/test split. Stream manifests and every sample payload hash/length were checked before decoding. Timing warmed each codec once and used 256 systematically spaced held-out samples, three repeats and one decoder thread; filesystem reads and serialization are outside the timed call. `perf_counter_ns` boundaries are described per row in `decoder_timing_per_sample.csv`.

Pre-reconstruction decoder time ends at the instrumented callback. The capture adapter time covers copying/transformation after that boundary, and the model-adapter time covers conversion to the shared token inputs. The timing boundary includes callback work needed to gather the IR; `extra_capture_work` marks this instrumentation cost. Native YUV and RGB endpoints are separately timed on the same sample/repeat harness. Stage timings are implementation measurements, not codec-algorithm constants.

| Condition | Stage | n | Decoder median / p95 ms | Capture adapter median ms | Model adapter median ms | Decoder + adapters median ms |
|---|---|---:|---:|---:|---:|---:|
| avif_q40 | decoded_rgb | 768 | 2.471 / 4.225 | 0.000 | 0.000 | 2.471 |
| avif_q40 | decoded_yuv | 768 | 1.434 / 2.908 | 0.000 | 0.000 | 1.434 |
| avif_q40 | pre_reconstruction_coefficients | 768 | 1.043 / 2.248 | 0.032 | 0.312 | 1.376 |
| avif_q60 | decoded_rgb | 768 | 3.154 / 5.411 | 0.000 | 0.000 | 3.154 |
| avif_q60 | decoded_yuv | 768 | 2.204 / 3.957 | 0.000 | 0.000 | 2.204 |
| avif_q60 | pre_reconstruction_coefficients | 768 | 1.721 / 3.264 | 0.035 | 0.309 | 2.090 |
| jxl_d3 | decoded_rgb | 768 | 1.929 / 4.738 | 0.000 | 0.000 | 1.929 |
| jxl_d6 | decoded_rgb | 768 | 3.858 / 5.731 | 0.000 | 0.000 | 3.858 |
| webp_q30 | decoded_rgb | 768 | 0.496 / 0.969 | 0.000 | 0.000 | 0.496 |
| webp_q30 | decoded_yuv | 768 | 0.481 / 0.907 | 0.000 | 0.000 | 0.481 |
| webp_q30 | pre_reconstruction_coefficients | 768 | 0.393 / 0.813 | 7.105 | 0.191 | 7.748 |
| webp_q75 | decoded_rgb | 768 | 0.721 / 1.396 | 0.000 | 0.000 | 0.721 |
| webp_q75 | decoded_yuv | 768 | 0.714 / 1.344 | 0.000 | 0.000 | 0.714 |
| webp_q75 | pre_reconstruction_coefficients | 768 | 0.602 / 1.271 | 7.339 | 0.206 | 8.273 |

## Probe methodology

Full RGB endpoints were evaluated with the three frozen Phase 4 RGBResNet18 checkpoints (seeds 42, 123 and 777). The shared BlockStageTransformer has 539,274 parameters, three 128-wide encoder layers and no codec-ID embedding; all four WebP/AV1 conditions used the same AdamW settings (learning rate 3e-4, weight decay 1e-4, batch size 32, at most 20 epochs, patience 6, CUDA mixed precision, workers 0, pin memory off). Checkpoints were selected on validation; test was evaluated once. After correcting WebP skip-byte initialization, the 120-image gate passed at 96.7% training accuracy after 62 epochs. Structured-stage results are **representation + shared structured probe**, not architecture-independent accessibility.

`U_probe_bits = (CE_prior - CE_model) / ln(2)`. It is a probe utility score and is not Shannon mutual information. The available raw-byte endpoint is the single existing Phase 4.5B byte-probe result per condition; it is context only, uses a different probe family and is not repeated across the three structured-probe seeds.

| Condition | Stage | Probe | Test accuracy mean ± SD | U-probe bits mean ± SD | Relative U mean | Relative accuracy mean |
|---|---|---|---:|---:|---:|---:|
| avif_q40 | decoded_rgb | RGBResNet18 | 0.814 ± 0.002 | 2.215 ± 0.033 | 1.000 | 1.000 |
| avif_q40 | pre_reconstruction_coefficients | BlockStageTransformer | 0.585 ± 0.008 | 1.556 ± 0.018 | 0.703 | 0.680 |
| avif_q40 | raw_bytes | byte | 0.184 ± 0.000 | 0.187 ± 0.000 | 0.083 | 0.117 |
| avif_q60 | decoded_rgb | RGBResNet18 | 0.819 ± 0.002 | 2.250 ± 0.034 | 1.000 | 1.000 |
| avif_q60 | pre_reconstruction_coefficients | BlockStageTransformer | 0.605 ± 0.003 | 1.612 ± 0.005 | 0.716 | 0.703 |
| avif_q60 | raw_bytes | byte | 0.168 ± 0.000 | 0.206 ± 0.000 | 0.090 | 0.095 |
| jxl_d3 | decoded_rgb | RGBResNet18 | 0.817 ± 0.002 | 2.236 ± 0.029 | 1.000 | 1.000 |
| jxl_d3 | raw_bytes | byte | 0.206 ± 0.000 | 0.221 ± 0.000 | 0.097 | 0.147 |
| jxl_d6 | decoded_rgb | RGBResNet18 | 0.810 ± 0.001 | 2.178 ± 0.027 | 1.000 | 1.000 |
| jxl_d6 | raw_bytes | byte | 0.204 ± 0.000 | 0.163 ± 0.000 | 0.074 | 0.146 |
| webp_q30 | decoded_rgb | RGBResNet18 | 0.813 ± 0.002 | 2.206 ± 0.026 | 1.000 | 1.000 |
| webp_q30 | pre_reconstruction_coefficients | BlockStageTransformer | 0.572 ± 0.001 | 1.508 ± 0.010 | 0.684 | 0.662 |
| webp_q30 | raw_bytes | byte | 0.244 ± 0.000 | 0.316 ± 0.000 | 0.142 | 0.201 |
| webp_q75 | decoded_rgb | RGBResNet18 | 0.819 ± 0.004 | 2.245 ± 0.032 | 1.000 | 1.000 |
| webp_q75 | pre_reconstruction_coefficients | BlockStageTransformer | 0.604 ± 0.002 | 1.605 ± 0.006 | 0.715 | 0.701 |
| webp_q75 | raw_bytes | byte | 0.225 ± 0.000 | 0.308 ± 0.000 | 0.135 | 0.174 |

## JXL findings

- **jxl_d6:** decoded_rgb: accuracy 0.810, U 2.178 bits. decoded_rgb median/p95 3.858/5.731 ms.
- **jxl_d3:** decoded_rgb: accuracy 0.817, U 2.236 bits. decoded_rgb median/p95 1.929/4.738 ms.
The exact cached JXL streams emitted zero `JXL_DEC_FRAME_PROGRESSION` events in the held-out samples; their encoder configuration has no progressive option. They therefore have no genuine pre-full RGB stage here. The pinned libjxl v0.12.0 one-shot RGB path is stable with an explicit one-thread resizable runner. The upstream progressive example crashes in `N_AVX2::DecodeGroupImpl` when fed the exact cached codestream in 1,024-byte chunks; a minimal C reproducer and sample are preserved in the ignored `phase47` cache. The partial-input path was excluded from results. No image was synthesized from full RGB.

## WebP findings

- **webp_q30:** decoded_rgb: accuracy 0.813, U 2.206 bits; pre_reconstruction_coefficients: accuracy 0.572, U 1.508 bits. decoded_rgb median/p95 0.496/0.969 ms; decoded_yuv median/p95 0.481/0.907 ms; pre_reconstruction_coefficients median/p95 0.393/0.813 ms.
- **webp_q75:** decoded_rgb: accuracy 0.819, U 2.245 bits; pre_reconstruction_coefficients: accuracy 0.604, U 1.605 bits. decoded_rgb median/p95 0.721/1.396 ms; decoded_yuv median/p95 0.714/1.344 ms; pre_reconstruction_coefficients median/p95 0.602/1.271 ms.
The opt-in libwebp patch returns at `VP8ProcessRow` after all 16 macroblock rows have passed `ParseResiduals`, before `ReconstructRow` and `FinishRow`. The IR preserves coefficient summaries, prediction-mode histograms, segment and quantizer information. Y2 WHT/DC handling has occurred; spatial IDCT and prediction application have not. The patch initializes the VP8 skip byte before its optional parse so the IR is deterministic; all six WebP probe seeds were rerun from rebuilt exact-stream feature maps. Sixteen held-out full RGB decodes matched the pre-fix build byte-for-byte.

## AVIF/AV1 findings

- **avif_q40:** decoded_rgb: accuracy 0.814, U 2.215 bits; pre_reconstruction_coefficients: accuracy 0.585, U 1.556 bits. decoded_rgb median/p95 2.471/4.225 ms; decoded_yuv median/p95 1.434/2.908 ms; pre_reconstruction_coefficients median/p95 1.043/2.248 ms.
- **avif_q60:** decoded_rgb: accuracy 0.819, U 2.250 bits; pre_reconstruction_coefficients: accuracy 0.605, U 1.612 bits. decoded_rgb median/p95 3.154/5.411 ms; decoded_yuv median/p95 2.204/3.957 ms; pre_reconstruction_coefficients median/p95 1.721/3.264 ms.
The libaom patch runs the actual parse-only visitor through AVIF/libavif, gathers AV1 block modes, qindex/segment/skip and low-frequency coefficient summaries, then exits before inverse transform/reconstruction and frame filters. Native YUV and RGB are decoded through the unmodified full path. This is a real early boundary, not the post-reconstruction inspect callback.

## Semantic exposure depth

SED is reported separately for U-probe and chance-corrected accuracy at τ = 0.50, 0.75 and 0.90. It uses only observed pre-RGB stages; no threshold is interpolated. `NR` means not reached before RGB. A fraction above 1 means the feature stage's measured decoder-plus-adapter cost exceeded the full RGB path even though its boundary precedes pixel reconstruction. The normalized AUC uses each seed's observed points; because only one historical raw-byte probe exists per condition, only its matching curve includes that raw-byte point. Structured and byte probes differ, so AUC is descriptive.

| Condition | Metric | τ | Mean ms among reaching seeds | Fraction of full RGB | Seeds reaching |
|---|---|---:|---:|---:|---:|
| jxl_d6 | relative_accuracy | 0.50 | — | — | 0/3 |
| jxl_d6 | relative_accuracy | 0.75 | — | — | 0/3 |
| jxl_d6 | relative_accuracy | 0.90 | — | — | 0/3 |
| jxl_d6 | relative_u_probe_utility | 0.50 | — | — | 0/3 |
| jxl_d6 | relative_u_probe_utility | 0.75 | — | — | 0/3 |
| jxl_d6 | relative_u_probe_utility | 0.90 | — | — | 0/3 |
| jxl_d3 | relative_accuracy | 0.50 | — | — | 0/3 |
| jxl_d3 | relative_accuracy | 0.75 | — | — | 0/3 |
| jxl_d3 | relative_accuracy | 0.90 | — | — | 0/3 |
| jxl_d3 | relative_u_probe_utility | 0.50 | — | — | 0/3 |
| jxl_d3 | relative_u_probe_utility | 0.75 | — | — | 0/3 |
| jxl_d3 | relative_u_probe_utility | 0.90 | — | — | 0/3 |
| webp_q30 | relative_accuracy | 0.50 | 7.748 | 15.615 | 3/3 |
| webp_q30 | relative_accuracy | 0.75 | — | — | 0/3 |
| webp_q30 | relative_accuracy | 0.90 | — | — | 0/3 |
| webp_q30 | relative_u_probe_utility | 0.50 | 7.748 | 15.615 | 3/3 |
| webp_q30 | relative_u_probe_utility | 0.75 | — | — | 0/3 |
| webp_q30 | relative_u_probe_utility | 0.90 | — | — | 0/3 |
| webp_q75 | relative_accuracy | 0.50 | 8.273 | 11.470 | 3/3 |
| webp_q75 | relative_accuracy | 0.75 | — | — | 0/3 |
| webp_q75 | relative_accuracy | 0.90 | — | — | 0/3 |
| webp_q75 | relative_u_probe_utility | 0.50 | 8.273 | 11.470 | 3/3 |
| webp_q75 | relative_u_probe_utility | 0.75 | — | — | 0/3 |
| webp_q75 | relative_u_probe_utility | 0.90 | — | — | 0/3 |
| avif_q40 | relative_accuracy | 0.50 | 1.376 | 0.557 | 3/3 |
| avif_q40 | relative_accuracy | 0.75 | — | — | 0/3 |
| avif_q40 | relative_accuracy | 0.90 | — | — | 0/3 |
| avif_q40 | relative_u_probe_utility | 0.50 | 1.376 | 0.557 | 3/3 |
| avif_q40 | relative_u_probe_utility | 0.75 | — | — | 0/3 |
| avif_q40 | relative_u_probe_utility | 0.90 | — | — | 0/3 |
| avif_q60 | relative_accuracy | 0.50 | 2.090 | 0.663 | 3/3 |
| avif_q60 | relative_accuracy | 0.75 | — | — | 0/3 |
| avif_q60 | relative_accuracy | 0.90 | — | — | 0/3 |
| avif_q60 | relative_u_probe_utility | 0.50 | 2.090 | 0.663 | 3/3 |
| avif_q60 | relative_u_probe_utility | 0.75 | — | — | 0/3 |
| avif_q60 | relative_u_probe_utility | 0.90 | — | — | 0/3 |

Normalized semantic-exposure AUC (trapezoidal utility vs measured normalized decode cost; descriptive only):

| Condition | U-probe AUC mean ± SD | Accuracy AUC mean ± SD |
|---|---:|---:|
| jxl_d6 | 0.179 ± 0.310 | 0.191 ± 0.331 |
| jxl_d3 | 0.183 ± 0.317 | 0.191 ± 0.331 |
| webp_q30 | 0.190 ± 0.330 | 0.200 ± 0.347 |
| webp_q75 | 0.189 ± 0.328 | 0.196 ± 0.339 |
| avif_q40 | 0.449 ± 0.121 | 0.445 ± 0.124 |
| avif_q60 | 0.377 ± 0.151 | 0.375 ± 0.152 |

## Rate-distortion efficiency vs accessibility

The six exact standard-codec points use full-Imagenette bpp/PSNR/SSIM from Phase 4.5B. Positive PSNR residual means observed PSNR exceeds the OLS baseline `PSNR ~ log₂(bpp)`; SSIM has an analogous separate OLS residual. Pearson and Spearman values below are descriptive with n=6; no significance claim is made.

| x | y | n | Pearson r | Spearman ρ |
|---|---|---:|---:|---:|
| rd_efficiency_residual | raw_u_probe_bits | 6 | -0.041 | -0.086 |
| rd_efficiency_residual | raw_accuracy | 6 | -0.465 | -0.429 |
| rd_efficiency_residual | sed50_u_fraction | 4 | -0.933 | -0.800 |
| rd_efficiency_residual | sed75_u_fraction | 0 | — | — |
| rd_efficiency_residual | sed90_u_fraction | 0 | — | — |
| rd_efficiency_residual | u_probe_auc | 6 | 0.868 | 0.943 |

## Rate-distortion efficiency vs exposure depth

SED/AUC correlations use each condition's mean over the three structured-probe seeds where a threshold was reached. The effective sample size can be below six because JXL has no progressive pre-full observation. These associations are exploratory and cannot establish that compression efficiency causes semantic exposure to move later.

In these points, RD residual versus SED50 fraction is Pearson -0.933 and Spearman -0.800 (n=4); RD residual versus normalized U-probe AUC is Pearson 0.868 and Spearman 0.943 (n=6). Higher AUC means utility appears earlier, so the observed direction does not support the idea that better RD efficiency pushes semantics deeper. The small sample and differing codec/probe behavior make this a weakening observation, not a general rejection.
| Condition | bpp | PSNR dB | SSIM | PSNR RD residual dB | raw U bits | SED50 ms / fraction | U-probe AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| webp_q30 | 0.598 | 31.011 | 0.897 | 0.074 | 0.316 | 7.748 / 15.615 | 0.190 |
| webp_q75 | 1.099 | 34.957 | 0.950 | -0.133 | 0.308 | 8.273 / 11.470 | 0.189 |
| avif_q40 | 0.590 | 31.945 | 0.911 | 1.090 | 0.187 | 1.376 / 0.557 | 0.449 |
| avif_q60 | 1.181 | 36.545 | 0.962 | 0.967 | 0.206 | 2.090 / 0.663 | 0.377 |
| jxl_d6 | 0.569 | 29.853 | 0.866 | -0.747 | 0.163 | — / — | 0.179 |
| jxl_d3 | 0.966 | 32.960 | 0.926 | -1.251 | 0.221 | — / — | 0.183 |

## Historical JPEG and PBC comparators

`historical_jpeg_pbc_comparators.csv` carries the existing JPEG q10 and PBC balanced Phase 4.5B frontier rows. Their decoder implementations, test timing subsets and probe families differ; they are context only and are not overlaid on the same-harness standard-codec plots.

## Systems interpretation

The early structured points include instrumentation and adapter costs, while full RGB points use the native decoder paths. The report therefore compares measured implementation paths and probe utility, not a codec-invariant computational lower bound. Raw-byte and structured probes also differ. No single global Pareto claim is made across the historical JPEG/PBC measurements.

## Strongest supported claims

- Exact WebP and AVIF streams yield deterministic structured stages before pixel reconstruction, with same-harness native YUV and RGB endpoints.
- The exact cached JXL operating points do not yield a valid pre-full stage through the pinned public API and are not counted as staged results.
- AVIF/AV1 exposes a measured early point with useful but incomplete utility; WebP's capture adapter dominates its online cost, while JXL is full-decode-only at these operating points.
- All codec-to-codec interpretation remains conditioned on the shared but non-identical probe families.

- **avif_q40:** pre-reconstruction test accuracy 0.585, relative U-probe utility 0.703; decoder plus adapters 1.376 ms vs full RGB 2.471 ms (0.557 of RGB median).
- **avif_q60:** pre-reconstruction test accuracy 0.605, relative U-probe utility 0.716; decoder plus adapters 2.090 ms vs full RGB 3.154 ms (0.663 of RGB median).
- **webp_q30:** pre-reconstruction test accuracy 0.572, relative U-probe utility 0.684; capture plus model adapters raise cumulative median to 7.748 ms vs full RGB 0.496 ms.
- **webp_q75:** pre-reconstruction test accuracy 0.604, relative U-probe utility 0.715; capture plus model adapters raise cumulative median to 8.273 ms vs full RGB 0.721 ms.

## Claims that remain unsupported

The six-point exploratory analysis does not establish that better rate-distortion efficiency pushes semantics deeper, that any probe score is an architecture-independent accessibility measure, or that these timings generalize to other implementations, images, hardware or video.

## Relevance to upcoming video work

The AV1 hook sits in libaom's codec-level parse-only tile path and its features describe a frame grid, so the hook itself can be reused behind a future video-frame/OBU adapter. The tested path is AVIF still-image decode only; frame references, inter-frame state, timestamps and video container behavior were not validated. No video or VLM experiment was run.

## Portfolio-safe summary

Instrumented open-source WebP/VP8 and AV1/libaom decoders to expose structured representations before pixel reconstruction, then measured classification utility and same-harness decode cost on exact Imagenette codec streams. Tested the cached JPEG XL files through the pinned public progression API; those streams exposed no progressive stage, so JXL is reported as full-decode-only at these operating points.

## RGB endpoint recheck

All 18 current full-RGB checkpoint evaluations used 3,925 held-out samples. Against the Phase 4.5B endpoint table, the maximum absolute accuracy delta was 0.00102 and the maximum absolute U-probe delta was 0.00156 bits. Eight repeated RGB decodes per condition were byte-identical. Small cross-library endpoint differences are retained in `full_decode_rechecks.csv` rather than silently treated as exact parity.

The structured training resource artifact contains 12 completed child rows, including process RSS/pagefile peaks, system commit headroom, disk/RAM/VRAM minima and CUDA allocation peaks. Parent-process disk/memory recovery snapshots are in `training_resource_recovery.csv`.
