# Phase 4.5B Findings

## Scope and provenance

This is a post-hoc analysis of the existing 13,394-sample Imagenette Phase 3/4 cache and saved checkpoints. It created no new codec operating points, performed no classifier training or retraining, launched no cloud resources, and did not run video/VLM work. All per-sample encoded streams were checked against their manifest lengths and SHA-256 values. Prepared 256×256 RGB hashes were checked against the dataset and codec manifests. PBC STORE and LZMA variants were decoded and confirmed pixel-identical across all three presets and all 13,394 samples.

Phase 1 measurements used a separate 95-image calibration set. This report uses new measurements over the exact Imagenette split for every required existing stream. The distinction and both sets of values are in `imagenette_vs_phase1_calibration.csv`; split aggregates are in `imagenette_rd_summary.csv`.

## Methodology audit

Raw byte results use the same `ByteStreamCNN` architecture. PBC parsed/canonical IR uses `PBCInstructionNet`; JPEG DCT IR uses `JPEGDCTNet`; RGB uses `RGBResNet18(width=32, random initialization)`. The staged frontier therefore compares a representation together with its trained probe; it is not a representation-only causal experiment. RGB endpoint checks use the already trained original-RGB reference checkpoint for each available seed. No checkpoint was selected using this analysis.

The cross-codec seed-42 raw-byte checks confirm the same ByteStreamCNN, AdamW, learning rate 1e-3, weight decay 1e-4, seed, split, zero truncation, and validation-selected checkpoint rule: `{"AdamW": true, "ByteStreamCNN": true, "batch_size_32": true, "device_cuda": true, "highest_validation_accuracy_checkpoint": true, "learning_rate_1e-3": true, "no_truncation": true, "same_shared_split_counts": true, "seed_42": true, "weight_decay_1e-4": true}`. PBC balanced/quality byte runs used the Phase 4 memory-safety override (batch 8, CPU); the modern codec and JPEG seed-42 raw-byte runs used batch 32, CUDA. This is a batch/device difference in cross-codec comparisons. All 55 saved classifier checkpoints were re-evaluated against their recorded test accuracy; see `checkpoint_accuracy_recheck.json`.

## Imagenette-aligned rate/distortion and decoded RGB endpoints

The full per-image rate/distortion output is `imagenette_rd_per_sample.csv.gz` (16 stream forms × 13,394 samples). The table below gives held-out test means; PBC LZMA and STORE rows share decoded-image quality but have different stream rates.

| Condition | Raw form | Imagenette bpp | PSNR dB | SSIM | Raw acc. | Raw CE nats | U bits | Decoded RGB acc. | Relative U | k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pbc_compression | pbc_lzma_bytes | 0.621 | 26.13 | 0.7712 | 0.1967 | 2.169 | 0.194 | 0.7359 | 0.115 | 3 |
| pbc_compression | pbc_store_bytes | 0.832 | 26.13 | 0.7712 | 0.3576 | 1.850 | 0.655 | 0.7359 | 0.388 | 3 |
| pbc_balanced | pbc_lzma_bytes | 1.398 | 29.43 | 0.8769 | 0.1846 | 2.198 | 0.152 | 0.7997 | 0.072 | 3 |
| pbc_balanced | pbc_store_bytes | 2.180 | 29.43 | 0.8769 | 0.3907 | 1.760 | 0.785 | 0.7997 | 0.371 | 3 |
| pbc_quality | pbc_lzma_bytes | 2.735 | 33.40 | 0.9418 | 0.1663 | 2.226 | 0.112 | 0.8143 | 0.050 | 3 |
| pbc_quality | pbc_store_bytes | 4.489 | 33.40 | 0.9418 | 0.3942 | 1.778 | 0.758 | 0.8143 | 0.342 | 3 |
| jpeg_q10 | jpeg_file_bytes | 0.448 | 26.25 | 0.7741 | 0.3664 | 1.777 | 0.760 | 0.8069 | 0.350 | 3 |
| jpeg_q60 | jpeg_file_bytes | 1.238 | 32.81 | 0.9279 | 0.3201 | 2.013 | 0.419 | 0.8196 | 0.186 | 3 |
| webp_q30 | codec_file_bytes | 0.598 | 31.01 | 0.8971 | 0.2436 | 2.085 | 0.316 | 0.8148 | 0.142 | 1 |
| webp_q75 | codec_file_bytes | 1.099 | 34.96 | 0.9496 | 0.2255 | 2.090 | 0.308 | 0.8229 | 0.135 | 1 |
| avif_q40 | codec_file_bytes | 0.590 | 31.95 | 0.9106 | 0.1839 | 2.174 | 0.187 | 0.8153 | 0.083 | 1 |
| avif_q60 | codec_file_bytes | 1.181 | 36.55 | 0.9620 | 0.1684 | 2.161 | 0.206 | 0.8209 | 0.090 | 1 |
| jxl_d6 | codec_file_bytes | 0.569 | 29.85 | 0.8663 | 0.2038 | 2.191 | 0.163 | 0.8092 | 0.074 | 1 |
| jxl_d3 | codec_file_bytes | 0.966 | 32.96 | 0.9261 | 0.2059 | 2.151 | 0.221 | 0.8191 | 0.097 | 1 |

The Phase 1 calibration set remains context only. For each required operating point the Imagenette alignment changes measured bpp and distortion; accessibility comparisons below use the Imagenette test values. All decoded-RGB endpoint evaluations use existing RGB reference checkpoints for seeds 42, 123, and 777. Endpoint predictions on validation/test are reported seed-wise in `decoded_rgb_endpoints.csv`; `decoded_rgb_endpoint_summary.csv` reports mean ± sample standard deviation, accuracy difference and normalized accessibility against original RGB, and relative U-probe utility. Per-image correctness is in `test_predictions_summary.csv.gz`.

| Condition | RGB seeds | Accuracy mean ± sample SD | Δ vs original RGB | Accuracy accessibility | U bits mean ± sample SD | Relative U |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| avif_q40 | 3 | 0.8138 ± 0.0013 | -0.0076 | 0.990 | 2.215 ± 0.033 | 0.981 |
| avif_q60 | 3 | 0.8191 ± 0.0021 | -0.0022 | 0.997 | 2.250 ± 0.034 | 0.997 |
| jpeg_q10 | 3 | 0.8069 ± 0.0037 | -0.0144 | 0.980 | 2.174 ± 0.030 | 0.963 |
| jpeg_q20 | 3 | 0.8161 ± 3.89e-04 | -0.0052 | 0.993 | 2.229 ± 0.034 | 0.988 |
| jpeg_q60 | 3 | 0.8196 ± 0.0029 | -0.0017 | 0.998 | 2.249 ± 0.033 | 0.997 |
| jpeg_q80 | 3 | 0.8199 ± 0.0037 | -0.0014 | 0.998 | 2.253 ± 0.034 | 0.998 |
| jxl_d3 | 3 | 0.8175 ± 0.0020 | -0.0038 | 0.995 | 2.236 ± 0.029 | 0.991 |
| jxl_d6 | 3 | 0.8097 ± 0.0011 | -0.0116 | 0.984 | 2.177 ± 0.025 | 0.965 |
| original_rgb | 3 | 0.8213 ± 9.65e-04 | 0.0000 | 1.000 | 2.257 ± 0.032 | 1.000 |
| pbc_balanced | 3 | 0.7997 ± 0.0047 | -0.0217 | 0.970 | 2.115 ± 0.037 | 0.937 |
| pbc_compression | 3 | 0.7359 ± 0.0041 | -0.0854 | 0.882 | 1.685 ± 0.013 | 0.746 |
| pbc_quality | 3 | 0.8143 ± 0.0034 | -0.0070 | 0.990 | 2.217 ± 0.032 | 0.982 |
| webp_q30 | 3 | 0.8128 ± 0.0022 | -0.0085 | 0.988 | 2.206 ± 0.026 | 0.978 |
| webp_q75 | 3 | 0.8189 ± 0.0035 | -0.0025 | 0.997 | 2.245 ± 0.032 | 0.995 |

## Convergence audit

Across `55` trained checkpoint histories, `33` stopped before their configured maximum epoch count. That records an early-stop event, not proof of convergence. For seed-42 raw-byte JPEG/WebP/AVIF/JXL, the validation traces and per-run best/final loss fields are in `training_convergence_audit.csv` and `raw_byte_validation_convergence.png`. Read them as validation behavior under this cosine-decay/early-stopping schedule: a validation peak followed by a decline gives no evidence that more epochs under the same schedule would help. The low/mid byte points are one-seed for WebP/AVIF/JXL; checkpoint seed variance is unmeasured there.

PBC balanced/quality byte models use batch 8 on CPU for the Phase 4 memory guard, while selected JPEG and standard-codec seed-42 models use batch 32 on CUDA. The common architecture/optimizer/LR/seed/split rules hold, but the different batch/device protocol weakens direct optimization-equivalence claims, especially for mid-rate PBC comparisons.

## Rate/distortion/accessibility analysis

Within-codec low-to-mid changes are:

| Codec | Δ bpp | Δ PSNR dB | Δ SSIM | Δ raw accuracy | Δ raw U bits | Δ decoded RGB accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| jpeg (jpeg_q10 → jpeg_q60) | 0.791 | 6.56 | 0.1538 | -0.0463 | -0.341 | 0.0127 |
| webp (webp_q30 → webp_q75) | 0.501 | 3.95 | 0.0526 | -0.0181 | -0.007 | 0.0060 |
| avif (avif_q40 → avif_q60) | 0.590 | 4.60 | 0.0515 | -0.0155 | 0.019 | 0.0054 |
| jxl (jxl_d6 → jxl_d3) | 0.397 | 3.11 | 0.0599 | 0.0020 | 0.058 | 0.0078 |

These comparisons show whether more bits and better reconstruction translate into more byte-probe utility; seed counts differ and WebP/AVIF/JXL remain exploratory single-seed evidence. They do not establish causality.

The rate/distortion model uses 14 raw-representation rows across 11 selected codec operating points: PBC STORE and LZMA for the three PBC settings, plus two points each for JPEG, WebP, AVIF, and JXL. Its `log(bpp)+PSNR` accuracy model has R² `0.332` and leave-one-out RMSE `0.0904`; adding codec identity gives R² `0.857` and leave-one-out RMSE `0.0556`. The corresponding U-probe models have R² `0.302` → `0.839` and leave-one-out RMSE `0.271` → `0.180`; the `log(bpp)+SSIM+codec` U model has R² `0.887`. In this small selected sample, bitrate and PSNR alone explain a limited share of probe performance, while codec indicators materially improve the descriptive fit. This leaves codec-conditioned residuals after rate/distortion adjustment; it does not identify causal codec effects. With 14 rows, correlated predictors, repeated PBC wrapper points, and 5 codec families, these are descriptive fits only. Do not interpret ordinary p-values or the fitted codec effects as strong evidence.

The rate/distortion-only residual table is a diagnostic, not a universal metric:

| Condition | Observed accuracy | Rate/distortion prediction | Residual |
| --- | ---: | ---: | ---: |
| pbc_quality | 0.1663 | 0.2843 | -0.1180 |
| pbc_balanced | 0.1846 | 0.3001 | -0.1155 |
| pbc_compression | 0.1967 | 0.2960 | -0.0993 |
| jxl_d6 | 0.2038 | 0.2327 | -0.0288 |
| avif_q40 | 0.1839 | 0.2029 | -0.0189 |
| jxl_d3 | 0.2059 | 0.2206 | -0.0147 |
| avif_q60 | 0.1684 | 0.1788 | -0.0104 |
| webp_q30 | 0.2436 | 0.2181 | 0.0254 |
| webp_q75 | 0.2255 | 0.1985 | 0.0270 |
| pbc_compression | 0.3576 | 0.3159 | 0.0418 |
| pbc_balanced | 0.3907 | 0.3302 | 0.0605 |
| pbc_quality | 0.3942 | 0.3178 | 0.0764 |
| jpeg_q60 | 0.3201 | 0.2397 | 0.0804 |
| jpeg_q10 | 0.3664 | 0.2721 | 0.0943 |

`rd_accessibility_model.json` stores coefficients, R²/adjusted R², rank, RMSE, and leave-one-out diagnostics for accuracy and U-probe, with and without codec indicators. `rd_adjusted_accessibility_residual.png` visualizes the in-sample diagnostic residuals.

Probe usable information is defined here as `U_probe_bits = (CE_train_prior − CE_model) / ln(2)`, where both cross-entropies are evaluated on held-out labels and the prior distribution is estimated from the train split. This is an empirical predictive diagnostic for each trained probe, **not Shannon mutual information**. Full per-image cross-entropy, true-class confidence, prediction, and correctness summaries are in `test_predictions_summary.csv.gz`; complete float16 logits live only in the ignored cache with an index and deterministic generation script.

## Cross-codec accessibility

The PBC and JPEG points with saved confirmation runs are three-seed means: PBC LZMA/STORE and JPEG q10/q60. JPEG q20/q80 raw-byte results are single seed. WebP, AVIF, and JXL raw-byte results are also single seed; their decoded-RGB endpoints use the three existing RGB checkpoint seeds. At the closest selected rates, JPEG q10 is 0.448 bpp and has 0.366 raw-byte accuracy, compared with WebP q30 at 0.598 bpp/0.244, AVIF q40 at 0.590 bpp/0.184, and JXL d6 at 0.569 bpp/0.204. JPEG q10 is higher in `7/10` classes than WebP q30 and `8/10` than AVIF q40 or `8/10` than JXL d6. At the mid-rate points, JPEG q60 is higher in `7/10` classes than WebP q75, `8/10` than AVIF q60, and `9/10` than JXL d3. These class counts come from seed 42; they do not measure modern-codec seed variance.

For PBC, STORE byte probes reach 0.358–0.394 accuracy across three seeds while same-pixel LZMA-wrapped streams reach 0.166–0.197. For JPEG, q10 is higher than q60 on raw-byte accuracy (0.366 vs 0.320), while decoded-RGB endpoints remain about 0.807–0.820. The six paired seed-42 comparisons and their image-bootstrap intervals are below; they compare trained probe outputs, not intrinsic codec properties.

## Paired cross-codec accessibility

Paired counts and deterministic image-bootstrap intervals use the same seed-42 test images. Exact McNemar values are descriptive because the codec probes are not multi-seed for WebP/AVIF/JXL.

| Pair | Both correct | A only | B only | Both wrong | Δ accuracy A−B | Paired bootstrap 95% CI | McNemar exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jpeg_q10 vs webp_q30 | 577 | 819 | 379 | 2150 | 0.1121 | [0.0953, 0.1289] | 1.09e-37 |
| jpeg_q10 vs avif_q40 | 453 | 943 | 269 | 2260 | 0.1717 | [0.1552, 0.1880] | 4.74e-88 |
| jpeg_q10 vs jxl_d6 | 446 | 950 | 354 | 2175 | 0.1518 | [0.1343, 0.1692] | 3.10e-63 |
| jpeg_q60 vs webp_q75 | 501 | 768 | 384 | 2272 | 0.0978 | [0.0813, 0.1146] | 4.60e-30 |
| jpeg_q60 vs avif_q60 | 369 | 900 | 292 | 2364 | 0.1549 | [0.1383, 0.1712] | 1.92e-72 |
| jpeg_q60 vs jxl_d3 | 390 | 879 | 418 | 2238 | 0.1175 | [0.0999, 0.1348] | 3.83e-38 |

The largest absolute paired accuracy difference in these six preselected comparisons is `jpeg_q10 vs avif_q40` (`Δ=0.1717`, 95% paired bootstrap CI [0.1552, 0.1880]). This measures the paired outputs of these trained probes, not an intrinsic codec property.

## Per-class and difficulty-conditioned results

`per_class_metrics.csv` contains class-wise accuracy, cross-entropy, support, full confusion counts, matched decoded-RGB differences, and differences versus the low/mid JPEG comparator. `per_class_raw_byte_accuracy_heatmap.png` aggregates accuracy by class; `class_accuracy_rank_correlations.csv` reports Spearman correlations between class-accuracy profiles. Median pairwise class-profile correlation excluding self-correlations is `0.637` (range `-0.213` to `0.964`). The JPEG byte models most often confuse English springer and chain saw with tench; WebP/AVIF often map several classes to French horn, while JXL often confuses gas pump/chain saw with garbage truck. JPEG q10’s advantage is broad across most classes, not all ten. The detailed table shows exceptions and seed counts.

`difficulty_conditioned.csv` uses matched original-RGB predictions: correct with confidence ≥0.75, correct below 0.75, or incorrect. It reports raw-byte accuracy in each bucket and byte-correct/RGB-wrong versus RGB-correct/byte-wrong overlap. For seed-42 JPEG q10, raw accuracy is `0.412` on `3004` high-confidence RGB-correct images and `0.159` on `697` RGB errors; it correctly classifies `111` RGB-error images, while missing `1943` images RGB classified correctly. The byte probe mainly succeeds on easier RGB cases but recovers a smaller complementary subset. This is a model-confidence stratification, not a ground-truth image-difficulty label. `per_image_rate_distortion_correlations.csv` reports Spearman associations between size/bpp, PSNR, SSIM and raw correctness, true-class probability, and cross-entropy. Across `32` condition/seed groups, bpp has `median -0.132, range -0.246 to -0.034 over 32 condition/seed groups` with correctness; SSIM has `median 0.170, range 0.003 to 0.248 over 32 condition/seed groups`. `per_image_rate_distortion_quantiles.csv` gives the outcomes in equal-count predictor quartiles. These are associations within the measured samples, not causal effects; file size can reflect image content complexity as well as codec behavior.

## Absolute decode-cost frontier

`absolute_frontier.csv` uses measured conventional preprocessing/decode milliseconds and test accuracy plus U-probe bits. PBC uses Phase 2 stage timings; JPEG uses the refreshed Phase 4 timing pass; WebP/AVIF/JXL decoded endpoints reuse Phase 1 full-decode medians from the 95-image calibration set. The timing source and implementation caveat are repeated on every row. Cross-codec points form an **implementation-level systems frontier**, not an algorithm-only codec comparison. Fully staged codecs (PBC/JPEG) are visually distinguished from endpoint-only WebP/AVIF/JXL.

## Dominated points

`pareto_analysis.csv` identifies all dominators for each point. There are `27` accuracy-dominated and `26` U-probe-dominated points among `30` measured representation points. The accuracy-efficient points are ``pbc_balanced::pbc_store_bytes`, `jpeg_q10::jpeg_file_bytes`, `jpeg_q60::decoded_rgb_reference``; the U-probe-efficient points are ``pbc_balanced::pbc_store_bytes`, `jpeg_q10::jpeg_file_bytes`, `jpeg_q60::decoded_rgb_reference`, `avif_q60::decoded_rgb_reference``. Accuracy-dominated point IDs: `pbc_compression::pbc_lzma_bytes, pbc_compression::pbc_store_bytes, pbc_compression::pbc_canonical_ir, pbc_compression::decoded_rgb_reference, pbc_balanced::pbc_lzma_bytes, pbc_balanced::pbc_canonical_ir, pbc_balanced::decoded_rgb_reference, pbc_quality::pbc_lzma_bytes, pbc_quality::pbc_store_bytes, pbc_quality::pbc_canonical_ir, pbc_quality::decoded_rgb_reference, jpeg_q10::jpeg_dct_ir, jpeg_q10::decoded_rgb_reference, jpeg_q60::jpeg_file_bytes, jpeg_q60::jpeg_dct_ir, webp_q30::codec_file_bytes, webp_q30::decoded_rgb_reference, webp_q75::codec_file_bytes, webp_q75::decoded_rgb_reference, avif_q40::codec_file_bytes, avif_q40::decoded_rgb_reference, avif_q60::codec_file_bytes, avif_q60::decoded_rgb_reference, jxl_d6::codec_file_bytes, jxl_d6::decoded_rgb_reference, jxl_d3::codec_file_bytes, jxl_d3::decoded_rgb_reference`. U-probe-dominated point IDs: `pbc_compression::pbc_lzma_bytes, pbc_compression::pbc_store_bytes, pbc_compression::pbc_canonical_ir, pbc_compression::decoded_rgb_reference, pbc_balanced::pbc_lzma_bytes, pbc_balanced::pbc_canonical_ir, pbc_balanced::decoded_rgb_reference, pbc_quality::pbc_lzma_bytes, pbc_quality::pbc_store_bytes, pbc_quality::pbc_canonical_ir, pbc_quality::decoded_rgb_reference, jpeg_q10::jpeg_dct_ir, jpeg_q10::decoded_rgb_reference, jpeg_q60::jpeg_file_bytes, jpeg_q60::jpeg_dct_ir, webp_q30::codec_file_bytes, webp_q30::decoded_rgb_reference, webp_q75::codec_file_bytes, webp_q75::decoded_rgb_reference, avif_q40::codec_file_bytes, avif_q40::decoded_rgb_reference, avif_q60::codec_file_bytes, jxl_d6::codec_file_bytes, jxl_d6::decoded_rgb_reference, jxl_d3::codec_file_bytes, jxl_d3::decoded_rgb_reference`. Every raw modern-codec file point is dominated at the measured 0 ms preprocessing coordinate; the closest-rate byte comparisons also favor JPEG q10 at a lower bpp than WebP q30, AVIF q40, and JXL d6.

Important examples:

- `pbc_compression::pbc_lzma_bytes` is accuracy-dominated by `jpeg_q10::jpeg_file_bytes;jpeg_q60::jpeg_file_bytes;webp_q30::codec_file_bytes;webp_q75::codec_file_bytes;jxl_d6::codec_file_bytes;jxl_d3::codec_file_bytes`.
- `pbc_compression::pbc_store_bytes` is accuracy-dominated by `jpeg_q10::jpeg_file_bytes`.
- `pbc_compression::pbc_canonical_ir` is accuracy-dominated by `jpeg_q10::jpeg_dct_ir;jpeg_q10::decoded_rgb_reference;jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference;webp_q30::decoded_rgb_reference;webp_q75::decoded_rgb_reference`.
- `pbc_compression::decoded_rgb_reference` is accuracy-dominated by `jpeg_q10::jpeg_dct_ir;jpeg_q10::decoded_rgb_reference;jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference;webp_q30::decoded_rgb_reference;webp_q75::decoded_rgb_reference;avif_q40::decoded_rgb_reference;avif_q60::decoded_rgb_reference`.
- `pbc_balanced::pbc_lzma_bytes` is accuracy-dominated by `pbc_compression::pbc_lzma_bytes;jpeg_q10::jpeg_file_bytes;jpeg_q60::jpeg_file_bytes;webp_q30::codec_file_bytes;webp_q75::codec_file_bytes;jxl_d6::codec_file_bytes;jxl_d3::codec_file_bytes`.
- `pbc_balanced::pbc_canonical_ir` is accuracy-dominated by `pbc_compression::pbc_canonical_ir;jpeg_q10::jpeg_dct_ir;jpeg_q10::decoded_rgb_reference;jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference;webp_q30::decoded_rgb_reference;webp_q75::decoded_rgb_reference;avif_q40::decoded_rgb_reference`.
- `pbc_balanced::decoded_rgb_reference` is accuracy-dominated by `jpeg_q10::decoded_rgb_reference;jpeg_q60::decoded_rgb_reference;webp_q30::decoded_rgb_reference;webp_q75::decoded_rgb_reference;avif_q40::decoded_rgb_reference;avif_q60::decoded_rgb_reference`.
- `pbc_quality::pbc_lzma_bytes` is accuracy-dominated by `pbc_compression::pbc_lzma_bytes;pbc_balanced::pbc_lzma_bytes;jpeg_q10::jpeg_file_bytes;jpeg_q60::jpeg_file_bytes;webp_q30::codec_file_bytes;webp_q75::codec_file_bytes;avif_q40::codec_file_bytes;avif_q60::codec_file_bytes;jxl_d6::codec_file_bytes;jxl_d3::codec_file_bytes`.
- `pbc_quality::pbc_store_bytes` is accuracy-dominated by `jpeg_q10::jpeg_dct_ir;jpeg_q10::decoded_rgb_reference;jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference`.
- `pbc_quality::pbc_canonical_ir` is accuracy-dominated by `pbc_compression::pbc_canonical_ir;pbc_compression::decoded_rgb_reference;pbc_balanced::pbc_canonical_ir;pbc_balanced::decoded_rgb_reference;jpeg_q10::jpeg_dct_ir;jpeg_q10::decoded_rgb_reference;jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference;webp_q30::decoded_rgb_reference;webp_q75::decoded_rgb_reference;avif_q40::decoded_rgb_reference;avif_q60::decoded_rgb_reference`.
- `pbc_quality::decoded_rgb_reference` is accuracy-dominated by `jpeg_q60::decoded_rgb_reference;webp_q75::decoded_rgb_reference;avif_q60::decoded_rgb_reference`.
- `jpeg_q10::jpeg_dct_ir` is accuracy-dominated by `jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference`.

- `pbc_compression::pbc_lzma_bytes` is U-probe-dominated by `jpeg_q10::jpeg_file_bytes;jpeg_q60::jpeg_file_bytes;webp_q30::codec_file_bytes;webp_q75::codec_file_bytes;avif_q60::codec_file_bytes;jxl_d3::codec_file_bytes`.
- `pbc_compression::pbc_store_bytes` is U-probe-dominated by `jpeg_q10::jpeg_file_bytes`.
- `pbc_compression::pbc_canonical_ir` is U-probe-dominated by `jpeg_q10::jpeg_dct_ir;jpeg_q10::decoded_rgb_reference;jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference;webp_q30::decoded_rgb_reference;webp_q75::decoded_rgb_reference`.
- `pbc_compression::decoded_rgb_reference` is U-probe-dominated by `jpeg_q10::jpeg_dct_ir;jpeg_q10::decoded_rgb_reference;jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference;webp_q30::decoded_rgb_reference;webp_q75::decoded_rgb_reference;avif_q40::decoded_rgb_reference;avif_q60::decoded_rgb_reference`.
- `pbc_balanced::pbc_lzma_bytes` is U-probe-dominated by `pbc_compression::pbc_lzma_bytes;jpeg_q10::jpeg_file_bytes;jpeg_q60::jpeg_file_bytes;webp_q30::codec_file_bytes;webp_q75::codec_file_bytes;avif_q40::codec_file_bytes;avif_q60::codec_file_bytes;jxl_d6::codec_file_bytes;jxl_d3::codec_file_bytes`.
- `pbc_balanced::pbc_canonical_ir` is U-probe-dominated by `pbc_compression::pbc_canonical_ir;jpeg_q10::jpeg_dct_ir;jpeg_q10::decoded_rgb_reference;jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference;webp_q30::decoded_rgb_reference;webp_q75::decoded_rgb_reference;avif_q40::decoded_rgb_reference`.
- `pbc_balanced::decoded_rgb_reference` is U-probe-dominated by `jpeg_q10::jpeg_dct_ir;jpeg_q10::decoded_rgb_reference;jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference;webp_q30::decoded_rgb_reference;webp_q75::decoded_rgb_reference;avif_q40::decoded_rgb_reference;avif_q60::decoded_rgb_reference`.
- `pbc_quality::pbc_lzma_bytes` is U-probe-dominated by `pbc_compression::pbc_lzma_bytes;pbc_balanced::pbc_lzma_bytes;jpeg_q10::jpeg_file_bytes;jpeg_q60::jpeg_file_bytes;webp_q30::codec_file_bytes;webp_q75::codec_file_bytes;avif_q40::codec_file_bytes;avif_q60::codec_file_bytes;jxl_d6::codec_file_bytes;jxl_d3::codec_file_bytes`.
- `pbc_quality::pbc_store_bytes` is U-probe-dominated by `pbc_balanced::pbc_store_bytes;jpeg_q10::jpeg_file_bytes;jpeg_q10::jpeg_dct_ir;jpeg_q10::decoded_rgb_reference;jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference`.
- `pbc_quality::pbc_canonical_ir` is U-probe-dominated by `pbc_compression::pbc_canonical_ir;pbc_compression::decoded_rgb_reference;pbc_balanced::decoded_rgb_reference;jpeg_q10::jpeg_dct_ir;jpeg_q10::decoded_rgb_reference;jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference;webp_q30::decoded_rgb_reference;webp_q75::decoded_rgb_reference;avif_q40::decoded_rgb_reference;avif_q60::decoded_rgb_reference`.
- `pbc_quality::decoded_rgb_reference` is U-probe-dominated by `jpeg_q60::decoded_rgb_reference;webp_q75::decoded_rgb_reference;avif_q60::decoded_rgb_reference`.
- `jpeg_q10::jpeg_dct_ir` is U-probe-dominated by `jpeg_q60::jpeg_dct_ir;jpeg_q60::decoded_rgb_reference`.

The refreshed JPEG numbers test the requested case directly: q60 DCT extraction is `0.5165 ms` at `0.7642 accuracy, while q60 RGB decode is `0.4546 ms` at `0.8196 accuracy. Under these measured means, q60 DCT is slower and less accurate, so q60 RGB dominates it on accuracy and preprocessing time. The Pareto table lists every point's dominators.

## What is genuinely interesting

- Raw-byte accessibility does not rise monotonically with rate or distortion: JPEG, WebP, and AVIF raw accuracy falls at the tested higher-fidelity point; JXL changes by `0.0020`. Within-image bpp correlations are also negative with byte correctness on these samples, consistent with larger files often representing more complex examples rather than being easier for the probe.
- The large PBC STORE/LZMA gap under pixel identity is a replicated, controlled wrapper effect from earlier phases, not a new discovery here. The stronger new evidence is that selected low-rate JPEG byte probes outperform near-rate WebP/AVIF/JXL probes on most classes, while rate/distortion-only fits explain only about one-third of accuracy/U variation in these 14 rows.
- JPEG DCT carries high probe utility, but q60 DCT is dominated by q60 RGB on both measured cost and performance. PBC IR, JPEG DCT, raw bytes, and RGB use different probe families, so these are representation-plus-probe observations rather than architecture-independent accessibility.

## What is not established

- No architecture-independent notion of neural accessibility has been validated.
- WebP/AVIF/JXL intermediate representations remain unmeasured; no intermediate points are fabricated here.
- WebP/AVIF/JXL raw-byte evidence is one seed; their codec-specific ranking uncertainty is not measured.
- Standard-codec differences cannot be causally attributed to entropy coding from these settings.
- The regression has 14 raw-representation rows across 11 selected codec-setting points; class and per-sample analyses are still conditional on the trained probes.
- Timing is implementation- and machine-dependent; WebP/AVIF/JXL timing source uses the Phase 1 calibration set, not Imagenette.
- The RGB reference is a modest random-initialized ResNet-style model, not a task ceiling.

## Go / no-go decision for image research

**continue only with narrow mechanistic experiments**. The current evidence still contains specific testable image mechanisms: PBC wrapper sensitivity under pixel identity and JPEG DCT utility close to the RGB probe. It does not justify a broad architecture sweep or claims about standard-codec entropy coding. If one narrow follow-up fails to reproduce across a second probe or split, shift most new effort to the planned video track.

## Recommended next experiments (not run)

1. Replicate the PBC STORE/LZMA pixel-identical control with one second, materially different byte probe. Scientific value high; compute low; risk of repeating the already-observed wrapper effect moderate.
2. Split JPEG DCT inputs into DC/low-frequency and high-frequency ablations at q10/q60. Scientific value high; compute low; risk of confirming the already strong DCT/RGB result moderate.
3. Re-evaluate the selected raw-byte conditions with a small second architecture using a fixed, matched batch protocol. Scientific value moderate-high; compute moderate; risk of finding similar rankings moderate.
4. Run byte-order/locality controls on Imagenette for the selected PBC/JPEG points. Scientific value moderate; compute low; risk of reproducing the early-testing order result moderate.
5. If one of these mechanisms survives, expose one public WebP/AVIF/JXL intermediate stage using a standard API and compare it with its existing file/RGB endpoints. Scientific value uncertain; compute/engineering cost high; risk of endpoint-only results is high.

## Reproduction

With the original ignored cache, checkpoints, source dataset, PBC checkout, and codec tools available:

Set `PBC_ROOT` to the local PBC checkout or pass `--pbc-root <path>` to the first command.

```powershell
python scripts/run_phase45b_analysis.py --device cuda
python scripts/report_phase45b_analysis.py
```

The first command performs decoding and inference only. It does not train or modify prior Phase 1–4 findings, caches, streams, or checkpoints. Summary tables, plots, and this report are new files under `results/Phase 4.5B Analysis/`.
