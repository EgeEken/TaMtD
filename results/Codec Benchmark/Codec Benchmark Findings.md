# Codec Benchmark Findings

## Dataset and preprocessing

- Dataset: local PBC training_set natural-image calibration set.
- Prepared source set: 95 images; this run evaluated 95. Selection: lexicographically sorted relative paths (casefold, then exact); exclude listed filenames, then take first --limit files.
- Excluded non-photographic files: googlewiki.png, mario64.png, monet.jpg, monkey.png, ratedistortion.png.
- Preprocessing: EXIF transpose, RGB conversion, centered square crop and Pillow BICUBIC resize to exactly 256×256. The prepared PNGs and SHA-256 hashes are in `sources/manifest.jsonl`; all codec runs read those same RGB pixel arrays.
- Source details: `C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC\training_set`. No CIFAR-10 images were used.

## Codec settings and software

- **JPEG:** Pillow/libjpeg-turbo 3.1.1; quality [1, 5, 10, 20, 30, 60, 80], 4:2:0, optimize=false, progressive=false
- **WEBP:** quality [1, 5, 10, 20, 30, 40, 50, 60, 75, 90, 100], method=6, lossy, libwebp subsampling default
- **AVIF:** quality [1, 5, 10, 20, 30, 40, 50, 60, 75, 90, 95], speed=6, subsampling=4:2:0, max_threads=1
- **JXL:** distance [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25], effort=7, num_threads=1
- **PNG:** lossless, Pillow compress_level=6, optimize=false
- **PBC:** compression, balanced, quality, high_quality; full effective config and provenance are in config.json and per-sample metadata

- Runtime: libavif 1.3.0, libjpeg-turbo 3.1.1, libwebp 1.5.0, matplotlib 3.10.7, numpy 2.2.6, opencv 4.12.0, pillow 11.3.0, python 3.13.5.
- AVIF and WebP use Pillow's bundled codec backends; JXL uses the recorded `cjxl`/`djxl` version in `config.json`.
- PBC provenance: multistage_pbc31 @ `3e9e8c94b7497ecefc5e482a66fde697dfad6e19`, dirty=True. Each preset's full config, configured LZMA use, and actual stream entropy mode are recorded in `config.json` and sample metadata.
- Git `status` returned `fatal: this operation must be run in a work tree`; the untracked-file fallback found files, so dirty=true is certain.
- Missing settings or runtime failures: none.

## Timing and metrics

- Each setting had 1 unmeasured warm-up encode/decode pair(s), followed by 3 timed encode and full-decode operations for every image. Module imports and PBC initialization are outside the timed regions. JXL timing includes temporary-file I/O and process startup.
- Latencies are repeated per-image measurements; aggregate median and p95 are across all timed operations for that setting.
- MSE, PSNR, and SSIM compare decoded RGB against the exact prepared 256×256 RGB source. SSIM uses 11×11 Gaussian windows (σ=1.5), RGB channels, and standard C1/C2 constants. PSNR uses peak 255.
- PBC encode is Python/NumPy/Numba-heavy; JPEG/WebP/AVIF/JXL use optimized native libraries/tools. Latency numbers describe these implementations and this machine, not codec algorithms in isolation. Thread settings are recorded per codec.

## PBC operating points

| PBC preset | Mean bytes | Mean bpp | PSNR dB | SSIM | Outer LZMA | Actual stream mode |
| --- | --- | --- | --- | --- | --- | --- |
| pbc_compression | 3436.7 | 0.4195 | 27.86 | 0.79257 | True | lzma: 95 |
| pbc_balanced | 6917.3 | 0.8444 | 30.25 | 0.86663 | True | lzma: 95 |
| pbc_quality | 15639.0 | 1.9091 | 33.59 | 0.93045 | True | lzma: 95 |
| pbc_high_quality | 27016.0 | 3.2979 | 36.33 | 0.95590 | True | lzma: 95 |

Rate matching chooses the nearest mean bpp. A rate match is called close when the absolute gap is at most `max(0.025 bpp, 10% of PBC bpp)`. Distortion matching reports the nearest mean PSNR point (close within 1 dB) and the separate nearest mean SSIM point (close within 0.02 SSIM).

| PBC | Codec | Nearest by rate | Rate-close | Nearest by PSNR | PSNR-close | Nearest by SSIM | SSIM-close |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pbc_balanced | avif | avif_q60 (0.847 bpp, Δ +0.3%) | yes | avif_q30 (Δ +0.80 dB; ΔSSIM +0.0084) | yes | avif_q30 (ΔSSIM +0.0084; ΔPSNR +0.80 dB) | yes |
| pbc_balanced | jpeg | jpeg_q60 (0.982 bpp, Δ +16.3%) | no | jpeg_q20 (Δ -0.01 dB; ΔSSIM -0.0096) | yes | jpeg_q20 (ΔSSIM -0.0096; ΔPSNR -0.01 dB) | yes |
| pbc_balanced | jxl | jxl_d3 (0.774 bpp, Δ -8.3%) | yes | jxl_d10 (Δ -0.25 dB; ΔSSIM -0.0259) | yes | jxl_d8 (ΔSSIM -0.0040; ΔPSNR +0.56 dB) | yes |
| pbc_balanced | webp | webp_q75 (0.801 bpp, Δ -5.1%) | yes | webp_q10 (Δ -0.37 dB; ΔSSIM -0.0241) | yes | webp_q20 (ΔSSIM +0.0055; ΔPSNR +0.77 dB) | yes |
| pbc_compression | avif | avif_q40 (0.407 bpp, Δ -2.9%) | yes | avif_q10 (Δ -0.66 dB; ΔSSIM -0.0212) | yes | avif_q10 (ΔSSIM -0.0212; ΔPSNR -0.66 dB) | no |
| pbc_compression | jpeg | jpeg_q10 (0.370 bpp, Δ -11.7%) | no | jpeg_q10 (Δ -0.19 dB; ΔSSIM -0.0099) | yes | jpeg_q10 (ΔSSIM -0.0099; ΔPSNR -0.19 dB) | yes |
| pbc_compression | jxl | jxl_d6 (0.457 bpp, Δ +8.9%) | yes | jxl_d20 (Δ -0.18 dB; ΔSSIM -0.0228) | yes | jxl_d16 (ΔSSIM -0.0107; ΔPSNR +0.19 dB) | yes |
| pbc_compression | webp | webp_q30 (0.426 bpp, Δ +1.5%) | yes | webp_q1 (Δ +0.16 dB; ΔSSIM -0.0090) | yes | webp_q1 (ΔSSIM -0.0090; ΔPSNR +0.16 dB) | yes |
| pbc_high_quality | avif | avif_q95 (2.790 bpp, Δ -15.4%) | no | avif_q60 (Δ +0.32 dB; ΔSSIM -0.0012) | yes | avif_q60 (ΔSSIM -0.0012; ΔPSNR +0.32 dB) | yes |
| pbc_high_quality | jpeg | jpeg_q80 (1.430 bpp, Δ -56.6%) | no | jpeg_q80 (Δ -0.40 dB; ΔSSIM -0.0074) | yes | jpeg_q80 (ΔSSIM -0.0074; ΔPSNR -0.40 dB) | yes |
| pbc_high_quality | jxl | jxl_d1 (1.650 bpp, Δ -50.0%) | no | jxl_d2 (Δ -0.08 dB; ΔSSIM -0.0053) | yes | jxl_d2 (ΔSSIM -0.0053; ΔPSNR -0.08 dB) | yes |
| pbc_high_quality | webp | webp_q100 (3.118 bpp, Δ -5.5%) | yes | webp_q75 (Δ -1.09 dB; ΔSSIM -0.0169) | no | webp_q90 (ΔSSIM +0.0144; ΔPSNR +2.76 dB) | yes |
| pbc_quality | avif | avif_q90 (2.080 bpp, Δ +9.0%) | yes | avif_q40 (Δ -0.76 dB; ΔSSIM -0.0214) | yes | avif_q50 (ΔSSIM +0.0041; ΔPSNR +1.07 dB) | yes |
| pbc_quality | jpeg | jpeg_q80 (1.430 bpp, Δ -25.1%) | no | jpeg_q60 (Δ +0.21 dB; ΔSSIM -0.0054) | yes | jpeg_q60 (ΔSSIM -0.0054; ΔPSNR +0.21 dB) | yes |
| pbc_quality | jxl | jxl_d1 (1.650 bpp, Δ -13.6%) | no | jxl_d4 (Δ -0.25 dB; ΔSSIM -0.0165) | yes | jxl_d3 (ΔSSIM +0.0002; ΔPSNR +0.93 dB) | yes |
| pbc_quality | webp | webp_q90 (1.612 bpp, Δ -15.6%) | no | webp_q50 (Δ +0.00 dB; ΔSSIM -0.0121) | yes | webp_q60 (ΔSSIM -0.0031; ΔPSNR +0.66 dB) | yes |

## Per-setting results

| Setting | Valid | Mean bytes | Median bytes | Mean bpp | Mean PSNR dB | Mean SSIM | Enc med ms | Enc p95 ms | Dec med ms | Dec p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avif_q1 | 95 | 701.7 | 613.0 | 0.0857 | 24.98 | 0.70178 | 13.472 | 26.676 | 4.252 | 5.876 |
| avif_q5 | 95 | 833.5 | 734.0 | 0.1018 | 26.07 | 0.73657 | 16.036 | 29.984 | 4.306 | 5.812 |
| avif_q10 | 95 | 1027.8 | 863.0 | 0.1255 | 27.20 | 0.77140 | 18.087 | 36.881 | 4.392 | 5.736 |
| avif_q20 | 95 | 1579.4 | 1351.0 | 0.1928 | 29.30 | 0.83234 | 23.553 | 43.926 | 4.405 | 5.735 |
| avif_q30 | 95 | 2292.3 | 1971.0 | 0.2798 | 31.05 | 0.87501 | 28.795 | 51.888 | 4.516 | 5.791 |
| avif_q40 | 95 | 3336.5 | 2869.0 | 0.4073 | 32.83 | 0.90902 | 30.674 | 52.893 | 4.346 | 5.465 |
| avif_q50 | 95 | 4771.6 | 4261.0 | 0.5825 | 34.67 | 0.93450 | 36.056 | 61.365 | 4.468 | 5.790 |
| avif_q60 | 95 | 6937.4 | 6519.0 | 0.8468 | 36.65 | 0.95473 | 40.105 | 72.540 | 4.547 | 5.846 |
| avif_q75 | 95 | 9810.8 | 9251.0 | 1.1976 | 38.45 | 0.96784 | 38.259 | 64.897 | 4.704 | 5.937 |
| avif_q90 | 95 | 17039.1 | 16041.0 | 2.0800 | 40.82 | 0.97994 | 48.632 | 89.647 | 5.074 | 7.036 |
| avif_q95 | 95 | 22851.8 | 22115.0 | 2.7895 | 41.62 | 0.98303 | 56.792 | 101.535 | 5.406 | 7.841 |
| jpeg_q1 | 95 | 1911.0 | 1866.0 | 0.2333 | 22.32 | 0.59733 | 0.408 | 0.588 | 0.424 | 0.580 |
| jpeg_q5 | 95 | 2276.1 | 2215.0 | 0.2778 | 24.70 | 0.69089 | 0.451 | 0.602 | 0.486 | 0.662 |
| jpeg_q10 | 95 | 3034.4 | 2919.0 | 0.3704 | 27.67 | 0.78263 | 0.447 | 0.606 | 0.492 | 0.640 |
| jpeg_q20 | 95 | 4319.9 | 4200.0 | 0.5273 | 30.24 | 0.85698 | 0.463 | 0.623 | 0.526 | 0.696 |
| jpeg_q30 | 95 | 5392.4 | 5320.0 | 0.6583 | 31.58 | 0.88743 | 0.470 | 0.637 | 0.542 | 0.677 |
| jpeg_q60 | 95 | 8042.8 | 7785.0 | 0.9818 | 33.80 | 0.92508 | 0.482 | 0.651 | 0.565 | 0.726 |
| jpeg_q80 | 95 | 11717.2 | 11461.0 | 1.4303 | 35.93 | 0.94852 | 0.322 | 0.620 | 0.351 | 0.706 |
| jxl_d25 | 95 | 1276.1 | 1158.0 | 0.1558 | 27.10 | 0.74782 | 154.718 | 178.656 | 39.752 | 51.808 |
| jxl_d20 | 95 | 1522.4 | 1340.0 | 0.1858 | 27.69 | 0.76979 | 158.981 | 185.098 | 41.203 | 58.630 |
| jxl_d16 | 95 | 1721.2 | 1624.0 | 0.2101 | 28.05 | 0.78184 | 80.725 | 109.383 | 42.914 | 64.612 |
| jxl_d12 | 95 | 2183.6 | 2072.0 | 0.2666 | 29.30 | 0.82111 | 80.470 | 110.822 | 44.159 | 64.946 |
| jxl_d10 | 95 | 2530.8 | 2362.0 | 0.3089 | 30.00 | 0.84072 | 77.029 | 109.383 | 40.803 | 62.579 |
| jxl_d8 | 95 | 3005.4 | 2850.0 | 0.3669 | 30.81 | 0.86262 | 74.789 | 104.789 | 41.148 | 65.736 |
| jxl_d6 | 95 | 3742.5 | 3570.0 | 0.4568 | 31.84 | 0.88629 | 75.310 | 97.464 | 40.225 | 54.454 |
| jxl_d4 | 95 | 5134.9 | 4986.0 | 0.6268 | 33.34 | 0.91397 | 83.562 | 114.206 | 42.443 | 63.798 |
| jxl_d3 | 95 | 6344.3 | 6196.0 | 0.7745 | 34.52 | 0.93067 | 75.379 | 101.382 | 41.773 | 58.161 |
| jxl_d2 | 95 | 8551.5 | 8303.0 | 1.0439 | 36.25 | 0.95061 | 79.298 | 99.633 | 45.397 | 64.523 |
| jxl_d1 | 95 | 13515.7 | 13337.0 | 1.6499 | 39.29 | 0.97188 | 80.693 | 99.552 | 45.900 | 57.105 |
| pbc_compression | 95 | 3436.7 | 2975.0 | 0.4195 | 27.86 | 0.79257 | 55.447 | 108.765 | 5.569 | 10.603 |
| pbc_balanced | 95 | 6917.3 | 5198.0 | 0.8444 | 30.25 | 0.86663 | 66.637 | 133.133 | 8.170 | 15.874 |
| pbc_quality | 95 | 15639.0 | 13752.0 | 1.9091 | 33.59 | 0.93045 | 80.345 | 164.560 | 13.758 | 25.938 |
| pbc_high_quality | 95 | 27016.0 | 26599.0 | 3.2979 | 36.33 | 0.95590 | 107.766 | 158.079 | 24.508 | 46.986 |
| png_lossless | 95 | 99180.2 | 99229.0 | 12.1070 | ∞ | 1.00000 | 7.266 | 10.430 | 1.801 | 2.675 |
| webp_q1 | 95 | 1319.8 | 1162.0 | 0.1611 | 28.02 | 0.78358 | 9.683 | 15.075 | 1.053 | 1.692 |
| webp_q5 | 95 | 1762.7 | 1590.0 | 0.2152 | 29.11 | 0.81909 | 10.078 | 15.895 | 1.040 | 1.707 |
| webp_q10 | 95 | 2150.6 | 1924.0 | 0.2625 | 29.88 | 0.84248 | 10.867 | 17.563 | 1.105 | 1.772 |
| webp_q20 | 95 | 2823.4 | 2516.0 | 0.3447 | 31.02 | 0.87216 | 12.470 | 19.742 | 1.363 | 2.295 |
| webp_q30 | 95 | 3486.6 | 3112.0 | 0.4256 | 31.98 | 0.89274 | 12.975 | 20.510 | 1.382 | 2.031 |
| webp_q40 | 95 | 4141.6 | 3730.0 | 0.5056 | 32.87 | 0.90744 | 13.998 | 22.434 | 1.365 | 2.203 |
| webp_q50 | 95 | 4781.0 | 4286.0 | 0.5836 | 33.60 | 0.91839 | 14.308 | 22.389 | 1.438 | 2.124 |
| webp_q60 | 95 | 5438.6 | 4876.0 | 0.6639 | 34.25 | 0.92731 | 15.170 | 24.721 | 1.550 | 2.329 |
| webp_q75 | 95 | 6564.1 | 5872.0 | 0.8013 | 35.24 | 0.93902 | 16.099 | 24.572 | 1.543 | 2.435 |
| webp_q90 | 95 | 13206.2 | 12328.0 | 1.6121 | 39.09 | 0.97030 | 20.435 | 32.078 | 1.743 | 2.694 |
| webp_q100 | 95 | 25539.8 | 24858.0 | 3.1177 | 41.65 | 0.98198 | 27.449 | 39.354 | 2.166 | 4.094 |

Machine-readable per-sample results are in `sample_results.csv`; `summary.json` and `summary.csv` contain aggregates and operating-point matches. The rate-distortion figure, latency figures, and representative grid are `rate_distortion.png`, `encode_latency.png`, `decode_latency.png`, and `visual_comparison_grid.png`.

## Findings and earlier CIFAR comparison

- PBC ranges from 0.420 bpp / 27.86 dB (`pbc_compression`) to 3.298 bpp / 36.33 dB (`pbc_high_quality`).
- At PBC compression rate, WebP q30, AVIF q40, and JXL distance 6 are close by the stated rate threshold; JPEG q10 is the nearest JPEG point but falls just outside it. Their measured PSNR is higher than PBC compression at similar rates.
- PBC balanced has close rate matches in WebP q75, AVIF q60, and JXL distance 3. JPEG q60 is nearest but is not close by the threshold.
- PBC quality has an AVIF q90 rate match; PBC high_quality has a WebP q100 rate match. No other tested conventional setting is close by rate to those presets.
- Distortion-matched points generally use less rate than PBC, most clearly around PBC high_quality. PBC median encode latency spans 55.4–107.8 ms and median decode spans 5.6–24.5 ms. JXL uses `cjxl`/`djxl` CLI processes, so its measured latency includes process startup and file staging and is not an in-process libjxl comparison.

Yes, the earlier 32×32 result was misleading as a general operating-point match. On CIFAR-10, PBC quality was about 7.34 bpp and JPEG q80 about 7.50 bpp, with PSNR 31.34 vs 30.73 dB. Here, PBC quality is 1.91 bpp / 33.59 dB, while JPEG q80 is 1.43 bpp / 35.93 dB (25.1% fewer bits for JPEG). JPEG q60 reaches 33.80 dB at 0.98 bpp, about 48.6% fewer bits than PBC quality. The dataset also changed from CIFAR-10 to local natural photos, so these measurements show that the old match does not transfer; they do not isolate resolution as the only cause.

## Phase 2 operating points

Primary shortlist: carry PBC `compression` and `balanced`; use every conventional point marked rate-close as the rate-control set. Use the nearest PSNR/SSIM points as a separate distortion-control set, not as bitrate matches. Keep PBC `quality` and `high_quality` as secondary points because fewer codecs have close rate matches there. Include JPEG q1, q5, q10 in the low-rate neural comparison; q10 is the nearest JPEG distortion match to PBC compression, while q1/q5 remain useful lower-rate controls. Do not label any nearest-but-not-close condition as matched. Keep PNG as the lossless ceiling/reference.

- **pbc_balanced** (0.844 bpp): rate — avif avif_q60 (0.847 bpp, +0.3%; close); jpeg jpeg_q60 (0.982 bpp, +16.3%; nearest only); jxl jxl_d3 (0.774 bpp, -8.3%; close); webp webp_q75 (0.801 bpp, -5.1%; close). PSNR — avif avif_q30 (+0.80 dB; close); jpeg jpeg_q20 (-0.01 dB; close); jxl jxl_d10 (-0.25 dB; close); webp webp_q10 (-0.37 dB; close). SSIM — avif avif_q30 (+0.0084; close); jpeg jpeg_q20 (-0.0096; close); jxl jxl_d8 (-0.0040; close); webp webp_q20 (+0.0055; close).
- **pbc_compression** (0.420 bpp): rate — avif avif_q40 (0.407 bpp, -2.9%; close); jpeg jpeg_q10 (0.370 bpp, -11.7%; nearest only); jxl jxl_d6 (0.457 bpp, +8.9%; close); webp webp_q30 (0.426 bpp, +1.5%; close). PSNR — avif avif_q10 (-0.66 dB; close); jpeg jpeg_q10 (-0.19 dB; close); jxl jxl_d20 (-0.18 dB; close); webp webp_q1 (+0.16 dB; close). SSIM — avif avif_q10 (-0.0212; nearest only); jpeg jpeg_q10 (-0.0099; close); jxl jxl_d16 (-0.0107; close); webp webp_q1 (-0.0090; close).
- **pbc_high_quality** (3.298 bpp): rate — avif avif_q95 (2.790 bpp, -15.4%; nearest only); jpeg jpeg_q80 (1.430 bpp, -56.6%; nearest only); jxl jxl_d1 (1.650 bpp, -50.0%; nearest only); webp webp_q100 (3.118 bpp, -5.5%; close). PSNR — avif avif_q60 (+0.32 dB; close); jpeg jpeg_q80 (-0.40 dB; close); jxl jxl_d2 (-0.08 dB; close); webp webp_q75 (-1.09 dB; nearest only). SSIM — avif avif_q60 (-0.0012; close); jpeg jpeg_q80 (-0.0074; close); jxl jxl_d2 (-0.0053; close); webp webp_q90 (+0.0144; close).
- **pbc_quality** (1.909 bpp): rate — avif avif_q90 (2.080 bpp, +9.0%; close); jpeg jpeg_q80 (1.430 bpp, -25.1%; nearest only); jxl jxl_d1 (1.650 bpp, -13.6%; nearest only); webp webp_q90 (1.612 bpp, -15.6%; nearest only). PSNR — avif avif_q40 (-0.76 dB; close); jpeg jpeg_q60 (+0.21 dB; close); jxl jxl_d4 (-0.25 dB; close); webp webp_q50 (+0.00 dB; close). SSIM — avif avif_q50 (+0.0041; close); jpeg jpeg_q60 (-0.0054; close); jxl jxl_d3 (+0.0002; close); webp webp_q60 (-0.0031; close).

This run is the full calibration run.
