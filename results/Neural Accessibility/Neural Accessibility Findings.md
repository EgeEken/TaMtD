# Neural Accessibility Findings

## Dataset and protocol

- Imagenette 320px was deterministically center-cropped/resized to 256×256 RGB using the Phase 1 EXIF transpose and Pillow BICUBIC preprocessing.
- Split seed: 42. Train: 8,522; validation: 947; official validation held out as test: 3,925. No test image was used for training or checkpoint selection.
- One seed was run. Checkpoints were selected by validation accuracy; each trained representation model used one final test evaluation.
- Model input: raw bytes use a shared byte CNN with no sequence cap; PBC P2/P3 use 64 patch tokens with 8×8 pooled masks/grid summaries and an MLP content encoder; JPEG DCT uses component grids in quantized coefficient space without IDCT; RGB uses one random-initialized width-32 ResNet-18-style classifier.
- Full per-model results and histories are in `model_results.json`; exact settings are in `config.json` and `training_settings.json`.

| Split | Samples |
| --- | ---: |
| Train | 8,522 |
| Validation | 947 |
| Official validation test | 3,925 |

## RGB reference and overfit checks

- Original prepared RGB test accuracy: **0.822**.
- PBC decoded RGB: compression 0.733, balanced 0.805, quality 0.818.
- JPEG decoded RGB: q10 0.811, q20 0.816, q60 0.823, q80 0.824.
- Tiny-set checks used a fixed stratified 128-sample train subset:
- byte_cnn: 0.977 training accuracy in 15 epochs (passed).
- pbc_instruction_set: 0.922 training accuracy in 15 epochs (passed).
- jpeg_dct_cnn: 0.977 training accuracy in 15 epochs (passed).
- rgb_resnet18_style: 0.906 training accuracy in 19 epochs (passed).

## PBC LZMA-to-STORE control

The forced-LZMA and STORE streams encode the same PBC body. The model sees the wrapper bytes for this control.

- pbc_compression: forced LZMA **0.189**, STORE **0.355** (+16.64 percentage points) after **0.1250 ms** to remove LZMA (2.6% of that preset's forced-LZMA full decode).
- pbc_balanced: forced LZMA **0.181**, STORE **0.395** (+21.40 percentage points) after **0.2359 ms** to remove LZMA (3.1% of that preset's forced-LZMA full decode).
- pbc_quality: forced LZMA **0.158**, STORE **0.406** (+24.76 percentage points) after **0.6286 ms** to remove LZMA (5.1% of that preset's forced-LZMA full decode).

## PBC parsed fields to canonical IR

- pbc_compression: parsed **0.502**, canonical **0.575** (+7.26 percentage points); canonical construction adds **0.2553 ms** in the Phase 2 timing.
- pbc_balanced: parsed **0.520**, canonical **0.568** (+4.82 percentage points); canonical construction adds **0.3504 ms** in the Phase 2 timing.
- pbc_quality: parsed **0.512**, canonical **0.568** (+5.61 percentage points); canonical construction adds **0.4653 ms** in the Phase 2 timing.

## JPEG file bytes to DCT IR

J1 cost includes file-to-coefficient extraction plus canonical DCT IR array construction. The q20 supplement measured only q20 on the original Phase 1 95-image set; Phase 2's full matrix was not repeated.

- jpeg_q10: file bytes **0.356**, DCT IR **0.759** (+40.33 percentage points); file-to-DCT IR is **0.4285 ms** versus **0.3973 ms** full Pillow decode (107.9%).
- jpeg_q20: file bytes **0.367**, DCT IR **0.747** (+38.01 percentage points); file-to-DCT IR is **0.4461 ms** versus **0.2029 ms** full Pillow decode (219.9%).
- jpeg_q60: file bytes **0.323**, DCT IR **0.765** (+44.18 percentage points); file-to-DCT IR is **0.4741 ms** versus **0.4422 ms** full Pillow decode (107.2%).
- jpeg_q80: file bytes **0.300**, DCT IR **0.766** (+46.62 percentage points); file-to-DCT IR is **0.5181 ms** versus **0.4741 ms** full Pillow decode (109.3%).

## Intermediate points and decode cost

Semantic accessibility is the project diagnostic `(model accuracy − 0.1) / (decoded RGB reference accuracy − 0.1)`. Values are not clamped; values below zero or above one are retained. The common RGB model has accessibility 1 by construction when its decoded reference is above chance.

- pbc_compression: best intermediate below full decode is `pbc_canonical_ir` at **0.575** (0.750 semantic accessibility) and **2.1712 ms** (45.6% of full decode).
- pbc_balanced: best intermediate below full decode is `pbc_canonical_ir` at **0.568** (0.664 semantic accessibility) and **4.3637 ms** (57.1% of full decode).
- pbc_quality: best intermediate below full decode is `pbc_canonical_ir` at **0.568** (0.652 semantic accessibility) and **9.3601 ms** (75.5% of full decode).
- jpeg_q10: best intermediate below full decode is `jpeg_file_bytes` at **0.356** (0.360 semantic accessibility) and **0.0000 ms** (0.0% of full decode).
- jpeg_q20: best intermediate below full decode is `jpeg_file_bytes` at **0.367** (0.373 semantic accessibility) and **0.0000 ms** (0.0% of full decode).
- jpeg_q60: best intermediate below full decode is `jpeg_file_bytes` at **0.323** (0.309 semantic accessibility) and **0.0000 ms** (0.0% of full decode).
- jpeg_q80: best intermediate below full decode is `jpeg_file_bytes` at **0.300** (0.276 semantic accessibility) and **0.0000 ms** (0.0% of full decode).

The plots show absolute preprocessing milliseconds and fraction of each implementation's full decode cost separately. PBC uses Python/NumPy/Numba research code; JPEG uses pyjpegio/libjpeg-turbo and Pillow/libjpeg-turbo. The ratios are implementation-level measurements from this machine, not algorithm-only comparisons. Phase 2 timing samples are the original 95 natural-image calibration sources; q20 used a targeted extension over those same sources. Classification accuracy comes from Imagenette test.

## Training feasibility

- Model parameter counts range from 340,266 to 2,801,450 across the measured classifiers.
- Mean training throughput across representation models: 1548.6 examples/s.
- Peak VRAM observed: 910 MB.
- Full row-level training metrics are in `training_feasibility.csv` and `training_feasibility.png`.

## Limitations

- This is a one-seed frontier; individual accuracy differences can reflect seed variance.
- No image augmentation was used. The RGB reference is a modest random-initialized model and is a measured ceiling for this run, not a task-level upper bound.
- PBC patch content is pooled to fixed-length 8×8/64-feature vectors before the patch-set model. It preserves the parsed/canonical arrays' structure but does not feed every variable-size grid cell as a separate token.
- JPEG full decode and coefficient access use different supported libraries, as in Phase 2. The frontier combines their measured stage costs and should be read as an implementation-level systems comparison.
- No claims are made that a classifier decodes images or learned a codec algorithm.

## Artifacts

- `summary.json`, `model_results.json`, `critical_comparisons.json`
- `frontier.csv`, `per_condition_metrics.csv`, `training_feasibility.csv`
- `frontier_absolute_latency.png`, `frontier_fraction_full_decode.png`, `semantic_accessibility.png`, `training_feasibility.png`
- `jpeg_q20_timing_extension.json`
