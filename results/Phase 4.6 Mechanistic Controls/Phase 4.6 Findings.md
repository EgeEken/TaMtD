# Phase 4.6 Findings

## Scope and protocol

Phase 4.6 tested only two questions: whether the Phase 4.5B low-rate byte ranking survives a different byte-model family, and which JPEG DCT frequency/component groups support the existing classifier result. The byte matrix used JPEG q10, WebP q30, AVIF q40, and JXL d6 with seeds 42, 123, and 777. DCT ablations used q10/q60 and the same seeds; full-DCT rows reuse prior results. All new training ran in separate CUDA child processes. No codec, image, resolution, or quality-point search was added.

`BytePatchTransformer` has 456,138 trainable parameters: 32-dimensional byte embeddings; 32-byte non-overlapping patches; a linear 1,024-to-96 patch projection; dynamic sinusoidal positions; four 4-head Transformer encoder layers with 256-wide feed-forward blocks and 0.1 dropout; padding-aware masks; adaptive average sequence reduction to at most 256 attention tokens; masked mean pooling; and a 10-class head. Every byte contributes to a patch before reduction; streams were not truncated. The 128-sample JPEG q10 overfit gate reached 93.0% in 12 epochs (target 90%).

Byte training used the shared Phase 4.5A length-bucketed policy: 1,048,576 padded bytes/tokens maximum, 32-sample physical-batch cap, 32-sample gradient-accumulation target, bucket multiplier 50, zero workers, no pinned memory, CUDA autocast, and no augmentation. Peak and minimum resource data are recorded per row. Windows process RSS was unavailable; host available memory and system commit were recorded instead.

## Raw-byte probe results

| Codec | ByteStreamCNN test accuracy | BytePatchTransformer test accuracy | CNN U-probe bits | Transformer U-probe bits | Transformer − CNN accuracy |
|---|---:|---:|---:|---:|---:|
| jpeg_q10 (n=3/3) | 0.366 ± 0.011 | 0.346 ± 0.004 | 0.760 ± 0.046 | 0.664 ± 0.033 | -0.020 |
| webp_q30 (n=1/3) | 0.244 ± 0.000 | 0.224 ± 0.016 | 0.316 ± 0.000 | 0.211 ± 0.158 | -0.019 |
| avif_q40 (n=1/3) | 0.184 ± 0.000 | 0.194 ± 0.009 | 0.187 ± 0.000 | 0.135 ± 0.131 | +0.010 |
| jxl_d6 (n=1/3) | 0.204 ± 0.000 | 0.214 ± 0.012 | 0.163 ± 0.000 | 0.161 ± 0.131 | +0.010 |

ByteStreamCNN has three JPEG q10 seeds but only seed 42 for WebP, AVIF, and JXL. The accuracy order matches exactly in both families (JPEG > WebP > JXL > AVIF; Spearman 1.00, Kendall 1.00). The U-probe order swaps AVIF and JXL (Spearman 0.80, Kendall 0.67). These four-codec rank correlations are descriptive, not inferential.

JPEG q10 minus the other codec test accuracies under BytePatchTransformer:

| Comparison | Per-seed differences | Mean ± sample SD |
|---|---|---:|
| JPEG q10 − webp_q30 | +0.107, +0.137, +0.121 | +0.122 ± 0.015 |
| JPEG q10 − avif_q40 | +0.161, +0.159, +0.137 | +0.152 ± 0.014 |
| JPEG q10 − jxl_d6 | +0.139, +0.144, +0.114 | +0.132 ± 0.016 |

**Probe dependence:** The JPEG-leading ordering persists with broadly comparable gaps in the attention probe; the result is not unique to ByteStreamCNN, within the limits of these four points and uneven legacy seed counts.

## JPEG DCT ablations

The fixed `JPEGDCTNet` received deterministic masks: DC-only keeps channel `u*8+v=0` in Y/Cb/Cr; low-4×4 keeps `u<4 and v<4` in all components; Y-only keeps all Y and zeros Cb/Cr; quantization metadata and tensor shapes are unchanged.

| Quality | Ablation | Test accuracy | Validation accuracy | CE (nats) | U-probe bits | U relative to full DCT | Accuracy difference vs RGB |
|---|---|---:|---:|---:|---:|---:|---:|
| jpeg_q10 | full | 0.762 ± 0.003 | 0.788 ± 0.010 | 0.776 ± 0.011 | 2.204 ± 0.016 | 1.000 ± 0.000 | -0.045 ± 0.007 |
| jpeg_q10 | dc_only | 0.755 ± 0.026 | 0.778 ± 0.011 | 0.758 ± 0.059 | 2.230 ± 0.085 | 1.012 ± 0.045 | -0.052 ± 0.025 |
| jpeg_q10 | low4x4 | 0.763 ± 0.010 | 0.782 ± 0.003 | 0.768 ± 0.024 | 2.215 ± 0.035 | 1.005 ± 0.009 | -0.044 ± 0.009 |
| jpeg_q10 | y_only | 0.694 ± 0.002 | 0.710 ± 0.008 | 0.997 ± 0.031 | 1.886 ± 0.045 | 0.856 ± 0.021 | -0.113 ± 0.006 |
| jpeg_q60 | full | 0.764 ± 0.002 | 0.782 ± 0.013 | 0.768 ± 0.014 | 2.215 ± 0.021 | 1.000 ± 0.000 | -0.055 ± 0.003 |
| jpeg_q60 | dc_only | 0.769 ± 0.003 | 0.791 ± 0.006 | 0.728 ± 0.015 | 2.273 ± 0.021 | 1.026 ± 0.010 | -0.050 ± 0.002 |
| jpeg_q60 | low4x4 | 0.763 ± 0.006 | 0.786 ± 0.008 | 0.763 ± 0.022 | 2.223 ± 0.032 | 1.004 ± 0.014 | -0.057 ± 0.003 |
| jpeg_q60 | y_only | 0.699 ± 0.005 | 0.723 ± 0.010 | 0.969 ± 0.035 | 1.925 ± 0.050 | 0.869 ± 0.016 | -0.121 ± 0.005 |

Mean test accuracy by quality and ablation:

- **jpeg_q10:** DC-only 0.755, low 4×4 0.763, full DCT 0.762, Y-only 0.694; low 4×4 − DC = +0.008, full − low 4×4 = -0.001, full − Y-only = +0.068 test-accuracy points.
- **jpeg_q60:** DC-only 0.769, low 4×4 0.763, full DCT 0.764, Y-only 0.699; low 4×4 − DC = -0.007, full − low 4×4 = +0.002, full − Y-only = +0.065 test-accuracy points.

| Quality | Contrast | Test-accuracy difference | U-probe difference (bits) |
|---|---|---:|---:|
| jpeg_q10 | dc_only_minus_full | -0.007 ± 0.026 | +0.026 ± 0.099 |
| jpeg_q10 | low4x4_minus_dc_only | +0.008 ± 0.033 | -0.015 ± 0.111 |
| jpeg_q10 | full_minus_low4x4 | -0.001 ± 0.012 | -0.011 ± 0.020 |
| jpeg_q10 | full_minus_y_only | +0.068 ± 0.001 | +0.319 ± 0.048 |
| jpeg_q60 | dc_only_minus_full | +0.005 ± 0.002 | +0.058 ± 0.023 |
| jpeg_q60 | low4x4_minus_dc_only | -0.007 ± 0.004 | -0.050 ± 0.011 |
| jpeg_q60 | full_minus_low4x4 | +0.002 ± 0.006 | -0.008 ± 0.032 |
| jpeg_q60 | full_minus_y_only | +0.065 ± 0.003 | +0.289 ± 0.034 |

Coefficient position, absolute-magnitude, and squared-energy fractions by Y/Cb/Cr are in `jpeg_dct_retained_energy.csv`. DC-only retains 1/64 (1.56%) of coefficient positions and about 81% of Y squared energy at both qualities, plus roughly 90% of chroma squared energy. Low-4×4 retains 16/64 (25%) of positions and over 99% of squared energy in each component. These are coefficient-energy statistics, not semantic-information measures.

## Interpretation

- **Representation observation:** BytePatchTransformer and DCT ablation values are listed above; the 95-image Phase 4.5B staged CNN baselines have the seed counts shown in the table.
- **Probe dependence:** The JPEG-leading ordering persists with broadly comparable gaps in the attention probe; the result is not unique to ByteStreamCNN, within the limits of these four points and uneven legacy seed counts.
- **Information mechanism:** DC-only is within 0.7 accuracy points of full DCT at q10 and 0.5 points above full DCT at q60. Adding low-4×4 frequencies changes mean accuracy by +0.8 points at q10 and −0.7 at q60; full DCT versus low-4×4 changes it by −0.1 and +0.2 points. Removing Cb/Cr lowers accuracy by 6.8 points at q10 and 6.5 at q60, and lowers U-probe by 0.319 and 0.289 bits. The classifier relies strongly on coarse block-level YCbCr appearance, with a material chroma contribution; frequencies above the low-4×4 grid add no consistent measured benefit. The q10 and q60 patterns are similar.
- **Systems relevance:** The Phase 4.5B timing result remains: q60 DCT extraction was about 0.5165 ms versus 0.4546 ms for RGB decode, with RGB also more accurate. These ablations measured model sensitivity, not an early-exit decoder speedup. Any lighter restricted-DCT decoder remains a future engineering hypothesis.

- **Resource safety:** All 30 new conditions ran on CUDA. Minimum free disk was 161.7 GB; minimum host-available memory was 10.8 GiB; minimum free VRAM was 5.48 GiB; peak system commit was 81.2%. These stayed above/below the configured guards. Windows process RSS was unavailable.

## Image-track recommendation

**C. Stop broad image research and shift the project primarily to video.** The JPEG byte advantage reproduces across two model families, but BytePatchTransformer accuracy is about 35% on JPEG q10 and 19–22% on the other selected formats. The DCT result is largely explained by coarse block-level coefficients, with a material chroma contribution; high-frequency AC coefficients add no consistent gain. The measured q60 DCT path was slower than RGB decode and less accurate, so the ablations establish model sensitivity but no systems advantage. Phase 4.6 answers its questions without justifying another broad image sweep.
