# TaMtD research archive

This is a chronological index of the completed image track through Phase 4.8. The linked findings files and the [consolidated research report](ta-mtd-research-report.md) remain the authoritative sources for complete tables, uncertainty estimates, and per-condition values. This archive records the questions, designs, durable conclusions, and limits needed to navigate that record.

## Early Testing — CIFAR-10 pipeline exploration

**Question and design.** Initial 32×32 CIFAR-10 experiments tested byte statistics, PBC byte models, coarse reconstruction, and whether secondary entropy coding changed accessibility while preserving pixels. These were pipeline probes, not the headline natural-image benchmark.

**Findings.** Histogram/statistical baselines often matched or beat byte CNNs on PBC streams, indicating that some apparent byte-model signal was order-insensitive. On the tested seeds, PBC STORE byte accuracy was roughly 22.5–25.9%; forced LZMA wrapping reduced accuracy despite pixel-identical reconstructions. In a coarse 8×8 reconstruction probe, ordered inputs reduced relative MSE by 24.1% for PBC quality and 26.9% for JPEG q80; shuffled inputs retained 8.5% and approximately 0% respectively. Reconstruction quality was insufficient to claim useful decoding.

**Conclusion and limits.** These results established controls and motivated the larger study. CIFAR resolution and the exploratory PBC source state preclude general codec-quality or rate-distortion claims. See [Early Testing findings](../../results/Early%20Testing/Early%20Testing%20Findings.md).

## Phase 1 — codec characterization

**Question and design.** Characterize codec rate, distortion, validity, and encode/decode timing on 95 deterministic 256×256 natural-image views. The sweep covered PBC presets; JPEG q1–q80; WebP, AVIF, JXL, and lossless PNG. Timings used repeated measurements and retained implementation/version caveats.

**Findings.** All 95 images decoded successfully. PBC compression, balanced, quality, and high_quality occupied approximately 0.4195, 0.8444, 1.9091, and 3.2979 bpp, with PSNR 27.86, 30.25, 33.59, and 36.33 dB. PBC compression was close in rate to WebP q30 (0.426 bpp), AVIF q40 (0.407 bpp), and JXL d6 (0.457 bpp). Similar quality labels were not treated as equivalent.

**Conclusion and limits.** The sweep selected measured operating points for later work and established the 256×256 protocol. Cross-library absolute latency includes implementation quality as well as codec work. See [Phase 1 findings](../../results/Codec%20Benchmark/Codec%20Benchmark%20Findings.md), [summary CSV](../../results/Codec%20Benchmark/summary.csv), and [summary JSON](../../results/Codec%20Benchmark/summary.json).

## Phase 2 — staged decoding

**Question and design.** Measure PBC unpacking, parsing, canonical IR construction, JPEG coefficient extraction, and full RGB decode on the same 95 images.

**Findings.** For PBC compression, LZMA unpacking took about 0.125 ms (2.6% of full decode); parsing took about 1.79 ms, with canonical IR adding about 0.255 ms. Canonical IR construction was about 44.8% of full decode; corresponding balanced and quality stages were about 57% and 73%. JPEG q10 coefficient extraction took about 0.338 ms versus 0.374 ms for full decode; q60 took 0.403 ms versus 0.442 ms. PBC STORE and LZMA variants were pixel-identical and canonical IR could be reserialized.

**Negative result and limit.** The available JPEG binding did not expose a separable coefficient-to-RGB stage, so no such timing was inferred. See [Staged Decoding findings](../../results/Staged%20Decoding/Staged%20Decoding%20Findings.md), [stage summary](../../results/Staged%20Decoding/stage_summary.csv), and [summary](../../results/Staged%20Decoding/summary.csv).

## Phase 3 — neural accessibility

**Question and design.** On 13,394 Imagenette 256×256 images (8,522/947/3,925 train/validation/test; seed 42), compare raw byte models, PBC canonical IR, JPEG DCT, and decoded RGB models. Tested families included ByteStreamCNN, PBCInstructionNet, JPEGDCTNet, and RGBResNet18.

**Findings.** PBC canonical-IR accuracy was approximately 56.8–57.5% across compression/balanced/quality; decoded RGB references ranged from 73.4% to 81.8%. JPEG q10–q80 DCT accuracy was approximately 74.7–76.6%, while raw JPEG byte accuracy was approximately 30.0–36.7%. DCT extraction was often as costly as or more costly than full RGB decoding.

**Conclusion and limits.** Structured codec representations expose useful information, but the one-seed, separately trained model families do not establish intrinsic representation superiority. See [Neural Accessibility findings](../../results/Neural%20Accessibility/Neural%20Accessibility%20Findings.md) and [summary JSON](../../results/Neural%20Accessibility/summary.json).

## Phase 4 — selected multi-seed confirmation

**Question and design.** Repeat selected PBC STORE/LZMA, JPEG byte/DCT, and RGB comparisons using seeds 42, 123, and 777; keep new WebP/AVIF/JXL byte conditions exploratory at seed 42.

**Findings.** STORE over LZMA byte accuracy gains were 16.09±0.69, 20.60±1.67, and 22.79±3.13 percentage points for PBC compression, balanced, and quality; all three seeds were positive. JPEG DCT over bytes gained 39.58±0.81 points at q10 and 44.43±0.67 at q60. DCT accuracies were about 76.21% and 76.44%, compared with RGB references around 80.7% and 82%.

**Conclusion and limits.** Secondary entropy coding materially changes byte accessibility without changing decoded pixels. Standard-codec byte comparisons remain one-seed exploratory. See [Phase 4 findings](../../results/Phase%204%20Confirmation/Phase%204%20Findings.md), [multi-seed summary](../../results/Phase%204%20Confirmation/multi_seed_summary.json), and [frontier CSV](../../results/Phase%204%20Confirmation/multi_seed_frontier.csv).

## Phase 4.5A — batching diagnostic

**Question and design.** An engineering-only 512-sample, one-epoch diagnostic evaluated token-capped byte batching for resource control; it was not a headline benchmark.

**Finding.** A 1,048,576-token cap reduced padded work by 40.8% and reached 11.38 examples/s under the diagnostic resource guards. See [Byte Batching Diagnostic](../../results/Phase%204.5A%20Diagnostics/Byte%20Batching%20Diagnostic.md).

## Phase 4.5B — post-hoc analysis

**Question and design.** Analyze retained per-image measurements and existing checkpoints; no classifier training was required for the main analysis.

**Findings.** Across 14 stream-form points, log bitrate plus PSNR explained raw byte accuracy with in-sample R²=.332 and leave-one-out RMSE=.0904. Adding codec family raised R² to .857 and reduced RMSE to .0556. For U utility, R² changed from .302 to .839 and RMSE from .271 to .180. The analysis also found that decoded-RGB endpoints cluster around .81 for most conditions.

**Conclusion and limit.** These are descriptive, small-sample fits with correlated predictors and repeated PBC decoded content; they are not causal codec effects. See [Phase 4.5B findings](../../results/Phase%204.5B%20Analysis/Phase%204.5B%20Findings.md), [regression metadata](../../results/Phase%204.5B%20Analysis/rd_accessibility_model.json), and the linked per-image and paired CSVs.

## Phase 4.6 — mechanistic controls

**Question and design.** Check whether byte-model architecture changes codec rankings and ablate JPEG DCT channels/frequencies.

**Findings.** A 456,138-parameter BytePatchTransformer reproduced the JPEG-leading rank across four selected settings, but seed coverage was uneven (only JPEG had three seeds). JPEG DC-only and low-4×4 inputs remained within roughly 0.7 points of full DCT accuracy for q10/q60. Y-only cost about 6.5–6.8 points; chroma carried about 0.289–0.319 bits of U utility. High-frequency AC did not show a consistent gain.

**Conclusion and limit.** Frequency/channel ablations identify accessible signal, not a demonstrated system speedup. See [Phase 4.6 findings](../../results/Phase%204.6%20Mechanistic%20Controls/Phase%204.6%20Findings.md), [summary JSON](../../results/Phase%204.6%20Mechanistic%20Controls/summary.json), and the ablation CSVs.

## Phase 4.7 — standard-codec stages

**Question and design.** Instrument AVIF and WebP before full reconstruction and probe their structured features; assess JXL event availability. Pinned versions and patches are in [`patches/phase47`](../../patches/phase47/) and build scripts under `scripts/`.

**Findings.** The three-seed structured probes reached 70.3% and 71.6% of RGB U utility for AVIF q40/q60 (relative accuracy 68.0%/70.3%). AVIF capture plus adapters took 1.376/2.090 ms versus 1.434/2.204 ms for native YUV and 2.471/3.154 ms for RGB, leaving little margin over YUV decode. WebP q30/q75 reached 68.4%/71.5% of RGB U utility (relative accuracy 66.2%/70.1%), but capture plus adapters took 7.748/8.273 ms versus RGB at 0.496/0.721 ms (15.6×/11.5× slower). JXL exposed no usable progressive event in the tested integration. These timings include instrumentation and adapters and describe the tested implementation paths.

**Invalidated result.** The initial semantic-exposure AUC used inconsistent endpoint sets and was withdrawn. The corrected same-endpoint analysis is recorded in Phase 4.8; it supports only two AVIF conditions, q40 at 0.377±0.004 and q60 at 0.290±0.002 across three seeds. No cross-codec AUC conclusion follows. See [Phase 4.7 findings](../../results/Phase%204.7%20Standard%20Codec%20Stages/Phase%204.7%20Findings.md), [stage audit](../../results/Phase%204.7%20Standard%20Codec%20Stages/decoder_stage_audit.md), and the [Phase 4.8 correction](../../results/Phase%204.8%20Extreme%20Compression/phase47_auc_supersession.csv).

## Phase 4.8 — extreme compression and analysis corrections

**Question and design.** Extend low-rate JPEG, WebP, and AVIF points; evaluate selected raw-byte and RGB conditions, validate the q1 DCT path, and correct Phase 4.7 analysis on consistent endpoints. Calibration used 200 stratified images with three timing repetitions; extraction/decoding succeeded with zero truncation.

**Findings.** The 16 JPEG/WebP/AVIF operating points show a pooled inverse association between PSNR residual from the fitted log-rate curve and seed-42 raw-byte accuracy (r=−.915; U also r=−.915). The association persists under alternate curve choices and adding two JXL points, but after codec-family means are removed it weakens to r=−.435 (U −.399) under the linear residual and is much smaller under the quadratic residual (r=−.122 and −.021). These points cover only three families and do not establish a universal or causal law.

At low rates, JPEG q1, WebP q5–q20, and AVIF q5–q20 provide measured byte accessibility and paired quality comparisons. The q1 DCT model reached 71.7% versus 68.8% for an RGB reference, but these are differently trained probes and extraction took 1.208 ms versus 0.264 ms for RGB decode; this is not evidence of intrinsic JPEG-DCT superiority. The corrected AUC finding from Phase 4.7 is limited to the two AVIF points above.

At comparable measured rate and distortion, raw byte accessibility still differed by codec. WebP q5 (0.299 bpp, 27.60 dB) and AVIF q20 (0.269 bpp, 27.88 dB) differed by 6.09 percentage points in raw accuracy. Near 0.6 bpp, JPEG q20, WebP q30, and AVIF q40 had decoded-RGB accuracy of 81.61%, 81.28%, and 81.38%, while raw-byte accuracy was 36.71%, 24.36%, and 18.39%, respectively. These are observed probe results, not controlled intrinsic-codec rankings.

**Conclusion and limit.** Rate, distortion, family, entropy structure, and model choice interact. The correction explicitly withdraws the earlier RD-efficiency-versus-AUC claim. See [Phase 4.8 findings](../../results/Phase%204.8%20Extreme%20Compression/Phase%204.8%20Findings.md), [operating-point summary](../../results/Phase%204.8%20Extreme%20Compression/operating_point_summary.csv), and the [corrected correlation table](../../results/Phase%204.8%20Extreme%20Compression/phase47_corrected_rd_semantic_correlations.csv).

## Cross-phase conclusions

### Normalized stage utility and decode cost

The normalized per-stage table is retained in [`normalized-stage-frontier.csv`](data/normalized-stage-frontier.csv), with definitions and figures in the [consolidated report](ta-mtd-research-report.md). Accuracy is chance-adjusted relative to each condition's RGB probe; U is relative to that RGB probe; cost is cumulative preprocessing time relative to RGB decode. These are representation-plus-probe measurements, not end-to-end speedups. Its one-seed AVIF/WebP rows are distinct from the later three-seed Phase 4.7 structured-probe results summarized above.

- PBC compression: LZMA bytes reached 15.2% relative accuracy and 11.5% relative U at the 0% decoder-work convention; STORE reached 40.5%/38.9% at 2.6% cost; canonical IR reached 74.8%/82.0% at 45.6%; RGB is 100%/100% at 100%. Balanced and quality canonical IR reached 67.3%/65.2% at 57.1% cost and 65.4%/62.2% at 75.5% cost. The Phase 4.5B staged probes used three seeds.
- JPEG q10 DCT reached 93.7% relative accuracy and 101.4% relative U at 84.4% decode cost; q60 DCT reached 92.3%/98.5% at 113.6% cost. q1 DCT reached 104.9%/145.5% at 457.8% cost, but this is a one-seed probe with a different architecture/training distribution.
- In the one-seed normalized frontier, AVIF q40/q60 pre-reconstruction features reached 66.8%/68.6% and 70.1%/70.6% relative accuracy/U at 55.7%/66.3% cost. WebP q30/q75 features reached 66.3%/67.6% and 69.4%/70.5% at 1,561.5%/1,147.0% cost. The separate Phase 4.7 three-seed probe reported the values in its phase summary above; both WebP paths were slower than RGB. JXL had no measured intermediate point.

The 0% raw-byte cost is only a decoder-stage accounting convention: it excludes file reads, tokenization, model inference, and end-to-end compute. Full RGB is normalized to 100% within each condition. Phase timing sources and sample counts differ, so cross-phase normalized costs are descriptive.

- PBC STORE and LZMA reconstruct pixel-identical images, while LZMA streams are 25–39% smaller and their byte-model accuracy is 16.09–22.79 percentage points lower across three seeds. This controlled coding-layer result shows that entropy coding can change byte accessibility while preserving decoded pixels; it does not isolate sequence-length effects or guarantee the same penalty for other architectures.
- Codec family explains substantial variation among raw-byte probes at comparable measured rate/distortion. The pooled PSNR-residual association is strong in these sampled points but weaker within codec families and sensitive to model/curve choice.
- Intermediate representations can preserve useful task signal. Their utility must be compared with the cost to extract and pack them. AVIF early features offered limited utility savings at near-YUV cost; the current WebP hook was slower than RGB decoding; JPEG coefficient extraction approached full decode cost; PBC stages had lower costs on the tested pipeline.
- No result shows a universal compressed-domain advantage over RGB. Differently trained models, one-seed exploratory conditions, codec-family imbalance, and implementation-specific timing constrain claims.
- The next work is video V0/V1 feasibility and syntax audit in the [Video Handoff](Video%20Handoff.md) and detailed [Video Track Research Plan](../Video%20Track%20Research%20Plan.md). It is a separate research track, not an automatic training authorization.
