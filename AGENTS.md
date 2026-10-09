# AGENTS.md

## Project: TaMtD

**TaMtD — Compressed-Domain Visual Understanding Across Codecs**

TaMtD is an independent research/engineering project studying how image-codec representation, entropy coding, and partial decoding affect the accessibility of visual information to neural models.

The current goal is **not** to prove that a neural model can fully replace a media decoder. The immediate goal is to build a rigorous, presentable benchmark that measures the trade-off between:

- how much conventional decoding/preprocessing has been performed,
- how much visual/semantic information is available to a learned model,
- how much latency that preprocessing costs,
- and how those trade-offs differ across codecs.

The long-term idea of learning directly from compressed media remains the motivation, but current work should prioritize clean experiments and interpretable results.

---

## Current repository state

The early exploratory stage is recorded under:

`results/Early Testing/`

Start by reading:

`results/Early Testing/Early Testing Findings.md`

Those experiments used **32x32 CIFAR-10** and are useful for pipeline validation and hypothesis generation only. They are **not** the headline benchmark and should not be used to make general claims about codec quality, rate-distortion behavior, or representative PBC-vs-JPEG operating points.

Important early observations:

- simple byte-level classification on PBC was often reproducible with order-insensitive statistics;
- PBC STORE was substantially easier to classify than PBC wrapped in forced LZMA despite identical decoded pixels;
- byte order mattered more for the coarse 8x8 reconstruction probe than for CIFAR classification;
- the 8x8 reconstruction outputs were not good enough to claim useful decoding;
- CIFAR's 32x32 resolution is too small and atypical for the intended codec study.

Treat these as **early testing findings**, not final conclusions.

---

# Current research question

The central practical question is:

> **How do codec structure, entropy coding, and partial decoding change the neural accessibility of visual information, and what preprocessing latency is required to reach a useful representation?**

The desired final product is a benchmark and set of plots/tables that can support a project report, GitHub presentation, and CV entry even if no state-of-the-art model or publishable research result emerges.

A strong final result does **not** require raw bitstreams to outperform fully decoded RGB.

A useful result can instead show an **accuracy–preprocessing-latency frontier** such as:

```text
JPEG file bytes
    -> entropy-decoded quantized DCT representation
    -> fully decoded RGB

PBC + LZMA
    -> bare PBC STORE body
    -> parsed PBC patch/instruction representation
    -> fully decoded RGB
```

The key measurement is how task performance changes as conventional decoding progresses and how much latency each stage costs.

---

# Headline benchmark scope

## Resolution

Headline experiments should use **256x256 natural images**.

Do not use 32x32 CIFAR as the main benchmark.

If compute or encoding time is a concern, reduce dataset size before reducing image resolution.

## Initial codecs

The intended benchmark set is:

- JPEG
- WebP
- AVIF
- JPEG XL / JXL
- PNG
- PBC

If one codec is difficult to install or expose reliably, do not let it block the entire benchmark. Record the issue and continue with the working codecs.

## Codec comparison rule

Never treat nominal codec quality values as equivalent across codecs.

Examples:

- JPEG quality 80 is not equivalent to WebP quality 80.
- Similar quality-number labels do not imply similar rate or distortion.
- The CIFAR result where PBC quality and JPEG q80 happened to have similar file sizes must not be generalized.

Choose comparison points from **measured operating characteristics**:

1. **rate matched** — approximately equal bits per pixel / encoded size; and/or
2. **distortion matched** — approximately equal PSNR/MSE/SSIM.

Report both where practical.

PBC should be evaluated prominently in the extreme-compression regime. JPEG q1-q10 should be included in calibration because this is often closer to PBC's intended operating range on realistic resolutions.

---

# Phase 1 — 256x256 codec characterization

This is the **next implementation task**.

Do not train a new neural architecture before this benchmark exists.

## Dataset

Use a small fixed natural-image calibration subset first:

- approximately 100-250 images;
- 256x256 RGB;
- deterministic preprocessing/cropping;
- same exact source view for every codec.

The dataset only needs to be large enough to establish codec operating points and timing distributions.

## Encode/decode sweep

Evaluate multiple settings for:

### PBC
- compression
- balanced
- quality
- high_quality

Use the local PBC checkout as the source of truth for exact preset configuration.

Record:
- PBC repository commit;
- branch;
- dirty state;
- full effective configuration.

### JPEG
Include a broad sweep emphasizing low qualities, e.g.:
- q1
- q5
- q10
- q20
- q30
- plus representative higher-quality points such as q60/q80 if useful.

Record exact encoder settings:
- implementation/library;
- subsampling;
- optimize/progressive flags;
- quality.

### WebP
Use several lossy settings spanning similar measured bpp values to PBC/JPEG.

### AVIF
Use several lossy settings spanning similar measured bpp values.

### JPEG XL / JXL
Use several lossy settings if the encoder/decoder can be integrated cleanly.

### PNG
Use as the lossless reference.

Do not invent a fake rate-matched PNG setting.

## Metrics

For every image and codec setting record at minimum:

- encoded bytes;
- bits per pixel;
- MSE against the exact original source RGB;
- PSNR;
- SSIM if straightforward and stable;
- encode latency;
- full decode latency;
- validity / decode success.

Timing summaries should include:
- median;
- p95;
- optionally mean.

Use warm-up runs and repeated timings where necessary so one-time initialization does not dominate.

### Timing caveat

Absolute latency across codecs can measure implementation quality as much as algorithmic complexity.

Always record the implementation/library and version where possible.

PBC is largely Python/research code while JPEG/WebP/AVIF/JXL/PNG may use optimized native libraries. Do not frame raw cross-library timing as a pure codec-algorithm comparison.

## Required Phase 1 outputs

Produce:

1. rate-distortion table;
2. PSNR vs bits/pixel plot;
3. decode latency vs bits/pixel or codec-setting plot;
4. encoded-size table;
5. fixed visual comparison grid;
6. machine-readable JSON/CSV result file;
7. selected rate-matched and distortion-matched operating points for later ML experiments.

Results should be saved under a new non-early-testing results area, e.g.:

`results/Codec Benchmark/`

Do not overwrite `results/Early Testing/`.

---

# Phase 2 — staged / partial decoding benchmarks

Once the 256x256 codec operating points are known, instrument codecs with useful intermediate representations.

## PBC decoding ladder

Desired stages:

```text
PBC + LZMA file
    -> LZMA unpack
    -> bare PBC STORE body
    -> PBC parse / structured patch representation
    -> patch application / rendering
    -> RGB pixels
```

Measure separately where feasible:

- LZMA unpack time;
- STORE body parsing time;
- patch/instruction parsing time;
- render/application time;
- total decode time.

Preserve the controlled invariant:

> PBC STORE and PBC+LZMA variants used for comparison must originate from the same raw PBC body and decode to pixel-identical images.

Relevant ML representations may include:

- full LZMA-wrapped bytes;
- bare STORE bytes;
- parsed patch/instruction tensors;
- decoded RGB.

## JPEG decoding ladder

Desired stages:

```text
JPEG file
    -> marker/header parse + entropy/Huffman decoding
    -> quantized DCT coefficient arrays
    -> dequantization + IDCT + upsampling/colorspace conversion
    -> RGB pixels
```

Prefer a standard library interface such as libjpeg/libjpeg-turbo coefficient access rather than writing a JPEG entropy decoder from scratch.

For the coefficient representation retain whatever information is required to interpret it correctly, e.g.:

- quantized DCT coefficients;
- quantization tables;
- component/sampling layout;
- image/block geometry.

Measure:

- file -> coefficient representation latency;
- coefficient representation -> RGB latency if separable;
- full decode latency.

## PNG decoding ladder — optional if cheap

A useful staged lossless comparison is:

```text
PNG file
    -> DEFLATE inflate
    -> filtered scanline bytes
    -> PNG unfilter
    -> pixels
```

Only implement this if it is clean and does not delay the main JPEG/PBC study.

## WebP / AVIF / JXL intermediate stages

Initially benchmark:
- raw file bytes;
- fully decoded pixels.

Do **not** spend excessive time reverse-engineering internal representations through non-public APIs before JPEG and PBC staged experiments are complete.

---

# Phase 3 — neural accessibility benchmark

Only after Phase 1 establishes sensible operating points.

The benchmark question is:

> How much task performance is available from each codec representation for a given amount of conventional preprocessing?

## Model hierarchy

Start cheap:

1. statistical baseline where applicable;
2. small byte/sequence model;
3. codec-intermediate representation model;
4. decoded-RGB CNN reference.

A ByteFormer-style model is allowed later, but do not introduce it merely because a small model underperforms.

The first purpose of ML here is **measurement**, not architecture competition.

## Recommended task

Use a natural-image classification task if a suitable 256x256 dataset with labels is available.

For every model/representation record:

- validation-selected test accuracy;
- preprocessing latency;
- training examples/second;
- peak VRAM;
- parameter count;
- encoded/intermediate representation size;
- truncation rate, which should preferably be zero.

Use one seed for exploratory sweeps.

Use three seeds only for the final selected comparisons that will appear in headline tables/plots.

## Central final plot

Target a plot with:

**x-axis: conventional preprocessing/decode latency**  
**y-axis: visual-task performance**

with staged points such as:

```text
JPEG bytes -> JPEG DCT -> RGB
PBC+LZMA -> PBC STORE -> parsed PBC -> RGB
```

This is the project's primary intended systems/research contribution.

---

# Success criteria

## Phase 1 success

Phase 1 is successful if:

- all feasible codecs encode/decode reproducibly at 256x256;
- rate-distortion curves are defensible;
- decode latency measurements are reproducible;
- codec versions/settings are recorded;
- genuine rate-matched/distortion-matched operating points can be selected;
- plots/tables are presentation quality.

PBC does **not** need to outperform every codec.

## Staged-decoding success

A strong result would show that an intermediate representation achieves a useful fraction of fully decoded visual-task performance at substantially lower preprocessing cost.

Examples of interesting outcomes:

- entropy decoding alone makes a large amount of semantic information accessible;
- a codec's native intermediate representation reaches near-RGB task accuracy without full reconstruction;
- entropy coding significantly reduces neural accessibility while preserving deterministic decodability;
- different codecs produce substantially different accuracy/preprocessing frontiers.

A negative result is also valid if measured cleanly, e.g.:
- useful visual understanding only appears after nearly full decode;
- intermediate states provide little benefit over raw bytes.

Do not reinterpret a negative result as success by changing metrics after the fact.

---

# Experimental discipline

- Use the same source image for all codec variants in a comparison.
- Keep train/validation/test splits fixed across representations.
- Select checkpoints on validation only.
- Evaluate test once per selected checkpoint.
- Record random seeds.
- Record codec/library versions.
- Record exact encoder parameters.
- Record exact source resolution and preprocessing.
- Record git commit hashes in result metadata when practical.
- Explicitly report truncation.
- Keep large dataset caches and checkpoints out of git.
- Commit compact JSON/CSV summaries, plots, configs, and documentation needed to reproduce conclusions.
- Do not push or commit on the user's behalf unless explicitly asked.
- Do not modify the external PBC repository unless explicitly requested.
- Do not use results from dirty/unrecorded codec configurations without recording the state.

---

# Repository organization

Existing reusable implementation belongs under:

`src/tamtd/`

Suggested areas as the benchmark expands:

```text
src/tamtd/codecs/
src/tamtd/data/
src/tamtd/models/
src/tamtd/training/
src/tamtd/analysis/
scripts/
configs/
tests/
results/
```

Keep early experiments intact under:

`results/Early Testing/`

Use a separate directory for the new 256x256 benchmark.

---

# Immediate implementation order

Follow this order unless the user explicitly redirects the project:

1. Build the 256x256 natural-image codec benchmark harness.
2. Integrate/verify JPEG, WebP, AVIF, JXL, PNG, and PBC adapters.
3. Run a small 100-250 image rate-distortion/timing calibration sweep.
4. Generate tables/plots and choose matched operating points.
5. Instrument staged PBC decoding.
6. Expose/timestamp JPEG quantized-DCT coefficient extraction.
7. Benchmark raw/intermediate/RGB representations with small models.
8. Produce the accuracy-preprocessing frontier.
9. Only then decide whether stronger byte models, ByteFormer, reconstruction, video, or VLM experiments are justified.

---

# First next step

The first concrete task is:

> **Implement and run a 256x256 natural-image codec characterization sweep on approximately 100-250 images.**

Do not start new model training before this is complete.

Expected engineering time is roughly a few hours if codec bindings are already available. AVIF/JXL setup and PBC runtime are the main likely sources of delay.

The immediate output should be enough to answer:

- What bpp ranges do the PBC presets actually occupy at 256x256?
- Which JPEG/WebP/AVIF/JXL settings are genuinely comparable by rate or distortion?
- How large are the decode-latency differences?
- Which codec operating points should be carried into the neural benchmark?

Once those are known, the project can proceed without repeating the 32x32 operating-point mistake.
