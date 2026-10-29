# Phase 4.7 decoder-stage audit

The exact six Phase 4.5B Imagenette stream sets were checked against the 13,394-row Phase 3 manifest for image identity, labels, split, source RGB digest, contiguous offsets, payload lengths and SHA-256 digests. No stream was re-encoded. The shared split is 8,522 train, 947 validation and 3,925 test samples.

## Pinned sources and builds

| Decoder | Revision | Build used |
|---|---|---|
| libjxl | `libjxl/libjxl` v0.12.0, `a7a9c787341cf703dede03c2009fa460cae5e5df` | Windows x64, Release shared, CMake 3.29.2, Ninja 1.12.0, GCC/G++ 13.2.0; pinned Brotli/Highway/skcms submodules; one-thread `JxlResizableParallelRunner`. |
| libwebp | `webmproject/libwebp` v1.6.0, `4fa21912338357f89e4fd51cf2368325b59e9bd9` | Windows x64 Release shared decoder; animation, mux, tools, tests and examples disabled. `libwebp-pre-reconstruction.patch` adds the early callback and initializes the VP8 skip byte. |
| libaom | `aomedia.googlesource.com/aom` v3.15.1, `44d0a57786f432d933ff64b653347c66f4d0fa1d` | Windows x64 Release decoder-only shared; AV1 encoder, multithreading, tests, examples and docs disabled. `libaom-pre-reconstruction.patch` adds the parse-only capture and early return. |
| libavif | `AOMediaCodec/libavif` v1.4.2, `c5240fc79fe5c2407e10afd35f5505ef6333ea49` | Windows x64 Release shared; system libaom decoder only; encoder, libyuv, apps and tests disabled. |

Bootstrap configuration and patch pins are in `scripts/build_phase47_codecs.ps1`, `scripts/build_phase47_codecs.sh`, and `patches/phase47/versions.json`. Source checkouts and generated libraries remain outside the repository.

## Exact stage boundaries

`decoder_stage_manifest.csv` records these operation states and whether the stage was timed/scored. Pre-reconstruction features are 16×16 grids of 62 float32 fields. They contain geometry, coefficient summaries, quantizer and mode data, and masks; they contain no pixels.

| Representation | Syntax / entropy parsed | Coefficients decoded | Inverse transform / prediction applied | Pixels reconstructed | Loop filters | RGB | Early endpoint available |
|---|---:|---:|---:|---:|---:|---:|---:|
| JXL progressive buffer | No usable event observed | — | — | — | — | — | No for these streams |
| JXL native full decode | Yes | Yes | Yes | Yes | Codec-complete output | Not necessarily | No |
| JXL RGB | Yes | Yes | Yes | Yes | Complete | Yes | No |
| WebP pre-reconstruction IR | Yes, through final VP8 macroblock row | Yes; Y2 WHT/DC handling is already done | Spatial IDCT and intra prediction application: no | No | No | No | Yes |
| WebP native YUV | Yes | Yes | Yes | Yes | Deblocking complete | No | No; full decode |
| WebP RGB | Yes | Yes | Yes | Yes | Deblocking complete | Yes | No; full decode |
| AV1/AVIF pre-reconstruction IR | AVIF/frame/tile syntax and entropy parse complete | Yes; captured by the parse-only tile visitor | Inverse transform and prediction application: no | No | No | No | Yes |
| AVIF native YUV | Yes | Yes | Yes | Yes | Loop/restoration filters complete | No | No; full decode |
| AVIF RGB | Yes | Yes | Yes | Yes | Loop/restoration filters complete | Yes | No; full decode |

### JPEG XL

The exact `jxl_d6` and `jxl_d3` streams were encoded by `cjxl v0.11.2` without a progressive option. The public API exposed full-image output but zero `JXL_DEC_FRAME_PROGRESSION` events across the held-out images. These operating points therefore have no valid pre-full RGB image stage. They are reported as bytes-to-full-native and bytes-to-RGB only; no resize or degradation of full RGB is treated as staging.

One-shot decode is stable when the pinned v0.12.0 API receives an explicit one-thread resizable runner. The official chunked progressive example crashes in `jxl::N_AVX2::DecodeGroupImpl` on an exact cached stream when supplied in 1,024-byte chunks. A minimal reproducer and stream are retained in the ignored Phase 4.7 cache. The chunked path is excluded from results; no output from it is used.

### WebP / VP8

The early hook is in `VP8ProcessRow` (`src/dec/frame_dec.c`) before `ReconstructRow` and `FinishRow`. At that point the last macroblock row has passed `VP8ParseIntraModeRow` and `VP8DecodeMB`/`ParseResiduals` (`src/dec/vp8_dec.c`); no spatial IDCT, intra prediction application, pixel reconstruction or filtering has run. The callback returns before those operations. The IR contains low-frequency summaries from decoded coefficient workspaces, quantizer/segment data, mode histograms, and sparsity summaries.

The source patch also initializes `VP8MBData.skip` before the optional skip-probability read. Without this initialization, streams with skip probabilities disabled left the byte undefined; the original maps included arbitrary values there. The earlier maps, six original WebP checkpoints and original matrix remain preserved under ignored `data/cache/.../phase47` paths. The fixed cache was rebuilt from the exact streams; eight held-out feature rows per WebP condition re-decode bitwise identically. Six WebP seeds were retrained. Sixteen held-out full RGB outputs matched the pre-fix build byte-for-byte, confirming that ordinary full decoding is unchanged.

### AVIF / AV1

The AVIF container is decoded by libavif and its system libaom backend. The libaom patch captures block mode, qindex/segment/skip and low-frequency coefficient summaries in the parse-only tile visitor, then exits before inverse transform, reconstruction and frame wrapup. The operation is in AV1 codec-level parsing and returns a frame grid, so the hook can be adapted to a video-frame/OBU input path. Video references, inter-frame state, timestamps and containers were not tested.

The native YUV and RGB rows use the ordinary full decode path. The public inspection callback occurs after reconstruction/filter wrapup and is not used as the early stage.

## Timing boundaries and validation

All stage timing uses in-memory exact stream bytes, one decoder thread, one warmup and 256 systematically spaced test samples with three repeats. File I/O and serialization are outside the measured call. `perf_counter_ns` boundaries are recorded per sample in `decoder_timing_per_sample.csv`.

For WebP/AV1 early paths, decoder time ends at the source callback/return after parse. Callback copy work, feature aggregation and model-token adaptation are separately recorded. Native YUV and RGB are measured through their full decoder paths on the same harness. Instrumentation cost is included where it runs inside the decoder endpoint and marked in timing metadata. These are implementation timings on the pinned builds and machine, not codec-invariant costs.

The unmodified RGB paths were checked on eight repeated samples per condition. After the WebP initialization fix, a separate before/after build comparison found byte-identical RGB output on eight held-out samples each for q30 and q75. RGB checkpoint endpoint scores were re-evaluated on the full 3,925-image test split; cross-library differences from Phase 4.5B are retained in `full_decode_rechecks.csv`.
