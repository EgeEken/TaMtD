# Staged Decoding Findings

## Dataset and settings

- Reused the exact Phase 1 source set: 95 deterministic 256×256 RGB images. Source manifest SHA-256: `98d384d3917caa140934238d7a92aebb2140f8eb851a524aa246a06c1b2c9e25`. Each PNG hash and RGB-pixel hash was checked against the Phase 1 manifest before encoding.
- PBC presets: compression, balanced, quality. Each image/preset was encoded once with LZMA disabled to obtain a STORE body, then raw-LZMA-wrapped from those exact body bytes. The forced wrapper never uses PBC's size-based STORE fallback.
- JPEG: Pillow/libjpeg-turbo 3.1.1 encoding at q1, q5, q10, q60, q80; pyjpegio 0.3.0 coefficient extraction using bundled libjpeg-turbo 3.2.0.
- PBC provenance and full effective configurations are in `config.json`. The external PBC checkout was read only.

## Stage boundaries and representation schema

PBC parsing uses the checkout's `BitReader`, PBC3 header structure, generated palette mask/bounds, and quantized grid indices. The **parsed fields** stage ends with decoded scalar fields, masks, and unsigned grid indices. **Canonical IR construction** maps these to typed arrays and adds the signed palette and residual grid values. **Patch application/render** initializes the PBC canvas, applies each signed-resampled patch, converts the internal color space to RGB, and performs the final resize when present. The standard `PBC3.decompress` call is measured independently for STORE and forced-LZMA streams.

PBC IR schema `tamtd.pbc-ir` v1 stores header fields and ordered patch metadata in canonical JSON inside a deterministic NPZ-like ZIP. Arrays keep each patch's mask (`uint8`), signed palette (`int16`), palette indices (`uint16`), and signed residual values (`int16`) as separate members. The stream can be rebuilt byte-for-byte from the IR. NPZ member order and timestamps are fixed; no pickle/object arrays are used.

JPEG coefficient access calls `jpegio.read(path, read_spatial=False)`, backed by libjpeg-turbo's coefficient API. Its output holds quantized coefficient arrays and tables without pixel-domain spatial arrays. Canonical IR schema `tamtd.jpeg-dct-ir` v1 stores signed `int16` coefficients in `(block_y, block_x, u, v)` natural DCT order; per-component sampling/block metadata; quantization tables; colorspace and image metadata; and Huffman table assignments. The binding exposes no standalone coefficient-to-RGB operation, so that stage is marked unavailable. We do not estimate it by subtracting unrelated timings.

## Timing method

- Each measured operation had 1 warm-up call(s) and 5 timed repetitions per image/condition. Imports and one-time PBC/JPEG initialization are outside the timed calls.
- PBC streams are decoded from in-memory bytes. JPEG coefficient extraction and Pillow full decode read the same cached JPEG file path; encoding and cache writes are outside measured regions. Filesystem read/open cost is included for both JPEG stages.
- Stage summaries report the median, p95, and mean across all repeated image operations. Cumulative rows sum separately measured stages; the direct complete-decode timing is measured independently and is the denominator for fractions. Small differences between cumulative component sums and direct full decode are expected.
- Canonical IR construction and serialization are measured separately from codec decoding. Serialization is an offline cache-preparation cost and is not included in fractions of full decode.

## Stage timing

`decode_stage_latency.png` plots incremental median stage times and direct full-decode medians (whisker to p95). `fraction_of_full_decode.png` plots cumulative stage time at each representation. The full per-condition stage table is in `stage_summary.csv`.

| Condition | Checkpoint | Incremental ms (median / p95) | Cumulative median ms | % full decode | Stored representation bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| pbc_compression_store | Parsed PBC instruction fields | 1.7909 / 4.0332 | 1.7909 | 38.6036 | 9997 |
| pbc_compression_store | Canonical PBC IR | 0.2553 / 0.7019 | 2.0766 | 44.7620 | 19865 |
| pbc_compression_store | Patch application and RGB render | 2.1719 / 4.5608 | 4.4601 | 96.1394 | 196608 |
| pbc_compression_store | Decoded RGB | 4.6392 / 9.8763 | 4.6392 | 100.0000 | 196608 |
| pbc_compression_forced_lzma | Bare PBC STORE body | 0.1250 / 0.3585 | 0.1250 | 2.6242 | 3861 |
| pbc_compression_forced_lzma | Parsed PBC instruction fields | 1.7909 / 4.0332 | 1.8969 | 39.8224 | 9997 |
| pbc_compression_forced_lzma | Canonical PBC IR | 0.2553 / 0.7019 | 2.2250 | 46.7103 | 19865 |
| pbc_compression_forced_lzma | Patch application and RGB render | 2.1719 / 4.5608 | 4.5993 | 96.5550 | 196608 |
| pbc_compression_forced_lzma | Decoded RGB | 4.7634 / 9.0688 | 4.7634 | 100.0000 | 196608 |
| pbc_balanced_store | Parsed PBC instruction fields | 3.7774 / 9.7201 | 3.7774 | 51.4843 | 14553 |
| pbc_balanced_store | Canonical PBC IR | 0.3504 / 0.8544 | 4.2044 | 57.3041 | 29170 |
| pbc_balanced_store | Patch application and RGB render | 2.1635 / 4.3492 | 6.7712 | 92.2884 | 196608 |
| pbc_balanced_store | Decoded RGB | 7.3370 / 14.6180 | 7.3370 | 100.0000 | 196608 |
| pbc_balanced_forced_lzma | Bare PBC STORE body | 0.2359 / 0.8431 | 0.2359 | 3.0862 | 8806 |
| pbc_balanced_forced_lzma | Parsed PBC instruction fields | 3.7774 / 9.7201 | 4.0149 | 52.5256 | 14553 |
| pbc_balanced_forced_lzma | Canonical PBC IR | 0.3504 / 0.8544 | 4.4348 | 58.0190 | 29170 |
| pbc_balanced_forced_lzma | Patch application and RGB render | 2.1635 / 4.3492 | 7.0125 | 91.7422 | 196608 |
| pbc_balanced_forced_lzma | Decoded RGB | 7.6437 / 14.4985 | 7.6437 | 100.0000 | 196608 |
| pbc_quality_store | Parsed PBC instruction fields | 8.2662 / 16.2526 | 8.2662 | 69.2526 | 24114 |
| pbc_quality_store | Canonical PBC IR | 0.4653 / 1.0650 | 8.7431 | 73.2480 | 48426 |
| pbc_quality_store | Patch application and RGB render | 2.0632 / 4.5357 | 11.0235 | 92.3527 | 196608 |
| pbc_quality_store | Decoded RGB | 11.9363 / 21.6862 | 11.9363 | 100.0000 | 196608 |
| pbc_quality_forced_lzma | Bare PBC STORE body | 0.6286 / 1.6163 | 0.6286 | 5.0731 | 23891 |
| pbc_quality_forced_lzma | Parsed PBC instruction fields | 8.2662 / 16.2526 | 8.9673 | 72.3700 | 24114 |
| pbc_quality_forced_lzma | Canonical PBC IR | 0.4653 / 1.0650 | 9.4116 | 75.9557 | 48426 |
| pbc_quality_forced_lzma | Patch application and RGB render | 2.0632 / 4.5357 | 11.5782 | 93.4412 | 196608 |
| pbc_quality_forced_lzma | Decoded RGB | 12.3909 / 22.6187 | 12.3909 | 100.0000 | 196608 |
| jpeg_q1 | JPEG quantized DCT coefficients | 0.3381 / 0.7830 | 0.3381 | 90.4736 | 3150 |
| jpeg_q1 | RGB reconstruction from coefficients | unavailable | n/a | n/a | n/a |
| jpeg_q1 | Decoded RGB | 0.3737 / 0.7324 | 0.3737 | 100.0000 | 196608 |
| jpeg_q5 | JPEG quantized DCT coefficients | 0.3512 / 0.7594 | 0.3512 | 90.5155 | 4139 |
| jpeg_q5 | RGB reconstruction from coefficients | unavailable | n/a | n/a | n/a |
| jpeg_q5 | Decoded RGB | 0.3880 / 0.7899 | 0.3880 | 100.0000 | 196608 |
| jpeg_q10 | JPEG quantized DCT coefficients | 0.3571 / 0.8073 | 0.3571 | 89.8817 | 5872 |
| jpeg_q10 | RGB reconstruction from coefficients | unavailable | n/a | n/a | n/a |
| jpeg_q10 | Decoded RGB | 0.3973 / 0.7811 | 0.3973 | 100.0000 | 196608 |
| jpeg_q60 | JPEG quantized DCT coefficients | 0.4032 / 0.9242 | 0.4032 | 91.1805 | 15602 |
| jpeg_q60 | RGB reconstruction from coefficients | unavailable | n/a | n/a | n/a |
| jpeg_q60 | Decoded RGB | 0.4422 / 0.8620 | 0.4422 | 100.0000 | 196608 |
| jpeg_q80 | JPEG quantized DCT coefficients | 0.4460 / 0.8874 | 0.4460 | 94.0730 | 21436 |
| jpeg_q80 | RGB reconstruction from coefficients | unavailable | n/a | n/a | n/a |
| jpeg_q80 | Decoded RGB | 0.4741 / 0.9563 | 0.4741 | 100.0000 | 196608 |

## Representation sizes

RGB size is 196608 bytes for every image. “Canonical array bytes” count typed NumPy payloads only, excluding Python object overhead. “Native array bytes” record pyjpegio's `int32` coefficient/table arrays before canonicalization; PBC has no separate native-array measurement.

| Condition | Encoded bytes (mean) | Canonical array bytes (mean) | Native array bytes (mean) | Serialized IR bytes (mean) | RGB bytes | Full decode median / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pbc_compression_store | 4441.8 | 43973.6 | n/a | 21804.2 | 196608 | 4.6392 / 9.8763 |
| pbc_compression_forced_lzma | 3436.7 | 43973.6 | n/a | 21804.2 | 196608 | 4.7634 / 9.0688 |
| pbc_balanced_store | 10609.6 | 104334.4 | n/a | 33777.3 | 196608 | 7.3370 / 14.6180 |
| pbc_balanced_forced_lzma | 6917.3 | 104334.4 | n/a | 33777.3 | 196608 | 7.6437 / 14.4985 |
| pbc_quality_store | 25637.7 | 204134.3 | n/a | 55447.8 | 196608 | 11.9363 / 21.6862 |
| pbc_quality_forced_lzma | 15639.0 | 204134.3 | n/a | 55447.8 | 196608 | 12.3909 / 22.6187 |
| jpeg_q1 | 1911.0 | 196864.0 | 393728.0 | 3247.0 | 196608 | 0.3737 / 0.7324 |
| jpeg_q5 | 2276.1 | 196864.0 | 393728.0 | 4284.4 | 196608 | 0.3880 / 0.7899 |
| jpeg_q10 | 3034.4 | 196864.0 | 393728.0 | 6113.8 | 196608 | 0.3973 / 0.7811 |
| jpeg_q60 | 8042.8 | 196864.0 | 393728.0 | 15578.6 | 196608 | 0.4422 / 0.8620 |
| jpeg_q80 | 11717.2 | 196864.0 | 393728.0 | 21428.2 | 196608 | 0.4741 / 0.9563 |

## Interpretation and Phase 3 input

1. **PBC LZMA unpack cost:** pbc_compression_forced_lzma: 0.1250 ms / 4.7634 ms full (2.6%); pbc_balanced_forced_lzma: 0.2359 ms / 7.6437 ms full (3.1%); pbc_quality_forced_lzma: 0.6286 ms / 12.3909 ms full (5.1%). Values are measured on this machine with Python's `lzma` implementation and must be read alongside the full PBC decoder timings.
2. **JPEG file-to-DCT cost:** jpeg_q1: 0.3381 ms / 0.3737 ms Pillow full decode (90.5%); jpeg_q5: 0.3512 ms / 0.3880 ms Pillow full decode (90.5%); jpeg_q10: 0.3571 ms / 0.3973 ms Pillow full decode (89.9%); jpeg_q60: 0.4032 ms / 0.4422 ms Pillow full decode (91.2%); jpeg_q80: 0.4460 ms / 0.4741 ms Pillow full decode (94.1%). `jpegio.read` includes file open/read, header/marker handling, entropy decoding, coefficient allocation, and Python-array exposure. Its coefficient-to-RGB successor is not independently available through this binding.
3. **Earlier structured representation:** by normalized checkpoint fraction, PBC reaches canonical IR at 44.8%, 57.3%, and 73.2% of direct STORE decode for compression, balanced, and quality; forced-LZMA PBC reaches it at 46.7%, 58.0%, and 76.0%. JPEG coefficient extraction takes 89.9%–94.1% of Pillow's full RGB decode. PBC therefore reaches its canonical IR earlier by this measure on this implementation. Absolute PBC timings are in milliseconds while JPEG timings are sub-millisecond, and the two implementations use different libjpeg-turbo versions; these are not algorithm-only comparisons.
4. **Representation sizes:** see the table above and `summary.csv`. PBC raw STORE/LZMA streams, parsed payload, canonical IR arrays/archive, JPEG file, canonical DCT arrays/archive, and RGB size are all recorded per image.
5. **Systems feasibility:** the PBC IR and JPEG DCT IR are usable structured inputs from a decoding and storage perspective. This phase measures cost and size only; it makes no claim about model accuracy or representation usefulness.
6. **Parsed PBC cost:** compare `store_body_parse`, `canonical_ir_build`, and `canonical_ir_serialize` against `forced_lzma_unpack` in the stage table. STORE parsing/canonicalization is the additional cost to obtain structured instructions after entropy unpack.
7. **Phase 3 representations:** use PBC `tamtd.pbc-ir` v1 for compression, balanced, and quality; use JPEG `tamtd.jpeg-dct-ir` v1 for q10, q60, and q80, with q1/q5 as low-rate controls. Keep encoded bytes and fully decoded RGB as reference conditions. No model training or reconstruction was run.

## Validation and limitations

- All 95 samples per condition were checked. PBC STORE and forced-LZMA bodies matched byte-for-byte after unpack; their full decodes matched pixel-for-pixel; canonical PBC IR reserialized to the original STORE stream and rendered pixel-identically. Serialization was deterministic.
- JPEG coefficient extraction was repeated and checked for deterministic metadata/arrays. Component block dimensions matched the header, and Pillow file and in-memory decodes matched pixel-for-pixel.
- pyjpegio and Pillow use bundled libjpeg-turbo versions 3.2.0 and 3.1.1 respectively. The coefficient-stage/full-decode ratio therefore compares the supported coefficient binding with Pillow's normal RGB path, not one identical library build.
- JPEG file I/O is included because pyjpegio's supported interface accepts a path. PBC uses byte buffers. Cache serialization sizes and timing are separate from decoding.
- Generated representations live under the ignored `data/cache/staged_decode/` directory and can be regenerated with `scripts/run_staged_decode_benchmark.py`; compact result tables and plots are stored here.

Per-sample data: `sample_results.jsonl`. Aggregates: `summary.json`, `summary.csv`, `stage_summary.csv`. Exact run metadata: `config.json`.
