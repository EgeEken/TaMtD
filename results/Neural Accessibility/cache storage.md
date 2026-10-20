# Phase 3 cache storage

The full-dataset training representation lives in `features/` and `streams/`:

- JPEG quantized DCT and quantization-table memmaps cover every sample and JPEG quality.
- PBC parsed and canonical model-feature memmaps and patch masks cover every sample and preset.
- Encoded JPEG and PBC streams support the raw-byte conditions.
- `manifest.jsonl` and `codec_manifest.jsonl` keep sample alignment and stream offsets.
- RGB training reads the original Imagenette files and applies EXIF transpose, RGB conversion, and the recorded 256x256 Pillow BICUBIC center fit.

The `ir/` directory stores full canonical IR NPZ bundles for a deterministic 95-image subset stratified across split and class. These bundles support schema verification and reproducibility; they are not the full-dataset training inputs. Full-dataset training uses the feature memmaps. The selected sample IDs are recorded in `ir/verification_subset.json`.

The cache builder writes stream data and feature rows before appending a manifest row. On resume, it verifies committed stream hashes, rolls back to the last fully valid sample, truncates uncommitted stream tails, and regenerates from that boundary.
