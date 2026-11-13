# Video research handoff

## Current status

The image track is complete through Phase 4.8. Its main conclusions and limits are in the [research archive](Research%20Archive.md) and [consolidated report](ta-mtd-research-report.md). No video dataset, video model, or VLM experiment was run as part of this cleanup.

The next work is a small H.264/AVC feasibility and representation audit. The workstation has FFmpeg 6.1.1 with `libx264`, `ffprobe`, and H.264 decoders available. The local `.venv` is Python 3.12.14 but this audit found no `torch` import there; establish a suitable environment before model work. Tool availability does not resolve dataset access/licensing or establish that FFmpeg exposes pre-reconstruction residuals.

The deleted Imagenette feature maps are image-model inputs and are not prerequisites for video V0/V1. Those feasibility stages need only a legally accessible small video sample, pinned encoder/decoder tools, and compact demux/syntax outputs; no classifier training or Imagenette cache rebuild is required. The dataset's current access and license still need verification before any sample is downloaded or processed.

Read the detailed [Video Track Research Plan](../Video%20Track%20Research%20Plan.md) before implementation. The handoff sequence is:

1. **H.264 data/encoder feasibility.** Verify dataset access and license for a small motion-sensitive action subset. Freeze sample IDs, source hashes, encoder version, preset, GOP, frame rate, resolution, rate-control, bitrate/quality, and audio policy.
2. **Demux and syntax extraction.** Build a small V0/V1 harness for MP4 bytes, elementary stream, NAL units, GOPs, timestamps, and frame boundaries. Verify hashes, alignment, parser validity, and measured throughput before scaling.
3. **Motion-vector/residual availability.** Test actual FFmpeg/libavcodec APIs and decoder flags. Record the exact decoder boundary where motion vectors or residual/transform coefficients become available. Do not assume decoder side data is pre-reconstruction, and do not implement a new H.264 entropy decoder.
4. **YUV/RGB controls.** Decode the same streams to reconstructed YUV and full RGB; isolate demux, decode, color conversion, frame sampling, I/O, and packing costs. Confirm RGB matches the reference decoder.
5. **Matched compact temporal probes.** Only after feature alignment is sound, compare motion-only, keyframe-only, residual/structured inputs, and matched sparse RGB on fixed IDs and budgets.
6. **End-to-end utility-cost frontier.** Include parser/decode, I/O, packing, CPU/GPU transfer, inference, memory, and token costs; report uncertainty and per-video latency distributions.
7. **VLM integration only if justified.** Add a pinned decoded-video VLM reference only after earlier measurements show a useful, reproducible structured signal and access/cost are acceptable.

Proceed beyond V0 only if a small licensed sample can be frozen with hashes and deterministic encoding. Proceed beyond V1 only if parser outputs align with source frames/GOPs, the decoder boundary is measured rather than inferred, and extraction throughput can support a meaningful utility-versus-cost comparison. Keep V0/V1 feasibility separate from classifier training and VLM inference; the full gates and later decision criteria are in the [Video Track Research Plan](../Video%20Track%20Research%20Plan.md).
