# Video track research plan

## Research question

> How much task/VLM utility becomes available at successive stages of a conventional video decode pipeline, and can models consume codec-native intermediate representations before paying the cost of full frame reconstruction and dense visual encoding?

For a fixed encoded H.264 video, measure how much action or event information is available from progressively decoded codec structure and what accuracy, latency, storage, and token costs are required to reach the decoded-frame baseline.

The aim is a measured accuracy–preprocessing-cost frontier. This work should establish whether compressed-domain signals are useful for a defined task; it should not assume a learned model can replace a standards-compliant decoder.

## Prior work and current gap

- [Compressed Video Action Recognition (CoViAR, CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Compressed_Video_Action_CVPR_2018_paper.html) studies RGB keyframes, motion vectors, and residual information for action recognition. It motivates separating appearance, motion, and residual ablations.
- [MM-ViT (WACV 2022)](https://openaccess.thecvf.com/content/WACV2022/html/Chen_MM-ViT_Multi-Modal_Video_Transformer_for_Compressed_Video_Action_Recognition_WACV_2022_paper.html) treats I-frames, motion vectors, residuals, and audio as distinct inputs and models their interactions.
- [Accurate and Fast Compressed Video Captioning (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Shen_Accurate_and_Fast_Compressed_Video_Captioning_ICCV_2023_paper.html) tests end-to-end captioning from compressed-video structure.
- [Efficient Motion-Aware Video MLLM (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Efficient_Motion-Aware_Video_MLLM_CVPR_2025_paper.html) combines fewer dense RGB frames with sparse motion-vector input and evaluates motion-sensitive reasoning.
- [Qwen3.5 documentation](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5) describes a native image/video model interface in Transformers. Use a pinned checkpoint and library version as a decoded-video reference, not as evidence that a compressed-input student will work. The Qwen team’s [release notes](https://qwen.ai/blog?email_hash=23463b99b62a72f26ed677cc556c44e8&id=qwen3.5) describe the model family.
- [Vision as LoRA (VoRA, 2025)](https://arxiv.org/abs/2503.20680) explores internalizing image features through vision-specific LoRA layers, block-wise distillation, and bidirectional visual attention. It is an image-oriented architectural reference; it does not establish compressed-video decoding or temporal motion reasoning.

The open project question is not whether those prior approaches can classify compressed video. It is how performance changes across explicit decode stages when the source encode, data split, model capacity, token budget, and timing boundary are controlled and fully recorded.

## Dataset and codec

Start with H.264/AVC elementary streams inside a fixed MP4 container. Use a motion-centric action task such as Something-Something V2, subject to dataset access and license review. Keep the official split and video IDs. Use a small fixed diagnostic subset to validate parsing and alignment before setting the scale of any training run. A captioning dataset can be a later secondary task after action recognition and timing are stable.

Freeze the encoder implementation, version, preset, GOP structure, frame rate, resolution, rate-control mode, bitrate/quality setting, and audio policy. Preserve the exact input file and its hash per sample. Compare multiple measured bitrate and distortion operating points; do not assume quality numbers match across encoders. Use the same encoded files for every representation and model ablation.

## Representation ladder and timing boundaries

Use standard demux/decoder APIs where they expose the needed information. Do not write a new H.264 entropy decoder. Record when an API does not expose a stage cleanly; do not relabel a combined operation as an isolated codec stage.

| Stage | Representation | Required timing boundary |
| --- | --- | --- |
| V0 | MP4/container bytes | File read through full byte buffer |
| V1 | Demuxed elementary stream with NAL, GOP, and frame boundaries/basic syntax | Demux start through packet/NAL and framing extraction |
| V2 | Entropy-decoded structure: frame types, block metadata/QP, motion vectors, and quantized residual/transform coefficients where accessible | NAL input through structured syntax output |
| V3 | Sparse keyframes paired with codec-native temporal information | V2 extraction plus keyframe selection and aligned feature packing |
| V4 | Reconstructed YUV frames | Entropy decode, inverse quantization/transforms, prediction, motion compensation, pixel reconstruction, and in-loop filtering through YUV output |
| V5 | Fully filtered, color-converted RGB frames | Full standards-compliant decode through RGB output |

Keep parser and decoder operations separate in timing records: demux; NAL/GOP/frame syntax parsing; entropy decoding; inverse quantization and inverse transforms; motion compensation; pixel reconstruction and in-loop filtering; YUV output; colorspace conversion; frame sampling; vision encoding; LLM prefill; generation; and end-to-end latency. Measure isolated stage time and cumulative time from V0. Include file I/O, allocation, tensor packing, and CPU/GPU transfer as separate columns when they occur. Report per-video and per-frame latency distributions, throughput, peak host memory, peak VRAM, bytes stored, token count, and truncation. For the decoded baseline, record frame sampling timestamps and number of frames. For compressed representations, record how much of the clip is represented and the resulting token count.

FFmpeg/libavcodec is a practical demux and decode baseline. Motion-vector side data may be exposed by decoder settings; quantized residual coefficients and finer intermediate states may require an instrumented decoder or a narrow codec library interface. First test whether those fields are available and stable before committing to an architecture that depends on them.

## Qwen3.5 baseline and compressed-domain student

The baseline is normal Qwen3.5 video input from decoded and sampled frames. Pin the checkpoint, processor, Transformers version, frame timestamps/count, prompt, context length, and inference precision. Treat the Qwen result as a reference point for the chosen task and budget.

The proposed compressed-domain student is:

```text
codec-native structured features
  -> lightweight codec encoder/tokenizer
  -> temporal compressor
  -> projector into Qwen multimodal/LLM embedding space
  -> initially frozen Qwen backbone
```

The student starts from parsed structures such as keyframe features, motion vectors, and residual/transform summaries. Raw entropy-coded bytes are a later endpoint, after the structured path has been validated. Preserve temporal positions and GOP boundaries in the feature schema.

Proposed training progression, only after a separate experiment approval:

1. Train the projector and codec encoder while keeping Qwen frozen.
2. Add feature-level distillation targets from Qwen’s normal decoded-video path.
3. Add task supervision using the fixed train/validation/test split.
4. Add LoRA adapters to selected Qwen layers only if the frozen-backbone stages fail to fit the student features.

The 2025 [VoRA](https://arxiv.org/abs/2503.20680) work suggests a parameter-efficient way to internalize image features with vision-specific LoRA, block-wise distillation, and bidirectional visual attention. VoRA is an image/vision architecture; it is not evidence that raw video bytes can simply be mapped through LoRA. A TaMtD adaptation would still need a structured temporal feature encoder and explicit timing/utility measurements.

## Model progression

1. **Decode and representation audit.** Verify every sample’s duration, frame count, frame alignment, decoder success, file hash, and stage timings. Confirm that V5 matches the reference decoder output.
2. **Small supervised probes.** Train linear/MLP probes on keyframes, motion vectors, residual features, and their combinations. Use one small fixed split to detect broken representation extraction.
3. **Compact temporal student.** Train one modest temporal model over the selected compressed inputs. Compare equal sample IDs and report parameter count, token budget, validation-selected checkpoint, single final test evaluation, and throughput.
4. **Reference comparison and ablations.** Evaluate a pinned Qwen3.5 checkpoint on a documented, fixed set of decoded video frames and a consistent prompt/task format. Compare against the compact student; then remove motion vectors, residuals, or keyframes one at a time. Only after these results are stable should multi-seed confirmation or captioning be considered.

Qwen is an inference baseline in this design, not current work. Evaluate model size, processor/library versions, quantization, frame limits, prompt, and context length before selecting a checkpoint. Keep inference cost and generation tokens separate from decode and vision preprocessing cost.

## Hypotheses and falsifiers

1. **Codec-native motion information retains temporal events that sparse RGB sampling misses at equal preprocessing and token budgets.** Falsified if motion inputs do not improve the held-out motion-sensitive task over matched sparse RGB or the gain disappears under fixed budgets.
2. **Useful task/VLM semantics appear before full motion compensation and pixel reconstruction.** Falsified if V2/V3 performance stays near the label-prior baseline, or if reaching useful performance costs as much preprocessing latency as V5 without a score gain.
3. **Appearance-heavy and motion-heavy tasks have different optimal decode exit points.** Falsified if both task families select the same representation stage within measured uncertainty after matching data, model capacity, and token cost.
4. **Structured codec representations need less VLM visual-token compute than dense RGB video.** Falsified if the compressed student needs comparable or greater visual tokens/compute for equal task performance, or falls well below the matched RGB baseline.
5. **Raw entropy-coded bytes remain less accessible than parsed structures until structure is exposed.** Falsified if the byte model matches the structured representation at the same compute and token budget without a representation-specific preprocessing advantage.
6. **Different video codecs expose different semantic-utility/decode-cost frontiers at comparable rate and distortion.** Falsified if codec frontiers overlap within uncertainty across the preselected rate/distortion operating points.

## Experimental controls and reporting

- Keep the split, encoded file, source hash, decoder version, and frame timestamps fixed across conditions.
- Select checkpoints on validation only and evaluate test once per selected checkpoint.
- Record seed, encoder settings, decoder/library versions, representation schema, tokenization, truncation, and invalid-stream counts.
- Report accuracy or task score beside preprocessing latency, model inference latency, throughput, bytes/video, token count, parameter count, peak host RAM, and peak VRAM.
- Include a decoded RGB baseline and a simple motion-only baseline. Report codec bitrate and distortion so quality shifts are visible.
- Use multiple seeds only for comparisons selected in advance for the headline table. Keep exploratory probes separate.

No video data, model training, Qwen inference, or cloud run is part of Phase 4.5A infrastructure readiness.
