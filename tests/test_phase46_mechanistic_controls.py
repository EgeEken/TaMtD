import numpy as np
import pytest
import torch

from tamtd.analysis.phase46 import (
    apply_jpeg_dct_ablation,
    compare_codec_rankings,
    paired_gap_summary,
    retained_energy_fractions,
)
from tamtd.models.accessibility import BytePatchTransformer, parameter_count
from tamtd.neural_accessibility import pad_byte_sequences
from tamtd.training.byte_batching import LengthBucketBatchSampler
from scripts.run_experiment import SafetyStop, _check_start, _safety_reason


def _dct_inputs():
    channels = torch.arange(64, dtype=torch.float32).view(64, 1, 1).expand(64, 2, 3)
    return {
        "y_coefficients": channels.clone(),
        "cb_coefficients": channels.clone(),
        "cr_coefficients": channels.clone(),
        "quantization_tables": torch.ones(2, 8, 8),
    }


def test_byte_patch_transformer_is_compact_variable_length_and_padding_aware():
    model = BytePatchTransformer().eval()
    assert 300_000 <= parameter_count(model) <= 1_000_000
    tokens = torch.full((2, 512), 256, dtype=torch.long)
    tokens[0, :173] = torch.arange(173) % 256
    tokens[1, :401] = torch.arange(401) % 256
    lengths = torch.tensor([173, 401])
    reduced, padding_mask, patch_counts = model.reduce_patches(tokens, lengths)
    assert patch_counts.tolist() == [6, 13]
    assert padding_mask[0].sum().item() == reduced.size(1) - 6
    single = torch.full((1, 256), 256, dtype=torch.long)
    single[0, :173] = tokens[0, :173]
    with torch.inference_mode():
        batched_logits = model(tokens, lengths)[0]
        single_logits = model(single, torch.tensor([173]))[0]
    assert torch.allclose(batched_logits, single_logits, atol=1e-6)


def test_byte_patch_reduction_uses_bytes_after_the_attention_token_limit():
    model = BytePatchTransformer().eval()
    width = 32 * 257
    tokens = torch.zeros((1, width), dtype=torch.long)
    lengths = torch.tensor([width])
    reduced, padding_mask, patch_counts = model.reduce_patches(tokens, lengths)
    changed = tokens.clone()
    changed[0, -1] = 255
    reduced_changed, _, _ = model.reduce_patches(changed, lengths)
    assert patch_counts.item() == 257
    assert reduced.shape[1] == model.max_attention_tokens
    assert not padding_mask.any()
    assert not torch.equal(reduced, reduced_changed)


def test_byte_patch_loader_batching_keeps_every_byte_and_respects_token_budget():
    streams = [bytes(index % 256 for index in range(length)) for length in (270, 511, 768, 1025, 1537)]
    lengths = [len(stream) for stream in streams]
    sampler = LengthBucketBatchSampler(lengths, 4096, 4, 42, True)
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(len(streams)))
    assert all(((max(lengths[index] for index in batch) + 255) // 256) * 256 * len(batch) <= 4096 for batch in batches)
    tokens, tensor_lengths, truncation = pad_byte_sequences([streams[index] for index in batches[0]])
    assert truncation["truncated_samples"] == 0
    assert tensor_lengths.tolist() == [lengths[index] for index in batches[0]]
    assert tokens.size(1) % 32 == 0


def test_dct_dc_and_low_frequency_masks_are_deterministic_and_exact():
    inputs = _dct_inputs()
    dc = apply_jpeg_dct_ablation(inputs, "dc_only")
    dc_again = apply_jpeg_dct_ablation(inputs, "dc_only")
    assert torch.equal(dc["y_coefficients"], dc_again["y_coefficients"])
    assert torch.equal(dc["y_coefficients"][0], inputs["y_coefficients"][0])
    assert dc["y_coefficients"][1:].eq(0).all()
    retained_low4 = {u * 8 + v for u in range(4) for v in range(4)}
    low4 = apply_jpeg_dct_ablation(inputs, "low4x4")
    assert all(torch.equal(low4["y_coefficients"][index], inputs["y_coefficients"][index]) for index in retained_low4)
    assert all(low4["y_coefficients"][index].eq(0).all() for index in set(range(64)) - retained_low4)
    assert torch.equal(low4["quantization_tables"], inputs["quantization_tables"])


def test_dct_y_only_keeps_luminance_and_zeros_both_chroma_components():
    inputs = _dct_inputs()
    masked = apply_jpeg_dct_ablation(inputs, "y_only")
    assert torch.equal(masked["y_coefficients"], inputs["y_coefficients"])
    assert masked["cb_coefficients"].eq(0).all()
    assert masked["cr_coefficients"].eq(0).all()
    assert torch.equal(masked["quantization_tables"], inputs["quantization_tables"])


def test_dct_retained_energy_and_codec_ranking_summaries():
    coefficients = np.ones((2, 64, 1, 1), dtype=np.int16)
    dc = retained_energy_fractions(coefficients, "dc_only", "y_coefficients")
    assert dc["retained_coefficient_fraction"] == 1 / 64
    assert dc["retained_absolute_magnitude_fraction"] == 1 / 64
    assert dc["retained_squared_energy_fraction"] == 1 / 64
    rank = compare_codec_rankings(
        {"jpeg": 0.4, "webp": 0.3, "avif": 0.2, "jxl": 0.1},
        {"jpeg": 0.5, "webp": 0.35, "avif": 0.2, "jxl": 0.15},
    )
    assert rank["reference_order"] == rank["candidate_order"] == ["jpeg", "webp", "avif", "jxl"]
    assert rank["spearman_rank_correlation"] == 1.0
    assert rank["kendall_rank_correlation"] == 1.0
    gaps = paired_gap_summary({"jpeg_q10": [0.4, 0.5, 0.6], "webp_q30": [0.3, 0.4, 0.5]})
    assert gaps["webp_q30"]["differences"] == pytest.approx([0.1, 0.1, 0.1])
    assert gaps["webp_q30"]["mean"] == pytest.approx(0.1)
    assert gaps["webp_q30"]["positive_seed_count"] == 3


def test_phase46_runner_enforces_host_memory_and_vram_guards():
    gib = 1024**3
    runtime = {
        "minimum_start_free_gb": 20,
        "stop_free_gb": 12,
        "windows_commit_start_max": 0.95,
        "windows_commit_stop": 0.985,
        "windows_commit_high": 0.95,
        "windows_commit_high_seconds": 300,
        "require_cuda": True,
        "minimum_memory_available_gib": 6,
        "stop_memory_available_gib": 2,
        "minimum_cuda_free_mb": 4096,
        "stop_cuda_free_mb": 1024,
    }
    snapshot = {
        "platform": "win32",
        "disk_free_bytes": 30_000_000_000,
        "system": {"memory_available_bytes": 8 * gib, "commit_used_bytes": 50, "commit_limit_bytes": 100},
        "gpu": {"vram_free_mb": 4096},
    }
    _check_start(snapshot, runtime)
    low_ram = {**snapshot, "system": {**snapshot["system"], "memory_available_bytes": 5 * gib}}
    with pytest.raises(SafetyStop, match="host memory"):
        _check_start(low_ram, runtime)
    low_vram = {**snapshot, "gpu": {"vram_free_mb": 1023}}
    reason, _ = _safety_reason(low_vram, runtime, None)
    assert reason == "free CUDA VRAM fell below the configured stop threshold"
