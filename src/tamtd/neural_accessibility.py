from __future__ import annotations

import math
from statistics import fmean, median

import numpy as np
import torch


def pad_byte_sequences(
    sequences: list[bytes | bytearray | np.ndarray],
    max_length: int | None = None,
    pad_multiple: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if not sequences:
        raise ValueError("at least one byte sequence is required")
    if pad_multiple < 1:
        raise ValueError("pad_multiple must be positive")
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    if np.any(lengths == 0):
        raise ValueError("byte sequences cannot be empty")
    target_length = int(lengths.max())
    if max_length is not None:
        if max_length < 1:
            raise ValueError("max_length must be positive")
        target_length = min(target_length, max_length)
    truncated = lengths > target_length
    lengths = np.minimum(lengths, target_length)
    width = int(math.ceil(target_length / pad_multiple) * pad_multiple)
    tokens = torch.full((len(sequences), width), 256, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        values = np.frombuffer(sequence, dtype=np.uint8) if isinstance(sequence, (bytes, bytearray)) else np.asarray(sequence, dtype=np.uint8)
        copy_length = min(len(values), target_length)
        if copy_length:
            tokens[index, :copy_length] = torch.from_numpy(values[:copy_length].astype(np.int64, copy=False))
    lengths_tensor = torch.from_numpy(lengths.copy())
    stats = {
        "sample_count": len(sequences),
        "max_input_length": int(np.max([len(sequence) for sequence in sequences])),
        "padded_width": width,
        "truncated_samples": int(truncated.sum()),
        "truncated_bytes": int(np.maximum(np.asarray([len(sequence) for sequence in sequences]) - target_length, 0).sum()),
        "truncation_rate": float(truncated.mean()),
    }
    return tokens, lengths_tensor, stats


def latency_summary(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        raise ValueError("latency summary needs at least one measurement")
    return {
        "median_ms": float(median(values_ms)),
        "p95_ms": float(np.percentile(values_ms, 95)),
        "mean_ms": float(fmean(values_ms)),
    }


def semantic_accessibility(
    model_accuracy: float,
    decoded_rgb_accuracy: float,
    chance_accuracy: float = 0.1,
) -> float | None:
    denominator = decoded_rgb_accuracy - chance_accuracy
    if denominator <= 0:
        return None
    return (model_accuracy - chance_accuracy) / denominator


def build_frontier_rows(
    model_results: list[dict],
    rgb_reference_accuracies: dict[str, float],
    preprocessing: dict[str, dict],
    condition_sizes: dict[str, dict],
    chance_accuracy: float = 0.1,
) -> list[dict]:
    lookup = {(row["condition"], row["representation"]): row for row in model_results}
    frontier = []
    for condition, timings in preprocessing.items():
        sizes = condition_sizes[condition]
        if condition.startswith("pbc_"):
            full_decode = timings["full_lzma_decode_ms"]
            stages = [
                ("pbc_lzma_bytes", 0.0, sizes["lzma_bytes"]),
                ("pbc_store_bytes", timings["lzma_unpack_ms"], sizes["store_bytes"]),
                (
                    "pbc_parsed_fields",
                    timings["lzma_unpack_ms"] + timings["store_body_parse_ms"],
                    sizes["parsed_bytes"],
                ),
                (
                    "pbc_canonical_ir",
                    timings["lzma_unpack_ms"] + timings["store_body_parse_ms"] + timings["canonical_ir_build_ms"],
                    sizes["canonical_bytes"],
                ),
                ("decoded_rgb", full_decode, sizes["rgb_bytes"]),
            ]
        else:
            full_decode = timings["full_rgb_decode_ms"]
            stages = [
                ("jpeg_file_bytes", 0.0, sizes["jpeg_bytes"]),
                ("jpeg_dct_ir", timings["file_to_dct_ms"], sizes["dct_bytes"]),
                ("decoded_rgb", full_decode, sizes["rgb_bytes"]),
            ]
        rgb_accuracy = rgb_reference_accuracies[condition]
        for representation, cumulative_ms, representation_bytes in stages:
            if representation == "decoded_rgb":
                accuracy = rgb_accuracy
                validation_accuracy = None
                family = "common_rgb_reference"
            else:
                result = lookup[(condition, representation)]
                accuracy = result["test_accuracy"]
                validation_accuracy = result["best_validation_accuracy"]
                family = result["model_family"]
            frontier.append(
                {
                    "condition": condition,
                    "codec_family": "pbc" if condition.startswith("pbc_") else "jpeg",
                    "representation": representation,
                    "model_family": family,
                    "test_accuracy": float(accuracy),
                    "best_validation_accuracy": validation_accuracy,
                    "decoded_rgb_reference_accuracy": float(rgb_accuracy),
                    "semantic_accessibility": semantic_accessibility(
                        float(accuracy), float(rgb_accuracy), chance_accuracy
                    ),
                    "cumulative_preprocessing_ms": float(cumulative_ms),
                    "fraction_full_decode": float(100.0 * cumulative_ms / full_decode),
                    "representation_bytes": int(representation_bytes),
                    "bpp": float(sizes["bpp"]),
                }
            )
    return frontier
