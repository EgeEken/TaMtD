from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr


DCT_ABLATIONS = ("full", "dc_only", "low4x4", "y_only")
DCT_COMPONENTS = ("y_coefficients", "cb_coefficients", "cr_coefficients")


def _coefficient_mask(ablation: str, component: str, device=None) -> torch.Tensor:
    if ablation not in DCT_ABLATIONS:
        raise ValueError(f"unknown JPEG DCT ablation: {ablation}")
    if component not in DCT_COMPONENTS:
        raise ValueError(f"unknown JPEG DCT component: {component}")
    mask = torch.zeros(64, dtype=torch.float32, device=device)
    if ablation == "full" or (ablation == "y_only" and component == "y_coefficients"):
        mask.fill_(1)
    elif ablation == "dc_only":
        mask[0] = 1
    elif ablation == "low4x4":
        mask[[u * 8 + v for u in range(4) for v in range(4)]] = 1
    return mask


def apply_jpeg_dct_ablation(inputs: dict[str, torch.Tensor], ablation: str) -> dict[str, torch.Tensor]:
    required = (*DCT_COMPONENTS, "quantization_tables")
    if any(name not in inputs for name in required):
        raise ValueError("JPEG DCT inputs are missing a component or quantization tables")
    output = dict(inputs)
    for name in DCT_COMPONENTS:
        coefficients = inputs[name]
        if coefficients.ndim not in (3, 4) or coefficients.shape[-3] != 64:
            raise ValueError("JPEG coefficients must have shape [64, y, x] or [batch, 64, y, x]")
        mask = _coefficient_mask(ablation, name, coefficients.device).to(coefficients.dtype)
        mask = mask.view(*([1] * (coefficients.ndim - 3)), 64, 1, 1)
        output[name] = coefficients * mask
    output["quantization_tables"] = inputs["quantization_tables"]
    return output


def retained_energy_fractions(coefficients: np.ndarray, ablation: str, component: str, chunk_size: int = 64) -> dict[str, float]:
    if ablation not in DCT_ABLATIONS:
        raise ValueError(f"unknown JPEG DCT ablation: {ablation}")
    if component not in DCT_COMPONENTS:
        raise ValueError(f"unknown JPEG DCT component: {component}")
    if coefficients.ndim != 4 or coefficients.shape[1] != 64:
        raise ValueError("coefficient arrays must have shape [samples, 64, block_y, block_x]")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    channel_mask = np.zeros(64, dtype=bool)
    if ablation == "full" or (ablation == "y_only" and component == "y_coefficients"):
        channel_mask[:] = True
    elif ablation == "dc_only":
        channel_mask[0] = True
    elif ablation == "low4x4":
        channel_mask[[u * 8 + v for u in range(4) for v in range(4)]] = True
    retained_positions = int(channel_mask.sum())
    total_abs = retained_abs = total_energy = retained_energy = 0.0
    for start in range(0, coefficients.shape[0], chunk_size):
        chunk = np.asarray(coefficients[start : start + chunk_size])
        total_abs += float(np.abs(chunk).sum(dtype=np.float64))
        total_energy += float(np.square(chunk, dtype=np.float64).sum(dtype=np.float64))
        if retained_positions:
            selected = np.take(chunk, np.flatnonzero(channel_mask), axis=1)
            retained_abs += float(np.abs(selected).sum(dtype=np.float64))
            retained_energy += float(np.square(selected, dtype=np.float64).sum(dtype=np.float64))
    return {
        "retained_coefficient_fraction": retained_positions / 64.0,
        "retained_absolute_magnitude_fraction": retained_abs / total_abs if total_abs else 0.0,
        "retained_squared_energy_fraction": retained_energy / total_energy if total_energy else 0.0,
    }


def compare_codec_rankings(reference: dict[str, float], candidate: dict[str, float]) -> dict:
    if set(reference) != set(candidate) or len(reference) < 2:
        raise ValueError("ranking inputs must contain the same two or more codecs")
    codecs = sorted(reference)
    reference_values = np.asarray([reference[codec] for codec in codecs], dtype=np.float64)
    candidate_values = np.asarray([candidate[codec] for codec in codecs], dtype=np.float64)
    return {
        "codecs": codecs,
        "reference_order": sorted(codecs, key=lambda codec: (-reference[codec], codec)),
        "candidate_order": sorted(codecs, key=lambda codec: (-candidate[codec], codec)),
        "spearman_rank_correlation": float(spearmanr(reference_values, candidate_values).statistic),
        "kendall_rank_correlation": float(kendalltau(reference_values, candidate_values).statistic),
    }


def paired_gap_summary(scores: dict[str, list[float] | tuple[float, ...]], focal: str = "jpeg_q10") -> dict[str, dict]:
    if focal not in scores or len(scores[focal]) < 2:
        raise ValueError("the focal score must have at least two paired seeds")
    focal_values = np.asarray(scores[focal], dtype=np.float64)
    summaries = {}
    for codec, values in scores.items():
        if codec == focal:
            continue
        paired = np.asarray(values, dtype=np.float64)
        if paired.shape != focal_values.shape:
            raise ValueError(f"{codec} scores must align with the focal seeds")
        differences = focal_values - paired
        summaries[codec] = {
            "differences": differences.tolist(),
            "mean": float(differences.mean()),
            "sample_std": float(differences.std(ddof=1)),
            "positive_seed_count": int(np.sum(differences > 0)),
            "seed_count": int(len(differences)),
        }
    return summaries
