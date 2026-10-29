from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr


_UNDECODED = {
    "entropy_syntax_decoded": False,
    "transform_coefficients_decoded": False,
    "inverse_transform_complete": False,
    "intra_prediction_applied": False,
    "pixel_reconstruction_complete": False,
    "loop_filters_complete": False,
    "rgb_conversion_complete": False,
}


def stage_operation_metadata(condition: str, stage: str) -> dict:
    family = condition.split("_", 1)[0]
    row = {"condition": condition, "stage": stage, **_UNDECODED}
    if stage == "raw_bytes":
        return {**row, "available": True, "boundary": "compressed stream bytes in memory", "api": "none", "early_exit": True, "extra_capture_work": False}
    if family == "jxl" and stage == "jxl_dc_progressive":
        unknown = {field: None for field in _UNDECODED}
        return {**row, **unknown, "available": False, "boundary": "JXL_DEC_FRAME_PROGRESSION / JxlDecoderFlushImage", "api": "public libjxl API", "early_exit": True, "extra_capture_work": False, "availability_note": "not emitted by the exact cached streams"}
    if family == "jxl" and stage == "decoded_rgb":
        return {**row, "entropy_syntax_decoded": True, "transform_coefficients_decoded": True, "inverse_transform_complete": True, "pixel_reconstruction_complete": True, "rgb_conversion_complete": True, "available": True, "boundary": "JXL_DEC_SUCCESS with full RGB output", "api": "public libjxl API", "early_exit": False, "extra_capture_work": False, "availability_note": "intra prediction and AV1/WebP-style loop filters are not applicable to JPEG XL"}
    if family == "webp" and stage == "pre_reconstruction_coefficients":
        return {**row, "entropy_syntax_decoded": True, "transform_coefficients_decoded": True, "available": True, "boundary": "VP8ProcessRow after final ParseResiduals, before ReconstructRow/FinishRow", "api": "libwebp source patch callback", "early_exit": True, "extra_capture_work": True, "availability_note": "Y2 WHT/DC handling has occurred; spatial IDCT and prediction application have not"}
    if family == "avif" and stage == "pre_reconstruction_coefficients":
        return {**row, "entropy_syntax_decoded": True, "transform_coefficients_decoded": True, "available": True, "boundary": "libaom parse-only tile visitor before reconstruction and frame wrapup", "api": "libaom source patch callback via libavif", "early_exit": True, "extra_capture_work": True, "availability_note": "AVIF container and AV1 frame/tile syntax are parsed; prediction, inverse transform, reconstruction and filters have not run"}
    if family in ("webp", "avif") and stage == "decoded_yuv":
        return {**row, "entropy_syntax_decoded": True, "transform_coefficients_decoded": True, "inverse_transform_complete": True, "intra_prediction_applied": True, "pixel_reconstruction_complete": True, "loop_filters_complete": True, "available": True, "boundary": "full native YUV output returned", "api": "public decoder API", "early_exit": False, "extra_capture_work": False}
    if family in ("webp", "avif") and stage == "decoded_rgb":
        return {**row, "entropy_syntax_decoded": True, "transform_coefficients_decoded": True, "inverse_transform_complete": True, "intra_prediction_applied": True, "pixel_reconstruction_complete": True, "loop_filters_complete": True, "rgb_conversion_complete": True, "available": True, "boundary": "full RGB output returned", "api": "public decoder API", "early_exit": False, "extra_capture_work": False}
    raise ValueError(f"unknown Phase 4.7 stage: {condition}/{stage}")


def summarize_timing(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    adapter_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["stage"])].append(float(row["decoder_ms"]))
        for field in ("capture_adapter_ms", "model_adapter_ms", "cumulative_decoder_ms", "cumulative_decoder_adapter_ms"):
            if row.get(field) is not None:
                adapter_groups[(row["condition"], row["stage"], field)].append(float(row[field]))
    summaries = []
    for (condition, stage), values in sorted(groups.items()):
        summary = {
            "condition": condition,
            "stage": stage,
            "n": len(values),
            "median_decoder_ms": float(np.median(values)),
            "p95_decoder_ms": float(np.percentile(values, 95)),
        }
        for (adapter_condition, adapter_stage, field), adapter_values in adapter_groups.items():
            if (adapter_condition, adapter_stage) == (condition, stage):
                summary[f"median_{field}"] = float(np.median(adapter_values))
                summary[f"p95_{field}"] = float(np.percentile(adapter_values, 95))
        summaries.append(summary)
    return summaries


def semantic_exposure_depth(
    stages: list[dict], full_rgb_ms: float, thresholds: tuple[float, ...] = (0.5, 0.75, 0.9)
) -> list[dict]:
    pre_rgb = sorted((row for row in stages if row["stage"] != "decoded_rgb"), key=lambda row: row["cumulative_ms"])
    result = []
    for metric in ("relative_u_probe_utility", "relative_accuracy"):
        for threshold in thresholds:
            reached = next((row for row in pre_rgb if row[metric] >= threshold), None)
            result.append(
                {
                    "metric": metric,
                    "threshold": threshold,
                    "status": "reached" if reached else "not_reached_before_rgb",
                    "stage": reached["stage"] if reached else None,
                    "milliseconds": float(reached["cumulative_ms"]) if reached else None,
                    "fraction_of_full_rgb": float(reached["cumulative_ms"] / full_rgb_ms) if reached and full_rgb_ms > 0 else None,
                }
            )
    return result


def semantic_exposure_auc(stages: list[dict], full_decode_ms: float) -> dict:
    ordered = sorted(stages, key=lambda row: row["cumulative_ms"])
    x = np.asarray([min(1.0, max(0.0, row["cumulative_ms"] / full_decode_ms)) for row in ordered], dtype=np.float64)
    utility = np.asarray([row["relative_u_probe_utility"] for row in ordered], dtype=np.float64)
    accuracy = np.asarray([row["relative_accuracy"] for row in ordered], dtype=np.float64)
    return {
        "u_probe_auc": float(np.trapezoid(utility, x)),
        "accuracy_auc": float(np.trapezoid(accuracy, x)),
        "normalized_cost_end": float(x[-1]) if len(x) else None,
        "n_stages": int(len(x)),
    }


def stage_utility_gain(stages: list[dict]) -> list[dict]:
    ordered = sorted(stages, key=lambda row: row.get("stage_order", row["cumulative_ms"]))
    result = []
    for left, right in zip(ordered, ordered[1:]):
        delta_ms = float(right["cumulative_ms"] - left["cumulative_ms"])
        result.append(
            {
                "from_stage": left["stage"],
                "to_stage": right["stage"],
                "delta_ms": delta_ms,
                "delta_u_probe_per_ms": float((right["u_probe_bits"] - left["u_probe_bits"]) / delta_ms) if delta_ms > 0 else None,
                "delta_accuracy_per_ms": float((right["accuracy"] - left["accuracy"]) / delta_ms) if delta_ms > 0 else None,
            }
        )
    return result


def rate_distortion_residuals(rows: list[dict]) -> list[dict]:
    log_bpp = np.log2(np.asarray([row["bpp"] for row in rows], dtype=np.float64))
    psnr = np.asarray([row["psnr"] for row in rows], dtype=np.float64)
    ssim = np.asarray([row["ssim"] for row in rows], dtype=np.float64)
    psnr_fit = np.polyfit(log_bpp, psnr, 1)
    ssim_fit = np.polyfit(log_bpp, ssim, 1)
    psnr_residual = psnr - np.polyval(psnr_fit, log_bpp)
    ssim_residual = ssim - np.polyval(ssim_fit, log_bpp)
    return [
        {
            **row,
            "rd_efficiency_residual": float(psnr_residual[index]),
            "ssim_efficiency_residual": float(ssim_residual[index]),
        }
        for index, row in enumerate(rows)
    ]


def descriptive_correlation(rows: list[dict], x_name: str, y_name: str) -> dict:
    valid = [(float(row[x_name]), float(row[y_name])) for row in rows if row.get(x_name) is not None and row.get(y_name) is not None]
    if len(valid) < 3:
        return {"x": x_name, "y": y_name, "n": len(valid), "pearson_r": None, "spearman_rho": None}
    x, y = np.asarray(valid, dtype=np.float64).T
    return {
        "x": x_name,
        "y": y_name,
        "n": int(len(valid)),
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_rho": float(spearmanr(x, y).statistic),
    }
