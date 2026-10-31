from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.analysis.phase45b import paired_correctness
from tamtd.analysis.phase47 import descriptive_correlation, rate_distortion_residuals, semantic_exposure_auc, semantic_exposure_pareto
from tamtd.analysis.phase48 import within_image_correlation


OUTPUT = PROJECT_ROOT / "results/Phase 4.8 Extreme Compression"
PHASE45B = PROJECT_ROOT / "results/Phase 4.5B Analysis"
PHASE47 = PROJECT_ROOT / "results/Phase 4.7 Standard Codec Stages"
NEW_CONDITIONS = ("jpeg_q1", "jpeg_q5", "webp_q5", "webp_q10", "webp_q20", "avif_q5", "avif_q10", "avif_q20")
HISTORICAL_CONDITIONS = ("jpeg_q10", "jpeg_q20", "jpeg_q60", "jpeg_q80", "webp_q30", "webp_q75", "avif_q40", "avif_q60")
CONDITION_ORDER = {
    "jpeg": ("jpeg_q1", "jpeg_q5", "jpeg_q10", "jpeg_q20", "jpeg_q60", "jpeg_q80"),
    "webp": ("webp_q5", "webp_q10", "webp_q20", "webp_q30", "webp_q75"),
    "avif": ("avif_q5", "avif_q10", "avif_q20", "avif_q40", "avif_q60"),
}
COLORS = {"jpeg": "#4c78a8", "webp": "#f58518", "avif": "#54a24b"}
MARKERS = {"jpeg": "o", "webp": "s", "avif": "^"}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_gzip_csv(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_gzip_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _num(value, default=None):
    if value in (None, ""):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None


def _correlation(rows: list[dict], x_field: str, y_field: str, group: str) -> dict:
    values = [(_num(row.get(x_field)), _num(row.get(y_field))) for row in rows]
    values = [(x, y) for x, y in values if x is not None and y is not None]
    x, y = np.asarray(values, dtype=np.float64).T if values else (np.asarray([]), np.asarray([]))
    return {
        "group": group,
        "x": x_field,
        "y": y_field,
        "n_operating_points": len(values),
        "pearson_r": float(pearsonr(x, y).statistic) if len(values) >= 3 and np.std(x) and np.std(y) else None,
        "spearman_rho": float(spearmanr(x, y).statistic) if len(values) >= 3 and np.std(x) and np.std(y) else None,
        "interpretation": "descriptive; operating points are selected settings, not independent codec replications",
    }


def _load_operating_points() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    historical_rd = _read_gzip_csv(PHASE45B / "imagenette_rd_per_sample.csv.gz")
    historical_raw = _read_csv(PHASE45B / "raw_byte_accessibility_table.csv")
    historical_raw_seed = _read_csv(PHASE45B / "probe_utility_summary.csv")
    historical_rgb_seed = _read_csv(PHASE45B / "decoded_rgb_endpoints.csv")
    new_raw = {}
    for condition in NEW_CONDITIONS:
        result_path = OUTPUT / "per_seed/seed_42" / f"{condition}__codec_file_bytes.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["test_metrics"]["n"] != 3925 or result["truncation"]["truncated_samples"] != 0:
            raise ValueError(f"incomplete or truncated raw-byte result for {condition}")
        new_raw[condition] = result

    rd_by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in historical_rd:
        if row["split"] == "test" and row["condition"] in HISTORICAL_CONDITIONS:
            rd_by_condition[row["condition"]].append(row)
    for row in _read_gzip_csv(OUTPUT / "new_test_rd_per_sample.csv.gz"):
        rd_by_condition[row["condition"]].append(row)

    raw_summary = {row["condition"]: row for row in historical_raw if row["representation"] in {"codec_file_bytes", "jpeg_file_bytes"}}
    historical_raw_seed42 = {
        row["condition"]: row for row in historical_raw_seed
        if row["representation"] in {"codec_file_bytes", "jpeg_file_bytes"} and int(row["seed"]) == 42
    }
    rgb_seed42 = {
        row["condition"]: row for row in historical_rgb_seed
        if row["split"] == "test" and int(row["seed"]) == 42 and row["representation"] == "decoded_rgb_reference"
    }
    new_rgb_rows = _read_csv(OUTPUT / "decoded_rgb_reference_per_seed.csv")
    rgb_rows_by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in new_rgb_rows:
        if row["stage"] == "decoded_rgb":
            rgb_rows_by_condition[row["condition"]].append(row)
            if int(row["seed"]) == 42:
                rgb_seed42[row["condition"]] = row

    operating_points = []
    per_image_rd = {}
    for condition in (*HISTORICAL_CONDITIONS, *NEW_CONDITIONS):
        samples = rd_by_condition[condition]
        if len(samples) != 3925:
            raise ValueError(f"expected 3,925 test rate-distortion rows for {condition}, found {len(samples)}")
        per_image_rd[condition] = {row["image_id"]: row for row in samples}
        summary = {}
        for field in ("bpp", "psnr_db", "ssim"):
            numbers = [_num(row.get(field)) for row in samples]
            numbers = [value for value in numbers if value is not None]
            mean, std = _mean_std(numbers)
            summary[field] = mean
            summary[f"{field}_sample_std"] = std
        family, quality = condition.split("_q", 1)
        historical = condition in HISTORICAL_CONDITIONS
        if historical:
            raw_aggregate = raw_summary[condition]
            raw_seed = historical_raw_seed42[condition]
            raw_accuracy_mean = _num(raw_aggregate["raw_test_accuracy_mean"])
            raw_accuracy_std = _num(raw_aggregate["raw_test_accuracy_std"])
            raw_u_mean = _num(raw_aggregate["raw_u_probe_bits_mean"])
            raw_u_std = _num(raw_aggregate["raw_u_probe_bits_std"])
            raw_seed_count = int(raw_aggregate["seed_count"])
            raw_accuracy_seed42 = float(raw_seed["accuracy"])
            raw_u_seed42 = float(raw_seed["u_probe_bits"])
        else:
            metrics = new_raw[condition]["test_metrics"]
            raw_accuracy_mean = raw_accuracy_seed42 = float(metrics["accuracy"])
            raw_u_mean = raw_u_seed42 = float(metrics["u_probe_bits"])
            raw_accuracy_std = raw_u_std = None
            raw_seed_count = 1
        rgb_seed = rgb_seed42.get(condition)
        if rgb_seed is None:
            raise ValueError(f"missing decoded RGB seed-42 endpoint for {condition}")
        if historical:
            rgb_summary = next(row for row in _read_csv(PHASE45B / "decoded_rgb_endpoint_summary.csv") if row["condition"] == condition and row["split"] == "test")
            rgb_mean = float(rgb_summary["accuracy_mean"])
            rgb_std = _num(rgb_summary["accuracy_sample_std"])
            rgb_seed_count = int(rgb_summary["seed_count"])
            rgb_seed_accuracy = float(rgb_seed["accuracy"])
        else:
            rgb_values = rgb_rows_by_condition[condition]
            rgb_mean, rgb_std = _mean_std([float(row["accuracy"]) for row in rgb_values])
            rgb_seed_count = len(rgb_values)
            rgb_seed_accuracy = float(rgb_seed["accuracy"])
        operating_points.append({
            "condition": condition,
            "codec_family": family,
            "quality": int(quality),
            "historical_or_new": "historical" if historical else "new_extreme",
            "raw_seed_count": raw_seed_count,
            "raw_accuracy_mean": raw_accuracy_mean,
            "raw_accuracy_seed_std": raw_accuracy_std,
            "raw_accuracy_seed42": raw_accuracy_seed42,
            "raw_u_probe_bits_mean": raw_u_mean,
            "raw_u_probe_bits_seed_std": raw_u_std,
            "raw_u_probe_bits_seed42": raw_u_seed42,
            "decoded_rgb_seed_count": rgb_seed_count,
            "decoded_rgb_accuracy_mean": rgb_mean,
            "decoded_rgb_accuracy_seed_std": rgb_std,
            "decoded_rgb_accuracy_seed42": rgb_seed_accuracy,
            **summary,
        })
    return operating_points, per_image_rd, {"raw_summary": raw_summary, "rgb_rows_by_condition": rgb_rows_by_condition}


def _load_predictions() -> dict[str, dict[str, dict]]:
    output: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in _read_gzip_csv(PHASE45B / "test_predictions_summary.csv.gz"):
        if row["representation"] in {"codec_file_bytes", "jpeg_file_bytes"} and row["model_family"] in {"byte", "byte_cnn"} and int(row["seed"]) == 42:
            output[row["model_condition"]][row["image_id"]] = row
    for condition in NEW_CONDITIONS:
        for row in _read_gzip_csv(OUTPUT / f"predictions/{condition}_seed42.csv.gz"):
            output[condition][row["image_id"]] = row
    return output


def _paired_and_within_image(operating_points: list[dict], rd: dict[str, dict[str, dict]]) -> tuple[list[dict], list[dict]]:
    predictions = _load_predictions()
    paired_rows = []
    within_rows = []
    for family, conditions in CONDITION_ORDER.items():
        for lower, higher in zip(conditions, conditions[1:]):
            pred_a, pred_b = predictions[lower], predictions[higher]
            ids = sorted(set(pred_a) & set(pred_b))
            labels = np.asarray([int(pred_a[image_id]["true_label"]) for image_id in ids], dtype=np.int64)
            paired = paired_correctness(
                labels,
                np.asarray([int(pred_a[image_id]["predicted_label"]) for image_id in ids]),
                np.asarray([int(pred_b[image_id]["predicted_label"]) for image_id in ids]),
                bootstrap_seed=4810 + len(paired_rows),
                bootstrap_replicates=2000,
            )
            rd_a, rd_b = rd[lower], rd[higher]
            aligned = [image_id for image_id in ids if image_id in rd_a and image_id in rd_b]
            output = {
                "codec_family": family,
                "lower_quality_condition": lower,
                "higher_quality_condition": higher,
                "paired_seed": 42,
                **paired,
                "mean_delta_bpp_higher_minus_lower": float(np.mean([float(rd_b[i]["bpp"]) - float(rd_a[i]["bpp"]) for i in aligned])),
                "mean_delta_psnr_db_higher_minus_lower": float(np.mean([float(rd_b[i]["psnr_db"]) - float(rd_a[i]["psnr_db"]) for i in aligned])),
                "mean_delta_ssim_higher_minus_lower": float(np.mean([float(rd_b[i]["ssim"]) - float(rd_a[i]["ssim"]) for i in aligned])),
                "paired_rate_distortion_images": len(aligned),
                "note": "quality-ordered paired test predictions; descriptive across independently trained per-setting probes",
            }
            paired_rows.append(output)

        rows = []
        for condition in conditions:
            for image_id, prediction in predictions[condition].items():
                rate_distortion = rd[condition][image_id]
                rows.append({
                    "image_id": image_id,
                    "bpp": float(rate_distortion["bpp"]),
                    "psnr_db": float(rate_distortion["psnr_db"]),
                    "ssim": float(rate_distortion["ssim"]),
                    "correct": int(prediction["correct"]),
                    "cross_entropy_nats": float(prediction["cross_entropy_nats"]),
                })
        for x_field, y_field in (("bpp", "correct"), ("psnr_db", "correct"), ("ssim", "correct"), ("bpp", "cross_entropy_nats")):
            result = within_image_correlation(rows, x_field, y_field, replicates=1000, seed=4800 + len(within_rows))
            within_rows.append({"codec_family": family, "x": x_field, "y": y_field, **result, "note": "each image is centered across that codec's measured settings; bootstrap resamples image identities"})
    return paired_rows, within_rows


def _correlations(operating_points: list[dict]) -> list[dict]:
    output = []
    for family in ("all", "jpeg", "webp", "avif"):
        rows = operating_points if family == "all" else [row for row in operating_points if row["codec_family"] == family]
        for x_field in ("bpp", "psnr_db", "ssim"):
            for y_field in ("raw_accuracy_seed42", "raw_u_probe_bits_seed42", "decoded_rgb_accuracy_seed42"):
                output.append(_correlation(rows, x_field, y_field, family))
    return output


def _matched_points(operating_points: list[dict]) -> tuple[list[dict], list[dict]]:
    adequate_rows = []
    candidates = []
    families = ("jpeg", "webp", "avif")
    for family_index, family_a in enumerate(families):
        for family_b in families[family_index + 1:]:
            rows_a = [row for row in operating_points if row["codec_family"] == family_a]
            rows_b = [row for row in operating_points if row["codec_family"] == family_b]
            pairs = [(a, b) for a in rows_a for b in rows_b]
            for basis in ("rate", "distortion"):
                if basis == "rate":
                    left, right = min(pairs, key=lambda pair: abs(math.log(pair[0]["bpp"] / pair[1]["bpp"])))
                    error = abs(left["bpp"] - right["bpp"]) / max(left["bpp"], right["bpp"])
                    adequate = error <= 0.10
                    error_name, threshold = "relative_bpp_difference", 0.10
                else:
                    left, right = min(pairs, key=lambda pair: abs(pair[0]["psnr_db"] - pair[1]["psnr_db"]))
                    error = abs(left["psnr_db"] - right["psnr_db"])
                    adequate = error <= 0.5
                    error_name, threshold = "absolute_psnr_difference_db", 0.5
                row = {
                    "codec_a": family_a,
                    "condition_a": left["condition"],
                    "codec_b": family_b,
                    "condition_b": right["condition"],
                    "matching_basis": basis,
                    "bpp_a": left["bpp"],
                    "bpp_b": right["bpp"],
                    "psnr_db_a": left["psnr_db"],
                    "psnr_db_b": right["psnr_db"],
                    "ssim_a": left["ssim"],
                    "ssim_b": right["ssim"],
                    error_name: error,
                    "match_threshold": threshold,
                    "adequate_match": adequate,
                    "raw_accuracy_seed42_a": left["raw_accuracy_seed42"],
                    "raw_accuracy_seed42_b": right["raw_accuracy_seed42"],
                    "raw_accuracy_delta_a_minus_b": left["raw_accuracy_seed42"] - right["raw_accuracy_seed42"],
                    "raw_u_probe_bits_seed42_a": left["raw_u_probe_bits_seed42"],
                    "raw_u_probe_bits_seed42_b": right["raw_u_probe_bits_seed42"],
                    "raw_u_probe_delta_bits_a_minus_b": left["raw_u_probe_bits_seed42"] - right["raw_u_probe_bits_seed42"],
                    "rgb_accuracy_seed42_a": left["decoded_rgb_accuracy_seed42"],
                    "rgb_accuracy_seed42_b": right["decoded_rgb_accuracy_seed42"],
                    "note": "observed settings only; no neural accessibility interpolation",
                }
                candidates.append(row)
                if adequate:
                    adequate_rows.append(row)
    return adequate_rows, candidates


def _phase47_corrections() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    probe_rows = _read_csv(PHASE47 / "stage_probe_per_seed.csv")
    timing_rows = _read_csv(PHASE47 / "decoder_timing_summary.csv")
    timing = {(row["condition"], row["stage"]): row for row in timing_rows}
    by_group: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in probe_rows:
        by_group[(row["condition"], int(row["seed"]))].append(row)

    per_seed = []
    pareto_rows = []
    corrected_conditions = sorted({row["condition"] for row in probe_rows})
    for (condition, seed), measured in sorted(by_group.items()):
        points = []
        for row in measured:
            stage = row["stage"]
            if stage == "raw_bytes":
                cost = 0.0
            else:
                timing_row = timing.get((condition, stage))
                if timing_row is None:
                    continue
                cost = float(timing_row["median_cumulative_decoder_adapter_ms"] or timing_row["median_decoder_ms"])
            points.append({
                **row,
                "cumulative_ms": cost,
                "relative_u_probe_utility": _num(row.get("relative_u_probe_utility")),
                "relative_accuracy": _num(row.get("relative_accuracy")),
            })
        full = next((point for point in points if point["stage"] == "decoded_rgb"), None)
        pre = next((point for point in points if point["stage"] == "pre_reconstruction_coefficients"), None)
        auc = semantic_exposure_auc(
            points,
            float(full["cumulative_ms"]) if full else 0,
            ("pre_reconstruction_coefficients", "decoded_rgb"),
        )
        per_seed.append({
            "condition": condition,
            "seed": seed,
            "full_rgb_ms": float(full["cumulative_ms"]) if full else None,
            "pre_reconstruction_ms": float(pre["cumulative_ms"]) if pre else None,
            "measured_stages": ";".join(point["stage"] for point in points),
            **auc,
            "raw_stage_in_auc": False,
            "raw_stage_exclusion_reason": "raw-byte result is available only for seed 42 and uses a different probe family",
        })
        for utility_field in ("u_probe_bits", "accuracy"):
            for point in semantic_exposure_pareto(points, utility_field):
                pareto_rows.append({
                    "condition": condition,
                    "seed": seed,
                    "utility_field": utility_field,
                    "stage": point["stage"],
                    "probe_family": point.get("model_family", point.get("probe_family")),
                    "cumulative_ms": point["cost_ms"],
                    "utility": point["utility"],
                    "nondominated": point["nondominated"],
                    "measured": True,
                })

    summary = []
    for condition in corrected_conditions:
        rows = [row for row in per_seed if row["condition"] == condition]
        valid = [row for row in rows if row["status"] == "available"]
        for metric in ("u_probe_auc", "accuracy_auc"):
            mean, std = _mean_std([float(row[metric]) for row in valid])
            summary.append({
                "condition": condition,
                "metric": metric,
                "status": "available" if valid else "unavailable",
                "mean": mean,
                "sample_std": std,
                "available_seed_count": len(valid),
                "total_seed_count": len(rows),
                "endpoint_convention": "pre_reconstruction_coefficients -> decoded_rgb; identical endpoints required for every seed",
                "interpolation": "linear between the two measured endpoints; descriptive normalized area, not a causal quantity",
                "unavailable_seeds": ";".join(str(row["seed"]) for row in rows if row["status"] != "available"),
            })

    old_auc = _read_csv(PHASE47 / "semantic_exposure_auc.csv")
    old_u_means = {row["condition"]: row for row in old_auc if row["seed"] == "mean" and row.get("u_probe_auc") not in (None, "")}
    old_accuracy_means = {row["condition"]: row for row in old_auc if row["seed"] == "mean" and row.get("accuracy_auc") not in (None, "")}
    supersession = []
    for condition in corrected_conditions:
        old_u = old_u_means.get(condition, {})
        old_accuracy = old_accuracy_means.get(condition, {})
        new = {row["metric"]: row for row in summary if row["condition"] == condition}
        supersession.append({
            "condition": condition,
            "historical_phase47_auc_mean_u": old_u.get("u_probe_auc"),
            "historical_phase47_auc_mean_accuracy": old_accuracy.get("accuracy_auc"),
            "historical_phase47_seed_count": old_u.get("n_seeds", old_accuracy.get("n_seeds")),
            "corrected_u_auc": new["u_probe_auc"]["mean"],
            "corrected_u_auc_status": new["u_probe_auc"]["status"],
            "corrected_u_auc_seed_count": new["u_probe_auc"]["available_seed_count"],
            "corrected_accuracy_auc": new["accuracy_auc"]["mean"],
            "corrected_accuracy_auc_status": new["accuracy_auc"]["status"],
            "corrected_accuracy_auc_seed_count": new["accuracy_auc"]["available_seed_count"],
            "historical_outputs_preserved": True,
        })

    rd_raw = [row for row in _read_csv(PHASE45B / "raw_byte_accessibility_table.csv") if row["condition"] in corrected_conditions and row["representation"] in {"codec_file_bytes", "jpeg_file_bytes"}]
    rd_inputs = [{
        "condition": row["condition"],
        "codec_family": row["codec_family"],
        "bpp": float(row["imagenette_bpp_mean"]),
        "psnr": float(row["imagenette_psnr_mean_db"]),
        "ssim": float(row["imagenette_ssim_mean"]),
        "raw_accuracy": float(row["raw_test_accuracy_mean"]),
        "raw_u_probe_bits": float(row["raw_u_probe_bits_mean"]),
    } for row in rd_raw]
    residuals = rate_distortion_residuals(rd_inputs)
    auc_by_condition = {row["condition"]: row for row in summary if row["metric"] == "u_probe_auc"}
    corrected_rd = []
    for row in residuals:
        corrected_rd.append({
            **row,
            "corrected_u_probe_auc": auc_by_condition[row["condition"]]["mean"],
            "corrected_u_probe_auc_status": auc_by_condition[row["condition"]]["status"],
            "corrected_u_probe_auc_seed_count": auc_by_condition[row["condition"]]["available_seed_count"],
        })
    correlations = []
    for y_field in ("raw_u_probe_bits", "raw_accuracy", "corrected_u_probe_auc"):
        valid = [row for row in corrected_rd if row.get(y_field) is not None]
        if y_field == "corrected_u_probe_auc":
            valid = [row for row in valid if row["corrected_u_probe_auc_status"] == "available"]
        correlations.append({
            **descriptive_correlation(valid, "rd_efficiency_residual", y_field),
            "rd_efficiency_residual": "PSNR residual from descriptive OLS on log2(bpp)",
            "note": "exploratory; no cross-seed inference when fewer than three comparable conditions are available",
        })

    avif_costs = []
    for condition in ("avif_q40", "avif_q60"):
        stages = {stage: timing[(condition, stage)] for stage in ("pre_reconstruction_coefficients", "decoded_yuv", "decoded_rgb")}
        early = float(stages["pre_reconstruction_coefficients"]["median_cumulative_decoder_adapter_ms"])
        yuv = float(stages["decoded_yuv"]["median_decoder_ms"])
        rgb = float(stages["decoded_rgb"]["median_decoder_ms"])
        avif_costs.append({
            "condition": condition,
            "pre_reconstruction_decoder_plus_adapters_ms": early,
            "full_yuv_decode_ms": yuv,
            "full_rgb_decode_ms": rgb,
            "early_savings_vs_yuv_percent": 100 * (yuv - early) / yuv,
            "early_savings_vs_rgb_percent": 100 * (rgb - early) / rgb,
            "yuv_inference_measured": False,
            "interpretation": "timing comparison only; no YUV classifier or matched YUV inference was measured",
        })
    return per_seed, summary, pareto_rows, supersession, corrected_rd, correlations, avif_costs


def _structured_outputs() -> tuple[list[dict], list[dict]]:
    comparison = []
    availability = []
    dct_path = OUTPUT / "per_seed/seed_42/jpeg_q1__jpeg_dct_ir.json"
    if dct_path.is_file():
        dct = json.loads(dct_path.read_text(encoding="utf-8"))
        byte_result = json.loads((OUTPUT / "per_seed/seed_42/jpeg_q1__codec_file_bytes.json").read_text(encoding="utf-8"))
        rgb = next(row for row in _read_csv(OUTPUT / "decoded_rgb_reference_per_seed.csv") if row["condition"] == "jpeg_q1" and int(row["seed"]) == 42)
        extraction = json.loads((OUTPUT / "jpeg_q1_dct_feature_cache.json").read_text(encoding="utf-8"))
        calibration = next(row for row in _read_csv(OUTPUT / "calibration_summary.csv") if row["condition"] == "jpeg_q1")
        byte_metrics = byte_result["test_metrics"]
        dct_metrics = dct["test_metrics"]
        rows = (
            ("raw_bytes", "compressed JPEG byte stream", byte_metrics["accuracy"], byte_metrics["u_probe_bits"], 0.0, 0.0, byte_metrics["inference_model_forward_ms_per_sample"], "raw ByteStreamCNN, seed 42"),
            ("pre_reconstruction_dct", "quantized DCT coefficients + quantization tables", dct_metrics["accuracy"], dct_metrics["u_probe_bits"], extraction["timing_summary"]["pre_reconstruction_decoder_plus_adapter_ms"]["median"], extraction["timing_summary"]["pre_reconstruction_decoder_plus_adapter_ms"]["p95"], dct["inference_model_forward_ms_per_sample"], "JPEGDCTNet, seed 42; pyjpegio entropy decode, canonical IR, model-array adapter"),
            ("decoded_rgb", "fully decoded RGB", float(rgb["accuracy"]), float(rgb["u_probe_bits"]), float(calibration["decode_ms_median"]), float(calibration["decode_ms_p95"]), float(rgb["inference_model_forward_ms_per_sample"]), "frozen RGBResNet18 seed 42; decode timing from 200-image calibration subset"),
        )
        for stage, representation, accuracy, utility, extraction_median, extraction_p95, inference_ms, note in rows:
            comparison.append({
                "condition": "jpeg_q1",
                "stage": stage,
                "status": "measured",
                "representation": representation,
                "seed": 42,
                "accuracy": accuracy,
                "u_probe_bits": utility,
                "extraction_or_decode_median_ms": extraction_median,
                "extraction_or_decode_p95_ms": extraction_p95,
                "model_forward_ms_per_sample": inference_ms,
                "sum_of_median_preprocess_and_inference_ms": extraction_median + inference_ms,
                "note": note,
            })
        availability.append({"codec_family": "jpeg", "condition": "jpeg_q1", "status": "completed", "representation": "quantized DCT coefficients and quantization tables", "reason": "existing pyjpegio/libjpeg coefficient reader and JPEGDCTNet supported the new stream layout"})
    else:
        availability.append({"codec_family": "jpeg", "condition": "jpeg_q1", "status": "not_run", "representation": "quantized DCT coefficients", "reason": "feature cache or training result missing"})
    for family, condition in (("webp", "webp_q5"), ("avif", "avif_q5")):
        availability.append({
            "codec_family": family,
            "condition": condition,
            "status": "not_run",
            "representation": "pre-reconstruction coefficient/token grid",
            "reason": "Phase 4.7 source instrumentation exists, but its compiled libwebp/libaom/libavif binaries are absent from the expected build root; rebuilding the native stack is outside this focused pass",
        })
    return comparison, availability


def _plot_operating_points(operating_points: list[dict]) -> None:
    for y_field, y_label, filename in (
        ("raw_accuracy_seed42", "Raw-byte test accuracy (seed 42)", "raw_byte_accuracy_vs_bpp.png"),
        ("raw_u_probe_bits_seed42", "Raw-byte U-probe bits (seed 42)", "raw_byte_u_probe_vs_bpp.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5.3), constrained_layout=True)
        for family, conditions in CONDITION_ORDER.items():
            rows = sorted((row for row in operating_points if row["codec_family"] == family), key=lambda row: row["quality"])
            ax.plot([row["bpp"] for row in rows], [row[y_field] for row in rows], color=COLORS[family], alpha=0.48, linewidth=1.1)
            for row in rows:
                ax.scatter(row["bpp"], row[y_field], color=COLORS[family], marker=MARKERS[family], s=48,
                           edgecolor="#222222" if row["historical_or_new"] == "new_extreme" else "none", linewidth=1.0,
                           label=family if row["condition"] == rows[0]["condition"] else None)
                if y_field == "raw_accuracy_seed42" and row["raw_seed_count"] > 1 and row["raw_accuracy_seed_std"] is not None:
                    ax.errorbar(row["bpp"], row[y_field], yerr=row["raw_accuracy_seed_std"], color=COLORS[family], capsize=2, linewidth=0.8)
                ax.annotate(row["condition"].replace(family + "_", ""), (row["bpp"], row[y_field]), xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Measured test-set bits per pixel")
        ax.set_ylabel(y_label)
        ax.set_title("Raw-byte accessibility across observed codec settings")
        ax.grid(alpha=0.22)
        ax.legend(title="Codec", fontsize=8)
        fig.savefig(OUTPUT / filename, dpi=200)
        plt.close(fig)

    for x_field, x_label, filename in (("psnr_db", "Test-set PSNR (dB)", "raw_accessibility_vs_psnr.png"), ("ssim", "Test-set SSIM", "raw_accessibility_vs_ssim.png")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
        for family in CONDITION_ORDER:
            for row in operating_points:
                if row["codec_family"] != family:
                    continue
                axes[0].scatter(row[x_field], row["raw_accuracy_seed42"], color=COLORS[family], marker=MARKERS[family], s=42)
                axes[1].scatter(row[x_field], row["raw_u_probe_bits_seed42"], color=COLORS[family], marker=MARKERS[family], s=42)
                axes[0].annotate(row["condition"], (row[x_field], row["raw_accuracy_seed42"]), xytext=(3, 3), textcoords="offset points", fontsize=6)
                axes[1].annotate(row["condition"], (row[x_field], row["raw_u_probe_bits_seed42"]), xytext=(3, 3), textcoords="offset points", fontsize=6)
        axes[0].set_ylabel("Raw-byte accuracy (seed 42)")
        axes[1].set_ylabel("Raw-byte U-probe bits (seed 42)")
        for ax in axes:
            ax.set_xlabel(x_label)
            ax.grid(alpha=0.2)
        fig.suptitle("Accessibility versus decoded-image distortion; observed points only")
        fig.savefig(OUTPUT / filename, dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    for family in CONDITION_ORDER:
        rows = sorted((row for row in operating_points if row["codec_family"] == family), key=lambda row: row["quality"])
        ax.errorbar([row["quality"] for row in rows], [row["decoded_rgb_accuracy_mean"] for row in rows],
                    yerr=[row["decoded_rgb_accuracy_seed_std"] or 0 for row in rows], color=COLORS[family],
                    marker=MARKERS[family], capsize=2, label=family)
    ax.set_xlabel("Codec quality setting (within family only)")
    ax.set_ylabel("Decoded RGB test accuracy")
    ax.set_title("Decoded RGB accuracy by codec setting (historical/new references)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(OUTPUT / "decoded_rgb_accuracy_vs_quality.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    for family in CONDITION_ORDER:
        rows = sorted((row for row in operating_points if row["codec_family"] == family), key=lambda row: row["bpp"])
        for row in rows:
            new_point = row["historical_or_new"] == "new_extreme"
            ax.errorbar(row["bpp"], row["psnr_db"], xerr=row["bpp_sample_std"], yerr=row["psnr_db_sample_std"],
                        color=COLORS[family], marker=MARKERS[family], markerfacecolor="none" if new_point else COLORS[family],
                        markersize=6, capsize=2, linestyle="none", label=family if row["condition"] == rows[0]["condition"] else None)
            ax.annotate(row["condition"], (row["bpp"], row["psnr_db"]), xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax.set_xlabel("Measured test-set bits per pixel")
    ax.set_ylabel("Test-set PSNR (dB)")
    ax.set_title("Imagenette rate–distortion points; hollow markers are new extreme settings")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(OUTPUT / "rate_distortion_extreme_points.png", dpi=200)
    plt.close(fig)


def _plot_corrected_frontier(per_seed: list[dict], pareto_rows: list[dict], timing_rows: list[dict]) -> None:
    fig, (ax, cost_ax) = plt.subplots(2, 1, figsize=(10, 7.4), gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    probe_rows = _read_csv(PHASE47 / "stage_probe_per_seed.csv")
    costs = {(row["condition"], row["stage"]): float(row["median_cumulative_decoder_adapter_ms"] or row["median_decoder_ms"]) for row in timing_rows if row["stage"] != "raw_bytes"}
    for row in probe_rows:
        condition, stage = row["condition"], row["stage"]
        x = 0.0 if stage == "raw_bytes" else costs.get((condition, stage))
        if x is None:
            continue
        family = condition.split("_", 1)[0]
        marker = {"raw_bytes": "o", "pre_reconstruction_coefficients": "s", "decoded_rgb": "^"}.get(stage, "x")
        ax.scatter(x, float(row["u_probe_bits"]), color=COLORS.get(family, "#888888"), marker=marker, alpha=0.75, s=34)
    ax.set_ylabel("Measured U-probe bits")
    ax.set_xlabel("Cumulative decoder + required adapter time (ms)")
    ax.set_title("Phase 4.7 corrected measured points; no connecting curve")
    ax.grid(alpha=0.2)
    ax.text(0.01, 0.98, "Circle raw bytes · square pre-reconstruction · triangle decoded RGB\nSeed-42 raw-byte points use a different probe family; YUV has no inference score.", transform=ax.transAxes, va="top", fontsize=8)

    families = ("avif", "webp")
    stage_names = ("pre_reconstruction_coefficients", "decoded_yuv", "decoded_rgb")
    ticks = []
    labels = []
    for family_index, family in enumerate(families):
        conditions = ("avif_q40", "avif_q60") if family == "avif" else ("webp_q30", "webp_q75")
        for condition_index, condition in enumerate(conditions):
            for stage_index, stage in enumerate(stage_names):
                match = next((row for row in timing_rows if row["condition"] == condition and row["stage"] == stage), None)
                if match is None:
                    continue
                cost = float(match["median_cumulative_decoder_adapter_ms"] or match["median_decoder_ms"])
                y = family_index * 2 + condition_index
                if stage == "decoded_yuv":
                    cost_ax.scatter(cost, y, marker="*", facecolors="none", edgecolors=COLORS[family], s=95)
                else:
                    marker = "s" if stage == "pre_reconstruction_coefficients" else "^"
                    cost_ax.scatter(cost, y, marker=marker, color=COLORS[family], s=40)
            ticks.append(family_index * 2 + condition_index)
            labels.append(condition)
    cost_ax.set_yticks(ticks, labels)
    cost_ax.set_xlabel("Measured median stage cost (ms); star = YUV decode cost only, no inference score")
    cost_ax.set_title("Decoder-cost reference points")
    cost_ax.grid(axis="x", alpha=0.2)
    fig.savefig(OUTPUT / "corrected_semantic_exposure_frontier.png", dpi=200)
    plt.close(fig)


def _write_findings(operating_points: list[dict], calibrations: list[dict], paired: list[dict], correlations: list[dict], within: list[dict], matched: list[dict], candidates: list[dict], auc_summary: list[dict], supersession: list[dict], avif_costs: list[dict], metadata: dict) -> None:
    by_condition = {row["condition"]: row for row in operating_points}
    new_table = [by_condition[condition] for condition in NEW_CONDITIONS]
    lines = [
        "# Phase 4.8 Findings",
        "",
        "## Scope and reproducibility",
        "",
        "This pass added JPEG q1/q5, WebP q5/q10/q20 and AVIF q5/q10/q20 to the exact prepared Imagenette 256×256 split. The raw-byte primary experiment used ByteStreamCNN, seed 42, the established length-bucket token-budget batching and CUDA. Each condition was encoded into its own resumable, hash-verified stream cache. No existing condition was retrained. Raw-byte test evaluation followed validation checkpoint selection; all eight new conditions had zero truncation.",
        "",
        f"Calibration used {metadata['calibration_samples']} stratified images ({metadata['calibration_per_class']} per class), three timed encodes and decodes per image after warmup. All {metadata['calibration_samples'] * len(NEW_CONDITIONS)} decodes succeeded. Encoder versions/settings and stream identities are in `calibration_metadata.json` and `calibration_per_sample.csv`. Historical q10+ measurements remain identified as historical, with their original unequal seed counts; new raw results are one-seed observations.",
        "",
        "## New operating points",
        "",
        "| Condition | bpp | PSNR dB | SSIM | Raw accuracy, seed 42 | U-probe bits, seed 42 | RGB accuracy, 3 seeds mean ± SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in new_table:
        sd = "—" if row["decoded_rgb_accuracy_seed_std"] is None else f"{row['decoded_rgb_accuracy_seed_std']:.3f}"
        lines.append(f"| {row['condition']} | {row['bpp']:.3f} | {row['psnr_db']:.2f} | {row['ssim']:.3f} | {row['raw_accuracy_seed42']:.3f} | {row['raw_u_probe_bits_seed42']:.3f} | {row['decoded_rgb_accuracy_mean']:.3f} ± {sd} |")
    lines += [
        "",
        "The low-quality settings occupy distinct encoded streams and operating points; calibration exact-stream checks did not identify redundant adjacent settings. JPEG q1→q5 increased mean calibration bpp while improving PSNR/SSIM. WebP q5→q20 and AVIF q5→q20 also increased rate and reconstruction quality. See the calibration tables for per-image variation and median/p95 timing.",
        "",
        "## Compression level and raw-byte accessibility",
        "",
        "| Condition | bpp | PSNR dB | Raw accuracy | U-probe bits | Raw seeds | Source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in operating_points:
        source = "new seed 42" if row["historical_or_new"] == "new_extreme" else "Phase 4.5B mean"
        lines.append(f"| {row['condition']} | {row['bpp']:.3f} | {row['psnr_db']:.2f} | {row['raw_accuracy_mean']:.3f} | {row['raw_u_probe_bits_mean']:.3f} | {row['raw_seed_count']} | {source} |")
    lines += [
        "",
        "The observed pattern is codec-specific rather than a universal inverse rate–accessibility law. JPEG's prior q10/q20 seed-42 accuracy was about 0.36–0.37; the new q1/q5 seed-42 results are lower (see table), so the low-quality extension does not show accessibility rising monotonically as rate falls. WebP and AVIF remain low in raw accuracy across the added and historical settings, with small, nonmonotonic U-probe changes. These are per-setting trained probes, not a controlled causal estimate of the effect of quality alone.",
        "",
        "The within-codec Pearson/Spearman table reports all measured points, and the paired table compares adjacent settings on the same 3,925 images. Within-image-centered correlations separately remove each image's average level and bootstrap image identities. These summaries reduce image-difficulty confounding but cannot separate quality from all other setting-dependent bitstream changes. Small setting counts, different historic seed counts, and the single-seed new sweep make correlations exploratory; no general law or causal mechanism is established.",
        "",
        "## Rate–distortion and cross-codec matching",
        "",
        "`matched_operating_point_candidates.csv` records the closest measured rate and distortion pairs plus errors. `paired_operating_point_comparisons.csv` includes only pairs meeting the predeclared thresholds: at most 10% relative bpp difference or at most 0.5 dB absolute PSNR difference. Accessibility is compared only at observed settings; no neural accessibility values are interpolated. Inadequate matches are excluded from the paired file and remain visible in the candidate file.",
        "",
        "Across the observed points, raw accessibility is associated with both codec family and compression setting. A pooled correlation can therefore reflect family clustering. The per-codec correlations and matched-point comparisons are the relevant descriptive views, but sample counts remain too small for confirmatory inference.",
        "",
        "## Phase 4.7 correction",
        "",
        "The historical Phase 4.7 AUC files and report were preserved unchanged. Corrected Phase 4.8 AUC requires the same `pre_reconstruction_coefficients → decoded_rgb` endpoints for every seed, with linear trapezoidal area between those two measured endpoints only. Raw bytes are excluded because that measurement exists only for seed 42 and uses a different probe family. Missing endpoints and nonmonotonic cost order are reported as unavailable. JXL has no measured pre-reconstruction stage; WebP's early decoder-plus-adapter cost is not ordered before RGB for all applicable paths. Neither is assigned a zero score.",
        "",
        "The corrected Pareto table retains every measured raw, structured and RGB point and labels nondominated points per seed and metric. The corrected AUC summary reports only available seeds, so no across-seed SD is calculated from different endpoint sets. The old q40/q60 values are superseded for interpretation by the Phase 4.8 corrected table.",
        "",
        "## AVIF early-exit and YUV reference",
        "",
        "| Condition | Pre-reconstruction + adapters | Full YUV | Full RGB | Savings vs YUV | Savings vs RGB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    maxima_index = lines.index("## Rate–distortion and cross-codec matching")
    maxima = [
        "Numerical maxima among the common seed-42 operating points are:",
        "",
        "| Codec | Highest raw accuracy | Highest U-probe |",
        "|---|---|---|",
    ]
    for family in CONDITION_ORDER:
        rows = [row for row in operating_points if row["codec_family"] == family]
        best_accuracy = max(rows, key=lambda row: row["raw_accuracy_seed42"])
        best_utility = max(rows, key=lambda row: row["raw_u_probe_bits_seed42"])
        maxima.append(f"| {family} | {best_accuracy['condition']} ({best_accuracy['raw_accuracy_seed42']:.3f}) | {best_utility['condition']} ({best_utility['raw_u_probe_bits_seed42']:.3f}) |")
    maxima += ["", "For JPEG, the accuracy maximum is q20 and the U-probe maximum is q10, both away from the extreme low and high ends. WebP accuracy is highest at q5 while U is highest at q30. AVIF's seed-42 maxima occur at q20 for both. These peak locations are exploratory because historical seed counts differ and the new points are single-seed.", ""]
    lines[maxima_index:maxima_index] = maxima

    structured, structured_status = _structured_outputs()
    structured_index = lines.index("## Phase 4.7 correction")
    if structured:
        dct = next(row for row in structured if row["stage"] == "pre_reconstruction_dct")
        raw = next(row for row in structured if row["stage"] == "raw_bytes")
        rgb = next(row for row in structured if row["stage"] == "decoded_rgb")
        section = [
            "## Extreme-setting structured comparison",
            "",
            "The existing JPEG q1 stream was evaluated with the validated JPEGDCTNet using the entropy-decoded quantized coefficients, sampling layout and quantization tables. The q1 DCT cache stores one model-ready array representation; it does not also store a second dense training grid or per-sample canonical NPZ bundle.",
            "",
            "| JPEG q1 stage | Accuracy | U-probe bits | Extraction/decode median, ms | Probe forward, ms/sample |",
            "|---|---:|---:|---:|---:|",
            f"| Raw bytes, seed 42 | {raw['accuracy']:.3f} | {raw['u_probe_bits']:.3f} | {raw['extraction_or_decode_median_ms']:.3f} | {raw['model_forward_ms_per_sample']:.3f} |",
            f"| Quantized DCT, seed 42 | {dct['accuracy']:.3f} | {dct['u_probe_bits']:.3f} | {dct['extraction_or_decode_median_ms']:.3f} | {dct['model_forward_ms_per_sample']:.3f} |",
            f"| Decoded RGB, seed 42 | {rgb['accuracy']:.3f} | {rgb['u_probe_bits']:.3f} | {rgb['extraction_or_decode_median_ms']:.3f} | {rgb['model_forward_ms_per_sample']:.3f} |",
            "",
            "DCT extraction includes entropy decoding, canonical coefficient-array construction, and model-array adaptation. RGB decode latency is from the deterministic calibration subset; model-forward timing is from the matched seed-42 test evaluation. These costs are reported separately, and their sum is only a descriptive per-sample estimate because the measurements use different timing batches.",
            "",
            "The entries are representation-plus-probe measurements: ByteStreamCNN was trained for raw bytes, JPEGDCTNet was trained on the DCT grid, and the RGBResNet18 endpoint is a previously trained frozen reference. The 0.717 DCT accuracy versus 0.688 RGB accuracy must not be interpreted as a representation-only advantage or a matched-probe comparison.",
            "",
            "The matching low-quality WebP q5 and AVIF q5 structured probes were not run: the Phase 4.7 instrumentation source is present, but the compiled libwebp/libaom/libavif binaries were absent from the expected build location. Rebuilding that native stack was outside this focused pass. The availability table records this limitation; no missing stage is assigned an accessibility score.",
            "",
        ]
        lines[structured_index:structured_index] = section
    for row in avif_costs:
        lines.append(f"| {row['condition']} | {row['pre_reconstruction_decoder_plus_adapters_ms']:.3f} ms | {row['full_yuv_decode_ms']:.3f} ms | {row['full_rgb_decode_ms']:.3f} ms | {row['early_savings_vs_yuv_percent']:.1f}% | {row['early_savings_vs_rgb_percent']:.1f}% |")
    lines += [
        "",
        "The pre-reconstruction path does expose an actual syntax/coefficient representation before inverse transform, prediction and reconstruction. However, its capture plus model-adapter cost is close to full YUV decode. Existing code and artifacts do not provide a matched classifier over native YUV; converting YUV to RGB and then using the RGB model would not be a YUV-inference control. Accordingly, savings relative to YUV and end-to-end latency benefit remain unestablished; the RGB comparison includes avoided color conversion. A future systems frontier must show decoder, decoder-plus-adapter, and neural inference costs separately.",
        "",
        "## Implications for video and remaining limits",
        "",
        "The low-quality sweep does not establish a stable monotonic relationship between compression efficiency and byte-model utility. It supports carrying codec-specific, rate-matched operating points and explicit byte/structured/decoded stages into video planning, while avoiding assumptions that a better rate–distortion point exposes more or less semantic information. The image work is sufficient for the narrow quality sweep; further broad image sweeps or architecture searches are not justified before the video track. A later targeted multi-seed confirmation is warranted only for the largest observed paired changes, not for the full matrix.",
        "",
        "### Scientific assessment",
        "",
        "**Supported:** all eight new streams were distinct on the deterministic calibration subset; all full test raw-byte runs completed without truncation; raw-byte accessibility differs by codec family and is not monotonic across the observed settings; corrected Phase 4.7 AUC requires matched endpoints and excludes unavailable/misordered paths; AVIF pre-reconstruction capture is close in cost to full YUV decode.",
        "",
        "**Suggestive:** JPEG's historical mid-low settings outperformed the new q1/q5 seed-42 probe, and per-image paired changes may identify quality transitions worth confirming. Codec-specific correlations may help select video conditions.",
        "",
        "**Unsupported:** a universal inverse compression-efficiency/accessibility law; a causal explanation attributing accessibility changes to entropy coding alone; practical end-to-end AVIF speedup before reconstruction; any claim about native-YUV neural inference; or a general trend inferred from selected settings and unequal seeds.",
        "",
        "## Artifact index",
        "",
        "Calibration: `calibration_per_sample.csv`, `calibration_summary.csv`, `calibration_metadata.json`. Full-set results: `operating_point_summary.csv`, `raw_byte_accessibility_per_condition.csv`, `paired_quality_changes.csv`, `within_image_centered_correlations.csv`, `descriptive_correlations.csv`, `matched_operating_point_candidates.csv`, and `paired_operating_point_comparisons.csv`. Structured evidence: `structured_representation_comparison.csv`, `structured_representation_availability.csv`, `jpeg_q1_dct_feature_cache.json`, and `jpeg_q1_dct_extraction_timing_per_sample.csv`. Phase 4.7 corrections: `phase47_corrected_auc_per_seed.csv`, `phase47_corrected_auc_summary.csv`, `phase47_corrected_pareto_points.csv`, `phase47_auc_supersession.csv`, and `phase47_avif_yuv_reassessment.csv`. Plots are PNG files in this directory. Training/resource diagnostics remain alongside the per-seed JSON and `training_resource_parent.json`.",
    ]
    rate_index = lines.index("## Rate–distortion and cross-codec matching")
    correlation_rows = {(row["group"], row["y"]): row for row in correlations if row["x"] == "bpp" and row["group"] in CONDITION_ORDER}
    correlation_section = [
        "",
        "Selected-point correlations use the common seed-42 raw results and are descriptive:",
        "",
        "| Codec | n settings | bpp vs raw accuracy r / ρ | bpp vs raw U r / ρ | bpp vs RGB accuracy r / ρ |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in CONDITION_ORDER:
        accuracy = correlation_rows[(family, "raw_accuracy_seed42")]
        utility = correlation_rows[(family, "raw_u_probe_bits_seed42")]
        rgb = correlation_rows[(family, "decoded_rgb_accuracy_seed42")]
        correlation_section.append(
            f"| {family} | {accuracy['n_operating_points']} | {accuracy['pearson_r']:.2f} / {accuracy['spearman_rho']:.2f} | {utility['pearson_r']:.2f} / {utility['spearman_rho']:.2f} | {rgb['pearson_r']:.2f} / {rgb['spearman_rho']:.2f} |"
        )
    correlation_section += [
        "",
        "The rate–accessibility association is not shared by all responses: WebP's bpp–accuracy relationship is negative while its bpp–U relationship is positive; AVIF's bpp–accuracy relationship is negative while bpp–U is approximately zero. Each coefficient uses only five or six selected settings, so it is not a general or causal relationship.",
        "",
        "After centering each image across its codec's quality settings, bpp–correctness correlations are much smaller than the between-setting summaries. The paired-bootstrap intervals condition on the fitted checkpoints and resample images; they do not capture training-seed variation. Per-image rate-complexity variation therefore explains part of the pooled pattern, but does not remove the codec/setting association.",
        "",
        "| Codec | Within-image r: bpp vs correct (95% image-bootstrap CI) | Within-image r: bpp vs per-image CE |",
        "|---|---:|---:|",
    ]
    within_lookup = {(row["codec_family"], row["x"], row["y"]): row for row in within}
    for family in CONDITION_ORDER:
        correctness = within_lookup[(family, "bpp", "correct")]
        cross_entropy = within_lookup[(family, "bpp", "cross_entropy_nats")]
        correlation_section.append(
            f"| {family} | {float(correctness['pearson_r']):.3f} [{float(correctness['pearson_ci95_low']):.3f}, {float(correctness['pearson_ci95_high']):.3f}] | {float(cross_entropy['pearson_r']):.3f} |"
        )
    correlation_section += [
        "",
        "| Codec pair | Match basis | Observed pair | Match error | Δ raw accuracy (A−B) | Δ raw U (A−B) bits |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in matched:
        error = row.get("relative_bpp_difference") if row["matching_basis"] == "rate" else row.get("absolute_psnr_difference_db")
        error_text = f"{float(error):.1%}" if row["matching_basis"] == "rate" else f"{float(error):.2f} dB"
        correlation_section.append(
            f"| {row['codec_a']} vs {row['codec_b']} | {row['matching_basis']} | {row['condition_a']} vs {row['condition_b']} | {error_text} | {float(row['raw_accuracy_delta_a_minus_b']):+.3f} | {float(row['raw_u_probe_delta_bits_a_minus_b']):+.3f} |"
        )
    correlation_section += [""]
    lines[rate_index + 1:rate_index + 1] = correlation_section

    auc_index = lines.index("## AVIF early-exit and YUV reference")
    auc_section = [
        "| Condition | Historical Phase 4.7 U AUC mean | Corrected U AUC mean ± SD | Seeds |",
        "|---|---:|---:|---:|",
    ]
    for row in supersession:
        corrected_auc = next(item for item in auc_summary if item["condition"] == row["condition"] and item["metric"] == "u_probe_auc")
        corrected = "unavailable" if row["corrected_u_auc"] is None else f"{float(row['corrected_u_auc']):.3f} ± {float(corrected_auc['sample_std'] or 0):.3f}"
        old = "—" if row["historical_phase47_auc_mean_u"] in (None, "") else f"{float(row['historical_phase47_auc_mean_u']):.3f}"
        auc_section.append(f"| {row['condition']} | {old} | {corrected} | {row['corrected_u_auc_seed_count']}/{row['historical_phase47_seed_count']} |")
    auc_section += [
        "",
        "The historical JXL U AUC means (0.179 and 0.183) included a single seed with a raw-byte endpoint while other seeds had RGB only; those corrected conditions are unavailable because no pre-reconstruction stage was emitted. AVIF's same-endpoint means fall to 0.377 and 0.290. WebP is unavailable because its measured pre-reconstruction decoder-plus-adapter cost is not strictly before the RGB endpoint. The old files remain as historical records.",
        "",
    ]
    lines[auc_index:auc_index] = auc_section
    (OUTPUT / "Phase 4.8 Findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not (OUTPUT / "decoded_rgb_reference_per_seed.csv").is_file():
        raise FileNotFoundError("decoded RGB references are incomplete; run scripts/evaluate_phase48_rgb.py first")
    operating_points, rd, _ = _load_operating_points()
    _write_csv(OUTPUT / "operating_point_summary.csv", operating_points)
    _write_csv(OUTPUT / "raw_byte_accessibility_per_condition.csv", operating_points)

    paired, within = _paired_and_within_image(operating_points, rd)
    correlations = _correlations(operating_points)
    matched, candidates = _matched_points(operating_points)
    _write_csv(OUTPUT / "paired_quality_changes.csv", paired)
    _write_csv(OUTPUT / "within_image_centered_correlations.csv", within)
    _write_csv(OUTPUT / "descriptive_correlations.csv", correlations)
    _write_csv(OUTPUT / "paired_operating_point_comparisons.csv", matched)
    _write_csv(OUTPUT / "matched_operating_point_candidates.csv", candidates)

    calibration = _read_csv(OUTPUT / "calibration_per_sample.csv")
    calibration_summary = _read_csv(OUTPUT / "calibration_summary.csv")
    _write_csv(OUTPUT / "calibration_summary.csv", calibration_summary)
    corrected_per_seed, corrected_summary, pareto, supersession, corrected_rd, auc_correlations, avif_costs = _phase47_corrections()
    _write_csv(OUTPUT / "phase47_corrected_auc_per_seed.csv", corrected_per_seed)
    _write_csv(OUTPUT / "phase47_corrected_auc_summary.csv", corrected_summary)
    _write_csv(OUTPUT / "phase47_corrected_pareto_points.csv", pareto)
    _write_csv(OUTPUT / "phase47_auc_supersession.csv", supersession)
    _write_csv(OUTPUT / "phase47_corrected_rd_efficiency.csv", corrected_rd)
    _write_csv(OUTPUT / "phase47_corrected_rd_semantic_correlations.csv", auc_correlations)
    _write_csv(OUTPUT / "phase47_avif_yuv_reassessment.csv", avif_costs)
    structured, structured_status = _structured_outputs()
    _write_csv(OUTPUT / "structured_representation_comparison.csv", structured)
    _write_csv(OUTPUT / "structured_representation_availability.csv", structured_status)

    _plot_operating_points(operating_points)
    _plot_corrected_frontier(corrected_per_seed, pareto, _read_csv(PHASE47 / "decoder_timing_summary.csv"))
    metadata_path = OUTPUT / "calibration_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _write_findings(operating_points, calibration_summary, paired, correlations, within, matched, candidates, corrected_summary, supersession, avif_costs, metadata)
    print(json.dumps({
        "conditions": len(operating_points),
        "paired_quality_comparisons": len(paired),
        "adequate_cross_codec_matches": len(matched),
        "inadequate_candidate_matches": sum(not row["adequate_match"] for row in candidates),
        "corrected_auc_available_seed_rows": sum(row["status"] == "available" for row in corrected_per_seed),
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
