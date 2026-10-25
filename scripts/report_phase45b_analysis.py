from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.analysis.phase45b import fit_linear_model, paired_correctness, pareto_efficient
from tamtd.runtime import cache_root, results_root
from run_phase45b_analysis import _discover_model_specs


RAW_REPRESENTATIONS = {"pbc_lzma_bytes", "pbc_store_bytes", "jpeg_file_bytes", "codec_file_bytes"}
RAW_OPS = [
    ("pbc_compression", "pbc_lzma_bytes"), ("pbc_compression", "pbc_store_bytes"),
    ("pbc_balanced", "pbc_lzma_bytes"), ("pbc_balanced", "pbc_store_bytes"),
    ("pbc_quality", "pbc_lzma_bytes"), ("pbc_quality", "pbc_store_bytes"),
    ("jpeg_q10", "jpeg_file_bytes"), ("jpeg_q20", "jpeg_file_bytes"),
    ("jpeg_q60", "jpeg_file_bytes"), ("jpeg_q80", "jpeg_file_bytes"),
    ("webp_q30", "codec_file_bytes"), ("webp_q75", "codec_file_bytes"),
    ("avif_q40", "codec_file_bytes"), ("avif_q60", "codec_file_bytes"),
    ("jxl_d6", "codec_file_bytes"), ("jxl_d3", "codec_file_bytes"),
]
PAIRED = [
    ("jpeg_q10", "jpeg_file_bytes", "webp_q30", "codec_file_bytes"),
    ("jpeg_q10", "jpeg_file_bytes", "avif_q40", "codec_file_bytes"),
    ("jpeg_q10", "jpeg_file_bytes", "jxl_d6", "codec_file_bytes"),
    ("jpeg_q60", "jpeg_file_bytes", "webp_q75", "codec_file_bytes"),
    ("jpeg_q60", "jpeg_file_bytes", "avif_q60", "codec_file_bytes"),
    ("jpeg_q60", "jpeg_file_bytes", "jxl_d3", "codec_file_bytes"),
]
COMPARATOR = {
    "pbc_compression": "jpeg_q10", "pbc_balanced": "jpeg_q60", "pbc_quality": "jpeg_q60",
    "jpeg_q10": "jpeg_q10", "jpeg_q20": "jpeg_q10", "jpeg_q60": "jpeg_q60", "jpeg_q80": "jpeg_q60",
    "webp_q30": "jpeg_q10", "webp_q75": "jpeg_q60", "avif_q40": "jpeg_q10", "avif_q60": "jpeg_q60",
    "jxl_d6": "jpeg_q10", "jxl_d3": "jpeg_q60",
}
LOW_MID = {
    "jpeg_q10": ("jpeg_q10", "jpeg_q60"), "webp": ("webp_q30", "webp_q75"),
    "avif": ("avif_q40", "avif_q60"), "jxl": ("jxl_d6", "jxl_d3"),
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _portable_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _save_csv(rows, path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _sample_std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _train_prior(labels: np.ndarray, classes: int = 10) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=classes).astype(np.float64)
    return counts / counts.sum()


def _prior_ce_for_labels(labels: np.ndarray, prior: np.ndarray) -> float:
    return float(-np.log(prior[np.asarray(labels, dtype=np.int64)]).mean())


def _write_enriched_endpoints(path: Path) -> pd.DataFrame:
    endpoints = pd.read_csv(path)
    enriched = [
        "original_rgb_accuracy", "original_rgb_u_probe_bits", "accuracy_difference_vs_original_rgb",
        "semantic_denominator_accuracy_minus_chance", "semantic_accessibility_vs_original_rgb",
        "u_probe_difference_vs_original_rgb_bits",
    ]
    endpoints = endpoints.drop(columns=[column for column in enriched if column in endpoints])
    baseline = endpoints[endpoints["condition"] == "original_rgb"][["seed", "split", "accuracy", "u_probe_bits"]].rename(columns={
        "accuracy": "original_rgb_accuracy", "u_probe_bits": "original_rgb_u_probe_bits",
    })
    endpoints = endpoints.merge(baseline, on=["seed", "split"], how="left", validate="many_to_one")
    endpoints["accuracy_difference_vs_original_rgb"] = endpoints["accuracy"] - endpoints["original_rgb_accuracy"]
    endpoints["semantic_denominator_accuracy_minus_chance"] = endpoints["original_rgb_accuracy"] - 0.1
    endpoints["semantic_accessibility_vs_original_rgb"] = np.where(
        endpoints["semantic_denominator_accuracy_minus_chance"] > 0,
        (endpoints["accuracy"] - 0.1) / endpoints["semantic_denominator_accuracy_minus_chance"],
        np.nan,
    )
    endpoints["u_probe_difference_vs_original_rgb_bits"] = endpoints["u_probe_bits"] - endpoints["original_rgb_u_probe_bits"]
    endpoints["relative_u_probe_utility_vs_original_rgb"] = np.where(
        endpoints["original_rgb_u_probe_bits"] > 0,
        endpoints["u_probe_bits"] / endpoints["original_rgb_u_probe_bits"],
        np.nan,
    )
    endpoints.to_csv(path, index=False)
    return endpoints


def _endpoint_summary(endpoints: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "accuracy", "cross_entropy_nats", "u_probe_bits", "brier_score",
        "accuracy_difference_vs_original_rgb", "semantic_accessibility_vs_original_rgb",
        "u_probe_difference_vs_original_rgb_bits", "relative_u_probe_utility_vs_original_rgb",
    ]
    aggregations = {"n_samples": ("n_samples", "first"), "seed_count": ("seed", "nunique")}
    for column in columns:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_sample_std"] = (column, "std")
    return endpoints.groupby(["condition", "codec_family", "representation", "split"], as_index=False).agg(**aggregations)


def _prediction_summaries(predictions: pd.DataFrame, train_labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = _train_prior(train_labels)
    prior_by_class = -np.log(prior)
    rows = []
    for (condition, representation, seed), group in predictions.groupby(["model_condition", "representation", "seed"], sort=True):
        labels = group["true_label"].to_numpy(dtype=np.int64)
        ce = group["cross_entropy_nats"].to_numpy(dtype=np.float64)
        prior_ce = float(np.mean(prior_by_class[labels]))
        rows.append({
            "condition": condition, "representation": representation, "seed": int(seed), "n_test": len(group),
            "accuracy": float(group["correct"].mean()), "cross_entropy_nats": float(ce.mean()),
            "class_prior_cross_entropy_nats": prior_ce, "test_label_entropy_nats": float(-np.sum(np.bincount(labels, minlength=10) / len(labels) * np.log(np.maximum(np.bincount(labels, minlength=10) / len(labels), 1e-300)))),
            "u_probe_nats": prior_ce - float(ce.mean()), "u_probe_bits": (prior_ce - float(ce.mean())) / math.log(2.0),
            "brier_score": float(group["brier_score"].mean()), "model_family": group["model_family"].iloc[0],
        })
    per_seed = pd.DataFrame(rows)
    return per_seed, predictions


def _raw_table(per_sample: pd.DataFrame, rd: pd.DataFrame, predictions: pd.DataFrame, probe: pd.DataFrame, endpoints: pd.DataFrame, phase1: pd.DataFrame) -> pd.DataFrame:
    rd_test = rd.loc[rd["split"] == "test"].groupby(["condition", "representation"], as_index=False).agg(
        imagenette_bpp_mean=("bpp", "mean"), imagenette_bpp_median=("bpp", "median"),
        imagenette_psnr_mean_db=("psnr_db", "mean"), imagenette_psnr_median_db=("psnr_db", "median"),
        imagenette_ssim_mean=("ssim", "mean"), imagenette_mse_mean=("mse", "mean"),
        encoded_bytes_mean=("encoded_bytes", "mean"), encoded_bytes_median=("encoded_bytes", "median"),
    )
    phase1_map = {row["codec_id"]: row for row in phase1.to_dict("records")}
    result = []
    for condition, representation in RAW_OPS:
        model = probe[(probe["condition"] == condition) & (probe["representation"] == representation)]
        rate = rd_test[(rd_test["condition"] == condition) & (rd_test["representation"] == representation)]
        if len(model) == 0 or len(rate) != 1:
            raise ValueError(f"missing aligned raw-byte analysis row: {condition}/{representation}")
        endpoint = probe[(probe["condition"] == condition) & (probe["representation"] == "decoded_rgb_reference")]
        end_seed = {int(row["seed"]): row for row in endpoint.to_dict("records")}
        per_seed = []
        for model_row in model.to_dict("records"):
            decoded = end_seed.get(int(model_row["seed"]))
            semantic = None if decoded is None or decoded["accuracy"] <= 0.1 else (model_row["accuracy"] - 0.1) / (decoded["accuracy"] - 0.1)
            relative_u = None if decoded is None or decoded["u_probe_bits"] <= 0 else model_row["u_probe_bits"] / decoded["u_probe_bits"]
            per_seed.append((model_row, decoded, semantic, relative_u))
        model_values = [item[0] for item in per_seed]
        decoded_values = [item[1] for item in per_seed if item[1] is not None]
        semantic_values = [item[2] for item in per_seed if item[2] is not None]
        relative_values = [item[3] for item in per_seed if item[3] is not None]
        rd_row = rate.iloc[0].to_dict()
        p1 = phase1_map.get(condition) if representation != "pbc_store_bytes" else None
        result.append({
            **rd_row,
            "codec_family": condition.split("_")[0], "seed_count": len(model_values),
            "raw_test_accuracy_mean": float(np.mean([row["accuracy"] for row in model_values])),
            "raw_test_accuracy_std": float(np.std([row["accuracy"] for row in model_values], ddof=1)) if len(model_values) > 1 else 0.0,
            "raw_cross_entropy_nats_mean": float(np.mean([row["cross_entropy_nats"] for row in model_values])),
            "raw_u_probe_bits_mean": float(np.mean([row["u_probe_bits"] for row in model_values])),
            "raw_u_probe_bits_std": float(np.std([row["u_probe_bits"] for row in model_values], ddof=1)) if len(model_values) > 1 else 0.0,
            "class_prior_cross_entropy_nats": float(np.mean([row["class_prior_cross_entropy_nats"] for row in model_values])),
            "raw_brier_score_mean": float(np.mean([row["brier_score"] for row in model_values])),
            "decoded_rgb_seed_count_matched": len(decoded_values),
            "decoded_rgb_accuracy_matched_seed_mean": float(np.mean([row["accuracy"] for row in decoded_values])) if decoded_values else None,
            "decoded_rgb_accuracy_all_seed_mean": float(endpoint["accuracy"].mean()) if len(endpoint) else None,
            "decoded_rgb_u_probe_bits_matched_seed_mean": float(np.mean([row["u_probe_bits"] for row in decoded_values])) if decoded_values else None,
            "semantic_accessibility_mean_per_seed": float(np.mean(semantic_values)) if semantic_values else None,
            "relative_probe_utility_mean_per_seed": float(np.mean(relative_values)) if relative_values else None,
            "relative_probe_utility_std_per_seed": float(np.std(relative_values, ddof=1)) if len(relative_values) > 1 else (0.0 if relative_values else None),
            "phase1_calibration_bpp": None if p1 is None else float(p1["mean_bpp"]),
            "phase1_calibration_psnr_db": None if p1 is None else float(p1["mean_psnr_db"]),
            "phase1_calibration_set_n": None if p1 is None else int(p1["samples"]),
            "phase1_comparison_note": "Phase 1 separate 95-image calibration set" if p1 is not None else "PBC STORE is an alternate wrapper; Phase 1 operating point refers to its LZMA file",
        })
    return pd.DataFrame(result)


def _within_codec_changes(raw: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("jpeg", "jpeg_q10", "jpeg_q60", "jpeg_file_bytes"),
        ("webp", "webp_q30", "webp_q75", "codec_file_bytes"),
        ("avif", "avif_q40", "avif_q60", "codec_file_bytes"),
        ("jxl", "jxl_d6", "jxl_d3", "codec_file_bytes"),
    ]
    rows = []
    for family, low, high, representation in pairs:
        a = raw[(raw["condition"] == low) & (raw["representation"] == representation)].iloc[0]
        b = raw[(raw["condition"] == high) & (raw["representation"] == representation)].iloc[0]
        endpoint_a = probe[(probe["condition"] == low) & (probe["representation"] == "decoded_rgb_reference")]
        endpoint_b = probe[(probe["condition"] == high) & (probe["representation"] == "decoded_rgb_reference")]
        endpoint_a = endpoint_a.set_index("seed").loc[sorted(set(endpoint_a["seed"]) & set(endpoint_b["seed"]))]
        endpoint_b = endpoint_b.set_index("seed").loc[endpoint_a.index]
        rows.append({
            "codec_family": family, "low_condition": low, "high_condition": high,
            "delta_bpp": float(b["imagenette_bpp_mean"] - a["imagenette_bpp_mean"]),
            "delta_psnr_db": float(b["imagenette_psnr_mean_db"] - a["imagenette_psnr_mean_db"]),
            "delta_ssim": float(b["imagenette_ssim_mean"] - a["imagenette_ssim_mean"]),
            "delta_raw_accuracy": float(b["raw_test_accuracy_mean"] - a["raw_test_accuracy_mean"]),
            "delta_raw_u_probe_bits": float(b["raw_u_probe_bits_mean"] - a["raw_u_probe_bits_mean"]),
            "delta_decoded_rgb_accuracy_matched_seeds": float(endpoint_b["accuracy"].mean() - endpoint_a["accuracy"].mean()),
            "raw_seed_count_low": int(a["seed_count"]), "raw_seed_count_high": int(b["seed_count"]),
            "rgb_seed_count": int(len(endpoint_a)),
        })
    return pd.DataFrame(rows)


def _paired_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition_a, representation_a, condition_b, representation_b in PAIRED:
        left = predictions[(predictions["model_condition"] == condition_a) & (predictions["representation"] == representation_a) & (predictions["seed"] == 42)].set_index("image_id").sort_index()
        right = predictions[(predictions["model_condition"] == condition_b) & (predictions["representation"] == representation_b) & (predictions["seed"] == 42)].set_index("image_id").sort_index()
        if len(left) == 0 or not left.index.equals(right.index) or not np.array_equal(left["true_label"].to_numpy(), right["true_label"].to_numpy()):
            raise ValueError(f"paired prediction rows do not align: {condition_a} vs {condition_b}")
        stats = paired_correctness(left["true_label"].to_numpy(), left["predicted_label"].to_numpy(), right["predicted_label"].to_numpy())
        rows.append({"condition_a": condition_a, "representation_a": representation_a, "condition_b": condition_b, "representation_b": representation_b, **stats})
    return pd.DataFrame(rows)


def _per_class(predictions: pd.DataFrame, class_names: dict[int, str]) -> pd.DataFrame:
    rows = []
    raw = predictions[predictions["representation"].isin(RAW_REPRESENTATIONS)]
    for (condition, representation, seed), group in raw.groupby(["model_condition", "representation", "seed"], sort=True):
        endpoint = predictions[(predictions["model_condition"] == condition) & (predictions["representation"] == "decoded_rgb_reference") & (predictions["seed"] == seed)]
        comparator_condition = COMPARATOR[condition]
        comparator = predictions[(predictions["model_condition"] == comparator_condition) & (predictions["representation"] == "jpeg_file_bytes") & (predictions["seed"] == seed)]
        endpoint_map = endpoint.set_index("image_id") if len(endpoint) else None
        comparator_map = comparator.set_index("image_id") if len(comparator) else None
        for class_id in range(10):
            class_rows = group[group["true_label"] == class_id]
            support = len(class_rows)
            predicted = class_rows["predicted_label"].to_numpy(dtype=np.int64)
            counts = np.bincount(predicted, minlength=10)
            class_accuracy = float(class_rows["correct"].mean()) if support else None
            if support and endpoint_map is not None:
                aligned = endpoint_map.loc[class_rows["image_id"]]
                endpoint_accuracy = float((aligned["predicted_label"].to_numpy() == class_id).mean())
            else:
                endpoint_accuracy = None
            if support and comparator_map is not None:
                aligned_cmp = comparator_map.loc[class_rows["image_id"]]
                comparator_accuracy = float((aligned_cmp["predicted_label"].to_numpy() == class_id).mean())
            else:
                comparator_accuracy = None
            row = {
                "condition": condition, "representation": representation, "seed": int(seed), "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "support": support, "accuracy": class_accuracy,
                "mean_cross_entropy_nats": float(class_rows["cross_entropy_nats"].mean()) if support else None,
                "decoded_rgb_accuracy": endpoint_accuracy,
                "difference_vs_decoded_rgb": None if class_accuracy is None or endpoint_accuracy is None else class_accuracy - endpoint_accuracy,
                "jpeg_comparator_condition": comparator_condition, "jpeg_comparator_accuracy": comparator_accuracy,
                "difference_vs_jpeg_comparator": None if class_accuracy is None or comparator_accuracy is None else class_accuracy - comparator_accuracy,
            }
            row.update({f"predicted_{target}": int(counts[target]) for target in range(10)})
            rows.append(row)
    return pd.DataFrame(rows)


def _difficulty(predictions: pd.DataFrame) -> pd.DataFrame:
    raw = predictions[predictions["representation"].isin(RAW_REPRESENTATIONS)]
    rgb = predictions[(predictions["model_condition"] == "original_rgb") & (predictions["representation"] == "decoded_rgb_reference")].set_index(["seed", "image_id"])
    rows = []
    for (condition, representation, seed), group in raw.groupby(["model_condition", "representation", "seed"], sort=True):
        reference = rgb.loc[int(seed)]
        aligned = group.set_index("image_id").join(reference[["predicted_label", "top1_probability"]].rename(columns={"predicted_label": "rgb_prediction", "top1_probability": "rgb_confidence"}), how="inner")
        if len(aligned) != len(group):
            raise ValueError(f"difficulty predictions are not aligned for {condition}/{seed}")
        true = aligned["true_label"].to_numpy()
        raw_correct = aligned["predicted_label"].to_numpy() == true
        rgb_correct = aligned["rgb_prediction"].to_numpy() == true
        confidence = aligned["rgb_confidence"].to_numpy()
        bucket = np.where(~rgb_correct, "rgb_incorrect", np.where(confidence >= 0.75, "rgb_correct_high_confidence", "rgb_correct_lower_confidence"))
        for name in ("rgb_correct_high_confidence", "rgb_correct_lower_confidence", "rgb_incorrect"):
            keep = bucket == name
            n = int(keep.sum())
            rows.append({
                "condition": condition, "representation": representation, "seed": int(seed), "difficulty_bucket": name,
                "n_samples": n, "raw_correct": int(raw_correct[keep].sum()),
                "raw_accuracy": float(raw_correct[keep].mean()) if n else None,
                "byte_correct_rgb_wrong": int(np.sum(raw_correct & ~rgb_correct & keep)),
                "rgb_correct_byte_wrong": int(np.sum(rgb_correct & ~raw_correct & keep)),
                "rgb_confidence_threshold": 0.75,
            })
    return pd.DataFrame(rows)


def _per_image_correlations(rd: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    raw = predictions[predictions["representation"].isin(RAW_REPRESENTATIONS)]
    merged = raw.merge(rd[rd["split"] == "test"], left_on=["model_condition", "representation", "image_id"], right_on=["condition", "representation", "image_id"], how="left", validate="many_to_one")
    if merged["bpp"].isna().any():
        raise ValueError("raw predictions could not be joined to Imagenette RD rows")
    rows = []
    predictors = ("encoded_bytes", "bpp", "psnr_db", "ssim")
    outcomes = ("correct", "true_class_probability", "cross_entropy_nats")
    for (condition, representation, seed), group in merged.groupby(["model_condition", "representation", "seed"], sort=True):
        for predictor in predictors:
            for outcome in outcomes:
                result = spearmanr(group[predictor].to_numpy(), group[outcome].to_numpy())
                rows.append({
                    "condition": condition, "representation": representation, "seed": int(seed), "predictor": predictor,
                    "outcome": outcome, "spearman_rho": float(result.statistic) if np.isfinite(result.statistic) else None,
                    "p_value_descriptive_only": float(result.pvalue) if np.isfinite(result.pvalue) else None, "n": len(group),
                })
    return pd.DataFrame(rows)


def _per_image_quantiles(rd: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    raw = predictions[predictions["representation"].isin(RAW_REPRESENTATIONS)]
    test_rd = rd[rd["split"] == "test"]
    merged = raw.merge(test_rd, left_on=["model_condition", "representation", "image_id"], right_on=["condition", "representation", "image_id"], how="left", validate="many_to_one")
    rows = []
    for (condition, representation, seed), group in merged.groupby(["model_condition", "representation", "seed"], sort=True):
        for predictor in ("bpp", "psnr_db", "ssim"):
            quartile = pd.qcut(group[predictor].rank(method="first"), 4, labels=False) + 1
            for number in sorted(quartile.unique()):
                selected = group.loc[quartile == number]
                rows.append({
                    "condition": condition, "representation": representation, "seed": int(seed),
                    "predictor": predictor, "quantile": int(number), "n": len(selected),
                    "predictor_mean": float(selected[predictor].mean()),
                    "predictor_min": float(selected[predictor].min()), "predictor_max": float(selected[predictor].max()),
                    "raw_accuracy": float(selected["correct"].mean()),
                    "true_class_probability_mean": float(selected["true_class_probability"].mean()),
                    "cross_entropy_nats_mean": float(selected["cross_entropy_nats"].mean()),
                })
    return pd.DataFrame(rows)


def _training_audit(results_dir: Path, cache_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    specs = _discover_model_specs(results_dir, cache_dir)
    settings = json.loads((results_dir / "Neural Accessibility" / "training_settings.json").read_text(encoding="utf-8"))
    confirmation_path = results_dir / "Phase 4 Confirmation" / "confirmation_per_seed.csv"
    confirmation = {}
    if confirmation_path.is_file():
        confirmed = pd.read_csv(confirmation_path)
        confirmation = {
            (row.condition, row.representation, int(row.seed)): row._asdict()
            for row in confirmed.itertuples(index=False)
        }
    split_counts = {}
    with (cache_dir / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            split = json.loads(line)["split"]
            split_counts[split] = split_counts.get(split, 0) + 1
    checkpoint_selection = settings.get("checkpoint_selection", "not recorded")
    families = {
        "pbc_lzma_bytes": "byte", "pbc_store_bytes": "byte", "jpeg_file_bytes": "byte", "codec_file_bytes": "byte",
        "pbc_parsed_fields": "pbc", "pbc_canonical_ir": "pbc", "jpeg_dct_ir": "jpeg_dct", "decoded_rgb_reference": "rgb",
    }
    rows = []
    for spec in specs:
        representation = spec["representation"]
        family = families[representation]
        configured = settings[family]
        training = spec.get("training_settings") or {}
        confirmed = confirmation.get((spec["condition"], representation, int(spec["seed"])), {})
        max_epochs = int(training.get("epochs", configured["epochs"]))
        patience = int(training.get("patience", configured["patience"]))
        history = spec.get("history") or []
        if not history:
            continue
        best_epoch = int(spec.get("best_epoch", max(history, key=lambda row: row["validation_accuracy"])["epoch"]))
        by_epoch = {int(row["epoch"]): row for row in history}
        best_entry = by_epoch[best_epoch]
        final_entry = history[-1]
        epochs_run = int(spec.get("epochs_run", len(history)))
        early_stopped = epochs_run < max_epochs
        truncation = spec.get("truncation") or {}
        batch_candidates = [training.get("batch_size"), spec.get("batch_size"), confirmed.get("training_batch_size"), configured["batch_size"]]
        batch_size = next(int(value) for value in batch_candidates if value is not None and not pd.isna(value))
        device_candidates = [spec.get("device"), confirmed.get("training_device")]
        device = next((str(value) for value in device_candidates if value is not None and not pd.isna(value)), "not recorded")
        rows.append({
            "condition": spec["condition"], "representation": representation, "model_family": spec.get("model_name", spec.get("model_family", family)),
            "seed": int(spec["seed"]), "max_epochs": max_epochs, "epochs_actually_run": epochs_run, "best_epoch": best_epoch,
            "best_validation_accuracy": float(spec.get("best_validation_accuracy", best_entry["validation_accuracy"])),
            "final_epoch_validation_accuracy": float(final_entry["validation_accuracy"]),
            "train_loss_at_best_epoch": float(best_entry["train_loss"]), "train_loss_final_epoch": float(final_entry["train_loss"]),
            "validation_drop_after_best": float(spec.get("best_validation_accuracy", best_entry["validation_accuracy"]) - final_entry["validation_accuracy"]),
            "early_stopping_triggered_before_max_epochs": early_stopped,
            "patience_accounting_consistent": bool(best_epoch + patience <= epochs_run) if early_stopped else False,
            "early_stopping_trigger": "patience exhausted" if early_stopped else "max epochs reached",
            "optimizer": (training.get("optimizer") or configured["optimizer"]).lower(),
            "learning_rate": float(training.get("learning_rate", configured["learning_rate"])),
            "weight_decay": float(training.get("weight_decay", configured["weight_decay"])),
            "batch_size": batch_size, "device": device,
            "train_samples": split_counts.get("train"), "validation_samples": split_counts.get("validation"),
            "test_samples": split_counts.get("test"), "dataset_manifest": _portable_path(cache_dir / "manifest.jsonl"),
            "checkpoint_selection_rule": checkpoint_selection,
            "truncated_samples": int(truncation.get("truncated_samples", 0)),
            "truncated_bytes": int(truncation.get("truncated_bytes", 0)),
            "checkpoint": _portable_path(spec["checkpoint"]),
            "history_source": _portable_path(spec["history_source"]) if spec.get("history_source") else None,
            "validation_curve": json.dumps(history, separators=(",", ":")),
        })
    return pd.DataFrame(rows), specs


def _rd_accessibility_models(raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    conditions = [
        ("pbc_compression", "pbc_lzma_bytes"), ("pbc_compression", "pbc_store_bytes"),
        ("pbc_balanced", "pbc_lzma_bytes"), ("pbc_balanced", "pbc_store_bytes"),
        ("pbc_quality", "pbc_lzma_bytes"), ("pbc_quality", "pbc_store_bytes"),
        ("jpeg_q10", "jpeg_file_bytes"), ("jpeg_q60", "jpeg_file_bytes"),
        ("webp_q30", "codec_file_bytes"), ("webp_q75", "codec_file_bytes"),
        ("avif_q40", "codec_file_bytes"), ("avif_q60", "codec_file_bytes"),
        ("jxl_d6", "codec_file_bytes"), ("jxl_d3", "codec_file_bytes"),
    ]
    data = raw.set_index(["condition", "representation"]).loc[conditions].reset_index()
    bpp = data["imagenette_bpp_mean"].to_numpy(dtype=np.float64)
    log_bpp = np.log(bpp)
    psnr = data["imagenette_psnr_mean_db"].to_numpy(dtype=np.float64)
    ssim = data["imagenette_ssim_mean"].to_numpy(dtype=np.float64)
    codec = data["codec_family"].to_numpy(dtype=str)
    models = {}
    residual_rows = []
    for response in ("raw_test_accuracy_mean", "raw_u_probe_bits_mean"):
        y = data[response].to_numpy(dtype=np.float64)
        rate_psnr = fit_linear_model({"log_bpp": log_bpp, "psnr_db": psnr}, y)
        rate_psnr_codec = fit_linear_model({"log_bpp": log_bpp, "psnr_db": psnr}, y, codec)
        rate_ssim_codec = fit_linear_model({"log_bpp": log_bpp, "ssim": ssim}, y, codec)
        models[response] = {
            "rate_distortion_only_log_bpp_psnr": rate_psnr,
            "rate_distortion_plus_codec_log_bpp_psnr": rate_psnr_codec,
            "rate_distortion_plus_codec_log_bpp_ssim": rate_ssim_codec,
        }
        for index, row in data.iterrows():
            residual_rows.append({
                "condition": row["condition"], "representation": row["representation"], "codec_family": row["codec_family"],
                "bpp": row["imagenette_bpp_mean"], "psnr_db": row["imagenette_psnr_mean_db"], "ssim": row["imagenette_ssim_mean"],
                "response": response, "observed": float(y[index]),
                "predicted_rate_distortion_only": float(rate_psnr["fitted"][index]),
                "residual_rate_distortion_only": float(rate_psnr["residuals"][index]),
                "leave_one_out_prediction_rate_distortion_only": float(rate_psnr["loo_predictions"][index]),
                "predicted_with_codec": float(rate_psnr_codec["fitted"][index]),
                "residual_with_codec": float(rate_psnr_codec["residuals"][index]),
            })
    serializable = {}
    for response, selection in models.items():
        serializable[response] = {}
        for model_name, fit in selection.items():
            serializable[response][model_name] = {key: value for key, value in fit.items() if key not in {"fitted", "residuals", "loo_predictions"}}
    return {
        "data": data[["condition", "representation", "codec_family", "imagenette_bpp_mean", "imagenette_psnr_mean_db", "imagenette_ssim_mean", "raw_test_accuracy_mean", "raw_u_probe_bits_mean"]].to_dict("records"),
        "fits": serializable,
        "warning": "Only 11 codec-setting points; predictors are correlated; coefficients, adjusted R-squared and leave-one-out residuals are descriptive. P-values are not used as confirmatory evidence.",
    }, pd.DataFrame(residual_rows)


def _frontier(results_dir: Path, rd: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    p4_frontier = pd.read_csv(results_dir / "Phase 4 Confirmation" / "multi_seed_frontier.csv")
    p4_timing = {(row["condition"], row["representation"]): float(row["preprocessing_ms"]) for row in p4_frontier.to_dict("records")}
    jpeg_timing = pd.read_csv(results_dir / "Phase 4 Confirmation" / "jpeg_timing_refresh.csv").set_index("condition")
    phase1 = pd.read_csv(results_dir / "Codec Benchmark" / "summary.csv").set_index("codec_id")
    stage_specs = []
    for condition in ("pbc_compression", "pbc_balanced", "pbc_quality"):
        for representation in ("pbc_lzma_bytes", "pbc_store_bytes", "pbc_canonical_ir", "decoded_rgb_reference"):
            stage_specs.append((condition, representation))
    for condition in ("jpeg_q10", "jpeg_q60"):
        stage_specs.extend((condition, rep) for rep in ("jpeg_file_bytes", "jpeg_dct_ir", "decoded_rgb_reference"))
    for condition in ("webp_q30", "webp_q75", "avif_q40", "avif_q60", "jxl_d6", "jxl_d3"):
        stage_specs.extend(((condition, "codec_file_bytes"), (condition, "decoded_rgb_reference")))
    rows = []
    for condition, representation in stage_specs:
        model_rep = representation
        if representation == "decoded_rgb_reference":
            metric_rows = probe[(probe["condition"] == condition) & (probe["representation"] == representation)]
        else:
            metric_rows = probe[(probe["condition"] == condition) & (probe["representation"] == model_rep)]
        if metric_rows.empty:
            raise ValueError(f"frontier model metrics missing for {condition}/{representation}")
        if condition.startswith("pbc_"):
            time_ms = p4_timing[(condition, representation)]
            timing_source = "Phase 2 PBC stage timing; 95-image calibration set"
            caveat = "Python/NumPy/Numba PBC implementation; not an algorithm-only comparison"
        elif condition.startswith("jpeg_"):
            timing_row = jpeg_timing.loc[condition]
            if representation == "jpeg_file_bytes":
                time_ms = 0.0
            elif representation == "jpeg_dct_ir":
                time_ms = float(timing_row["file_to_dct_ir_median_ms"])
            else:
                time_ms = float(timing_row["full_rgb_decode_median_ms"])
            timing_source = "Phase 4 refreshed JPEG timing; 95-image calibration set"
            caveat = "pyjpegio/libjpeg-turbo coefficient access versus Pillow/libjpeg-turbo RGB decode"
        else:
            if representation == "codec_file_bytes":
                time_ms = 0.0
            else:
                time_ms = float(phase1.loc[condition, "median_decode_ms"])
            timing_source = "Phase 1 full decode median; 95-image calibration set"
            caveat = "Pillow bundled/native backend or cjxl/djxl process timing; implementation-level only"
        family = condition.split("_")[0]
        point = f"{condition}::{representation}"
        rows.append({
            "point_id": point, "condition": condition, "codec_family": family, "representation": representation,
            "model_family": metric_rows["model_family"].iloc[0], "preprocessing_ms": time_ms,
            "test_accuracy_mean": float(metric_rows["accuracy"].mean()), "test_accuracy_std": _sample_std(metric_rows["accuracy"]),
            "u_probe_bits_mean": float(metric_rows["u_probe_bits"].mean()), "u_probe_bits_std": _sample_std(metric_rows["u_probe_bits"]),
            "cross_entropy_nats_mean": float(metric_rows["cross_entropy_nats"].mean()), "seed_count": len(metric_rows),
            "timing_source": timing_source, "timing_caveat": caveat,
            "fully_staged_codec": family in {"pbc", "jpeg"}, "reference_only": False,
        })
    frontier = pd.DataFrame(rows)
    for utility in ("test_accuracy_mean", "u_probe_bits_mean"):
        efficient = pareto_efficient(frontier["preprocessing_ms"].to_numpy(), frontier[utility].to_numpy())
        suffix = "accuracy" if utility.startswith("test") else "u_probe"
        frontier[f"pareto_efficient_{suffix}"] = efficient
        dominators = []
        for index, row in frontier.iterrows():
            mask = (frontier["preprocessing_ms"] <= row["preprocessing_ms"]) & (frontier[utility] >= row[utility]) & (
                (frontier["preprocessing_ms"] < row["preprocessing_ms"]) | (frontier[utility] > row[utility])
            )
            dominators.append(";".join(frontier.loc[mask, "point_id"].tolist()))
        frontier[f"dominated_by_{suffix}"] = dominators
    return frontier


def _write_frontier_tables(frontier: pd.DataFrame, out_dir: Path) -> None:
    frontier.to_csv(out_dir / "absolute_frontier.csv", index=False)
    frontier[[
        "point_id", "condition", "codec_family", "representation", "preprocessing_ms", "test_accuracy_mean",
        "pareto_efficient_accuracy", "dominated_by_accuracy", "u_probe_bits_mean", "pareto_efficient_u_probe",
        "dominated_by_u_probe", "seed_count", "timing_source", "timing_caveat",
    ]].to_csv(out_dir / "pareto_analysis.csv", index=False)


def _plot_rd_accessibility(raw: pd.DataFrame, out_dir: Path, x: str, xlabel: str, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"pbc": "#8c564b", "jpeg": "#1f77b4", "webp": "#ff7f0e", "avif": "#2ca02c", "jxl": "#9467bd"}
    for index, row in enumerate(raw.to_dict("records")):
        family = row["codec_family"]
        acc = row["raw_test_accuracy_mean"]
        err = row["raw_test_accuracy_std"]
        ax.errorbar(row[x], acc, yerr=err, fmt="o", color=colors.get(family, "black"), capsize=2)
        if family == "pbc":
            preset = row["condition"].removeprefix("pbc_")
            stream = "STORE" if row["representation"] == "pbc_store_bytes" else "LZMA"
            label = f"PBC {preset} {stream}"
        elif family == "jpeg":
            label = f"JPEG {row['condition'].removeprefix('jpeg_')}"
        else:
            label = f"{family.upper()} {row['condition'].split('_', 1)[1]}"
        label += f" k={row['seed_count']}"
        ax.annotate(label, (row[x], acc), xytext=(4, 4 + (index % 4 - 1.5) * 8), textcoords="offset points", fontsize=5.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Raw-byte test accuracy (fraction)")
    ax.set_title(f"{title}\nImagenette 256×256, held-out test n=3,925; k shown per point")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def _plot_decoded_vs_raw(raw: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"pbc": "#8c564b", "jpeg": "#1f77b4", "webp": "#ff7f0e", "avif": "#2ca02c", "jxl": "#9467bd"}
    for row in raw.to_dict("records"):
        x = row["decoded_rgb_accuracy_matched_seed_mean"]
        y = row["raw_test_accuracy_mean"]
        ax.errorbar(x, y, xerr=0, yerr=row["raw_test_accuracy_std"], fmt="o", color=colors.get(row["codec_family"], "black"), capsize=2)
        ax.annotate(f"{row['condition']} k={row['seed_count']}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6)
    limits = [0.1, 0.9]
    ax.plot(limits, limits, linestyle="--", color="gray", linewidth=1, label="equal accuracy")
    ax.set_xlim(limits)
    ax.set_ylim(0.1, 0.5)
    ax.set_xlabel("Matched decoded-RGB test accuracy (fraction)")
    ax.set_ylabel("Raw-byte test accuracy (fraction)")
    ax.set_title("Raw-byte accuracy vs matched decoded-RGB endpoint\nImagenette test n=3,925; seed counts shown per point")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "decoded_rgb_vs_raw_accessibility.png", dpi=180)
    plt.close(fig)


def _plot_residuals(residuals: pd.DataFrame, out_dir: Path) -> None:
    values = residuals[residuals["response"] == "raw_test_accuracy_mean"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="black", linewidth=1)
    x = np.arange(len(values))
    ax.bar(x, values["residual_rate_distortion_only"], color=[{"pbc":"#8c564b","jpeg":"#1f77b4","webp":"#ff7f0e","avif":"#2ca02c","jxl":"#9467bd"}.get(c,"gray") for c in values["codec_family"]])
    ax.set_xticks(x, [f"{c}\n{r}" for c, r in zip(values["condition"], values["representation"])], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Observed minus rate/distortion-only predicted accuracy (fraction)")
    ax.set_title("Rate/distortion-adjusted accessibility residual\n11 codec settings / 14 raw representations; Imagenette test")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "rd_adjusted_accessibility_residual.png", dpi=180)
    plt.close(fig)


def _plot_frontier(frontier: pd.DataFrame, out_dir: Path, utility: str, ylabel: str, filename: str, efficient_column: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"pbc": "#8c564b", "jpeg": "#1f77b4", "webp": "#ff7f0e", "avif": "#2ca02c", "jxl": "#9467bd"}
    markers = {"pbc": "o", "jpeg": "s", "webp": "^", "avif": "D", "jxl": "P"}
    for family, group in frontier.groupby("codec_family"):
        ax.scatter(group["preprocessing_ms"], group[utility], c=colors.get(family, "black"), marker=markers.get(family, "o"), label=family, alpha=0.82)
        for row in group.to_dict("records"):
            if row[efficient_column]:
                ax.annotate(row["representation"].replace("_reference", ""), (row["preprocessing_ms"], row[utility]), xytext=(4, 4), textcoords="offset points", fontsize=6)
    efficient = frontier[frontier[efficient_column]].sort_values("preprocessing_ms")
    ax.plot(efficient["preprocessing_ms"], efficient[utility], color="gray", linewidth=1, linestyle="--", label="global Pareto envelope")
    ax.set_xlabel("Measured conventional preprocessing/decode cost (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title("Absolute implementation-level accessibility frontier\nImagenette test; seed count per point in CSV; timing set is 95-image calibration")
    ax.grid(alpha=0.25)
    ax.legend(title="Codec family", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def _plot_convergence(audit: pd.DataFrame, out_dir: Path) -> dict:
    conditions = ["jpeg_q10", "jpeg_q60", "webp_q30", "webp_q75", "avif_q40", "avif_q60", "jxl_d6", "jxl_d3"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for condition in conditions:
        row = audit[(audit["condition"] == condition) & (audit["seed"] == 42) & (audit["representation"].isin(["jpeg_file_bytes", "codec_file_bytes"]))]
        if row.empty:
            continue
        history = json.loads(row.iloc[0]["validation_curve"])
        epochs = [point["epoch"] for point in history]
        values = [point["validation_accuracy"] for point in history]
        ax.plot(epochs, values, marker="o", markersize=3, label=f"{condition} (n={len(history)})")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation accuracy (fraction)")
    ax.set_title("Seed-42 raw-byte ByteStreamCNN validation curves\nImagenette validation n=947; no training performed during Phase 4.5B")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "raw_byte_validation_convergence.png", dpi=180)
    plt.close(fig)
    return {"conditions": conditions, "training_performed": False}


def _plot_class_heatmap(class_metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    aggregated = class_metrics.groupby(["condition", "representation", "class_id"], as_index=False)["accuracy"].mean()
    pivot = aggregated.pivot(index=["condition", "representation"], columns="class_id", values="accuracy")
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=max(0.5, float(np.nanmax(pivot.to_numpy()))), cmap="magma")
    ax.set_yticks(np.arange(len(pivot)), [f"{a} / {b}" for a, b in pivot.index], fontsize=7)
    ax.set_xticks(np.arange(10), [str(i) for i in range(10)])
    ax.set_xlabel("Imagenette class ID")
    ax.set_ylabel("Raw-byte model condition / representation")
    ax.set_title("Per-class raw-byte accuracy\nMeans over available seeds; test n=3,925; single seed for WebP/AVIF/JXL")
    fig.colorbar(image, ax=ax, label="Class accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_raw_byte_accuracy_heatmap.png", dpi=180)
    plt.close(fig)
    return aggregated


def _plot_paired_overlap(paired: pd.DataFrame, out_dir: Path) -> None:
    labels = [f"{row.condition_a} vs {row.condition_b}" for row in paired.itertuples()]
    columns = ["both_correct", "a_only_correct", "b_only_correct", "both_wrong"]
    colors = ["#4c9f70", "#377eb8", "#e41a1c", "#bdbdbd"]
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(paired))
    for column, color in zip(columns, colors):
        values = paired[column].to_numpy() / paired["n"].to_numpy()
        ax.bar(np.arange(len(paired)), values, bottom=bottom, label=column.replace("_", " "), color=color)
        bottom += values
    ax.set_xticks(np.arange(len(paired)), labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Share of paired test images")
    ax.set_title("Paired correctness overlap, raw-byte probes\nSeed 42; Imagenette held-out test n=3,925; exact McNemar and paired bootstrap in CSV")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "paired_codec_correctness_overlap.png", dpi=180)
    plt.close(fig)


def _plot_rate_distortion_comparison(rd: pd.DataFrame, raw: pd.DataFrame, results_dir: Path, out_dir: Path) -> pd.DataFrame:
    phase1 = pd.read_csv(results_dir / "Codec Benchmark" / "summary.csv")
    phase1_map = phase1.set_index("codec_id")
    rows = []
    for condition in sorted(rd["condition"].unique()):
        representation = "pbc_lzma_bytes" if condition.startswith("pbc_") else ("jpeg_file_bytes" if condition.startswith("jpeg_") else "codec_file_bytes")
        per_split = rd[(rd["condition"] == condition) & (rd["representation"] == representation)].groupby("split").agg(
            imagenette_mean_bpp=("bpp", "mean"), imagenette_mean_psnr_db=("psnr_db", "mean"), imagenette_mean_ssim=("ssim", "mean"), n=("image_id", "count"),
        ).reset_index()
        p1 = phase1_map.loc[condition] if condition in phase1_map.index else None
        for row in per_split.to_dict("records"):
            rows.append({
                "condition": condition, "split": row["split"], "n": int(row["n"]),
                "imagenette_mean_bpp": row["imagenette_mean_bpp"], "imagenette_mean_psnr_db": row["imagenette_mean_psnr_db"], "imagenette_mean_ssim": row["imagenette_mean_ssim"],
                "phase1_calibration_n": None if p1 is None else int(p1["samples"]),
                "phase1_mean_bpp": None if p1 is None else float(p1["mean_bpp"]),
                "phase1_mean_psnr_db": None if p1 is None else float(p1["mean_psnr_db"]),
                "phase1_mean_ssim": None if p1 is None else float(p1["mean_ssim"]),
                "phase1_dataset": "separate 95-image calibration set",
                "imagenette_dataset": "Phase 3/4 Imagenette split; exact prepared RGB hashes checked",
            })
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "imagenette_vs_phase1_calibration.csv", index=False)
    return result


def _class_rank_matrix(class_metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    values = class_metrics.groupby(["condition", "representation", "class_id"], as_index=False)["accuracy"].mean()
    values["model"] = values["condition"] + "/" + values["representation"]
    pivot = values.pivot(index="class_id", columns="model", values="accuracy")
    correlation = pivot.corr(method="spearman")
    correlation.to_csv(out_dir / "class_accuracy_rank_correlations.csv")
    return correlation


def _top_confusions(class_metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    raw = class_metrics.groupby(["condition", "representation", "class_id"], as_index=False)[[f"predicted_{i}" for i in range(10)]].sum()
    for row in raw.to_dict("records"):
        actual = int(row["class_id"])
        candidates = [(int(row[f"predicted_{predicted}"]), predicted) for predicted in range(10) if predicted != actual]
        count, predicted = max(candidates)
        rows.append({"condition": row["condition"], "representation": row["representation"], "true_class": actual, "most_common_wrong_class": predicted, "count": count})
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "top_class_confusions.csv", index=False)
    return result


def _format(value, digits=3):
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    return f"{number:.2e}" if 0 < abs(number) < 0.001 else f"{number:.{digits}f}"


def _write_findings(
    path: Path,
    rd_summary: pd.DataFrame,
    raw: pd.DataFrame,
    changes: pd.DataFrame,
    probe: pd.DataFrame,
    paired: pd.DataFrame,
    class_metrics: pd.DataFrame,
    class_corr: pd.DataFrame,
    confusions: pd.DataFrame,
    audit: pd.DataFrame,
    frontier: pd.DataFrame,
    rd_models: dict,
    residuals: pd.DataFrame,
    class_names: dict[int, str],
    summary: dict,
    endpoint_summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    selected = raw[
        raw["condition"].isin(["pbc_compression", "pbc_balanced", "pbc_quality", "jpeg_q10", "jpeg_q60", "webp_q30", "webp_q75", "avif_q40", "avif_q60", "jxl_d6", "jxl_d3"])
    ]
    table = [
        "| Condition | Raw form | Imagenette bpp | PSNR dB | SSIM | Raw acc. | Raw CE nats | U bits | Decoded RGB acc. | Relative U | k |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected.to_dict("records"):
        table.append(
            f"| {row['condition']} | {row['representation']} | {_format(row['imagenette_bpp_mean'], 3)} | {_format(row['imagenette_psnr_mean_db'], 2)} | {_format(row['imagenette_ssim_mean'], 4)} | {_format(row['raw_test_accuracy_mean'], 4)} | {_format(row['raw_cross_entropy_nats_mean'], 3)} | {_format(row['raw_u_probe_bits_mean'], 3)} | {_format(row['decoded_rgb_accuracy_matched_seed_mean'], 4)} | {_format(row['relative_probe_utility_mean_per_seed'], 3)} | {row['seed_count']} |"
        )
    endpoint_test = endpoint_summary[endpoint_summary["split"] == "test"]
    endpoint_table = [
        "| Condition | RGB seeds | Accuracy mean ± sample SD | Δ vs original RGB | Accuracy accessibility | U bits mean ± sample SD | Relative U |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in endpoint_test.to_dict("records"):
        endpoint_table.append(
            f"| {row['condition']} | {row['seed_count']} | {_format(row['accuracy_mean'],4)} ± {_format(row['accuracy_sample_std'],4)} | {_format(row['accuracy_difference_vs_original_rgb_mean'],4)} | {_format(row['semantic_accessibility_vs_original_rgb_mean'],3)} | {_format(row['u_probe_bits_mean'],3)} ± {_format(row['u_probe_bits_sample_std'],3)} | {_format(row['relative_u_probe_utility_vs_original_rgb_mean'],3)} |"
        )
    matched_classes = {}
    for condition in ("jpeg_q10", "jpeg_q60", "webp_q30", "webp_q75", "avif_q40", "avif_q60", "jxl_d6", "jxl_d3"):
        representation = "jpeg_file_bytes" if condition.startswith("jpeg_") else "codec_file_bytes"
        matched_classes[condition] = class_metrics[
            (class_metrics["condition"] == condition)
            & (class_metrics["representation"] == representation)
            & (class_metrics["seed"] == 42)
        ].set_index("class_id")["accuracy"]
    class_pairs = [("jpeg_q10", "webp_q30"), ("jpeg_q10", "avif_q40"), ("jpeg_q10", "jxl_d6"), ("jpeg_q60", "webp_q75"), ("jpeg_q60", "avif_q60"), ("jpeg_q60", "jxl_d3")]
    class_win_counts = {
        f"{left}_vs_{right}": int((matched_classes[left] > matched_classes[right]).sum())
        for left, right in class_pairs
    }
    def rho_summary(predictor: str, outcome: str) -> str:
        values = correlations.loc[
            (correlations["predictor"] == predictor) & (correlations["outcome"] == outcome), "spearman_rho"
        ].dropna()
        return f"median {_format(values.median(),3)}, range {_format(values.min(),3)} to {_format(values.max(),3)} over {len(values)} condition/seed groups"
    bpp_correct_rho = rho_summary("bpp", "correct")
    ssim_correct_rho = rho_summary("ssim", "correct")
    n_correlation_groups = len(correlations.loc[
        (correlations["predictor"] == "bpp") & (correlations["outcome"] == "correct")
    ])
    difficulty_q10 = summary["difficulty_conditioned_jpeg_q10_seed42"]
    difficulty_q10 = {row["difficulty_bucket"]: row for row in difficulty_q10}
    high_confidence_q10 = difficulty_q10["rgb_correct_high_confidence"]
    rgb_error_q10 = difficulty_q10["rgb_incorrect"]
    rgb_correct_byte_wrong_q10 = sum(
        row["rgb_correct_byte_wrong"]
        for bucket, row in difficulty_q10.items()
        if bucket.startswith("rgb_correct_")
    )
    dominated_acc = frontier[~frontier["pareto_efficient_accuracy"]]
    dominated_u = frontier[~frontier["pareto_efficient_u_probe"]]
    efficient_acc = frontier.loc[frontier["pareto_efficient_accuracy"], "point_id"].tolist()
    efficient_u = frontier.loc[frontier["pareto_efficient_u_probe"], "point_id"].tolist()
    dominated_accuracy_ids = dominated_acc["point_id"].tolist()
    dominated_utility_ids = dominated_u["point_id"].tolist()
    changes_lines = [
        "| Codec | Δ bpp | Δ PSNR dB | Δ SSIM | Δ raw accuracy | Δ raw U bits | Δ decoded RGB accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in changes.to_dict("records"):
        changes_lines.append(f"| {row['codec_family']} ({row['low_condition']} → {row['high_condition']}) | {_format(row['delta_bpp'],3)} | {_format(row['delta_psnr_db'],2)} | {_format(row['delta_ssim'],4)} | {_format(row['delta_raw_accuracy'],4)} | {_format(row['delta_raw_u_probe_bits'],3)} | {_format(row['delta_decoded_rgb_accuracy_matched_seeds'],4)} |")
    paired_lines = [
        "| Pair | Both correct | A only | B only | Both wrong | Δ accuracy A−B | Paired bootstrap 95% CI | McNemar exact p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in paired.to_dict("records"):
        paired_lines.append(f"| {row['condition_a']} vs {row['condition_b']} | {row['both_correct']} | {row['a_only_correct']} | {row['b_only_correct']} | {row['both_wrong']} | {_format(row['accuracy_difference_a_minus_b'],4)} | [{_format(row['bootstrap_ci95_low'],4)}, {_format(row['bootstrap_ci95_high'],4)}] | {_format(row['mcnemar_exact_p'],4)} |")
    residual_accuracy = residuals[residuals["response"] == "raw_test_accuracy_mean"].sort_values("residual_rate_distortion_only")
    residual_lines = [
        "| Condition | Observed accuracy | Rate/distortion prediction | Residual |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in residual_accuracy.to_dict("records"):
        residual_lines.append(f"| {row['condition']} | {_format(row['observed'],4)} | {_format(row['predicted_rate_distortion_only'],4)} | {_format(row['residual_rate_distortion_only'],4)} |")
    acc_fit = rd_models["fits"]["raw_test_accuracy_mean"]
    u_fit = rd_models["fits"]["raw_u_probe_bits_mean"]
    acc_rd = acc_fit["rate_distortion_only_log_bpp_psnr"]
    acc_codec = acc_fit["rate_distortion_plus_codec_log_bpp_psnr"]
    u_rd = u_fit["rate_distortion_only_log_bpp_psnr"]
    u_codec = u_fit["rate_distortion_plus_codec_log_bpp_psnr"]
    n_early = int(audit["early_stopping_triggered_before_max_epochs"].sum())
    cross_seed = audit[(audit["seed"] == 42) & audit["condition"].isin(["jpeg_q10", "jpeg_q60", "webp_q30", "webp_q75", "avif_q40", "avif_q60", "jxl_d6", "jxl_d3"]) & audit["representation"].isin(["jpeg_file_bytes", "codec_file_bytes"])]
    protocol_same = {
        "ByteStreamCNN": bool(cross_seed["model_family"].str.contains("ByteStreamCNN").all()),
        "AdamW": bool(cross_seed["optimizer"].str.lower().eq("adamw").all()),
        "learning_rate_1e-3": bool(np.allclose(cross_seed["learning_rate"].astype(float), 1e-3)),
        "weight_decay_1e-4": bool(np.allclose(cross_seed["weight_decay"].astype(float), 1e-4)),
        "seed_42": bool(cross_seed["seed"].eq(42).all()),
        "batch_size_32": bool(cross_seed["batch_size"].astype(int).eq(32).all()),
        "device_cuda": bool(cross_seed["device"].str.lower().eq("cuda").all()),
        "same_shared_split_counts": bool(
            cross_seed["train_samples"].eq(8522).all()
            and cross_seed["validation_samples"].eq(947).all()
            and cross_seed["test_samples"].eq(3925).all()
            and cross_seed["dataset_manifest"].nunique() == 1
        ),
        "highest_validation_accuracy_checkpoint": bool(cross_seed["checkpoint_selection_rule"].str.contains("highest validation accuracy", case=False).all()),
        "no_truncation": bool(cross_seed["truncated_samples"].eq(0).all() and cross_seed["truncated_bytes"].eq(0).all()),
    }
    pbc_mid = audit[(audit["condition"] == "pbc_balanced") & (audit["representation"].isin(["pbc_lzma_bytes", "pbc_store_bytes"])) & (audit["seed"] == 42)]
    batch_protocol_note = "PBC balanced/quality byte runs used the Phase 4 memory-safety override (batch 8, CPU); the modern codec and JPEG seed-42 raw-byte runs used batch 32, CUDA. This is a batch/device difference in cross-codec comparisons." if not pbc_mid.empty else "Batch/device overrides are listed in the convergence audit."
    best_pair = paired.reindex(paired["accuracy_difference_a_minus_b"].abs().sort_values(ascending=False).index).iloc[0]
    class_correlations = class_corr.to_numpy(dtype=np.float64)
    class_correlations = class_correlations[~np.eye(len(class_correlations), dtype=bool)]
    corr_min, corr_med = float(np.nanmin(class_correlations)), float(np.nanmedian(class_correlations))
    corr_max = float(np.nanmax(class_correlations))
    doc = f"""# Phase 4.5B Findings

## Scope and provenance

This is a post-hoc analysis of the existing 13,394-sample Imagenette Phase 3/4 cache and saved checkpoints. It created no new codec operating points, performed no classifier training or retraining, launched no cloud resources, and did not run video/VLM work. All per-sample encoded streams were checked against their manifest lengths and SHA-256 values. Prepared 256×256 RGB hashes were checked against the dataset and codec manifests. PBC STORE and LZMA variants were decoded and confirmed pixel-identical across all three presets and all 13,394 samples.

Phase 1 measurements used a separate 95-image calibration set. This report uses new measurements over the exact Imagenette split for every required existing stream. The distinction and both sets of values are in `imagenette_vs_phase1_calibration.csv`; split aggregates are in `imagenette_rd_summary.csv`.

## Methodology audit

Raw byte results use the same `ByteStreamCNN` architecture. PBC parsed/canonical IR uses `PBCInstructionNet`; JPEG DCT IR uses `JPEGDCTNet`; RGB uses `RGBResNet18(width=32, random initialization)`. The staged frontier therefore compares a representation together with its trained probe; it is not a representation-only causal experiment. RGB endpoint checks use the already trained original-RGB reference checkpoint for each available seed. No checkpoint was selected using this analysis.

The cross-codec seed-42 raw-byte checks confirm the same ByteStreamCNN, AdamW, learning rate 1e-3, weight decay 1e-4, seed, split, zero truncation, and validation-selected checkpoint rule: `{json.dumps(protocol_same, sort_keys=True)}`. {batch_protocol_note} All 55 saved classifier checkpoints were re-evaluated against their recorded test accuracy; see `checkpoint_accuracy_recheck.json`.

## Imagenette-aligned rate/distortion and decoded RGB endpoints

The full per-image rate/distortion output is `imagenette_rd_per_sample.csv.gz` (16 stream forms × 13,394 samples). The table below gives held-out test means; PBC LZMA and STORE rows share decoded-image quality but have different stream rates.

{chr(10).join(table)}

The Phase 1 calibration set remains context only. For each required operating point the Imagenette alignment changes measured bpp and distortion; accessibility comparisons below use the Imagenette test values. All decoded-RGB endpoint evaluations use existing RGB reference checkpoints for seeds 42, 123, and 777. Endpoint predictions on validation/test are reported seed-wise in `decoded_rgb_endpoints.csv`; `decoded_rgb_endpoint_summary.csv` reports mean ± sample standard deviation, accuracy difference and normalized accessibility against original RGB, and relative U-probe utility. Per-image correctness is in `test_predictions_summary.csv.gz`.

{chr(10).join(endpoint_table)}

## Convergence audit

Across `{len(audit)}` trained checkpoint histories, `{n_early}` stopped before their configured maximum epoch count. That records an early-stop event, not proof of convergence. For seed-42 raw-byte JPEG/WebP/AVIF/JXL, the validation traces and per-run best/final loss fields are in `training_convergence_audit.csv` and `raw_byte_validation_convergence.png`. Read them as validation behavior under this cosine-decay/early-stopping schedule: a validation peak followed by a decline gives no evidence that more epochs under the same schedule would help. The low/mid byte points are one-seed for WebP/AVIF/JXL; checkpoint seed variance is unmeasured there.

PBC balanced/quality byte models use batch 8 on CPU for the Phase 4 memory guard, while selected JPEG and standard-codec seed-42 models use batch 32 on CUDA. The common architecture/optimizer/LR/seed/split rules hold, but the different batch/device protocol weakens direct optimization-equivalence claims, especially for mid-rate PBC comparisons.

## Rate/distortion/accessibility analysis

Within-codec low-to-mid changes are:

{chr(10).join(changes_lines)}

These comparisons show whether more bits and better reconstruction translate into more byte-probe utility; seed counts differ and WebP/AVIF/JXL remain exploratory single-seed evidence. They do not establish causality.

The rate/distortion model uses 14 raw-representation rows across 11 selected codec operating points: PBC STORE and LZMA for the three PBC settings, plus two points each for JPEG, WebP, AVIF, and JXL. Its `log(bpp)+PSNR` accuracy model has R² `{_format(acc_rd['r_squared'],3)}` and leave-one-out RMSE `{_format(acc_rd['loo_rmse'],4)}`; adding codec identity gives R² `{_format(acc_codec['r_squared'],3)}` and leave-one-out RMSE `{_format(acc_codec['loo_rmse'],4)}`. The corresponding U-probe models have R² `{_format(u_rd['r_squared'],3)}` → `{_format(u_codec['r_squared'],3)}` and leave-one-out RMSE `{_format(u_rd['loo_rmse'],3)}` → `{_format(u_codec['loo_rmse'],3)}`; the `log(bpp)+SSIM+codec` U model has R² `{_format(u_fit['rate_distortion_plus_codec_log_bpp_ssim']['r_squared'],3)}`. In this small selected sample, bitrate and PSNR alone explain a limited share of probe performance, while codec indicators materially improve the descriptive fit. This leaves codec-conditioned residuals after rate/distortion adjustment; it does not identify causal codec effects. With 14 rows, correlated predictors, repeated PBC wrapper points, and 5 codec families, these are descriptive fits only. Do not interpret ordinary p-values or the fitted codec effects as strong evidence.

The rate/distortion-only residual table is a diagnostic, not a universal metric:

{chr(10).join(residual_lines)}

`rd_accessibility_model.json` stores coefficients, R²/adjusted R², rank, RMSE, and leave-one-out diagnostics for accuracy and U-probe, with and without codec indicators. `rd_adjusted_accessibility_residual.png` visualizes the in-sample diagnostic residuals.

Probe usable information is defined here as `U_probe_bits = (CE_train_prior − CE_model) / ln(2)`, where both cross-entropies are evaluated on held-out labels and the prior distribution is estimated from the train split. This is an empirical predictive diagnostic for each trained probe, **not Shannon mutual information**. Full per-image cross-entropy, true-class confidence, prediction, and correctness summaries are in `test_predictions_summary.csv.gz`; complete float16 logits live only in the ignored cache with an index and deterministic generation script.

## Cross-codec accessibility

The PBC and JPEG points with saved confirmation runs are three-seed means: PBC LZMA/STORE and JPEG q10/q60. JPEG q20/q80 raw-byte results are single seed. WebP, AVIF, and JXL raw-byte results are also single seed; their decoded-RGB endpoints use the three existing RGB checkpoint seeds. At the closest selected rates, JPEG q10 is 0.448 bpp and has 0.366 raw-byte accuracy, compared with WebP q30 at 0.598 bpp/0.244, AVIF q40 at 0.590 bpp/0.184, and JXL d6 at 0.569 bpp/0.204. JPEG q10 is higher in `{class_win_counts['jpeg_q10_vs_webp_q30']}/10` classes than WebP q30 and `{class_win_counts['jpeg_q10_vs_avif_q40']}/10` than AVIF q40 or `{class_win_counts['jpeg_q10_vs_jxl_d6']}/10` than JXL d6. At the mid-rate points, JPEG q60 is higher in `{class_win_counts['jpeg_q60_vs_webp_q75']}/10` classes than WebP q75, `{class_win_counts['jpeg_q60_vs_avif_q60']}/10` than AVIF q60, and `{class_win_counts['jpeg_q60_vs_jxl_d3']}/10` than JXL d3. These class counts come from seed 42; they do not measure modern-codec seed variance.

For PBC, STORE byte probes reach 0.358–0.394 accuracy across three seeds while same-pixel LZMA-wrapped streams reach 0.166–0.197. For JPEG, q10 is higher than q60 on raw-byte accuracy (0.366 vs 0.320), while decoded-RGB endpoints remain about 0.807–0.820. The six paired seed-42 comparisons and their image-bootstrap intervals are below; they compare trained probe outputs, not intrinsic codec properties.

## Paired cross-codec accessibility

Paired counts and deterministic image-bootstrap intervals use the same seed-42 test images. Exact McNemar values are descriptive because the codec probes are not multi-seed for WebP/AVIF/JXL.

{chr(10).join(paired_lines)}

The largest absolute paired accuracy difference in these six preselected comparisons is `{best_pair['condition_a']} vs {best_pair['condition_b']}` (`Δ={_format(best_pair['accuracy_difference_a_minus_b'],4)}`, 95% paired bootstrap CI [{_format(best_pair['bootstrap_ci95_low'],4)}, {_format(best_pair['bootstrap_ci95_high'],4)}]). This measures the paired outputs of these trained probes, not an intrinsic codec property.

## Per-class and difficulty-conditioned results

`per_class_metrics.csv` contains class-wise accuracy, cross-entropy, support, full confusion counts, matched decoded-RGB differences, and differences versus the low/mid JPEG comparator. `per_class_raw_byte_accuracy_heatmap.png` aggregates accuracy by class; `class_accuracy_rank_correlations.csv` reports Spearman correlations between class-accuracy profiles. Median pairwise class-profile correlation excluding self-correlations is `{_format(corr_med,3)}` (range `{_format(corr_min,3)}` to `{_format(corr_max,3)}`). The JPEG byte models most often confuse English springer and chain saw with tench; WebP/AVIF often map several classes to French horn, while JXL often confuses gas pump/chain saw with garbage truck. JPEG q10’s advantage is broad across most classes, not all ten. The detailed table shows exceptions and seed counts.

`difficulty_conditioned.csv` uses matched original-RGB predictions: correct with confidence ≥0.75, correct below 0.75, or incorrect. It reports raw-byte accuracy in each bucket and byte-correct/RGB-wrong versus RGB-correct/byte-wrong overlap. For seed-42 JPEG q10, raw accuracy is `{_format(high_confidence_q10['raw_accuracy'],3)}` on `{high_confidence_q10['n_samples']}` high-confidence RGB-correct images and `{_format(rgb_error_q10['raw_accuracy'],3)}` on `{rgb_error_q10['n_samples']}` RGB errors; it correctly classifies `{rgb_error_q10['byte_correct_rgb_wrong']}` RGB-error images, while missing `{rgb_correct_byte_wrong_q10}` images RGB classified correctly. The byte probe mainly succeeds on easier RGB cases but recovers a smaller complementary subset. This is a model-confidence stratification, not a ground-truth image-difficulty label. `per_image_rate_distortion_correlations.csv` reports Spearman associations between size/bpp, PSNR, SSIM and raw correctness, true-class probability, and cross-entropy. Across `{n_correlation_groups}` condition/seed groups, bpp has `{bpp_correct_rho}` with correctness; SSIM has `{ssim_correct_rho}`. `per_image_rate_distortion_quantiles.csv` gives the outcomes in equal-count predictor quartiles. These are associations within the measured samples, not causal effects; file size can reflect image content complexity as well as codec behavior.

## Absolute decode-cost frontier

`absolute_frontier.csv` uses measured conventional preprocessing/decode milliseconds and test accuracy plus U-probe bits. PBC uses Phase 2 stage timings; JPEG uses the refreshed Phase 4 timing pass; WebP/AVIF/JXL decoded endpoints reuse Phase 1 full-decode medians from the 95-image calibration set. The timing source and implementation caveat are repeated on every row. Cross-codec points form an **implementation-level systems frontier**, not an algorithm-only codec comparison. Fully staged codecs (PBC/JPEG) are visually distinguished from endpoint-only WebP/AVIF/JXL.

## Dominated points

`pareto_analysis.csv` identifies all dominators for each point. There are `{len(dominated_acc)}` accuracy-dominated and `{len(dominated_u)}` U-probe-dominated points among `{len(frontier)}` measured representation points. The accuracy-efficient points are `{', '.join(f'`{x}`' for x in efficient_acc)}`; the U-probe-efficient points are `{', '.join(f'`{x}`' for x in efficient_u)}`. Accuracy-dominated point IDs: `{', '.join(dominated_accuracy_ids)}`. U-probe-dominated point IDs: `{', '.join(dominated_utility_ids)}`. Every raw modern-codec file point is dominated at the measured 0 ms preprocessing coordinate; the closest-rate byte comparisons also favor JPEG q10 at a lower bpp than WebP q30, AVIF q40, and JXL d6.

Important examples:

{chr(10).join(f"- `{row['point_id']}` is accuracy-dominated by `{row['dominated_by_accuracy']}`." for row in dominated_acc.to_dict('records')[:12]) or '- No accuracy-dominated points under the computed mean-point rule.'}

{chr(10).join(f"- `{row['point_id']}` is U-probe-dominated by `{row['dominated_by_u_probe']}`." for row in dominated_u.to_dict('records')[:12]) or '- No U-probe-dominated points under the computed mean-point rule.'}

The refreshed JPEG numbers test the requested case directly: q60 DCT extraction is `{_format(float(frontier.loc[frontier['point_id']=='jpeg_q60::jpeg_dct_ir','preprocessing_ms'].iloc[0]),4)} ms` at `{_format(float(frontier.loc[frontier['point_id']=='jpeg_q60::jpeg_dct_ir','test_accuracy_mean'].iloc[0]),4)} accuracy, while q60 RGB decode is `{_format(float(frontier.loc[frontier['point_id']=='jpeg_q60::decoded_rgb_reference','preprocessing_ms'].iloc[0]),4)} ms` at `{_format(float(frontier.loc[frontier['point_id']=='jpeg_q60::decoded_rgb_reference','test_accuracy_mean'].iloc[0]),4)} accuracy. Under these measured means, q60 DCT is slower and less accurate, so q60 RGB dominates it on accuracy and preprocessing time. The Pareto table lists every point's dominators.

## What is genuinely interesting

- Raw-byte accessibility does not rise monotonically with rate or distortion: JPEG, WebP, and AVIF raw accuracy falls at the tested higher-fidelity point; JXL changes by `{_format(float(changes.loc[changes['codec_family']=='jxl','delta_raw_accuracy'].iloc[0]),4)}`. Within-image bpp correlations are also negative with byte correctness on these samples, consistent with larger files often representing more complex examples rather than being easier for the probe.
- The large PBC STORE/LZMA gap under pixel identity is a replicated, controlled wrapper effect from earlier phases, not a new discovery here. The stronger new evidence is that selected low-rate JPEG byte probes outperform near-rate WebP/AVIF/JXL probes on most classes, while rate/distortion-only fits explain only about one-third of accuracy/U variation in these 14 rows.
- JPEG DCT carries high probe utility, but q60 DCT is dominated by q60 RGB on both measured cost and performance. PBC IR, JPEG DCT, raw bytes, and RGB use different probe families, so these are representation-plus-probe observations rather than architecture-independent accessibility.

## What is not established

- No architecture-independent notion of neural accessibility has been validated.
- WebP/AVIF/JXL intermediate representations remain unmeasured; no intermediate points are fabricated here.
- WebP/AVIF/JXL raw-byte evidence is one seed; their codec-specific ranking uncertainty is not measured.
- Standard-codec differences cannot be causally attributed to entropy coding from these settings.
- The regression has 14 raw-representation rows across 11 selected codec-setting points; class and per-sample analyses are still conditional on the trained probes.
- Timing is implementation- and machine-dependent; WebP/AVIF/JXL timing source uses the Phase 1 calibration set, not Imagenette.
- The RGB reference is a modest random-initialized ResNet-style model, not a task ceiling.

## Go / no-go decision for image research

**continue only with narrow mechanistic experiments**. The current evidence still contains specific testable image mechanisms: PBC wrapper sensitivity under pixel identity and JPEG DCT utility close to the RGB probe. It does not justify a broad architecture sweep or claims about standard-codec entropy coding. If one narrow follow-up fails to reproduce across a second probe or split, shift most new effort to the planned video track.

## Recommended next experiments (not run)

1. Replicate the PBC STORE/LZMA pixel-identical control with one second, materially different byte probe. Scientific value high; compute low; risk of repeating the already-observed wrapper effect moderate.
2. Split JPEG DCT inputs into DC/low-frequency and high-frequency ablations at q10/q60. Scientific value high; compute low; risk of confirming the already strong DCT/RGB result moderate.
3. Re-evaluate the selected raw-byte conditions with a small second architecture using a fixed, matched batch protocol. Scientific value moderate-high; compute moderate; risk of finding similar rankings moderate.
4. Run byte-order/locality controls on Imagenette for the selected PBC/JPEG points. Scientific value moderate; compute low; risk of reproducing the early-testing order result moderate.
5. If one of these mechanisms survives, expose one public WebP/AVIF/JXL intermediate stage using a standard API and compare it with its existing file/RGB endpoints. Scientific value uncertain; compute/engineering cost high; risk of endpoint-only results is high.

## Reproduction

With the original ignored cache, checkpoints, source dataset, PBC checkout, and codec tools available:

Set `PBC_ROOT` to the local PBC checkout or pass `--pbc-root <path>` to the first command.

```powershell
python scripts/run_phase45b_analysis.py --device cuda
python scripts/report_phase45b_analysis.py
```

The first command performs decoding and inference only. It does not train or modify prior Phase 1–4 findings, caches, streams, or checkpoints. Summary tables, plots, and this report are new files under `results/Phase 4.5B Analysis/`.
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 4.5B post-hoc analysis artifacts and generate tables/plots/findings.")
    parser.add_argument("--results-root", type=Path, default=results_root())
    parser.add_argument("--cache-dir", type=Path, default=cache_root() / "neural_accessibility" / "imagenette_320")
    args = parser.parse_args()
    results_dir = args.results_root.resolve()
    cache_dir = args.cache_dir.resolve()
    out_dir = results_dir / "Phase 4.5B Analysis"
    required = ["imagenette_rd_per_sample.csv.gz", "test_predictions_summary.csv.gz", "decoded_rgb_endpoints.csv", "checkpoint_accuracy_recheck.json"]
    missing = [name for name in required if not (out_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"run scripts/run_phase45b_analysis.py first; missing {missing}")
    accuracy_path = out_dir / "checkpoint_accuracy_recheck.json"
    accuracy_recheck = json.loads(accuracy_path.read_text(encoding="utf-8"))
    for check in accuracy_recheck["checks"]:
        check["checkpoint"] = _portable_path(check["checkpoint"])
    _write_json(accuracy_path, accuracy_recheck)
    metadata_path = out_dir / "generation_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("manifest", "codec_manifest", "checkpoint_accuracy_recheck"):
            if metadata.get(key):
                metadata[key] = _portable_path(metadata[key])
        if metadata.get("rate_distortion", {}).get("path"):
            metadata["rate_distortion"]["path"] = _portable_path(metadata["rate_distortion"]["path"])
        for key in ("summary_path", "logits_cache"):
            if metadata.get("predictions", {}).get(key):
                metadata["predictions"][key] = _portable_path(metadata["predictions"][key])
        _write_json(metadata_path, metadata)
    rows = [json.loads(line) for line in (cache_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    train_labels = np.asarray([row["label"] for row in rows if row["split"] == "train"], dtype=np.int64)
    label_names = {int(row["label"]): row.get("label_name", str(row["label"])) for row in rows}
    rd = pd.read_csv(out_dir / "imagenette_rd_per_sample.csv.gz")
    jpeg_mask = rd["condition"].str.startswith("jpeg_") & rd["representation"].eq("codec_file_bytes")
    if jpeg_mask.any():
        rd.loc[jpeg_mask, "representation"] = "jpeg_file_bytes"
        rd.to_csv(out_dir / "imagenette_rd_per_sample.csv.gz", index=False, compression="gzip")
    predictions = pd.read_csv(out_dir / "test_predictions_summary.csv.gz")
    if predictions.duplicated(["image_id", "model_condition", "representation", "seed"]).any():
        raise ValueError("test predictions contain duplicate model/image rows")
    rd_summary = rd.groupby(["condition", "representation", "split"], as_index=False).agg(
        n=("image_id", "count"), mean_encoded_bytes=("encoded_bytes", "mean"), median_encoded_bytes=("encoded_bytes", "median"),
        std_encoded_bytes=("encoded_bytes", "std"), mean_bpp=("bpp", "mean"), median_bpp=("bpp", "median"), std_bpp=("bpp", "std"),
        mean_mse=("mse", "mean"), median_mse=("mse", "median"), mean_psnr_db=("psnr_db", "mean"), median_psnr_db=("psnr_db", "median"),
        std_psnr_db=("psnr_db", "std"), mean_ssim=("ssim", "mean"), median_ssim=("ssim", "median"), std_ssim=("ssim", "std"),
    )
    rd_summary.to_csv(out_dir / "imagenette_rd_summary.csv", index=False)
    probe, _ = _prediction_summaries(predictions, train_labels)
    probe.to_csv(out_dir / "probe_utility_summary.csv", index=False)
    endpoints = _write_enriched_endpoints(out_dir / "decoded_rgb_endpoints.csv")
    endpoint_summary = _endpoint_summary(endpoints)
    endpoint_summary.to_csv(out_dir / "decoded_rgb_endpoint_summary.csv", index=False)
    phase1 = pd.read_csv(results_dir / "Codec Benchmark" / "summary.csv")
    raw = _raw_table(rd, rd, predictions, probe, endpoints, phase1)
    raw.to_csv(out_dir / "raw_byte_accessibility_table.csv", index=False)
    changes = _within_codec_changes(raw, probe)
    changes.to_csv(out_dir / "within_codec_low_to_mid_changes.csv", index=False)
    paired = _paired_table(predictions)
    paired.to_csv(out_dir / "paired_codec_comparisons.csv", index=False)
    class_metrics = _per_class(predictions, label_names)
    class_metrics.to_csv(out_dir / "per_class_metrics.csv", index=False)
    class_corr = _class_rank_matrix(class_metrics, out_dir)
    class_corr_values = class_corr.to_numpy(dtype=np.float64)
    class_corr_values = class_corr_values[~np.eye(len(class_corr_values), dtype=bool)]
    class_profile_corr_median = float(np.nanmedian(class_corr_values))
    confusions = _top_confusions(class_metrics, out_dir)
    difficulty = _difficulty(predictions)
    difficulty.to_csv(out_dir / "difficulty_conditioned.csv", index=False)
    correlations = _per_image_correlations(rd, predictions)
    correlations.to_csv(out_dir / "per_image_rate_distortion_correlations.csv", index=False)
    quantiles = _per_image_quantiles(rd, predictions)
    quantiles.to_csv(out_dir / "per_image_rate_distortion_quantiles.csv", index=False)
    audit, specs = _training_audit(results_dir, cache_dir)
    audit.to_csv(out_dir / "training_convergence_audit.csv", index=False)
    rd_models, residuals = _rd_accessibility_models(raw)
    _write_json(out_dir / "rd_accessibility_model.json", rd_models)
    residuals.to_csv(out_dir / "rd_accessibility_residuals.csv", index=False)
    calibration = _plot_rate_distortion_comparison(rd, raw, results_dir, out_dir)
    frontier = _frontier(results_dir, rd, probe)
    _write_frontier_tables(frontier, out_dir)
    _plot_rd_accessibility(raw, out_dir, "imagenette_bpp_mean", "Imagenette test mean bitrate (bits per pixel)", "bpp_vs_raw_accessibility.png", "Imagenette bitrate vs raw-byte accessibility")
    _plot_rd_accessibility(raw, out_dir, "imagenette_psnr_mean_db", "Imagenette test mean PSNR (dB)", "psnr_vs_raw_accessibility.png", "Imagenette PSNR vs raw-byte accessibility")
    _plot_rd_accessibility(raw, out_dir, "imagenette_ssim_mean", "Imagenette test mean SSIM (unitless)", "ssim_vs_raw_accessibility.png", "Imagenette SSIM vs raw-byte accessibility")
    _plot_decoded_vs_raw(raw, out_dir)
    _plot_residuals(residuals, out_dir)
    _plot_frontier(frontier, out_dir, "test_accuracy_mean", "Held-out test accuracy (fraction)", "absolute_accuracy_decode_cost_frontier.png", "pareto_efficient_accuracy")
    _plot_frontier(frontier, out_dir, "u_probe_bits_mean", "Probe usable-information diagnostic (bits/example)", "probe_utility_decode_cost_frontier.png", "pareto_efficient_u_probe")
    convergence = _plot_convergence(audit, out_dir)
    class_profile = _plot_class_heatmap(class_metrics, out_dir)
    _plot_paired_overlap(paired, out_dir)
    best_pair = paired.reindex(paired["accuracy_difference_a_minus_b"].abs().sort_values(ascending=False).index).iloc[0].to_dict()
    model_fits = rd_models["fits"]
    dominated_accuracy = frontier[~frontier["pareto_efficient_accuracy"]]
    dominated_utility = frontier[~frontier["pareto_efficient_u_probe"]]
    phase1_comparison = calibration[calibration["split"] == "test"]
    summary = {
        "phase": "Phase 4.5B Post-hoc Analysis",
        "analysis_only": True,
        "training_performed": False,
        "new_codec_operating_points": 0,
        "dataset": "Imagenette 320px distribution, prepared 256x256 RGB; official validation held-out test split",
        "split_counts": {split: int(sum(row["split"] == split for row in rows)) for split in ("train", "validation", "test")},
        "rate_distortion_per_sample_rows": int(len(rd)),
        "per_image_rate_distortion_quantile_rows": int(len(quantiles)),
        "recomputed_classifier_checkpoints": int(len(json.loads((out_dir / "checkpoint_accuracy_recheck.json").read_text(encoding="utf-8"))["checks"])),
        "decoded_rgb_endpoint_conditions": sorted(endpoints["condition"].unique().tolist()),
        "decoded_rgb_endpoint_summary_test": endpoint_summary[endpoint_summary["split"] == "test"].to_dict("records"),
        "imagenette_test_rd_summary": rd_summary[rd_summary["split"] == "test"].to_dict("records"),
        "raw_byte_accessibility": raw.to_dict("records"),
        "within_codec_low_to_mid": changes.to_dict("records"),
        "difficulty_conditioned_jpeg_q10_seed42": difficulty[
            (difficulty["condition"] == "jpeg_q10") & (difficulty["seed"] == 42)
        ].to_dict("records"),
        "strongest_preselected_paired_difference": best_pair,
        "regression_models": model_fits,
        "rate_distortion_only_residual_accuracy": residuals[residuals["response"] == "raw_test_accuracy_mean"].to_dict("records"),
        "accuracy_dominated_frontier_points": dominated_accuracy[["point_id", "dominated_by_accuracy"]].to_dict("records"),
        "u_probe_dominated_frontier_points": dominated_utility[["point_id", "dominated_by_u_probe"]].to_dict("records"),
        "class_accuracy_profile_rank_median_excluding_self": class_profile_corr_median,
        "training_histories": int(len(audit)),
        "early_stopped_histories": int(audit["early_stopping_triggered_before_max_epochs"].sum()),
        "seed42_cross_codec_protocol": "same ByteStreamCNN, AdamW, LR=1e-3, weight_decay=1e-4, split, validation selection and no truncation; batch/device differs for PBC balanced/quality safety overrides",
        "calibration_comparison": phase1_comparison.to_dict("records"),
        "figures": ["bpp_vs_raw_accessibility.png", "psnr_vs_raw_accessibility.png", "ssim_vs_raw_accessibility.png", "decoded_rgb_vs_raw_accessibility.png", "rd_adjusted_accessibility_residual.png", "absolute_accuracy_decode_cost_frontier.png", "probe_utility_decode_cost_frontier.png", "raw_byte_validation_convergence.png", "per_class_raw_byte_accuracy_heatmap.png", "paired_codec_correctness_overlap.png"],
        "class_names": label_names,
        "frontier_scope": "implementation-level systems comparison; model family and timing implementation differ by representation/codec",
        "metric_definition": "U_probe_bits = (held-out CE under train-split empirical class prior minus model held-out CE)/ln(2); empirical predictive diagnostic, not mutual information",
        "reproduction": ["python scripts/run_phase45b_analysis.py --device cuda", "python scripts/report_phase45b_analysis.py"],
    }
    _write_json(out_dir / "summary.json", summary)
    _write_findings(out_dir / "Phase 4.5B Findings.md", rd_summary, raw, changes, probe, paired, class_metrics, class_corr, confusions, audit, frontier, rd_models, residuals, label_names, summary, endpoint_summary, correlations)
    print(json.dumps({"analysis_dir": str(out_dir), "rd_summary_rows": len(rd_summary), "raw_rows": len(raw), "paired_rows": len(paired), "frontier_points": len(frontier), "training_rows": len(audit), "plots": len(summary["figures"])}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create Phase 4.5B tables, plots, Pareto analysis, and findings.")
    parser.add_argument("--results-root", type=Path, default=results_root())
    parser.add_argument("--cache-dir", type=Path, default=cache_root() / "neural_accessibility" / "imagenette_320")
    args = parser.parse_args()
    main()
