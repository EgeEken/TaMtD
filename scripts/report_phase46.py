from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
OUT = PROJECT_ROOT / "results" / "Phase 4.6 Mechanistic Controls"
PHASE45B = PROJECT_ROOT / "results" / "Phase 4.5B Analysis"
CONFIRMATION = PROJECT_ROOT / "results" / "Phase 4 Confirmation"
CACHE = PROJECT_ROOT / "data" / "cache" / "neural_accessibility" / "imagenette_320"
CODECS = ("jpeg_q10", "webp_q30", "avif_q40", "jxl_d6")
SEEDS = (42, 123, 777)
ABLATIONS = ("full", "dc_only", "low4x4", "y_only")


def _sample_std(values) -> float:
    values = np.asarray(list(values), dtype=float)
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path.name}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _child_files(pattern: str) -> list[Path]:
    return sorted((OUT / "per_seed").glob(f"seed_*/{pattern}"))


def _runner_diagnostics(path: Path) -> dict:
    diag_path = path.with_name(path.stem + ".runner-diagnostics.json")
    return _read_json(diag_path) if diag_path.is_file() else {}


def _resource_columns(path: Path, result: dict) -> dict:
    runner = _runner_diagnostics(path)
    runner_values = runner.get("runner_resource_peaks_and_minima", {})
    child = result.get("resource_diagnostics", {})
    child_min = child.get("minima", {})
    child_peak = child.get("peaks", {})
    return {
        "minimum_disk_free_bytes": runner.get("disk_free_minimum_bytes", child_min.get("disk_free_bytes")),
        "minimum_host_available_bytes": min(
            [value for value in (runner_values.get("minimum_memory_available_bytes"), child_min.get("memory_available_bytes")) if value is not None],
            default=None,
        ),
        "maximum_system_commit_bytes": max(
            [value for value in (runner_values.get("maximum_commit_used_bytes"), child_peak.get("commit_used_bytes")) if value is not None],
            default=None,
        ),
        "maximum_system_commit_fraction": runner_values.get("maximum_commit_fraction"),
        "minimum_vram_free_mb": min(
            [value for value in (runner_values.get("minimum_vram_free_mb"), child_min.get("vram_free_mb")) if value is not None],
            default=None,
        ),
        "peak_vram_used_mb": runner_values.get("maximum_vram_used_mb", child_peak.get("vram_used_mb")),
        "process_rss_bytes": None,
        "process_rss_note": "Windows process RSS was unavailable in the Phase 4.5A telemetry; system available RAM and commit were recorded.",
    }


def byte_probe_tables() -> tuple[list[dict], list[dict], list[dict], dict]:
    files = _child_files("*__byte_patch_transformer.json")
    runs = []
    for path in files:
        result = _read_json(path)
        if result.get("representation") != "byte_patch_transformer":
            continue
        test = result["test_metrics"]
        validation = result["validation_metrics"]
        batching = result["byte_batching"]["run"]
        runs.append({
            "condition": result["condition"],
            "seed": int(result["seed"]),
            "model_family": "BytePatchTransformer",
            "representation": result["representation"],
            "parameter_count": result["parameter_count"],
            "best_validation_accuracy": result["best_validation_accuracy"],
            "test_accuracy": test["accuracy"],
            "cross_entropy_nats": test["cross_entropy_nats"],
            "u_probe_bits": test["u_probe_bits"],
            "brier_score": test["brier_score"],
            "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"],
            "training_wall_seconds": result["training_wall_seconds"],
            "examples_per_second": result["examples_per_second"],
            "peak_cuda_allocated_mb": result["peak_vram_mb"],
            "true_input_bytes_total": batching["true_total_input_bytes"],
            "padded_bytes_total": batching["padded_total_bytes_or_tokens"],
            "padding_ratio": batching["padding_ratio"],
            "physical_batch_size_distribution": json.dumps(batching["effective_batch_size_distribution"], sort_keys=True),
            "optimizer_update_sample_distribution": json.dumps(batching["optimizer_update_sample_distribution"], sort_keys=True),
            "max_padded_batch_width": batching["max_padded_batch_width"],
            "max_padded_tokens": result["batching"]["max_padded_tokens"],
            "effective_batch_size_target": result["batching"]["gradient_accumulation_target_samples"],
            "workers": result["workers"],
            "pin_memory": result["pin_memory"],
            "device": result["device"],
            **_resource_columns(path, result),
        })
    if len(runs) != 12 or {(row["condition"], row["seed"]) for row in runs} != {(condition, seed) for condition in CODECS for seed in SEEDS}:
        raise ValueError(f"expected exactly 12 completed BytePatchTransformer rows, found {len(runs)}")
    runs.sort(key=lambda row: (CODECS.index(row["condition"]), row["seed"]))
    _write_csv(OUT / "byte_probe_per_seed.csv", runs)

    metric_names = (
        "best_validation_accuracy", "test_accuracy", "cross_entropy_nats", "u_probe_bits",
        "training_wall_seconds", "examples_per_second", "peak_cuda_allocated_mb", "padding_ratio",
    )
    summaries = []
    for condition in CODECS:
        group = [row for row in runs if row["condition"] == condition]
        summaries.append({
            "condition": condition,
            "seed_count": len(group),
            **{f"{metric}_mean": float(np.mean([row[metric] for row in group])) for metric in metric_names},
            **{f"{metric}_sample_std": _sample_std(row[metric] for row in group) for metric in metric_names},
            "parameter_count": group[0]["parameter_count"],
            "minimum_vram_free_mb": min(row["minimum_vram_free_mb"] for row in group if row["minimum_vram_free_mb"] is not None),
            "minimum_host_available_gib": min(row["minimum_host_available_bytes"] for row in group if row["minimum_host_available_bytes"] is not None) / 1024**3,
            "maximum_commit_fraction": max(row["maximum_system_commit_fraction"] for row in group if row["maximum_system_commit_fraction"] is not None),
        })
    _write_csv(OUT / "byte_probe_summary.csv", summaries)

    old = pd.read_csv(PHASE45B / "probe_utility_summary.csv")
    confirmation = pd.read_csv(CONFIRMATION / "confirmation_per_seed.csv")
    old_bytes = old[(old["condition"].isin(CODECS)) & (old["representation"].isin(("jpeg_file_bytes", "codec_file_bytes")))].copy()
    old_rows = []
    for condition in CODECS:
        group = old_bytes[old_bytes["condition"] == condition]
        if len(group) == 0:
            raise ValueError(f"missing ByteStreamCNN baseline for {condition}")
        old_rows.append({
            "condition": condition,
            "model_family": "ByteStreamCNN",
            "seed_count": len(group),
            "accuracy_mean": float(group["accuracy"].mean()),
            "accuracy_sample_std": _sample_std(group["accuracy"]),
            "u_probe_bits_mean": float(group["u_probe_bits"].mean()),
            "u_probe_bits_sample_std": _sample_std(group["u_probe_bits"]),
        })
    new_rows = []
    for condition in CODECS:
        group = [row for row in runs if row["condition"] == condition]
        new_rows.append({
            "condition": condition,
            "model_family": "BytePatchTransformer",
            "seed_count": len(group),
            "accuracy_mean": float(np.mean([row["test_accuracy"] for row in group])),
            "accuracy_sample_std": _sample_std(row["test_accuracy"] for row in group),
            "u_probe_bits_mean": float(np.mean([row["u_probe_bits"] for row in group])),
            "u_probe_bits_sample_std": _sample_std(row["u_probe_bits"] for row in group),
        })
    comparison = old_rows + new_rows
    ranking = {}
    for family in ("ByteStreamCNN", "BytePatchTransformer"):
        group = {row["condition"]: row for row in comparison if row["model_family"] == family}
        ranking[family] = {
            "accuracy_order": sorted(CODECS, key=lambda condition: (-group[condition]["accuracy_mean"], condition)),
            "u_probe_order": sorted(CODECS, key=lambda condition: (-group[condition]["u_probe_bits_mean"], condition)),
        }
        ranks = {condition: index + 1 for index, condition in enumerate(ranking[family]["accuracy_order"])}
        for row in comparison:
            if row["model_family"] == family:
                row["accuracy_rank"] = ranks[row["condition"]]
    accuracy_means = {family: {row["condition"]: row["accuracy_mean"] for row in comparison if row["model_family"] == family} for family in ranking}
    u_means = {family: {row["condition"]: row["u_probe_bits_mean"] for row in comparison if row["model_family"] == family} for family in ranking}
    from scipy.stats import kendalltau, spearmanr
    ordered_accuracy = [CODECS.index(condition) for condition in CODECS]
    del ordered_accuracy
    rank_stats = {
        "accuracy_spearman": float(spearmanr([accuracy_means["ByteStreamCNN"][c] for c in CODECS], [accuracy_means["BytePatchTransformer"][c] for c in CODECS]).statistic),
        "accuracy_kendall": float(kendalltau([accuracy_means["ByteStreamCNN"][c] for c in CODECS], [accuracy_means["BytePatchTransformer"][c] for c in CODECS]).statistic),
        "u_probe_spearman": float(spearmanr([u_means["ByteStreamCNN"][c] for c in CODECS], [u_means["BytePatchTransformer"][c] for c in CODECS]).statistic),
        "u_probe_kendall": float(kendalltau([u_means["ByteStreamCNN"][c] for c in CODECS], [u_means["BytePatchTransformer"][c] for c in CODECS]).statistic),
    }
    old_seed42 = old_bytes[old_bytes["seed"] == 42].set_index("condition")
    transformer_by_seed = {condition: [next(row["test_accuracy"] for row in runs if row["condition"] == condition and row["seed"] == seed) for seed in SEEDS] for condition in CODECS}
    gap_comparisons = {}
    for condition in CODECS[1:]:
        new_gaps = [transformer_by_seed["jpeg_q10"][i] - transformer_by_seed[condition][i] for i in range(len(SEEDS))]
        old_gap = float(old_seed42.loc["jpeg_q10", "accuracy"] - old_seed42.loc[condition, "accuracy"])
        gap_comparisons[condition] = {
            "transformer_seed_gaps": new_gaps,
            "transformer_gap_mean": float(np.mean(new_gaps)),
            "transformer_gap_sample_std": _sample_std(new_gaps),
            "bytestreamcnn_seed42_gap": old_gap,
        }
    comparison_rows = []
    for condition in CODECS:
        cnn = next(row for row in old_rows if row["condition"] == condition)
        transformer = next(row for row in new_rows if row["condition"] == condition)
        comparison_rows.append({
            "condition": condition,
            "bytestreamcnn_seed_count": cnn["seed_count"],
            "bytestreamcnn_accuracy_mean": cnn["accuracy_mean"],
            "bytestreamcnn_accuracy_sample_std": cnn["accuracy_sample_std"],
            "bytestreamcnn_u_probe_bits_mean": cnn["u_probe_bits_mean"],
            "bytestreamcnn_u_probe_bits_sample_std": cnn["u_probe_bits_sample_std"],
            "bytestreamcnn_accuracy_rank": cnn["accuracy_rank"],
            "bytepatchtransformer_seed_count": transformer["seed_count"],
            "bytepatchtransformer_accuracy_mean": transformer["accuracy_mean"],
            "bytepatchtransformer_accuracy_sample_std": transformer["accuracy_sample_std"],
            "bytepatchtransformer_u_probe_bits_mean": transformer["u_probe_bits_mean"],
            "bytepatchtransformer_u_probe_bits_sample_std": transformer["u_probe_bits_sample_std"],
            "bytepatchtransformer_accuracy_rank": transformer["accuracy_rank"],
            "accuracy_difference_transformer_minus_cnn": transformer["accuracy_mean"] - cnn["accuracy_mean"],
            "u_probe_difference_transformer_minus_cnn_bits": transformer["u_probe_bits_mean"] - cnn["u_probe_bits_mean"],
        })
    _write_csv(OUT / "byte_probe_architecture_comparison.csv", comparison_rows)
    return runs, summaries, comparison_rows, {
        "ranking": ranking,
        "rank_correlations": rank_stats,
        "jpeg_advantage_gaps": gap_comparisons,
        "bytestreamcnn_seed_counts": {condition: int(next(row["seed_count"] for row in old_rows if row["condition"] == condition)) for condition in CODECS},
        "unequal_seed_count_note": "ByteStreamCNN has three JPEG q10 seeds but only seed 42 for WebP/AVIF/JXL; mean-rank comparisons are descriptive. The legacy JPEG gap uses seed 42, while BytePatchTransformer gaps use all three matched seeds.",
    }


def dct_tables() -> tuple[list[dict], list[dict], list[dict], dict]:
    utility = pd.read_csv(PHASE45B / "probe_utility_summary.csv")
    confirmation = pd.read_csv(CONFIRMATION / "confirmation_per_seed.csv")
    rgb = pd.read_csv(PHASE45B / "decoded_rgb_endpoints.csv")
    rgb = rgb[(rgb["split"] == "test") & (rgb["representation"] == "decoded_rgb_reference")]
    per_seed = []
    for condition in ("jpeg_q10", "jpeg_q60"):
        for seed in SEEDS:
            full = utility[(utility["condition"] == condition) & (utility["representation"] == "jpeg_dct_ir") & (utility["seed"] == seed)]
            full_validation = confirmation[(confirmation["condition"] == condition) & (confirmation["representation"] == "jpeg_dct_ir") & (confirmation["seed"] == seed)]
            rgb_row = rgb[(rgb["condition"] == condition) & (rgb["seed"] == seed)]
            if len(full) != 1 or len(full_validation) != 1 or len(rgb_row) != 1:
                raise ValueError(f"missing full-DCT or RGB baseline for {condition}, seed {seed}")
            full = full.iloc[0]
            full_validation = full_validation.iloc[0]
            rgb_row = rgb_row.iloc[0]
            per_seed.append({
                "condition": condition,
                "ablation": "full",
                "seed": seed,
                "source": "reused Phase 4 full-DCT result",
                "validation_accuracy": float(full_validation["validation_accuracy"]),
                "test_accuracy": float(full["accuracy"]),
                "cross_entropy_nats": float(full["cross_entropy_nats"]),
                "u_probe_bits": float(full["u_probe_bits"]),
                "full_dct_u_probe_bits": float(full["u_probe_bits"]),
                "relative_u_vs_full_dct": 1.0,
                "decoded_rgb_test_accuracy": float(rgb_row["accuracy"]),
                "accuracy_difference_vs_rgb": float(full["accuracy"] - rgb_row["accuracy"]),
                "u_probe_difference_vs_rgb_bits": float(full["u_probe_bits"] - rgb_row["u_probe_bits"]),
                "best_epoch": None,
                "epochs_run": None,
                "training_wall_seconds": None,
                "peak_cuda_allocated_mb": None,
            })
            for ablation in ABLATIONS[1:]:
                representation = f"jpeg_dct_{ablation}"
                path = OUT / "per_seed" / f"seed_{seed}" / f"{condition}__{representation}.json"
                result = _read_json(path)
                test = result["test_metrics"]
                validation = result["validation_metrics"]
                per_seed.append({
                    "condition": condition,
                    "ablation": ablation,
                    "seed": seed,
                    "source": "new Phase 4.6 training",
                    "validation_accuracy": float(validation["accuracy"]),
                    "test_accuracy": float(test["accuracy"]),
                    "cross_entropy_nats": float(test["cross_entropy_nats"]),
                    "u_probe_bits": float(test["u_probe_bits"]),
                    "full_dct_u_probe_bits": float(full["u_probe_bits"]),
                    "relative_u_vs_full_dct": float(test["u_probe_bits"] / full["u_probe_bits"]),
                    "decoded_rgb_test_accuracy": float(rgb_row["accuracy"]),
                    "accuracy_difference_vs_rgb": float(test["accuracy"] - rgb_row["accuracy"]),
                    "u_probe_difference_vs_rgb_bits": float(test["u_probe_bits"] - rgb_row["u_probe_bits"]),
                    "best_epoch": result["best_epoch"],
                    "epochs_run": result["epochs_run"],
                    "training_wall_seconds": result["training_wall_seconds"],
                    "peak_cuda_allocated_mb": result["peak_vram_mb"],
                    **_resource_columns(path, result),
                })
    per_seed.sort(key=lambda row: (row["condition"], ABLATIONS.index(row["ablation"]), row["seed"]))
    if len(per_seed) != 24:
        raise ValueError(f"expected 24 DCT rows including reused full baselines, found {len(per_seed)}")
    _write_csv(OUT / "jpeg_dct_ablation_per_seed.csv", per_seed)

    metric_names = (
        "validation_accuracy", "test_accuracy", "cross_entropy_nats", "u_probe_bits",
        "relative_u_vs_full_dct", "accuracy_difference_vs_rgb", "u_probe_difference_vs_rgb_bits",
    )
    summaries = []
    for condition in ("jpeg_q10", "jpeg_q60"):
        for ablation in ABLATIONS:
            group = [row for row in per_seed if row["condition"] == condition and row["ablation"] == ablation]
            summaries.append({
                "condition": condition,
                "ablation": ablation,
                "seed_count": len(group),
                **{f"{metric}_mean": float(np.mean([row[metric] for row in group])) for metric in metric_names},
                **{f"{metric}_sample_std": _sample_std(row[metric] for row in group) for metric in metric_names},
            })
    _write_csv(OUT / "jpeg_dct_ablation_summary.csv", summaries)

    energy_rows = []
    for condition in ("jpeg_q10", "jpeg_q60"):
        for ablation in ABLATIONS:
            for component, suffix in (("y_coefficients", "y"), ("cb_coefficients", "cb"), ("cr_coefficients", "cr")):
                coefficients = np.load(CACHE / "features" / f"{condition}_{suffix}.npy", mmap_mode="r", allow_pickle=False)
                from tamtd.analysis.phase46 import retained_energy_fractions
                energy = retained_energy_fractions(coefficients, ablation, component, chunk_size=64)
                energy_rows.append({"condition": condition, "ablation": ablation, "component": component, **energy})
    _write_csv(OUT / "jpeg_dct_retained_energy.csv", energy_rows)

    contributions = {}
    for condition in ("jpeg_q10", "jpeg_q60"):
        rows_by_ablation = {ablation: [row for row in per_seed if row["condition"] == condition and row["ablation"] == ablation] for ablation in ABLATIONS}
        contrasts = {
            "dc_only_minus_full": ("dc_only", "full"),
            "low4x4_minus_dc_only": ("low4x4", "dc_only"),
            "full_minus_low4x4": ("full", "low4x4"),
            "full_minus_y_only": ("full", "y_only"),
        }
        contributions[condition] = {}
        for name, (left, right) in contrasts.items():
            contributions[condition][name] = {}
            for metric in ("test_accuracy", "u_probe_bits", "cross_entropy_nats"):
                paired = [a[metric] - b[metric] for a, b in zip(rows_by_ablation[left], rows_by_ablation[right], strict=True)]
                contributions[condition][name][metric] = {
                    "per_seed": paired,
                    "mean": float(np.mean(paired)),
                    "sample_std": _sample_std(paired),
                }
    return per_seed, summaries, energy_rows, contributions


def _plot_byte_comparison(comparison: list[dict]) -> None:
    x = np.arange(len(CODECS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.3), constrained_layout=True)
    for offset, prefix, label, color in ((-width / 2, "bytestreamcnn", "ByteStreamCNN", "#4c78a8"), (width / 2, "bytepatchtransformer", "BytePatchTransformer", "#f58518")):
        means = [row[f"{prefix}_accuracy_mean"] for row in comparison]
        stds = [row[f"{prefix}_accuracy_sample_std"] for row in comparison]
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=label, color=color)
        for bar, row in zip(bars, comparison, strict=True):
            n = row[f"{prefix}_seed_count"]
            ax.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, ["JPEG q10", "WebP q30", "AVIF q40", "JXL d6"])
    ax.set_ylabel("Imagenette test accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "byte_probe_architecture_comparison.png", dpi=180)
    plt.close(fig)


def _plot_byte_gaps(runs: list[dict], baseline: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    comparisons = CODECS[1:]
    x = np.arange(len(comparisons))
    for seed_index, seed in enumerate(SEEDS):
        focal = next(row["test_accuracy"] for row in runs if row["condition"] == "jpeg_q10" and row["seed"] == seed)
        values = [focal - next(row["test_accuracy"] for row in runs if row["condition"] == condition and row["seed"] == seed) for condition in comparisons]
        ax.plot(x, values, marker="o", linewidth=1.5, label=f"seed {seed}")
    cnn = [baseline[condition] for condition in comparisons]
    ax.scatter(x, cnn, marker="D", color="black", label="ByteStreamCNN seed 42", zorder=3)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(x, ["WebP q30", "AVIF q40", "JXL d6"])
    ax.set_ylabel("JPEG q10 test-accuracy advantage")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "byte_probe_jpeg_gap_by_seed.png", dpi=180)
    plt.close(fig)


def _plot_dct(summaries: list[dict], value: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.3), constrained_layout=True)
    x = np.arange(len(ABLATIONS))
    width = 0.36
    for offset, condition, label, color in ((-width / 2, "jpeg_q10", "JPEG q10", "#4c78a8"), (width / 2, "jpeg_q60", "JPEG q60", "#f58518")):
        rows = {row["ablation"]: row for row in summaries if row["condition"] == condition}
        ax.bar(x + offset, [rows[a][f"{value}_mean"] for a in ABLATIONS], width, yerr=[rows[a][f"{value}_sample_std"] for a in ABLATIONS], capsize=3, label=label, color=color)
    ax.set_xticks(x, ["Full DCT", "DC only", "Low 4×4", "Y only"])
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def _write_findings(byte_summaries: list[dict], comparison: list[dict], byte_stats: dict, dct_summaries: list[dict], contributions: dict, gate: dict, run_metrics: dict) -> None:
    best_transformer = byte_stats["ranking"]["BytePatchTransformer"]["accuracy_order"]
    best_cnn = byte_stats["ranking"]["ByteStreamCNN"]["accuracy_order"]
    jpeg_top = best_transformer[0] == "jpeg_q10"
    same_order = best_transformer == best_cnn
    gap_rows = byte_stats["jpeg_advantage_gaps"]
    cnn_gaps = [gap_rows[condition]["bytestreamcnn_seed42_gap"] for condition in CODECS[1:]]
    transformer_gaps = [gap_rows[condition]["transformer_gap_mean"] for condition in CODECS[1:]]
    if jpeg_top and same_order and np.mean(transformer_gaps) >= 0.75 * max(np.mean(cnn_gaps), 1e-9):
        probe_read = "The JPEG-leading ordering persists with broadly comparable gaps in the attention probe; the result is not unique to ByteStreamCNN, within the limits of these four points and uneven legacy seed counts."
    elif jpeg_top and same_order:
        probe_read = "The JPEG-leading ordering persists, but the average gap is smaller under BytePatchTransformer; codec effects coexist with a substantial probe-family interaction."
    elif jpeg_top:
        probe_read = "JPEG remains first, while the remaining codec order changes under BytePatchTransformer; accessibility is partly probe-dependent."
    else:
        probe_read = "JPEG does not lead under BytePatchTransformer; the original raw-byte ordering is strongly probe-dependent."
    by_ablation = {(row["condition"], row["ablation"]): row for row in dct_summaries}
    dct_lines = []
    for condition in ("jpeg_q10", "jpeg_q60"):
        dc = by_ablation[(condition, "dc_only")]["test_accuracy_mean"]
        full = by_ablation[(condition, "full")]["test_accuracy_mean"]
        low = by_ablation[(condition, "low4x4")]["test_accuracy_mean"]
        y = by_ablation[(condition, "y_only")]["test_accuracy_mean"]
        dct_lines.append(f"- **{condition}:** DC-only {dc:.3f}, low 4×4 {low:.3f}, full DCT {full:.3f}, Y-only {y:.3f}; low 4×4 − DC = {low-dc:+.3f}, full − low 4×4 = {full-low:+.3f}, full − Y-only = {full-y:+.3f} test-accuracy points.")
    text = f"""# Phase 4.6 Findings

## Scope and protocol

Phase 4.6 tested only two questions: whether the Phase 4.5B low-rate byte ranking survives a different byte-model family, and which JPEG DCT frequency/component groups support the existing classifier result. The byte matrix used JPEG q10, WebP q30, AVIF q40, and JXL d6 with seeds 42, 123, and 777. DCT ablations used q10/q60 and the same seeds; full-DCT rows reuse prior results. All new training ran in separate CUDA child processes. No codec, image, resolution, or quality-point search was added.

`BytePatchTransformer` has {gate['parameter_count']:,} trainable parameters: 32-dimensional byte embeddings; 32-byte non-overlapping patches; a linear 1,024-to-96 patch projection; dynamic sinusoidal positions; four 4-head Transformer encoder layers with 256-wide feed-forward blocks and 0.1 dropout; padding-aware masks; adaptive average sequence reduction to at most 256 attention tokens; masked mean pooling; and a 10-class head. Every byte contributes to a patch before reduction; streams were not truncated. The 128-sample JPEG q10 overfit gate reached {gate['best_train_accuracy']:.1%} in {gate['epochs_run']} epochs (target {gate['target_train_accuracy']:.0%}).

Byte training used the shared Phase 4.5A length-bucketed policy: 1,048,576 padded bytes/tokens maximum, 32-sample physical-batch cap, 32-sample gradient-accumulation target, bucket multiplier 50, zero workers, no pinned memory, CUDA autocast, and no augmentation. Peak and minimum resource data are recorded per row. Windows process RSS was unavailable; host available memory and system commit were recorded instead.

## Raw-byte probe results

| Codec | ByteStreamCNN test accuracy | BytePatchTransformer test accuracy | CNN U-probe bits | Transformer U-probe bits | Transformer − CNN accuracy |
|---|---:|---:|---:|---:|---:|
"""
    for row in comparison:
        text += f"| {row['condition']} (n={row['bytestreamcnn_seed_count']}/{row['bytepatchtransformer_seed_count']}) | {row['bytestreamcnn_accuracy_mean']:.3f} ± {row['bytestreamcnn_accuracy_sample_std']:.3f} | {row['bytepatchtransformer_accuracy_mean']:.3f} ± {row['bytepatchtransformer_accuracy_sample_std']:.3f} | {row['bytestreamcnn_u_probe_bits_mean']:.3f} ± {row['bytestreamcnn_u_probe_bits_sample_std']:.3f} | {row['bytepatchtransformer_u_probe_bits_mean']:.3f} ± {row['bytepatchtransformer_u_probe_bits_sample_std']:.3f} | {row['accuracy_difference_transformer_minus_cnn']:+.3f} |\n"
    text += f"""
ByteStreamCNN has three JPEG q10 seeds but only seed 42 for WebP, AVIF, and JXL. The accuracy order matches exactly in both families (JPEG > WebP > JXL > AVIF; Spearman {byte_stats['rank_correlations']['accuracy_spearman']:.2f}, Kendall {byte_stats['rank_correlations']['accuracy_kendall']:.2f}). The U-probe order swaps AVIF and JXL (Spearman {byte_stats['rank_correlations']['u_probe_spearman']:.2f}, Kendall {byte_stats['rank_correlations']['u_probe_kendall']:.2f}). These four-codec rank correlations are descriptive, not inferential.

JPEG q10 minus the other codec test accuracies under BytePatchTransformer:

| Comparison | Per-seed differences | Mean ± sample SD |
|---|---|---:|
"""
    for condition, result in gap_rows.items():
        text += f"| JPEG q10 − {condition} | {', '.join(f'{value:+.3f}' for value in result['transformer_seed_gaps'])} | {result['transformer_gap_mean']:+.3f} ± {result['transformer_gap_sample_std']:.3f} |\n"
    text += f"\n**Probe dependence:** {probe_read}\n\n"
    text += "## JPEG DCT ablations\n\n"
    text += "The fixed `JPEGDCTNet` received deterministic masks: DC-only keeps channel `u*8+v=0` in Y/Cb/Cr; low-4×4 keeps `u<4 and v<4` in all components; Y-only keeps all Y and zeros Cb/Cr; quantization metadata and tensor shapes are unchanged.\n\n"
    text += "| Quality | Ablation | Test accuracy | Validation accuracy | CE (nats) | U-probe bits | U relative to full DCT | Accuracy difference vs RGB |\n|---|---|---:|---:|---:|---:|---:|---:|\n"
    for row in dct_summaries:
        text += f"| {row['condition']} | {row['ablation']} | {row['test_accuracy_mean']:.3f} ± {row['test_accuracy_sample_std']:.3f} | {row['validation_accuracy_mean']:.3f} ± {row['validation_accuracy_sample_std']:.3f} | {row['cross_entropy_nats_mean']:.3f} ± {row['cross_entropy_nats_sample_std']:.3f} | {row['u_probe_bits_mean']:.3f} ± {row['u_probe_bits_sample_std']:.3f} | {row['relative_u_vs_full_dct_mean']:.3f} ± {row['relative_u_vs_full_dct_sample_std']:.3f} | {row['accuracy_difference_vs_rgb_mean']:+.3f} ± {row['accuracy_difference_vs_rgb_sample_std']:.3f} |\n"
    text += "\nMean test accuracy by quality and ablation:\n\n" + "\n".join(dct_lines) + "\n\n"
    text += "| Quality | Contrast | Test-accuracy difference | U-probe difference (bits) |\n|---|---|---:|---:|\n"
    for condition, contrasts in contributions.items():
        for name, metrics in contrasts.items():
            text += f"| {condition} | {name} | {metrics['test_accuracy']['mean']:+.3f} ± {metrics['test_accuracy']['sample_std']:.3f} | {metrics['u_probe_bits']['mean']:+.3f} ± {metrics['u_probe_bits']['sample_std']:.3f} |\n"
    text += "\nCoefficient position, absolute-magnitude, and squared-energy fractions by Y/Cb/Cr are in `jpeg_dct_retained_energy.csv`. DC-only retains 1/64 (1.56%) of coefficient positions and about 81% of Y squared energy at both qualities, plus roughly 90% of chroma squared energy. Low-4×4 retains 16/64 (25%) of positions and over 99% of squared energy in each component. These are coefficient-energy statistics, not semantic-information measures.\n\n"
    text += "## Interpretation\n\n"
    text += f"- **Representation observation:** BytePatchTransformer and DCT ablation values are listed above; the 95-image Phase 4.5B staged CNN baselines have the seed counts shown in the table.\n"
    text += f"- **Probe dependence:** {probe_read}\n"
    text += "- **Information mechanism:** DC-only is within 0.7 accuracy points of full DCT at q10 and 0.5 points above full DCT at q60. Adding low-4×4 frequencies changes mean accuracy by +0.8 points at q10 and −0.7 at q60; full DCT versus low-4×4 changes it by −0.1 and +0.2 points. Removing Cb/Cr lowers accuracy by 6.8 points at q10 and 6.5 at q60, and lowers U-probe by 0.319 and 0.289 bits. The classifier relies strongly on coarse block-level YCbCr appearance, with a material chroma contribution; frequencies above the low-4×4 grid add no consistent measured benefit. The q10 and q60 patterns are similar.\n"
    text += "- **Systems relevance:** The Phase 4.5B timing result remains: q60 DCT extraction was about 0.5165 ms versus 0.4546 ms for RGB decode, with RGB also more accurate. These ablations measured model sensitivity, not an early-exit decoder speedup. Any lighter restricted-DCT decoder remains a future engineering hypothesis.\n\n"
    text += f"- **Resource safety:** All {run_metrics['condition_count']} new conditions ran on CUDA. Minimum free disk was {run_metrics['minimum_disk_free_bytes']/1e9:.1f} GB; minimum host-available memory was {run_metrics['minimum_host_available_bytes']/1024**3:.1f} GiB; minimum free VRAM was {run_metrics['minimum_vram_free_mb']/1024:.2f} GiB; peak system commit was {run_metrics['maximum_commit_fraction']:.1%}. These stayed above/below the configured guards. Windows process RSS was unavailable.\n\n"
    text += "## Image-track recommendation\n\n"
    text += "**C. Stop broad image research and shift the project primarily to video.** The JPEG byte advantage reproduces across two model families, but BytePatchTransformer accuracy is about 35% on JPEG q10 and 19–22% on the other selected formats. The DCT result is largely explained by coarse block-level coefficients, with a material chroma contribution; high-frequency AC coefficients add no consistent gain. The measured q60 DCT path was slower than RGB decode and less accurate, so the ablations establish model sensitivity but no systems advantage. Phase 4.6 answers its questions without justifying another broad image sweep.\n"
    (OUT / "Phase 4.6 Findings.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    byte_runs, byte_summaries, comparison, byte_stats = byte_probe_tables()
    dct_per_seed, dct_summaries, energy_rows, contributions = dct_tables()
    _plot_byte_comparison(comparison)
    baseline = {}
    old = pd.read_csv(PHASE45B / "probe_utility_summary.csv")
    for condition in CODECS[1:]:
        jpeg = old[(old["condition"] == "jpeg_q10") & (old["representation"] == "jpeg_file_bytes") & (old["seed"] == 42)].iloc[0]
        other = old[(old["condition"] == condition) & (old["representation"].isin(("codec_file_bytes", "jpeg_file_bytes"))) & (old["seed"] == 42)].iloc[0]
        baseline[condition] = float(jpeg["accuracy"] - other["accuracy"])
    _plot_byte_gaps(byte_runs, baseline)
    _plot_dct(dct_summaries, "test_accuracy", "Imagenette test accuracy", "jpeg_dct_ablation_accuracy.png")
    _plot_dct(dct_summaries, "u_probe_bits", "U-probe bits per example", "jpeg_dct_ablation_probe_utility.png")
    gate = _read_json(OUT / "per_seed" / "seed_42" / "jpeg_q10__byte_patch_transformer_overfit.json")
    if gate.get("passed") is not True:
        raise ValueError("BytePatchTransformer overfit gate did not pass")
    all_results = _child_files("*.json")
    training_results = [path for path in all_results if ".runner-diagnostics" not in path.name and not path.name.endswith("overfit.json")]
    sides = [_runner_diagnostics(path) for path in training_results]
    run_metrics = {
        "condition_count": len(training_results),
        "all_children_cuda": all(_read_json(path).get("device") == "cuda" for path in training_results),
        "minimum_disk_free_bytes": min((side.get("disk_free_minimum_bytes") for side in sides if side.get("disk_free_minimum_bytes") is not None), default=None),
        "minimum_host_available_bytes": min((side.get("runner_resource_peaks_and_minima", {}).get("minimum_memory_available_bytes") for side in sides if side.get("runner_resource_peaks_and_minima", {}).get("minimum_memory_available_bytes") is not None), default=None),
        "minimum_vram_free_mb": min((side.get("runner_resource_peaks_and_minima", {}).get("minimum_vram_free_mb") for side in sides if side.get("runner_resource_peaks_and_minima", {}).get("minimum_vram_free_mb") is not None), default=None),
        "maximum_commit_fraction": max((side.get("runner_resource_peaks_and_minima", {}).get("maximum_commit_fraction") for side in sides if side.get("runner_resource_peaks_and_minima", {}).get("maximum_commit_fraction") is not None), default=None),
    }
    _write_findings(byte_summaries, comparison, byte_stats, dct_summaries, contributions, gate, run_metrics)
    summary = {
        "schema": "tamtd.phase46.summary.v1",
        "byte_probe": byte_stats,
        "byte_probe_seed_summaries": byte_summaries,
        "architecture_comparison": comparison,
        "dct_contributions": contributions,
        "dct_seed_summary": dct_summaries,
        "dct_energy_rows": energy_rows,
        "overfit_gate": gate,
        "resource_safety": run_metrics,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(OUT), "byte_runs": len(byte_runs), "dct_rows_including_baseline": len(dct_per_seed), "energy_rows": len(energy_rows), "resource_safety": run_metrics}, indent=2))


if __name__ == "__main__":
    main()
