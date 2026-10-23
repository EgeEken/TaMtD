import csv
import json
import math
import statistics
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "Phase 4 Confirmation"
PHASE3_DIR = PROJECT_ROOT / "results" / "Neural Accessibility"
CACHE_ROOT = PROJECT_ROOT / "data" / "cache" / "neural_accessibility" / "imagenette_320"
PHASE4_CACHE = CACHE_ROOT / "phase4_streams"
CHANCE = 0.1
SEEDS = (42, 123, 777)
PBC_PRESETS = ("compression", "balanced", "quality")
JPEG_QUALITIES = (10, 60)
CROSS_CODEC = ("webp_q30", "webp_q75", "avif_q40", "avif_q60", "jxl_d6", "jxl_d3")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: list[float | None]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None, None
    return statistics.mean(finite), statistics.stdev(finite) if len(finite) > 1 else 0.0


def _phase3_rows() -> dict[tuple[str, str, int], dict]:
    rows = _read_json(PHASE3_DIR / "model_results.json")
    return {(row["condition"], row["representation"], int(row["seed"])): row for row in rows}


def _phase4_row(seed: int, condition: str, representation: str) -> dict:
    path = OUTPUT_DIR / "per_seed" / f"seed_{seed}" / f"{condition}__{representation}.json"
    return _read_json(path)


def _training_row(seed: int, condition: str, representation: str, phase3: dict) -> dict:
    if (condition, representation) == ("pbc_balanced", "pbc_lzma_bytes"):
        variant = OUTPUT_DIR / "per_seed" / f"seed_{seed}" / f"{condition}__{representation}__cpu_batch8.json"
        if variant.is_file():
            return _read_json(variant)
    path = OUTPUT_DIR / "per_seed" / f"seed_{seed}" / f"{condition}__{representation}.json"
    if path.is_file():
        return _read_json(path)
    if seed == 42 and (condition, representation, seed) in phase3:
        return phase3[(condition, representation, seed)]
    return _read_json(path)


def _decoded_refs(seed: int) -> dict[str, dict[str, float]]:
    if seed == 42:
        test_values = _read_json(PHASE3_DIR / "decoded_rgb_references.json")
        validation = _read_json(OUTPUT_DIR / "per_seed" / "seed_42" / "decoded_rgb_validation.json")
        return {
            condition: {
                "validation_accuracy": float(validation[condition]["validation_accuracy"]),
                "test_accuracy": float(test_values[condition]),
            }
            for condition in ("pbc_compression", "pbc_balanced", "pbc_quality", "jpeg_q10", "jpeg_q60")
        }
    rgb = _phase4_row(seed, "original_rgb", "decoded_rgb_reference")
    return rgb["decoded_rgb_references"]


def _condition_rows(phase3: dict) -> tuple[list[dict], dict[tuple[str, str, int], dict]]:
    per_seed = []
    lookup = {}
    training_settings = _read_json(PHASE3_DIR / "training_settings.json")
    phase3_device = _read_json(PHASE3_DIR / "config.json").get("device")
    for seed in SEEDS:
        references = _decoded_refs(seed)
        trained = []
        for preset in PBC_PRESETS:
            condition = f"pbc_{preset}"
            trained.extend((condition, representation) for representation in ("pbc_lzma_bytes", "pbc_store_bytes", "pbc_canonical_ir"))
        for quality in JPEG_QUALITIES:
            condition = f"jpeg_q{quality}"
            trained.extend((condition, representation) for representation in ("jpeg_file_bytes", "jpeg_dct_ir"))
        trained.append(("original_rgb", "decoded_rgb_reference"))
        for condition, representation in trained:
            row = _training_row(seed, condition, representation, phase3)
            reference_condition = condition if condition != "original_rgb" else "original_rgb"
            if condition.startswith("pbc_") or condition.startswith("jpeg_"):
                reference = references[reference_condition]
                ref_accuracy = float(reference["test_accuracy"])
                validation_accuracy = float(row["best_validation_accuracy"])
                if representation == "decoded_rgb_reference":
                    validation_accuracy = float(reference["validation_accuracy"])
                ref_kind = "codec-decoded RGB"
            else:
                ref_accuracy = float(row["test_accuracy"])
                validation_accuracy = float(row["best_validation_accuracy"])
                ref_kind = "prepared source RGB"
            semantic = (float(row["test_accuracy"]) - CHANCE) / (ref_accuracy - CHANCE)
            family = "byte" if representation in {"pbc_lzma_bytes", "pbc_store_bytes", "jpeg_file_bytes", "codec_file_bytes"} else "pbc" if representation == "pbc_canonical_ir" else "jpeg_dct" if representation == "jpeg_dct_ir" else "rgb"
            batch_size = row.get("training_settings", {}).get("batch_size", training_settings[family]["batch_size"])
            device = row.get("device", phase3_device)
            output = {
                "seed": seed,
                "condition": condition,
                "representation": representation,
                "test_accuracy": float(row["test_accuracy"]),
                "validation_accuracy": validation_accuracy,
                "reference_test_accuracy": ref_accuracy,
                "semantic_accessibility": semantic,
                "reference_kind": ref_kind,
                "training_wall_seconds": row.get("training_wall_seconds"),
                "examples_per_second": row.get("examples_per_second"),
                "peak_vram_mb": row.get("peak_vram_mb"),
                "training_batch_size": batch_size,
                "training_device": device,
            }
            per_seed.append(output)
            lookup[(condition, representation, seed)] = output
        for condition, reference in references.items():
            output = {
                "seed": seed,
                "condition": condition,
                "representation": "decoded_rgb_reference",
                "test_accuracy": float(reference["test_accuracy"]),
                "validation_accuracy": float(reference["validation_accuracy"]),
                "reference_test_accuracy": float(reference["test_accuracy"]),
                "semantic_accessibility": 1.0,
                "reference_kind": "codec-decoded RGB",
                "training_wall_seconds": None,
                "examples_per_second": None,
                "peak_vram_mb": None,
                "training_batch_size": None,
                "training_device": None,
            }
            per_seed.append(output)
            lookup[(condition, "decoded_rgb_reference", seed)] = output
    return per_seed, lookup


def _confirmation_summary(per_seed: list[dict], lookup: dict) -> list[dict]:
    keys = []
    for preset in PBC_PRESETS:
        condition = f"pbc_{preset}"
        keys.extend((condition, rep) for rep in ("pbc_lzma_bytes", "pbc_store_bytes", "pbc_canonical_ir", "decoded_rgb_reference"))
    for quality in JPEG_QUALITIES:
        condition = f"jpeg_q{quality}"
        keys.extend((condition, rep) for rep in ("jpeg_file_bytes", "jpeg_dct_ir", "decoded_rgb_reference"))
    keys.append(("original_rgb", "decoded_rgb_reference"))
    results = []
    for condition, representation in keys:
        rows = [lookup[(condition, representation, seed)] for seed in SEEDS]
        accuracy_mean, accuracy_std = _stats([row["test_accuracy"] for row in rows])
        validation_mean, validation_std = _stats([row["validation_accuracy"] for row in rows])
        semantic_mean, semantic_std = _stats([row["semantic_accessibility"] for row in rows])
        reference_mean, _ = _stats([row["reference_test_accuracy"] for row in rows])
        time_mean, _ = _stats([row["training_wall_seconds"] for row in rows])
        throughput_mean, _ = _stats([row["examples_per_second"] for row in rows])
        vram_mean, _ = _stats([row["peak_vram_mb"] for row in rows])
        from_means = None if accuracy_mean is None or reference_mean is None else (accuracy_mean - CHANCE) / (reference_mean - CHANCE)
        results.append({
            "condition": condition,
            "representation": representation,
            "seed_count": len(rows),
            "test_accuracy_mean": accuracy_mean,
            "test_accuracy_std": accuracy_std,
            "validation_accuracy_mean": validation_mean,
            "validation_accuracy_std": validation_std,
            "reference_test_accuracy_mean": reference_mean,
            "semantic_accessibility_mean_per_seed": semantic_mean,
            "semantic_accessibility_std_per_seed": semantic_std,
            "semantic_accessibility_from_means": from_means,
            "training_wall_seconds_mean": time_mean,
            "examples_per_second_mean": throughput_mean,
            "peak_vram_mb_mean": vram_mean,
            "training_batch_sizes": sorted({lookup[(condition, representation, seed)]["training_batch_size"] for seed in SEEDS if lookup[(condition, representation, seed)]["training_batch_size"] is not None}),
            "training_devices": sorted({lookup[(condition, representation, seed)]["training_device"] for seed in SEEDS if lookup[(condition, representation, seed)]["training_device"] is not None}),
        })
    return results


def _cache_mean_bpp(condition: str, core_rows: list[dict]) -> tuple[float, float]:
    if condition in {"pbc_compression_lzma", "pbc_compression_store", "pbc_balanced_lzma", "pbc_balanced_store", "pbc_quality_lzma", "pbc_quality_store", "jpeg_q10", "jpeg_q60"}:
        lengths = [int(row["streams"][condition]["length"]) for row in core_rows]
    else:
        additions = _read_jsonl(PHASE4_CACHE / f"{condition}.jsonl")
        lengths = [int(row["stream"]["length"]) for row in additions]
    return statistics.mean(lengths) * 8 / (256 * 256), statistics.mean(lengths)


def _raw_byte_rows(phase3: dict, codec_summary: dict, core_rows: list[dict]) -> list[dict]:
    points = [
        ("low-rate", "PBC LZMA", "pbc_compression", "pbc_lzma_bytes", "pbc_compression_lzma", "pbc_compression"),
        ("low-rate", "PBC STORE", "pbc_compression", "pbc_store_bytes", "pbc_compression_store", "pbc_compression"),
        ("low-rate", "JPEG", "jpeg_q10", "jpeg_file_bytes", "jpeg_q10", "jpeg_q10"),
        ("low-rate", "WebP", "webp_q30", "codec_file_bytes", "webp_q30", "webp_q30"),
        ("low-rate", "AVIF", "avif_q40", "codec_file_bytes", "avif_q40", "avif_q40"),
        ("low-rate", "JXL", "jxl_d6", "codec_file_bytes", "jxl_d6", "jxl_d6"),
        ("mid-rate", "PBC LZMA", "pbc_balanced", "pbc_lzma_bytes", "pbc_balanced_lzma", "pbc_balanced"),
        ("mid-rate", "PBC STORE", "pbc_balanced", "pbc_store_bytes", "pbc_balanced_store", "pbc_balanced"),
        ("mid-rate", "JPEG", "jpeg_q60", "jpeg_file_bytes", "jpeg_q60", "jpeg_q60"),
        ("mid-rate", "WebP", "webp_q75", "codec_file_bytes", "webp_q75", "webp_q75"),
        ("mid-rate", "AVIF", "avif_q60", "codec_file_bytes", "avif_q60", "avif_q60"),
        ("mid-rate", "JXL", "jxl_d3", "codec_file_bytes", "jxl_d3", "jxl_d3"),
    ]
    results = []
    for rate_band, codec, condition, representation, cache_name, phase1_name in points:
        model = _training_row(42, condition, representation, phase3)
        bpp, mean_bytes = _cache_mean_bpp(cache_name, core_rows)
        phase1 = codec_summary.get(phase1_name, {})
        phase1_bpp = None if representation == "pbc_store_bytes" else float(phase1["mean_bpp"])
        phase1_psnr = float(phase1["mean_psnr_db"])
        results.append({
            "rate_band": rate_band,
            "codec": codec,
            "condition": condition,
            "representation": representation,
            "seed": 42,
            "test_accuracy": float(model["test_accuracy"]),
            "phase4_mean_bpp": bpp,
            "phase4_mean_bytes": mean_bytes,
            "phase1_mean_bpp": phase1_bpp,
            "phase1_mean_psnr_db": phase1_psnr,
            "phase1_setting": phase1.get("settings"),
        })
    return results


def _preprocessing_times(timing: dict) -> dict[str, float]:
    stage = _read_json(PROJECT_ROOT / "results" / "Staged Decoding" / "summary.json")
    conditions = {item["condition"]: item for item in stage["conditions"]}
    output = {}
    for preset in PBC_PRESETS:
        forced = conditions[f"pbc_{preset}_forced_lzma"]
        store = conditions[f"pbc_{preset}_store"]
        unpack = forced["stage_timings_ms"]["forced_lzma_unpack"]["median"]
        parse = store["stage_timings_ms"]["store_body_parse"]["median"]
        canonical = store["stage_timings_ms"]["canonical_ir_build"]["median"]
        full = forced["full_decode_ms"]["median"]
        output[f"pbc_{preset}"] = {"lzma_bytes": 0.0, "store_bytes": unpack, "canonical_ir": unpack + parse + canonical, "decoded_rgb": full}
    for quality in JPEG_QUALITIES:
        condition = f"jpeg_q{quality}"
        measured = timing["conditions"][condition]
        output[condition] = {
            "jpeg_file_bytes": 0.0,
            "jpeg_dct_ir": float(measured["file_to_dct_ir_ms"]["median"]),
            "decoded_rgb": float(measured["full_rgb_decode_ms"]["median"]),
        }
    return output


def _frontier_rows(lookup: dict, timing: dict) -> list[dict]:
    times = _preprocessing_times(timing)
    rows = []
    for preset in PBC_PRESETS:
        condition = f"pbc_{preset}"
        for representation, stage in (("pbc_lzma_bytes", "lzma_bytes"), ("pbc_store_bytes", "store_bytes"), ("pbc_canonical_ir", "canonical_ir"), ("decoded_rgb_reference", "decoded_rgb")):
            values = [lookup[(condition, representation, seed)]["test_accuracy"] for seed in SEEDS]
            mean, std = _stats(values)
            rows.append({"condition": condition, "representation": representation, "preprocessing_ms": times[condition][stage], "test_accuracy_mean": mean, "test_accuracy_std": std})
    for quality in JPEG_QUALITIES:
        condition = f"jpeg_q{quality}"
        for representation, stage in (("jpeg_file_bytes", "jpeg_file_bytes"), ("jpeg_dct_ir", "jpeg_dct_ir"), ("decoded_rgb_reference", "decoded_rgb")):
            values = [lookup[(condition, representation, seed)]["test_accuracy"] for seed in SEEDS]
            mean, std = _stats(values)
            rows.append({"condition": condition, "representation": representation, "preprocessing_ms": times[condition][stage], "test_accuracy_mean": mean, "test_accuracy_std": std})
    return rows


def _plot_frontier(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
    colors = {
        "pbc_compression": "#4c78a8",
        "pbc_balanced": "#f58518",
        "pbc_quality": "#54a24b",
        "jpeg_q10": "#b279a2",
        "jpeg_q60": "#e45756",
    }
    markers = {"pbc_lzma_bytes": "o", "jpeg_file_bytes": "o", "pbc_store_bytes": "s", "pbc_canonical_ir": "D", "jpeg_dct_ir": "D", "decoded_rgb_reference": "^"}
    for condition in colors:
        values = [row for row in rows if row["condition"] == condition]
        values.sort(key=lambda row: row["preprocessing_ms"])
        xs = [row["preprocessing_ms"] for row in values]
        ys = [row["test_accuracy_mean"] for row in values]
        ax.plot(xs, ys, color=colors[condition], alpha=0.7, linewidth=1.5, label=condition)
        for row in values:
            ax.errorbar(
                row["preprocessing_ms"], row["test_accuracy_mean"], yerr=row["test_accuracy_std"],
                fmt=markers[row["representation"]], color=colors[condition], capsize=3, markersize=6,
            )
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("Conventional preprocessing latency per image (ms; symlog)")
    ax.set_ylabel("Imagenette test accuracy (mean ± sample SD across seeds)")
    ax.set_title("Phase 4 multi-seed codec accessibility frontier")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_raw_bytes(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.7), constrained_layout=True)
    colors = {"PBC LZMA": "#4c78a8", "PBC STORE": "#72b7b2", "JPEG": "#e45756", "WebP": "#54a24b", "AVIF": "#f58518", "JXL": "#b279a2"}
    markers = {"low-rate": "o", "mid-rate": "s"}
    label_offsets = {
        ("PBC LZMA", "pbc_compression"): ((36, 12), "left"),
        ("AVIF", "avif_q40"): ((8, -12), "left"),
        ("PBC STORE", "pbc_balanced"): ((-8, -12), "right"),
    }
    for row in rows:
        ax.scatter(row["phase4_mean_bpp"], row["test_accuracy"], s=70, marker=markers[row["rate_band"]], color=colors[row["codec"]], label=row["codec"])
        offset, alignment = label_offsets.get((row["codec"], row["condition"]), ((5, 5), "left"))
        ax.annotate(f"{row['codec']} {row['condition']}", (row["phase4_mean_bpp"], row["test_accuracy"]), xytext=offset, textcoords="offset points", ha=alignment, fontsize=7)
    handles = []
    for name, color in colors.items():
        handles.append(plt.Line2D([], [], color=color, marker="o", linestyle="", label=name))
    for name, marker in markers.items():
        handles.append(plt.Line2D([], [], color="black", marker=marker, linestyle="", label=name))
    ax.legend(handles=handles, fontsize=8, ncol=2)
    ax.set_xlabel("Mean encoded bitrate on Phase 3 Imagenette cache (bits/pixel)")
    ax.set_ylabel("Raw-byte model test accuracy (seed 42)")
    ax.set_title("Raw-byte accessibility at Phase 1 low and mid-rate points")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _timing_rows(timing: dict) -> list[dict]:
    rows = []
    for quality in (10, 20, 60, 80):
        condition = timing["conditions"][f"jpeg_q{quality}"]
        rows.append({
            "condition": f"jpeg_q{quality}",
            "file_to_dct_ir_median_ms": condition["file_to_dct_ir_ms"]["median"],
            "file_to_dct_ir_p95_ms": condition["file_to_dct_ir_ms"]["p95"],
            "full_rgb_decode_median_ms": condition["full_rgb_decode_ms"]["median"],
            "full_rgb_decode_p95_ms": condition["full_rgb_decode_ms"]["p95"],
            "median_ratio_dct_to_full_decode": condition["ratio_dct_to_full_decode"],
        })
    return rows


def _plot_timing(rows: list[dict], path: Path) -> None:
    labels = [row["condition"].replace("jpeg_", "") for row in rows]
    dct = [row["file_to_dct_ir_median_ms"] for row in rows]
    rgb = [row["full_rgb_decode_median_ms"] for row in rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    ax.bar(x - width / 2, dct, width, label="File to DCT IR", color="#4c78a8")
    ax.bar(x + width / 2, rgb, width, label="Full RGB decode", color="#f58518")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median time per image (ms)")
    ax.set_xlabel("JPEG quality")
    ax.set_title("Consistent JPEG timing refresh on 95 calibration images")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _mean_sd(mean: float | None, std: float | None, multiplier: float = 1.0, digits: int = 2) -> str:
    if mean is None:
        return "n/a"
    return f"{mean * multiplier:.{digits}f} ± {std * multiplier:.{digits}f}"


def _write_findings(summary: list[dict], per_seed: list[dict], lookup: dict, raw: list[dict], timing_rows: list[dict], timing: dict) -> None:
    by_key = {(row["condition"], row["representation"]): row for row in summary}
    lines = [
        "# Phase 4 Findings",
        "",
        "## Scope and protocol",
        "",
        "- Phase 3 seed 42 results were reused where present. The longer PBC byte conditions use one reduced batch size per condition across all three seeds after GPU runs approached the system commit limit; all bytes and the model architecture were retained. Balanced LZMA/STORE and both quality byte streams ran on CPU to keep system commit below the safety stop. Each training ran in a fresh Python process with workers=0 and pin_memory=False.",
        "- Phase 4A includes PBC compression/balanced/quality byte, STORE, canonical IR, and common RGB references; JPEG q10/q60 file bytes, DCT IR, and decoded RGB references; and the original RGB reference.",
        "- Phase 4B adds raw-byte-only WebP q30/q75, AVIF q40/q60, and JXL distance 6/3 at seed 42. The selections follow the Phase 1 measured low/mid rate matches.",
        "- Checkpoints, logs, and codec streams remain in ignored cache/result storage and are not part of the compact report artifacts.",
        "",
        "## Multi-seed confirmation",
        "",
        "Test and validation columns show mean ± sample standard deviation across seeds 42, 123, and 777. For decoded RGB references, validation accuracy is measured with the common RGB model selected on original validation data; it is not used for checkpoint selection.",
        "",
        "| Condition | Representation | Batch/device | Test accuracy | Validation accuracy | Semantic accessibility | Mean train time (s) | Mean examples/s | Mean peak VRAM (MB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        semantic = row["semantic_accessibility_from_means"]
        semantic_pair = _mean_sd(row["semantic_accessibility_mean_per_seed"], row["semantic_accessibility_std_per_seed"], 1, 3)
        test = _mean_sd(row["test_accuracy_mean"], row["test_accuracy_std"], 100, 2) + "%"
        validation = _mean_sd(row["validation_accuracy_mean"], row["validation_accuracy_std"], 100, 2) + "%"
        if row["representation"] == "decoded_rgb_reference":
            semantic_text = f"{semantic_pair} (from means {semantic:.3f})" if semantic is not None else semantic_pair
        else:
            semantic_text = f"{semantic_pair} (from means {semantic:.3f})" if semantic is not None else semantic_pair
        train_time = "n/a" if row["training_wall_seconds_mean"] is None else f"{row['training_wall_seconds_mean']:.1f}"
        throughput = "n/a" if row["examples_per_second_mean"] is None else f"{row['examples_per_second_mean']:.1f}"
        vram = "n/a" if row["peak_vram_mb_mean"] is None else f"{row['peak_vram_mb_mean']:.1f}"
        batch_sizes = ", ".join(str(value) for value in row["training_batch_sizes"]) or "n/a"
        devices = ", ".join(row["training_devices"]) or "n/a"
        batch_device = f"{batch_sizes}/{devices}" if batch_sizes != "n/a" else "n/a"
        lines.append(
            f"| {row['condition']} | {row['representation']} | {batch_device} | {test} | {validation} | {semantic_text} | "
            f"{train_time} | {throughput} | {vram} |"
        )
    lines.extend(["", "Semantic accessibility uses `(model accuracy − 0.1) / (matched decoded-RGB accuracy − 0.1)`. The `from means` value applies that formula to the three-seed mean accuracies, as specified.", ""])

    comparison_lines = ["| Comparison | Seed-wise result |", "| --- | --- |"]
    for preset in PBC_PRESETS:
        condition = f"pbc_{preset}"
        lzma = [row for row in per_seed if row["condition"] == condition and row["representation"] == "pbc_lzma_bytes"]
        store = [row for row in per_seed if row["condition"] == condition and row["representation"] == "pbc_store_bytes"]
        deltas = [b["test_accuracy"] - a["test_accuracy"] for a, b in zip(lzma, store)]
        mean, sd = _stats(deltas)
        comparison_lines.append(f"| {condition}: STORE − LZMA | {_mean_sd(mean, sd, 100, 2)} percentage points; positive in {sum(value > 0 for value in deltas)}/{len(deltas)} seeds |")
    for quality in JPEG_QUALITIES:
        condition = f"jpeg_q{quality}"
        raw_rows = [row for row in per_seed if row["condition"] == condition and row["representation"] == "jpeg_file_bytes"]
        dct_rows = [row for row in per_seed if row["condition"] == condition and row["representation"] == "jpeg_dct_ir"]
        deltas = [d["test_accuracy"] - b["test_accuracy"] for b, d in zip(raw_rows, dct_rows)]
        mean, sd = _stats(deltas)
        comparison_lines.append(f"| {condition}: DCT IR − file bytes | {_mean_sd(mean, sd, 100, 2)} percentage points; positive in {sum(value > 0 for value in deltas)}/{len(deltas)} seeds |")
        decoded = [row for row in per_seed if row["condition"] == condition and row["representation"] == "decoded_rgb_reference"]
        gaps = [ref["test_accuracy"] - dct["test_accuracy"] for ref, dct in zip(decoded, dct_rows)]
        gap_mean, gap_sd = _stats(gaps)
        comparison_lines.append(f"| {condition}: decoded RGB − DCT IR | {_mean_sd(gap_mean, gap_sd, 100, 2)} percentage points |")
    lines.extend(["## Direct checks of the Phase 3 claims", "", *comparison_lines, ""])

    pbc_access = []
    for preset in PBC_PRESETS:
        row = by_key[(f"pbc_{preset}", "pbc_canonical_ir")]
        pbc_access.append(f"{preset}: {row['semantic_accessibility_from_means']:.3f} semantic accessibility at {row['test_accuracy_mean']:.3f} mean test accuracy")
    stage_times = _preprocessing_times(timing)
    compression_access = by_key[("pbc_compression", "pbc_canonical_ir")]["semantic_accessibility_from_means"]
    q10_access = by_key[("jpeg_q10", "jpeg_dct_ir")]["semantic_accessibility_from_means"]
    compression_fraction = stage_times["pbc_compression"]["canonical_ir"] / stage_times["pbc_compression"]["decoded_rgb"]
    q10_fraction = stage_times["jpeg_q10"]["jpeg_dct_ir"] / stage_times["jpeg_q10"]["decoded_rgb"]
    low_store_deltas = [
        lookup[("pbc_compression", "pbc_store_bytes", seed)]["test_accuracy"] - lookup[("pbc_compression", "pbc_lzma_bytes", seed)]["test_accuracy"]
        for seed in SEEDS
    ]
    lines.extend([
        "### PBC LZMA to STORE",
        "",
        f"The paired changes above measure the same PBC body with and without the outer LZMA wrapper. Compression improves by {_mean_sd(*_stats(low_store_deltas), 100, 2)} percentage points on average. This controlled result supports a conclusion about PBC's wrapper and this byte CNN, not all entropy-coded formats.",
        "",
        "### PBC canonical IR",
        "",
        "Canonical IR semantic accessibility calculated from multi-seed means: " + "; ".join(pbc_access) + ".",
        "",
        "### Intermediate preprocessing trade-off",
        "",
        f"At PBC compression, canonical IR reaches {compression_access:.3f} semantic accessibility after {compression_fraction:.1%} of that preset's full decode time. JPEG q10 DCT IR reaches {q10_access:.3f} after {q10_fraction:.1%} of its full RGB decode time. The absolute timings are in the refreshed table; PBC has a lower relative-to-full-decode cost at this point, while JPEG exposes more class information.",
        "",
        "### JPEG DCT IR",
        "",
        "The paired q10/q60 DCT-minus-file changes and DCT-to-decoded-RGB gaps are in the table above. The refreshed JPEG timing shows DCT access and full RGB decode in the same pass and environment; it does not include JPEG encode time.",
        "",
        "## Cross-codec raw-byte extension",
        "",
        "The low-rate points were selected near PBC compression and the mid-rate points near PBC balanced/JPEG q60 using Phase 1's 95-image rate measurements. Model comparisons use seed 42 for all codecs. Phase 4 mean bpp is measured on the Imagenette cache; Phase 1 rate/PSNR are shown as calibration context.",
        "",
        "| Rate band | Codec | Setting | Phase 1 bpp | Phase 1 PSNR (dB) | Imagenette bpp | Raw-byte test accuracy |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in raw:
        p1_bpp = "n/a" if row["phase1_mean_bpp"] is None else f"{row['phase1_mean_bpp']:.3f}"
        p1_psnr = "n/a" if row["phase1_mean_psnr_db"] is None else f"{row['phase1_mean_psnr_db']:.2f}"
        lines.append(f"| {row['rate_band']} | {row['codec']} | {row['condition']} | {p1_bpp} | {p1_psnr} | {row['phase4_mean_bpp']:.3f} | {row['test_accuracy']:.3f} |")
    lines.extend(["", "The one-seed cross-codec results are exploratory. JPEG q10 and q60, and the selected WebP/AVIF/JXL operating points, have different codec syntax and rate-distortion behavior; quality numbers are not treated as equivalent. The controlled PBC LZMA-to-STORE comparison supports an effect of outer wrapping within the same PBC body. The other codecs do not have an equivalent paired raw-versus-unwrapped condition here, so these results do not establish that entropy coding is generally harmful.", ""])

    lines.extend([
        "## Refreshed JPEG timing",
        "",
        f"The pass used {timing['source_count']} fixed prepared 256×256 images, {timing['warmups_per_image']} warmup and {timing['repeats_per_image']} timed repetitions per image. Environment: Python {timing['environment']['python']}, Pillow {timing['environment']['pillow']}, libjpeg-turbo {timing['environment']['libjpeg_turbo']}, pyjpegio {timing['environment']['pyjpegio']}, NumPy {timing['environment']['numpy']}; {timing['environment']['os']}.",
        "",
        "| JPEG | File → DCT IR median (ms) | p95 (ms) | Full RGB median (ms) | p95 (ms) | Median ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in timing_rows:
        lines.append(f"| {row['condition']} | {row['file_to_dct_ir_median_ms']:.4f} | {row['file_to_dct_ir_p95_ms']:.4f} | {row['full_rgb_decode_median_ms']:.4f} | {row['full_rgb_decode_p95_ms']:.4f} | {row['median_ratio_dct_to_full_decode']:.3f}× |")
    fresh_q20 = next(row["median_ratio_dct_to_full_decode"] for row in timing_rows if row["condition"] == "jpeg_q20")
    other_ratios = [row["median_ratio_dct_to_full_decode"] for row in timing_rows if row["condition"] != "jpeg_q20"]
    lines.extend([
        "",
        f"The earlier q20 extension reported a 2.20× ratio. The refreshed q20 ratio is {fresh_q20:.3f}×; q10/q60/q80 ratios span {min(other_ratios):.3f}×–{max(other_ratios):.3f}×. The q20 anomaly {'remains slower than all three comparison qualities in this pass' if fresh_q20 > max(other_ratios) else 'does not remain an outlier relative to q10, q60, and q80 in this pass'}.",
        "",
        "## Interpretation and limitations",
        "",
        "The three-seed results answer whether the Phase 3 PBC/JPEG differences hold across these model seeds. The cross-codec raw-byte extension uses one seed and one classifier family, so it describes this Imagenette setup rather than a general codec ranking. Absolute preprocessing times remain implementation-level measurements. No raw-byte model is claimed to decode images, and these results do not demonstrate state-of-the-art accuracy.",
        "",
        "The strongest current claim is that, for this fixed Imagenette benchmark and byte CNN, removing PBC's outer LZMA wrapper changes classification accessibility substantially while preserving the PBC body, and JPEG's quantized DCT representation exposes more class information than JPEG file bytes. Whether those effects generalize to other datasets and architectures remains open.",
        "The evidence supports continuing TaMtD as a measured compressed-domain vision systems study: it yields reproducible questions about representation structure, task accessibility, and preprocessing cost without requiring a decoder-replacement claim.",
        "",
        "## Artifacts",
        "",
        "- `confirmation_summary.csv`, `confirmation_per_seed.csv`, `multi_seed_summary.json`",
        "- `byte_level_result_table.csv`, `jpeg_timing_refresh.csv`",
        "- `multi_seed_accessibility_frontier.png`, `cross_codec_raw_bytes.png`, `jpeg_timing_refresh.png`",
        "- `resource_diagnostics.json` and `config.json`",
    ])
    (OUTPUT_DIR / "Phase 4 Findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase3 = _phase3_rows()
    per_seed, lookup = _condition_rows(phase3)
    summary = _confirmation_summary(per_seed, lookup)
    codec_summary_rows = list(csv.DictReader((PROJECT_ROOT / "results/Codec Benchmark/summary.csv").open(encoding="utf-8", newline="")))
    codec_summary = {row["codec_id"]: row for row in codec_summary_rows}
    core_rows = _read_jsonl(CACHE_ROOT / "codec_manifest.jsonl")
    raw = _raw_byte_rows(phase3, codec_summary, core_rows)
    timing = _read_json(OUTPUT_DIR / "jpeg_timing_refresh.json")
    frontier = _frontier_rows(lookup, timing)
    timing_rows = _timing_rows(timing)

    _write_csv(OUTPUT_DIR / "confirmation_per_seed.csv", per_seed)
    _write_csv(OUTPUT_DIR / "confirmation_summary.csv", summary)
    _write_csv(OUTPUT_DIR / "byte_level_result_table.csv", raw)
    _write_csv(OUTPUT_DIR / "jpeg_timing_refresh.csv", timing_rows)
    _write_csv(OUTPUT_DIR / "multi_seed_frontier.csv", frontier)
    _plot_frontier(frontier, OUTPUT_DIR / "multi_seed_accessibility_frontier.png")
    _plot_raw_bytes(raw, OUTPUT_DIR / "cross_codec_raw_bytes.png")
    _plot_timing(timing_rows, OUTPUT_DIR / "jpeg_timing_refresh.png")

    baseline_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dataset = _read_json(PHASE3_DIR / "summary.json")["dataset"]
    training_settings = _read_json(PHASE3_DIR / "training_settings.json")
    config = {
        "phase": "Phase 4 Confirmation and Cross-Codec Extension",
        "baseline_git_commit": baseline_commit,
        "dataset": dataset,
        "seeds": list(SEEDS),
        "reused_phase3_seed_42": True,
        "phase4a_conditions": {"pbc": list(PBC_PRESETS), "jpeg_qualities": list(JPEG_QUALITIES), "representations": ["pbc_lzma_bytes", "pbc_store_bytes", "pbc_canonical_ir", "jpeg_file_bytes", "jpeg_dct_ir", "decoded_rgb_reference"]},
        "cross_codec_seed": 42,
        "cross_codec_settings": {
            "webp_q30": {"quality": 30, "method": 6, "lossless": False, "exact": False, "subsampling": "libwebp default", "phase1_mean_bpp": float(codec_summary["webp_q30"]["mean_bpp"])},
            "webp_q75": {"quality": 75, "method": 6, "lossless": False, "exact": False, "subsampling": "libwebp default", "phase1_mean_bpp": float(codec_summary["webp_q75"]["mean_bpp"])},
            "avif_q40": {"quality": 40, "speed": 6, "subsampling": "4:2:0", "max_threads": 1, "codec": "auto", "range": "full", "phase1_mean_bpp": float(codec_summary["avif_q40"]["mean_bpp"])},
            "avif_q60": {"quality": 60, "speed": 6, "subsampling": "4:2:0", "max_threads": 1, "codec": "auto", "range": "full", "phase1_mean_bpp": float(codec_summary["avif_q60"]["mean_bpp"])},
            "jxl_d6": {"distance": 6, "effort": 7, "num_threads": 1, "phase1_mean_bpp": float(codec_summary["jxl_d6"]["mean_bpp"])},
            "jxl_d3": {"distance": 3, "effort": 7, "num_threads": 1, "phase1_mean_bpp": float(codec_summary["jxl_d3"]["mean_bpp"])},
        },
        "operating_point_rationale": {
            "low_rate_target_bpp": float(codec_summary["pbc_compression"]["mean_bpp"]),
            "mid_rate_target_bpp": float(codec_summary["pbc_balanced"]["mean_bpp"]),
            "selection": "nearest Phase 1 measured rate points; exact codec settings and Phase 1 bpp/PSNR are recorded per setting",
        },
        "training_settings": training_settings,
        "training_batch_size_overrides": {
            "pbc_balanced/pbc_lzma_bytes": {"batch_size": 8, "seeds": list(SEEDS), "reason": "GPU runs approached 98.5% system commit before all three seed rows completed"},
            "pbc_balanced/pbc_store_bytes": {"batch_size": 8, "seeds": list(SEEDS), "reason": "batch-16 run reached 98.8% system commit before result evaluation"},
            "pbc_quality/pbc_lzma_bytes": {"batch_size": 8, "seeds": list(SEEDS), "reason": "long PBC byte stream; matched to a condition that remained below the system commit guard"},
            "pbc_quality/pbc_store_bytes": {"batch_size": 8, "seeds": list(SEEDS), "reason": "longest PBC byte stream; matched across seeds to keep commit below the safety guard"},
        },
        "training_device_overrides": {
            "pbc_balanced/pbc_lzma_bytes": {"device": "cpu", "seeds": list(SEEDS), "reason": "GPU runs reached at least 97.7% system commit before result evaluation"},
            "pbc_balanced/pbc_store_bytes": {"device": "cpu", "seeds": list(SEEDS), "reason": "GPU run reached 98.8% system commit at batch size 16"},
            "pbc_quality/pbc_lzma_bytes": {"device": "cpu", "seeds": list(SEEDS), "reason": "longer PBC byte stream; grouped with CPU memory-safe runs"},
            "pbc_quality/pbc_store_bytes": {"device": "cpu", "seeds": list(SEEDS), "reason": "longest PBC byte stream; grouped with CPU memory-safe runs"},
        },
        "loader_safety": {"workers": 0, "pin_memory": False, "minimum_start_free_bytes": 20_000_000_000, "stop_below_free_bytes": 12_000_000_000, "stop_at_commit_fraction": 0.985, "stop_if_commit_at_or_above_95_percent_for_seconds": 300},
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "multi_seed_summary.json").write_text(
        json.dumps({"seeds": list(SEEDS), "summary": summary, "per_seed": per_seed, "frontier": frontier}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_findings(summary, per_seed, lookup, raw, timing_rows, timing)
    print(json.dumps({"summary_rows": len(summary), "per_seed_rows": len(per_seed), "raw_byte_rows": len(raw), "frontier_points": len(frontier), "output": str(OUTPUT_DIR / "Phase 4 Findings.md")}, indent=2))


if __name__ == "__main__":
    main()
