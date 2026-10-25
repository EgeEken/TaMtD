import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tamtd.runtime import results_root


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _find_model(models: list[dict], condition: str, representation: str) -> dict:
    return next(row for row in models if row["condition"] == condition and row["representation"] == representation)


def _condition_name(condition: str) -> str:
    return condition.replace("pbc_", "PBC ").replace("jpeg_q", "JPEG q")


def _build_plots(summary: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    frontier = summary["frontier"]
    colors = {
        "pbc_compression": "#3366cc",
        "pbc_balanced": "#109618",
        "pbc_quality": "#dc3912",
        "jpeg_q10": "#990099",
        "jpeg_q20": "#ff9900",
        "jpeg_q60": "#0099c6",
        "jpeg_q80": "#dd4477",
    }
    markers = {
        "pbc_lzma_bytes": "o",
        "pbc_store_bytes": "s",
        "pbc_parsed_fields": "^",
        "pbc_canonical_ir": "D",
        "jpeg_file_bytes": "o",
        "jpeg_dct_ir": "s",
        "decoded_rgb": "X",
    }
    stage_labels = {
        "pbc_lzma_bytes": "PBC forced LZMA bytes",
        "pbc_store_bytes": "PBC STORE bytes",
        "pbc_parsed_fields": "PBC parsed fields",
        "pbc_canonical_ir": "PBC canonical IR",
        "jpeg_file_bytes": "JPEG file bytes",
        "jpeg_dct_ir": "JPEG DCT IR",
        "decoded_rgb": "decoded RGB",
    }
    for metric, x_key, x_label, filename in (
        ("test_accuracy", "cumulative_preprocessing_ms", "Cumulative preprocessing latency (ms)", "frontier_absolute_latency.png"),
        ("test_accuracy", "fraction_full_decode", "Fraction of full codec decode cost paid (%)", "frontier_fraction_full_decode.png"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        for ax, family in zip(axes, ("pbc", "jpeg")):
            condition_names = sorted(
                {row["condition"] for row in frontier if row["codec_family"] == family},
                key=lambda name: (name.startswith("jpeg") or name.startswith("pbc"), name),
            )
            for condition in condition_names:
                points = [row for row in frontier if row["condition"] == condition]
                ax.plot(
                    [row[x_key] for row in points],
                    [row[metric] * 100 for row in points],
                    color=colors[condition],
                    linewidth=1.8,
                    alpha=0.9,
                )
                for point in points:
                    ax.scatter(
                        point[x_key], point[metric] * 100,
                        marker=markers[point["representation"]],
                        s=58,
                        color=colors[condition],
                        edgecolor="white",
                        linewidth=0.6,
                        label=_condition_name(condition) if point is points[0] else None,
                        zorder=3,
                    )
            ax.axhline(10, color="#777777", linestyle=":", linewidth=0.9, label="10% chance")
            ax.set_title("PBC implementation" if family == "pbc" else "JPEG implementation")
            ax.set_xlabel(x_label)
            ax.grid(True, alpha=0.22)
            ax.legend(fontsize=8, loc="lower right")
        axes[0].set_ylabel("Imagenette test accuracy (%)")
        stage_handles = [
            Line2D([], [], color="#555555", marker=markers[stage], linestyle="None", label=stage_labels[stage])
            for stage in markers
        ]
        figure.legend(handles=stage_handles, loc="lower center", ncol=4, fontsize=8)
        figure.suptitle("Neural accessibility across decode stages", fontsize=14)
        figure.subplots_adjust(top=0.88, bottom=0.18, wspace=0.25)
        figure.savefig(output_dir / filename, dpi=200)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, family, stages in (
        (axes[0], "pbc", ["pbc_lzma_bytes", "pbc_store_bytes", "pbc_parsed_fields", "pbc_canonical_ir", "decoded_rgb"]),
        (axes[1], "jpeg", ["jpeg_file_bytes", "jpeg_dct_ir", "decoded_rgb"]),
    ):
        conditions = sorted({row["condition"] for row in frontier if row["codec_family"] == family})
        positions = {stage: index for index, stage in enumerate(stages)}
        for condition in conditions:
            points = [row for row in frontier if row["condition"] == condition]
            ax.plot(
                [positions[row["representation"]] for row in points],
                [row["semantic_accessibility"] for row in points],
                marker="o",
                linewidth=1.7,
                markersize=5,
                color=colors[condition],
                label=_condition_name(condition),
            )
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1, label="decoded RGB reference")
        ax.axhline(0, color="#777777", linestyle=":", linewidth=0.8)
        ax.set_xticks(range(len(stages)), [stage_labels[stage] for stage in stages], rotation=18, ha="right")
        ax.set_title("PBC" if family == "pbc" else "JPEG")
        ax.set_ylabel("Semantic accessibility (unclamped)")
        ax.grid(True, axis="y", alpha=0.22)
        ax.legend(fontsize=8, loc="best")
    figure.suptitle("Fraction of decoded-domain semantic performance recovered", fontsize=14)
    figure.savefig(output_dir / "semantic_accessibility.png", dpi=200)
    plt.close(figure)

    feasibility = summary["training_feasibility"]
    columns = [
        ("condition", "Condition"), ("representation", "Representation"), ("model_family", "Model"),
        ("parameter_count", "Params"), ("examples_per_second", "Examples/s"),
        ("peak_vram_mb", "Peak VRAM MB"), ("training_wall_seconds", "Train s"), ("test_accuracy", "Test acc"),
    ]
    height = max(6.0, 0.34 * len(feasibility) + 1.8)
    figure, ax = plt.subplots(figsize=(16, height))
    ax.axis("off")
    cells = []
    for row in feasibility:
        cells.append([
            _condition_name(row["condition"]), row["representation"], row["model_family"],
            f"{row['parameter_count']:,}", _fmt(row["examples_per_second"], 1),
            _fmt(row["peak_vram_mb"], 0), _fmt(row["training_wall_seconds"], 1),
            f"{row['test_accuracy']:.3f}",
        ])
    table = ax.table(
        cellText=cells,
        colLabels=[label for _, label in columns],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.12, 0.18, 0.15, 0.10, 0.12, 0.11, 0.10, 0.09],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1, 1.28)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dde7f0")
            cell.set_text_props(weight="bold")
    ax.set_title("Training feasibility by representation", fontsize=14, pad=18)
    figure.tight_layout()
    figure.savefig(output_dir / "training_feasibility.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _critical_comparisons(summary: dict) -> dict:
    models = summary["model_results"]
    preprocessing = summary["preprocessing"]
    frontier = summary["frontier"]
    frontier_lookup = {(row["condition"], row["representation"]): row for row in frontier}

    def semantic_delta(left: tuple[str, str], right: tuple[str, str]):
        left_value = frontier_lookup[left]["semantic_accessibility"]
        right_value = frontier_lookup[right]["semantic_accessibility"]
        return None if left_value is None or right_value is None else left_value - right_value
    result = {"pbc_entropy_control": {}, "pbc_parsed_to_canonical": {}, "jpeg_bytes_to_dct": {}, "best_intermediate_under_full_decode": {}}
    for preset in ("compression", "balanced", "quality"):
        condition = f"pbc_{preset}"
        lzma = _find_model(models, condition, "pbc_lzma_bytes")
        store = _find_model(models, condition, "pbc_store_bytes")
        parsed = _find_model(models, condition, "pbc_parsed_fields")
        canonical = _find_model(models, condition, "pbc_canonical_ir")
        timing = preprocessing[condition]
        result["pbc_entropy_control"][condition] = {
            "lzma_test_accuracy": lzma["test_accuracy"],
            "store_test_accuracy": store["test_accuracy"],
            "accuracy_delta_store_minus_lzma": store["test_accuracy"] - lzma["test_accuracy"],
            "semantic_accessibility_delta_store_minus_lzma": semantic_delta(
                (condition, "pbc_store_bytes"), (condition, "pbc_lzma_bytes")
            ),
            "lzma_to_store_median_ms": timing["lzma_unpack_ms"],
            "fraction_full_decode_percent": 100 * timing["lzma_unpack_ms"] / timing["full_lzma_decode_ms"],
        }
        result["pbc_parsed_to_canonical"][condition] = {
            "parsed_test_accuracy": parsed["test_accuracy"],
            "canonical_test_accuracy": canonical["test_accuracy"],
            "accuracy_delta_canonical_minus_parsed": canonical["test_accuracy"] - parsed["test_accuracy"],
            "semantic_accessibility_delta_canonical_minus_parsed": semantic_delta(
                (condition, "pbc_canonical_ir"), (condition, "pbc_parsed_fields")
            ),
            "additional_canonical_ir_build_ms": timing["canonical_ir_build_ms"],
        }
    for quality in (10, 20, 60, 80):
        condition = f"jpeg_q{quality}"
        raw = _find_model(models, condition, "jpeg_file_bytes")
        dct = _find_model(models, condition, "jpeg_dct_ir")
        timing = preprocessing[condition]
        result["jpeg_bytes_to_dct"][condition] = {
            "file_bytes_test_accuracy": raw["test_accuracy"],
            "dct_ir_test_accuracy": dct["test_accuracy"],
            "accuracy_delta_dct_minus_file_bytes": dct["test_accuracy"] - raw["test_accuracy"],
            "semantic_accessibility_delta_dct_minus_file_bytes": semantic_delta(
                (condition, "jpeg_dct_ir"), (condition, "jpeg_file_bytes")
            ),
            "file_to_coefficients_ms": timing["file_to_coefficients_ms"],
            "canonical_ir_build_ms": timing["canonical_ir_build_ms"],
            "file_to_dct_ir_ms": timing["file_to_dct_ms"],
            "full_rgb_decode_ms": timing["full_rgb_decode_ms"],
            "fraction_full_decode_percent": 100 * timing["file_to_dct_ms"] / timing["full_rgb_decode_ms"],
        }
    for condition in summary["rgb_reference_accuracies"]:
        if condition == "original_rgb":
            continue
        rows = [row for row in frontier if row["condition"] == condition and row["representation"] != "decoded_rgb"]
        full_cost = 100.0
        rows = [row for row in rows if row["fraction_full_decode"] < full_cost]
        best = max(rows, key=lambda row: row["test_accuracy"]) if rows else None
        result["best_intermediate_under_full_decode"][condition] = best
    return result


def _write_findings(summary: dict, comparisons: dict, output_dir: Path) -> None:
    dataset = summary["dataset"]
    models = summary["model_results"]
    counts = {row["condition"]: row["parameter_count"] for row in models}
    mean_train_speed = [row["examples_per_second"] for row in models]
    vram = [row["peak_vram_mb"] for row in models if row["peak_vram_mb"] is not None]
    pbc_entropy_lines = []
    for condition, values in comparisons["pbc_entropy_control"].items():
        pbc_entropy_lines.append(
            f"- {condition}: forced LZMA **{values['lzma_test_accuracy']:.3f}**, STORE **{values['store_test_accuracy']:.3f}** "
            f"({values['accuracy_delta_store_minus_lzma'] * 100:+.2f} percentage points) after **{values['lzma_to_store_median_ms']:.4f} ms** "
            f"to remove LZMA ({values['fraction_full_decode_percent']:.1f}% of that preset's forced-LZMA full decode)."
        )
    pbc_ir_lines = []
    for condition, values in comparisons["pbc_parsed_to_canonical"].items():
        pbc_ir_lines.append(
            f"- {condition}: parsed **{values['parsed_test_accuracy']:.3f}**, canonical **{values['canonical_test_accuracy']:.3f}** "
            f"({values['accuracy_delta_canonical_minus_parsed'] * 100:+.2f} percentage points); canonical construction adds "
            f"**{values['additional_canonical_ir_build_ms']:.4f} ms** in the Phase 2 timing."
        )
    jpeg_lines = []
    for condition, values in comparisons["jpeg_bytes_to_dct"].items():
        jpeg_lines.append(
            f"- {condition}: file bytes **{values['file_bytes_test_accuracy']:.3f}**, DCT IR **{values['dct_ir_test_accuracy']:.3f}** "
            f"({values['accuracy_delta_dct_minus_file_bytes'] * 100:+.2f} percentage points); file-to-DCT IR is "
            f"**{values['file_to_dct_ir_ms']:.4f} ms** versus **{values['full_rgb_decode_ms']:.4f} ms** full Pillow decode "
            f"({values['fraction_full_decode_percent']:.1f}%)."
        )
    semantic_lines = []
    for condition, row in comparisons["best_intermediate_under_full_decode"].items():
        if row is None:
            semantic_lines.append(f"- {condition}: canonical DCT/PBC IR requires at least the measured full-decode latency; no intermediate point was below it.")
        else:
            semantic_lines.append(
                f"- {condition}: best intermediate below full decode is `{row['representation']}` at **{row['test_accuracy']:.3f}** "
                f"({row['semantic_accessibility']:.3f} semantic accessibility) and **{row['cumulative_preprocessing_ms']:.4f} ms** "
                f"({row['fraction_full_decode']:.1f}% of full decode)."
            )
    overfit = json.loads((output_dir / "overfit_sanity.json").read_text(encoding="utf-8"))
    overfit_lines = [
        f"- {name}: {result['best_train_accuracy']:.3f} training accuracy in {result['epochs_run']} epochs ({'passed' if result['passed'] else 'failed'})."
        for name, result in overfit["models"].items()
    ]
    references = summary["rgb_reference_accuracies"]
    content = f"""# Neural Accessibility Findings

## Dataset and protocol

- Imagenette 320px was deterministically center-cropped/resized to 256×256 RGB using the Phase 1 EXIF transpose and Pillow BICUBIC preprocessing.
- Split seed: {dataset['seed']}. Train: {dataset['train_samples']:,}; validation: {dataset['validation_samples']:,}; official validation held out as test: {dataset['test_samples']:,}. No test image was used for training or checkpoint selection.
- One seed was run. Checkpoints were selected by validation accuracy; each trained representation model used one final test evaluation.
- Model input: raw bytes use a shared byte CNN with no sequence cap; PBC P2/P3 use 64 patch tokens with 8×8 pooled masks/grid summaries and an MLP content encoder; JPEG DCT uses component grids in quantized coefficient space without IDCT; RGB uses one random-initialized width-32 ResNet-18-style classifier.
- Full per-model results and histories are in `model_results.json`; exact settings are in `config.json` and `training_settings.json`.

| Split | Samples |
| --- | ---: |
| Train | {dataset['train_samples']:,} |
| Validation | {dataset['validation_samples']:,} |
| Official validation test | {dataset['test_samples']:,} |

## RGB reference and overfit checks

- Original prepared RGB test accuracy: **{references['original_rgb']:.3f}**.
- PBC decoded RGB: {', '.join(f"{name.replace('pbc_', '')} {value:.3f}" for name, value in references.items() if name.startswith('pbc_'))}.
- JPEG decoded RGB: {', '.join(f"q{name.split('q')[1]} {value:.3f}" for name, value in references.items() if name.startswith('jpeg_'))}.
- Tiny-set checks used a fixed stratified 128-sample train subset:
{chr(10).join(overfit_lines)}

## PBC LZMA-to-STORE control

The forced-LZMA and STORE streams encode the same PBC body. The model sees the wrapper bytes for this control.

{chr(10).join(pbc_entropy_lines)}

## PBC parsed fields to canonical IR

{chr(10).join(pbc_ir_lines)}

## JPEG file bytes to DCT IR

J1 cost includes file-to-coefficient extraction plus canonical DCT IR array construction. The q20 supplement measured only q20 on the original Phase 1 95-image set; Phase 2's full matrix was not repeated.

{chr(10).join(jpeg_lines)}

## Intermediate points and decode cost

Semantic accessibility is the project diagnostic `(model accuracy − 0.1) / (decoded RGB reference accuracy − 0.1)`. Values are not clamped; values below zero or above one are retained. The common RGB model has accessibility 1 by construction when its decoded reference is above chance.

{chr(10).join(semantic_lines)}

The plots show absolute preprocessing milliseconds and fraction of each implementation's full decode cost separately. PBC uses Python/NumPy/Numba research code; JPEG uses pyjpegio/libjpeg-turbo and Pillow/libjpeg-turbo. The ratios are implementation-level measurements from this machine, not algorithm-only comparisons. Phase 2 timing samples are the original 95 natural-image calibration sources; q20 used a targeted extension over those same sources. Classification accuracy comes from Imagenette test.

## Training feasibility

- Model parameter counts range from {min(counts.values()):,} to {max(counts.values()):,} across the measured classifiers.
- Mean training throughput across representation models: {sum(mean_train_speed) / max(len(mean_train_speed), 1):.1f} examples/s.
- Peak VRAM observed: {max(vram):.0f} MB.
- Full row-level training metrics are in `training_feasibility.csv` and `training_feasibility.png`.

## Limitations

- This is a one-seed frontier; individual accuracy differences can reflect seed variance.
- No image augmentation was used. The RGB reference is a modest random-initialized model and is a measured ceiling for this run, not a task-level upper bound.
- PBC patch content is pooled to fixed-length 8×8/64-feature vectors before the patch-set model. It preserves the parsed/canonical arrays' structure but does not feed every variable-size grid cell as a separate token.
- JPEG full decode and coefficient access use different supported libraries, as in Phase 2. The frontier combines their measured stage costs and should be read as an implementation-level systems comparison.
- No claims are made that a classifier decodes images or learned a codec algorithm.

## Artifacts

- `summary.json`, `model_results.json`, `critical_comparisons.json`
- `frontier.csv`, `per_condition_metrics.csv`, `training_feasibility.csv`
- `frontier_absolute_latency.png`, `frontier_fraction_full_decode.png`, `semantic_accessibility.png`, `training_feasibility.png`
- `jpeg_q20_timing_extension.json`
"""
    (output_dir / "Neural Accessibility Findings.md").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=results_root() / "Neural Accessibility")
    args = parser.parse_args()
    output_dir = args.results_dir.resolve()
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparisons = _critical_comparisons(summary)
    summary["critical_comparisons"] = comparisons
    summary["hardware"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    feasibility = [
        {
            key: row.get(key)
            for key in (
                "condition", "representation", "model_family", "parameter_count",
                "examples_per_second", "peak_vram_mb", "training_wall_seconds", "test_accuracy",
            )
        }
        for row in summary["model_results"]
    ]
    summary["training_feasibility"] = feasibility
    _write_json(output_dir / "critical_comparisons.json", comparisons)
    _write_json(summary_path, summary)
    _build_plots(summary, output_dir)

    frontier = summary["frontier"]
    _write_csv(
        output_dir / "frontier.csv",
        frontier,
        [
            "condition", "codec_family", "representation", "model_family", "test_accuracy",
            "best_validation_accuracy", "decoded_rgb_reference_accuracy", "semantic_accessibility",
            "cumulative_preprocessing_ms", "fraction_full_decode", "representation_bytes", "bpp",
        ],
    )
    metrics = []
    for model in summary["model_results"]:
        if model["condition"] == "original_rgb":
            continue
        endpoint = next(
            row for row in frontier
            if row["condition"] == model["condition"] and row["representation"] == model["representation"]
        )
        metrics.append({**model, **endpoint})
    _write_csv(
        output_dir / "per_condition_metrics.csv",
        metrics,
        [
            "condition", "representation", "model_family", "parameter_count", "best_validation_accuracy",
            "test_accuracy", "training_wall_seconds", "examples_per_second", "peak_vram_mb",
            "cumulative_preprocessing_ms", "fraction_full_decode", "representation_bytes", "bpp",
        ],
    )
    _write_csv(
        output_dir / "training_feasibility.csv",
        feasibility,
        ["condition", "representation", "model_family", "parameter_count", "examples_per_second", "peak_vram_mb", "training_wall_seconds", "test_accuracy"],
    )
    _write_json(summary_path, summary)
    _write_findings(summary, comparisons, output_dir)
    phase2_config = json.loads((results_root() / "Staged Decoding" / "config.json").read_text(encoding="utf-8"))
    config = {
        "dataset": summary["dataset"],
        "device": summary["device"],
        "training_settings": summary["training_settings"],
        "pbc_presets": ["compression", "balanced", "quality"],
        "jpeg_qualities": [10, 20, 60, 80],
        "jpeg_settings": {"quality": "condition quality", "subsampling": "4:2:0", "optimize": False, "progressive": False},
        "pbc_effective_configurations": phase2_config["settings"]["pbc_effective_configurations"],
        "pbc_git_state": phase2_config["settings"]["pbc_git_state"],
        "preprocessing_timing_sources": summary["preprocessing"],
        "model_architectures": {
            "byte": "ByteStreamCNN: embedding, stride-8/4/2 Conv1D downsampling, three masked depthwise/pointwise residual blocks, masked mean pooling",
            "pbc": "PBCInstructionNet: 24 geometry features, 144 pooled mask/grid/palette/value features, two-layer 128-wide Transformer set encoder, 64-patch padding",
            "jpeg_dct": "JPEGDCTNet: component-specific 2D stems, 4:2:0 chroma upsampling, quantization-table projection, residual fusion; no IDCT",
            "rgb": "RGBResNet18(width=32), random initialization, ImageNet RGB normalization",
        },
    }
    _write_json(output_dir / "config.json", config)
    print(json.dumps({"findings": str(output_dir / "Neural Accessibility Findings.md"), "plots": 4, "model_count": len(feasibility)}, indent=2))


if __name__ == "__main__":
    main()
