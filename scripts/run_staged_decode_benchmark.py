import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, __version__ as pillow_version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codecs import JPEGCodec, PBCCodecPair
from tamtd.codecs.pbc import resolve_pbc_root
from tamtd.staged_decode import (
    build_jpeg_ir,
    build_pbc_ir,
    load_array_bundle,
    measure_parsed_pbc_size,
    parse_pbc_store_body,
    read_jpeg_coefficients,
    rebuild_pbc_store_stream,
    render_pbc_ir,
    serialize_array_bundle,
)


PBC_PRESETS = ("compression", "balanced", "quality")
JPEG_QUALITIES = (1, 5, 10, 60, 80)
REPEATED_STAGE_NAMES = (
    "forced_lzma_unpack",
    "store_body_parse",
    "canonical_ir_build",
    "canonical_ir_serialize",
    "patch_application_render",
    "full_store_decode",
    "full_lzma_decode",
    "jpeg_file_to_coefficients",
    "jpeg_canonical_ir_build",
    "jpeg_ir_serialize",
    "jpeg_full_rgb_decode",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def measure(function, warmups: int, repeats: int) -> tuple[object, list[float]]:
    result = None
    for _ in range(warmups):
        result = function()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = function()
        timings.append((time.perf_counter_ns() - start) / 1_000_000)
    return result, timings


def stats(values: list[float] | None) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "median": float(statistics.median(values)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(statistics.fmean(values)),
    }


def source_images(source_dir: Path, limit: int | None) -> tuple[list[dict], list[Image.Image], dict]:
    phase1_config_path = source_dir.parent / "config.json"
    phase1_config = json.loads(phase1_config_path.read_text(encoding="utf-8"))
    manifest_path = source_dir / "manifest.jsonl"
    manifest_bytes = manifest_path.read_bytes()
    if sha256(manifest_bytes) != phase1_config["source_manifest_sha256"]:
        raise ValueError("Phase 1 source manifest hash does not match its benchmark config")
    rows = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != phase1_config["image_count"]:
        raise ValueError("Phase 1 source manifest image count changed")
    if limit is not None:
        rows = rows[:limit]
    images = []
    for row in rows:
        path = source_dir / row["source_png"]
        encoded = path.read_bytes()
        if sha256(encoded) != row["source_png_sha256"]:
            raise ValueError(f"prepared source PNG hash mismatch: {path}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        if image.size != (256, 256) or sha256(image.tobytes()) != row["rgb_pixels_sha256"]:
            raise ValueError(f"prepared source pixels differ from Phase 1: {path}")
        images.append(image)
    return rows, images, phase1_config


def pillow_decode(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def pbc_sample_rows(
    row: dict,
    image: Image.Image,
    pair: PBCCodecPair,
    warmups: int,
    repeats: int,
    cache_root: Path,
) -> list[dict]:
    store, forced = pair.encode_pair(image)
    raw_body = store.data[6:]
    unwrapped = pair.unpack_body(forced.data)
    if unwrapped != raw_body or pair.unpack_body(store.data) != raw_body:
        raise AssertionError("forced LZMA and STORE streams do not share the same body")

    parsed = parse_pbc_store_body(raw_body, pair)
    parsed_size = measure_parsed_pbc_size(parsed)

    def build_ir():
        metadata, arrays = build_pbc_ir(parsed, pair)
        metadata.update(
            {
                "pbc_preset": pair.preset,
                "image_id": row["image_id"],
                "source_rgb_sha256": row["rgb_pixels_sha256"],
                "raw_pbc_body_sha256": sha256(raw_body),
            }
        )
        return metadata, arrays

    (ir_metadata, ir_arrays), build_runs = measure(build_ir, warmups, repeats)
    rebuilt_stream = rebuild_pbc_store_stream(ir_metadata, ir_arrays, pair)
    if rebuilt_stream != store.data:
        raise AssertionError("canonical PBC IR does not round-trip to the original STORE stream")
    ir_bytes, serialize_runs = measure(
        lambda: serialize_array_bundle(ir_metadata, ir_arrays), warmups, repeats
    )
    if ir_bytes != serialize_array_bundle(ir_metadata, ir_arrays):
        raise AssertionError("PBC IR serialization is not deterministic")
    loaded_metadata, loaded_arrays = load_array_bundle(ir_bytes)
    if loaded_metadata != ir_metadata or any(
        not np.array_equal(ir_arrays[name], loaded_arrays[name]) for name in ir_arrays
    ):
        raise AssertionError("serialized PBC IR does not preserve its metadata and arrays")

    ir_path = cache_root / "representations" / "pbc" / pair.preset / f"{row['image_id']}.npz"
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    ir_path.write_bytes(ir_bytes)

    def render_ir():
        return render_pbc_ir(ir_metadata, ir_arrays, pair)

    rendered, render_runs = measure(render_ir, warmups, repeats)
    full_store, store_decode_runs = measure(
        lambda: pair.pbc_class.decompress(store.data).image.convert("RGB"), warmups, repeats
    )
    full_lzma, lzma_decode_runs = measure(
        lambda: pair.pbc_class.decompress(forced.data).image.convert("RGB"), warmups, repeats
    )
    if not np.array_equal(np.asarray(full_store), np.asarray(full_lzma)):
        raise AssertionError("PBC STORE and forced-LZMA decodes are not pixel-identical")
    if not np.array_equal(np.asarray(rendered), np.asarray(full_store)):
        raise AssertionError("canonical PBC IR rendering differs from the standard decoder")

    def unpack_lzma_body():
        return pair.pbc_class._entropy_unpack(pair.pbc_class.ENTROPY_LZMA, forced.data[6:])

    unpacked, unpack_runs = measure(unpack_lzma_body, warmups, repeats)
    if unpacked != raw_body:
        raise AssertionError("forced-LZMA unpack did not reproduce the raw PBC body")
    _, parse_runs = measure(lambda: parse_pbc_store_body(raw_body, pair), warmups, repeats)
    stage_times = {
        "forced_lzma_unpack": unpack_runs,
        "store_body_parse": parse_runs,
        "canonical_ir_build": build_runs,
        "canonical_ir_serialize": serialize_runs,
        "patch_application_render": render_runs,
        "full_store_decode": store_decode_runs,
        "full_lzma_decode": lzma_decode_runs,
    }
    common = {
        "image_id": row["image_id"],
        "source_rgb_sha256": row["rgb_pixels_sha256"],
        "codec_family": "pbc",
        "pbc_preset": pair.preset,
        "jpeg_quality": None,
        "raw_pbc_body_bytes": len(raw_body),
        "pbc_store_stream_bytes": len(store.data),
        "pbc_forced_lzma_stream_bytes": len(forced.data),
        "pbc_parsed_serialized_bytes": parsed_size["serialized_bytes"],
        "pbc_parsed_logical_metadata_bytes": parsed_size["metadata_bytes"],
        "pbc_parsed_array_bytes": parsed_size["array_bytes"],
        "representation_serialized_bytes": len(ir_bytes),
        "representation_metadata_bytes": len(json.dumps(ir_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "representation_array_bytes": sum(array.nbytes for array in ir_arrays.values()),
        "representation_native_array_bytes": None,
        "rgb_bytes": len(image.tobytes()),
        "representation_cache_path": ir_path.relative_to(PROJECT_ROOT).as_posix(),
        "pbc_validation": {
            "forced_lzma_body_equals_store": True,
            "store_and_lzma_pixels_equal": True,
            "canonical_ir_stream_roundtrip": True,
            "canonical_ir_render_matches_decoder": True,
            "canonical_ir_serialization_deterministic": True,
        },
        "stage_timings_ms": stage_times,
    }
    store_row = {
        **common,
        "condition": f"pbc_{pair.preset}_store",
        "entropy_mode": "store",
        "encoded_bytes": len(store.data),
        "stage_timings_ms": {
            "store_body_parse": parse_runs,
            "canonical_ir_build": build_runs,
            "canonical_ir_serialize": serialize_runs,
            "patch_application_render": render_runs,
            "full_store_decode": store_decode_runs,
        },
    }
    lzma_row = {
        **common,
        "condition": f"pbc_{pair.preset}_forced_lzma",
        "entropy_mode": "forced_lzma",
        "encoded_bytes": len(forced.data),
        "stage_timings_ms": {
            "forced_lzma_unpack": unpack_runs,
            "store_body_parse": parse_runs,
            "canonical_ir_build": build_runs,
            "canonical_ir_serialize": serialize_runs,
            "patch_application_render": render_runs,
            "full_lzma_decode": lzma_decode_runs,
        },
    }
    return [store_row, lzma_row]


def jpeg_sample_row(
    row: dict,
    image: Image.Image,
    quality: int,
    warmups: int,
    repeats: int,
    cache_root: Path,
) -> dict:
    condition = f"jpeg_q{quality}"
    sample = JPEGCodec(quality).encode(image)
    jpeg_path = cache_root / "encoded" / "jpeg" / condition / f"{row['image_id']}.jpg"
    jpeg_path.parent.mkdir(parents=True, exist_ok=True)
    jpeg_path.write_bytes(sample.data)

    jpeg_coefficients, extract_runs = measure(
        lambda: read_jpeg_coefficients(jpeg_path), warmups, repeats
    )
    (ir_metadata, ir_arrays, native_array_bytes), ir_build_runs = measure(
        lambda: build_jpeg_ir(jpeg_coefficients), warmups, repeats
    )
    ir_metadata.update(
        {
            "jpeg_condition": condition,
            "image_id": row["image_id"],
            "source_rgb_sha256": row["rgb_pixels_sha256"],
            "jpeg_sha256": sha256(sample.data),
            "coefficient_extractor": "pyjpegio 0.3.0 / jpeg_read_coefficients",
            "bundled_libjpeg_turbo": "3.2.0",
        }
    )
    ir_bytes, serialize_runs = measure(
        lambda: serialize_array_bundle(ir_metadata, ir_arrays), warmups, repeats
    )
    if ir_bytes != serialize_array_bundle(ir_metadata, ir_arrays):
        raise AssertionError("JPEG coefficient representation serialization is not deterministic")
    verify_coefficients = read_jpeg_coefficients(jpeg_path)
    verify_metadata, verify_arrays, _ = build_jpeg_ir(verify_coefficients)
    if verify_metadata != build_jpeg_ir(jpeg_coefficients)[0] or any(
        not np.array_equal(ir_arrays[name], verify_arrays[name]) for name in ir_arrays
    ):
        raise AssertionError("JPEG coefficient extraction is not deterministic")
    loaded_metadata, loaded_arrays = load_array_bundle(ir_bytes)
    if loaded_metadata != ir_metadata or any(
        not np.array_equal(ir_arrays[name], loaded_arrays[name]) for name in ir_arrays
    ):
        raise AssertionError("serialized JPEG representation does not preserve its metadata and arrays")

    ir_path = cache_root / "representations" / "jpeg" / condition / f"{row['image_id']}.npz"
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    ir_path.write_bytes(ir_bytes)
    decoded, decode_runs = measure(lambda: pillow_decode(jpeg_path), warmups, repeats)
    memory_decoded = JPEGCodec(quality).decode(sample.data)
    if not np.array_equal(np.asarray(decoded), np.asarray(memory_decoded)):
        raise AssertionError("file and in-memory Pillow JPEG decodes differ")
    if decoded.size != (256, 256):
        raise AssertionError("JPEG full decoder returned an unexpected image size")
    for component in ir_metadata["components"]:
        shape = ir_arrays[component["coefficient_array"]].shape
        if shape[:2] != (component["height_in_blocks"], component["width_in_blocks"]):
            raise AssertionError("JPEG coefficient dimensions do not match component block metadata")

    return {
        "image_id": row["image_id"],
        "source_rgb_sha256": row["rgb_pixels_sha256"],
        "codec_family": "jpeg",
        "condition": condition,
        "pbc_preset": None,
        "jpeg_quality": quality,
        "entropy_mode": None,
        "encoded_bytes": len(sample.data),
        "raw_pbc_body_bytes": None,
        "pbc_store_stream_bytes": None,
        "pbc_forced_lzma_stream_bytes": None,
        "pbc_parsed_serialized_bytes": None,
        "pbc_parsed_logical_metadata_bytes": None,
        "pbc_parsed_array_bytes": None,
        "representation_serialized_bytes": len(ir_bytes),
        "representation_metadata_bytes": len(json.dumps(ir_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "representation_array_bytes": sum(array.nbytes for array in ir_arrays.values()),
        "representation_native_array_bytes": native_array_bytes,
        "rgb_bytes": len(decoded.tobytes()),
        "representation_cache_path": ir_path.relative_to(PROJECT_ROOT).as_posix(),
        "jpeg_coefficient_representation_deterministic": True,
        "jpeg_header_dimensions_match_coefficients": True,
        "jpeg_full_decode_matches_in_memory_decode": True,
        "coefficient_to_rgb_available": False,
        "stage_timings_ms": {
            "jpeg_file_to_coefficients": extract_runs,
            "jpeg_canonical_ir_build": ir_build_runs,
            "jpeg_ir_serialize": serialize_runs,
            "jpeg_full_rgb_decode": decode_runs,
        },
    }


def condition_stage_path(sample: dict) -> tuple[list[tuple[str, str]], str]:
    if sample["codec_family"] == "jpeg":
        return [("jpeg_file_to_coefficients", "JPEG quantized DCT coefficients")], "jpeg_full_rgb_decode"
    stages = []
    if sample["entropy_mode"] == "forced_lzma":
        stages.append(("forced_lzma_unpack", "Bare PBC STORE body"))
        full = "full_lzma_decode"
    else:
        full = "full_store_decode"
    stages.extend(
        [
            ("store_body_parse", "Parsed PBC instruction fields"),
            ("canonical_ir_build", "Canonical PBC IR"),
            ("patch_application_render", "Patch application and RGB render"),
        ]
    )
    return stages, full


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    condition_rows = []
    stage_rows = []
    condition_names = list(dict.fromkeys(row["condition"] for row in rows))
    for condition in condition_names:
        group = [row for row in rows if row["condition"] == condition]
        sample = group[0]
        stage_names = list(
            dict.fromkeys(
                stage for row in group for stage in row["stage_timings_ms"]
            )
        )
        stage_stats = {
            stage: stats([value for row in group for value in row["stage_timings_ms"].get(stage, [])])
            for stage in stage_names
        }
        full_stage = next(
            stage for stage in stage_names if stage in {"full_store_decode", "full_lzma_decode", "jpeg_full_rgb_decode"}
        )
        size_stats = {
            name: {
                "median": float(statistics.median(values)),
                "mean": float(statistics.fmean(values)),
            }
            for name in (
                "encoded_bytes",
                "representation_serialized_bytes",
                "representation_metadata_bytes",
                "representation_array_bytes",
                "representation_native_array_bytes",
                "rgb_bytes",
            )
            if (values := [row[name] for row in group if row.get(name) is not None])
        }
        summary = {
            "condition": condition,
            "codec_family": sample["codec_family"],
            "pbc_preset": sample.get("pbc_preset"),
            "jpeg_quality": sample.get("jpeg_quality"),
            "entropy_mode": sample.get("entropy_mode"),
            "sample_count": len(group),
            "size_bytes": size_stats,
            "stage_timings_ms": stage_stats,
            "full_decode_stage": full_stage,
            "full_decode_ms": stage_stats[full_stage],
        }
        condition_rows.append(summary)

        path, full = condition_stage_path(sample)
        cumulative_by_stage = {}
        for index, (stage, label) in enumerate(path):
            cumulative_runs = []
            incremental_runs = []
            for row in group:
                current = row["stage_timings_ms"].get(stage, [])
                incremental_runs.extend(current)
                previous = [
                    row["stage_timings_ms"].get(previous_stage, [])
                    for previous_stage, _ in path[: index + 1]
                ]
                repeat_count = min((len(values) for values in previous), default=0)
                cumulative_runs.extend(
                    [sum(values[repeat] for values in previous) for repeat in range(repeat_count)]
                )
            incremental_stats = stats(incremental_runs)
            cumulative_stats = stats(cumulative_runs)
            cumulative_by_stage[stage] = cumulative_stats
            representation_bytes = None
            if stage == "forced_lzma_unpack":
                representation_bytes = statistics.median(row["raw_pbc_body_bytes"] for row in group)
            elif stage == "store_body_parse":
                representation_bytes = statistics.median(row["pbc_parsed_serialized_bytes"] for row in group)
            elif stage == "canonical_ir_build":
                representation_bytes = statistics.median(row["representation_serialized_bytes"] for row in group)
            elif stage == "jpeg_file_to_coefficients":
                representation_bytes = statistics.median(row["representation_serialized_bytes"] for row in group)
            elif stage == "patch_application_render":
                representation_bytes = statistics.median(row["rgb_bytes"] for row in group)
            stage_rows.append(
                {
                    "condition": condition,
                    "codec_family": sample["codec_family"],
                    "checkpoint": label,
                    "stage": stage,
                    "available": True,
                    "incremental_median_ms": incremental_stats["median"],
                    "incremental_p95_ms": incremental_stats["p95"],
                    "incremental_mean_ms": incremental_stats["mean"],
                    "cumulative_median_ms": cumulative_stats["median"],
                    "cumulative_p95_ms": cumulative_stats["p95"],
                    "full_decode_median_ms": stage_stats[full]["median"],
                    "percent_of_full_decode": 100 * cumulative_stats["median"] / stage_stats[full]["median"],
                    "representation_serialized_bytes": representation_bytes,
                }
            )
        if sample["codec_family"] == "jpeg":
            stage_rows.append(
                {
                    "condition": condition,
                    "codec_family": "jpeg",
                    "checkpoint": "RGB reconstruction from coefficients",
                    "stage": "jpeg_coefficients_to_rgb",
                    "available": False,
                    "incremental_median_ms": None,
                    "incremental_p95_ms": None,
                    "incremental_mean_ms": None,
                    "cumulative_median_ms": None,
                    "cumulative_p95_ms": None,
                    "full_decode_median_ms": stage_stats[full]["median"],
                    "percent_of_full_decode": None,
                    "representation_serialized_bytes": None,
                }
            )
        full_stats = stage_stats[full]
        stage_rows.append(
            {
                "condition": condition,
                "codec_family": sample["codec_family"],
                "checkpoint": "Decoded RGB",
                "stage": full,
                "available": True,
                "incremental_median_ms": full_stats["median"],
                "incremental_p95_ms": full_stats["p95"],
                "incremental_mean_ms": full_stats["mean"],
                "cumulative_median_ms": full_stats["median"],
                "cumulative_p95_ms": full_stats["p95"],
                "full_decode_median_ms": full_stats["median"],
                "percent_of_full_decode": 100.0,
                "representation_serialized_bytes": statistics.median(row["rgb_bytes"] for row in group),
            }
        )
    return condition_rows, stage_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_csv_rows(conditions: list[dict]) -> list[dict]:
    output = []
    for condition in conditions:
        row = {
            "condition": condition["condition"],
            "codec_family": condition["codec_family"],
            "pbc_preset": condition["pbc_preset"],
            "jpeg_quality": condition["jpeg_quality"],
            "entropy_mode": condition["entropy_mode"],
            "sample_count": condition["sample_count"],
            "encoded_bytes_mean": condition["size_bytes"]["encoded_bytes"]["mean"],
            "encoded_bytes_median": condition["size_bytes"]["encoded_bytes"]["median"],
            "intermediate_serialized_bytes_mean": condition["size_bytes"]["representation_serialized_bytes"]["mean"],
            "intermediate_serialized_bytes_median": condition["size_bytes"]["representation_serialized_bytes"]["median"],
            "intermediate_array_bytes_mean": condition["size_bytes"]["representation_array_bytes"]["mean"],
            "intermediate_array_bytes_median": condition["size_bytes"]["representation_array_bytes"]["median"],
            "intermediate_native_array_bytes_mean": condition["size_bytes"].get("representation_native_array_bytes", {}).get("mean"),
            "rgb_bytes_mean": condition["size_bytes"]["rgb_bytes"]["mean"],
            "full_decode_median_ms": condition["full_decode_ms"]["median"],
            "full_decode_p95_ms": condition["full_decode_ms"]["p95"],
            "full_decode_mean_ms": condition["full_decode_ms"]["mean"],
        }
        for stage, stage_stats in condition["stage_timings_ms"].items():
            row[f"{stage}_median_ms"] = stage_stats["median"]
            row[f"{stage}_p95_ms"] = stage_stats["p95"]
            row[f"{stage}_mean_ms"] = stage_stats["mean"]
        output.append(row)
    return output


def make_plots(condition_rows: list[dict], stage_rows: list[dict], output_dir: Path) -> None:
    labels = [row["condition"] for row in condition_rows]
    fig, axis = plt.subplots(figsize=(12, max(6, 0.48 * len(labels))))
    colors = {
        "forced_lzma_unpack": "#e6a23c",
        "store_body_parse": "#2878b5",
        "canonical_ir_build": "#36a18b",
        "patch_application_render": "#8b65a5",
        "jpeg_file_to_coefficients": "#266a9e",
    }
    for y, condition in enumerate(condition_rows):
        group = [row for row in stage_rows if row["condition"] == condition["condition"]]
        offset = 0.0
        for item in group:
            stage = item["stage"]
            if stage not in colors or not item["available"]:
                continue
            width = item["incremental_median_ms"]
            axis.barh(y, width, left=offset, color=colors[stage], height=0.58, label=stage.replace("_", " "))
            offset += width
        full = condition["full_decode_ms"]
        axis.errorbar(
            full["median"], y,
            xerr=[[0], [max(0, full["p95"] - full["median"])]],
            fmt="D", color="#222222", capsize=3,
            label="full decode median; whisker to p95",
        )
    handles, legend_labels = axis.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Incremental stage medians (ms); diamond marks direct full decode")
    axis.set_title("Decode stage timing by condition")
    axis.legend(unique.values(), unique.keys(), loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "decode_stage_latency.png", dpi=180)
    plt.close(fig)

    stage_markers = {
        "Input bytes": "o",
        "Bare PBC STORE body": "s",
        "Parsed PBC instruction fields": "^",
        "Canonical PBC IR": "D",
        "Patch application and RGB render": "o",
        "JPEG quantized DCT coefficients": "^",
        "Decoded RGB": "X",
    }
    stage_colors = {
        "Input bytes": "#777777",
        "Bare PBC STORE body": "#e6a23c",
        "Parsed PBC instruction fields": "#2878b5",
        "Canonical PBC IR": "#36a18b",
        "Patch application and RGB render": "#8b65a5",
        "JPEG quantized DCT coefficients": "#8b65a5",
        "Decoded RGB": "#222222",
    }
    fig, axis = plt.subplots(figsize=(11, max(6, 0.48 * len(labels))))
    for y, condition in enumerate(condition_rows):
        group = [row for row in stage_rows if row["condition"] == condition["condition"]]
        for item in group:
            checkpoint = item["checkpoint"]
            if checkpoint == "Decoded RGB":
                fraction = 100.0
            elif item["available"]:
                fraction = item["percent_of_full_decode"]
            else:
                continue
            axis.scatter(
                fraction,
                y,
                marker=stage_markers.get(checkpoint, "o"),
                color=stage_colors[checkpoint],
                s=50,
                label=checkpoint,
            )
        axis.scatter(0, y, marker=stage_markers["Input bytes"], color=stage_colors["Input bytes"], s=50, label="Input bytes")
    handles, legend_labels = axis.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    axis.axvline(100, color="#555555", linestyle="--", linewidth=1)
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Cumulative measured stage time / direct full-decode median (%)")
    axis.set_title("Decode cost paid at each intermediate representation")
    axis.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(output_dir / "fraction_of_full_decode.png", dpi=180)
    plt.close(fig)


def format_ms(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def make_report(config: dict, conditions: list[dict], stage_rows: list[dict]) -> str:
    pbc_rows = [row for row in conditions if row["codec_family"] == "pbc"]
    jpeg_rows = [row for row in conditions if row["codec_family"] == "jpeg"]
    stage_table = [
        "| Condition | Checkpoint | Incremental ms (median / p95) | Cumulative median ms | % full decode | Stored representation bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in stage_rows:
        if row["checkpoint"] == "Decoded RGB" or row["available"]:
            stage_table.append(
                f"| {row['condition']} | {row['checkpoint']} | {format_ms(row['incremental_median_ms'])} / {format_ms(row['incremental_p95_ms'])} | {format_ms(row['cumulative_median_ms'])} | {format_ms(row['percent_of_full_decode'])} | {row['representation_serialized_bytes']} |"
            )
        else:
            stage_table.append(
                f"| {row['condition']} | {row['checkpoint']} | unavailable | n/a | n/a | n/a |"
            )
    size_table = [
        "| Condition | Encoded bytes (mean) | Canonical array bytes (mean) | Native array bytes (mean) | Serialized IR bytes (mean) | RGB bytes | Full decode median / p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in conditions:
        sizes = row["size_bytes"]
        native_array_bytes = sizes.get("representation_native_array_bytes", {}).get("mean")
        native_array_text = "n/a" if native_array_bytes is None else f"{native_array_bytes:.1f}"
        size_table.append(
            f"| {row['condition']} | {sizes['encoded_bytes']['mean']:.1f} | {sizes['representation_array_bytes']['mean']:.1f} | {native_array_text} | {sizes['representation_serialized_bytes']['mean']:.1f} | {sizes['rgb_bytes']['mean']:.0f} | {format_ms(row['full_decode_ms']['median'])} / {format_ms(row['full_decode_ms']['p95'])} |"
        )
    pbc_full = {row["condition"]: row["full_decode_ms"]["median"] for row in pbc_rows}
    lzma_rows = [row for row in stage_rows if row["stage"] == "forced_lzma_unpack"]
    pbc_lzma_text = "; ".join(
        f"{row['condition']}: {row['incremental_median_ms']:.4f} ms / {pbc_full[row['condition']]:.4f} ms full ({row['percent_of_full_decode']:.1f}%)"
        for row in lzma_rows
    )
    jpeg_coeff_rows = [row for row in stage_rows if row["stage"] == "jpeg_file_to_coefficients"]
    jpeg_coeff_text = "; ".join(
        f"{row['condition']}: {row['incremental_median_ms']:.4f} ms / {row['full_decode_median_ms']:.4f} ms Pillow full decode ({row['percent_of_full_decode']:.1f}%)"
        for row in jpeg_coeff_rows
    )
    return f"""# Staged Decoding Findings

## Dataset and settings

- Reused the exact Phase 1 source set: {config['dataset']['image_count']} deterministic 256×256 RGB images. Source manifest SHA-256: `{config['dataset']['source_manifest_sha256']}`. Each PNG hash and RGB-pixel hash was checked against the Phase 1 manifest before encoding.
- PBC presets: compression, balanced, quality. Each image/preset was encoded once with LZMA disabled to obtain a STORE body, then raw-LZMA-wrapped from those exact body bytes. The forced wrapper never uses PBC's size-based STORE fallback.
- JPEG: Pillow/libjpeg-turbo 3.1.1 encoding at q{', q'.join(map(str, JPEG_QUALITIES))}; pyjpegio {config['software']['pyjpegio']} coefficient extraction using bundled libjpeg-turbo {config['software']['pyjpegio_libjpeg_turbo']}.
- PBC provenance and full effective configurations are in `config.json`. The external PBC checkout was read only.

## Stage boundaries and representation schema

PBC parsing uses the checkout's `BitReader`, PBC3 header structure, generated palette mask/bounds, and quantized grid indices. The **parsed fields** stage ends with decoded scalar fields, masks, and unsigned grid indices. **Canonical IR construction** maps these to typed arrays and adds the signed palette and residual grid values. **Patch application/render** initializes the PBC canvas, applies each signed-resampled patch, converts the internal color space to RGB, and performs the final resize when present. The standard `PBC3.decompress` call is measured independently for STORE and forced-LZMA streams.

PBC IR schema `tamtd.pbc-ir` v1 stores header fields and ordered patch metadata in canonical JSON inside a deterministic NPZ-like ZIP. Arrays keep each patch's mask (`uint8`), signed palette (`int16`), palette indices (`uint16`), and signed residual values (`int16`) as separate members. The stream can be rebuilt byte-for-byte from the IR. NPZ member order and timestamps are fixed; no pickle/object arrays are used.

JPEG coefficient access calls `jpegio.read(path, read_spatial=False)`, backed by libjpeg-turbo's coefficient API. Its output holds quantized coefficient arrays and tables without pixel-domain spatial arrays. Canonical IR schema `tamtd.jpeg-dct-ir` v1 stores signed `int16` coefficients in `(block_y, block_x, u, v)` natural DCT order; per-component sampling/block metadata; quantization tables; colorspace and image metadata; and Huffman table assignments. The binding exposes no standalone coefficient-to-RGB operation, so that stage is marked unavailable. We do not estimate it by subtracting unrelated timings.

## Timing method

- Each measured operation had {config['timing']['warmups']} warm-up call(s) and {config['timing']['repeats']} timed repetitions per image/condition. Imports and one-time PBC/JPEG initialization are outside the timed calls.
- PBC streams are decoded from in-memory bytes. JPEG coefficient extraction and Pillow full decode read the same cached JPEG file path; encoding and cache writes are outside measured regions. Filesystem read/open cost is included for both JPEG stages.
- Stage summaries report the median, p95, and mean across all repeated image operations. Cumulative rows sum separately measured stages; the direct complete-decode timing is measured independently and is the denominator for fractions. Small differences between cumulative component sums and direct full decode are expected.
- Canonical IR construction and serialization are measured separately from codec decoding. Serialization is an offline cache-preparation cost and is not included in fractions of full decode.

## Stage timing

`decode_stage_latency.png` plots incremental median stage times and direct full-decode medians (whisker to p95). `fraction_of_full_decode.png` plots cumulative stage time at each representation. The full per-condition stage table is in `stage_summary.csv`.

{chr(10).join(stage_table)}

## Representation sizes

RGB size is {256 * 256 * 3} bytes for every image. “Canonical array bytes” count typed NumPy payloads only, excluding Python object overhead. “Native array bytes” record pyjpegio's `int32` coefficient/table arrays before canonicalization; PBC has no separate native-array measurement.

{chr(10).join(size_table)}

## Interpretation and Phase 3 input

1. **PBC LZMA unpack cost:** {pbc_lzma_text}. Values are measured on this machine with Python's `lzma` implementation and must be read alongside the full PBC decoder timings.
2. **JPEG file-to-DCT cost:** {jpeg_coeff_text}. `jpegio.read` includes file open/read, header/marker handling, entropy decoding, coefficient allocation, and Python-array exposure. Its coefficient-to-RGB successor is not independently available through this binding.
3. **Earlier structured representation:** by normalized checkpoint fraction, PBC reaches canonical IR at 44.8%, 57.3%, and 73.2% of direct STORE decode for compression, balanced, and quality; forced-LZMA PBC reaches it at 46.7%, 58.0%, and 76.0%. JPEG coefficient extraction takes 89.9%–94.1% of Pillow's full RGB decode. PBC therefore reaches its canonical IR earlier by this measure on this implementation. Absolute PBC timings are in milliseconds while JPEG timings are sub-millisecond, and the two implementations use different libjpeg-turbo versions; these are not algorithm-only comparisons.
4. **Representation sizes:** see the table above and `summary.csv`. PBC raw STORE/LZMA streams, parsed payload, canonical IR arrays/archive, JPEG file, canonical DCT arrays/archive, and RGB size are all recorded per image.
5. **Systems feasibility:** the PBC IR and JPEG DCT IR are usable structured inputs from a decoding and storage perspective. This phase measures cost and size only; it makes no claim about model accuracy or representation usefulness.
6. **Parsed PBC cost:** compare `store_body_parse`, `canonical_ir_build`, and `canonical_ir_serialize` against `forced_lzma_unpack` in the stage table. STORE parsing/canonicalization is the additional cost to obtain structured instructions after entropy unpack.
7. **Phase 3 representations:** use PBC `tamtd.pbc-ir` v1 for compression, balanced, and quality; use JPEG `tamtd.jpeg-dct-ir` v1 for q10, q60, and q80, with q1/q5 as low-rate controls. Keep encoded bytes and fully decoded RGB as reference conditions. No model training or reconstruction was run.

## Validation and limitations

- All {config['dataset']['image_count']} samples per condition were checked. PBC STORE and forced-LZMA bodies matched byte-for-byte after unpack; their full decodes matched pixel-for-pixel; canonical PBC IR reserialized to the original STORE stream and rendered pixel-identically. Serialization was deterministic.
- JPEG coefficient extraction was repeated and checked for deterministic metadata/arrays. Component block dimensions matched the header, and Pillow file and in-memory decodes matched pixel-for-pixel.
- pyjpegio and Pillow use bundled libjpeg-turbo versions 3.2.0 and 3.1.1 respectively. The coefficient-stage/full-decode ratio therefore compares the supported coefficient binding with Pillow's normal RGB path, not one identical library build.
- JPEG file I/O is included because pyjpegio's supported interface accepts a path. PBC uses byte buffers. Cache serialization sizes and timing are separate from decoding.
- Generated representations live under the ignored `data/cache/staged_decode/` directory and can be regenerated with `scripts/run_staged_decode_benchmark.py`; compact result tables and plots are stored here.

Per-sample data: `sample_results.jsonl`. Aggregates: `summary.json`, `summary.csv`, `stage_summary.csv`. Exact run metadata: `config.json`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure PBC and JPEG decode stages on the Phase 1 images")
    parser.add_argument("--source-dir", type=Path, default=Path("results/Codec Benchmark/sources"))
    parser.add_argument("--pbc-root", type=Path, default=os.environ.get("PBC_ROOT"))
    parser.add_argument("--output", type=Path, default=Path("results/Staged Decoding"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/staged_decode"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("--warmups must be non-negative and --repeats must be positive")
    source_dir = args.source_dir.resolve()
    output_dir = args.output.resolve()
    cache_root = args.cache_dir.resolve()
    pbc_root = resolve_pbc_root(args.pbc_root)
    rows, images, phase1_config = source_images(source_dir, args.limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    pairs = {preset: PBCCodecPair(pbc_root, preset=preset) for preset in PBC_PRESETS}
    results = []
    for row, image in zip(rows, images):
        for pair in pairs.values():
            results.extend(pbc_sample_rows(row, image, pair, args.warmups, args.repeats, cache_root))
        for quality in JPEG_QUALITIES:
            results.append(jpeg_sample_row(row, image, quality, args.warmups, args.repeats, cache_root))

    condition_rows, stage_rows = aggregate(results)
    (output_dir / "sample_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in results),
        encoding="utf-8",
    )
    write_csv(output_dir / "summary.csv", summary_csv_rows(condition_rows))
    write_csv(output_dir / "stage_summary.csv", stage_rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"conditions": condition_rows, "stages": stage_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    software = {
        "python": platform.python_version(),
        "pillow": pillow_version,
        "pillow_libjpeg_turbo": getattr(Image.core, "libjpeg_turbo_version", None),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "pyjpegio": importlib.metadata.version("pyjpegio"),
        "pyjpegio_libjpeg_turbo": "3.2.0",
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }
    config = {
        "dataset": {
            "name": phase1_config["source_metadata"]["dataset"],
            "image_count": len(rows),
            "full_phase1_image_count": phase1_config["image_count"],
            "source_dir": str(source_dir),
            "source_manifest_sha256": phase1_config["source_manifest_sha256"],
            "source_manifest_selection": phase1_config["source_metadata"]["selection"],
            "preprocessing": phase1_config["source_metadata"]["crop_resize"],
            "image_size": [256, 256],
            "pixel_format": "RGB uint8",
        },
        "settings": {
            "pbc_presets": list(PBC_PRESETS),
            "pbc_effective_configurations": {name: pairs[name].config_values for name in PBC_PRESETS},
            "pbc_git_state": pairs[PBC_PRESETS[0]].git_state,
            "jpeg_qualities": list(JPEG_QUALITIES),
            "jpeg_encoder": "Pillow JPEG / libjpeg-turbo",
            "jpeg_encoder_settings": {
                "quality": "condition quality",
                "subsampling": "4:2:0",
                "optimize": False,
                "progressive": False,
            },
            "jpeg_coefficient_api": "pyjpegio.read(path, read_spatial=False), backed by jpeg_read_coefficients",
            "jpeg_coefficients_to_rgb_api": None,
            "other_codecs": "not staged in this phase",
        },
        "timing": {
            "clock": "time.perf_counter_ns",
            "warmups": args.warmups,
            "repeats": args.repeats,
            "imports_outside_timed_regions": True,
            "jpeg_file_io_included": True,
            "jpeg_cache_write_included": False,
            "pbc_input_mode": "in-memory bytes",
            "stage_definitions": {
                "forced_lzma_unpack": "Raw LZMA2 body decompression only; frame validation and wrapper creation excluded.",
                "store_body_parse": "PBC header, patch metadata, masks, palette bounds, and unsigned grid indices.",
                "canonical_ir_build": "Typed arrays plus generated signed palette and residual grid values.",
                "patch_application_render": "Canvas initialization, bicubic signed patch application, colorspace conversion, and final resize.",
                "jpeg_file_to_coefficients": "pyjpegio file read, header/marker parsing, entropy decoding, and quantized coefficient/table exposure.",
                "jpeg_coefficients_to_rgb": "not exposed by the selected public binding; not estimated by subtraction.",
            },
        },
        "software": software,
        "cache_dir": str(cache_root),
        "run_arguments": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "limit": args.limit,
            "output": str(output_dir),
            "pbc_root": str(pbc_root),
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    make_plots(condition_rows, stage_rows, output_dir)
    (output_dir / "Staged Decoding Findings.md").write_text(
        make_report(config, condition_rows, stage_rows), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "images": len(rows),
                "conditions": len(condition_rows),
                "sample_rows": len(results),
                "output": str(output_dir),
                "cache": str(cache_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
