from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codec_stage_ir import SCHEMA, SCHEMA_VERSION, TOKEN_GRID, adapt_common_tokens, validate_ir, verify_stream_alignment
from tamtd.models.accessibility import RGBResNet18
from tamtd.phase47.decoder_bindings import CodecLibraries
from tamtd.training.telemetry import ResourceMonitor, collect_resource_snapshot


CONDITIONS = ("jxl_d6", "jxl_d3", "webp_q30", "webp_q75", "avif_q40", "avif_q60")
RGB_SEEDS = (42, 123, 777)
FEATURE_CONDITIONS = {"webp_q30", "webp_q75", "avif_q40", "avif_q60"}
NUM_CLASSES = 10


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def _load_rgb_models(cache_root: Path, device: torch.device) -> dict[int, RGBResNet18]:
    model_paths = {
        42: cache_root / "checkpoints" / "original_rgb__decoded_rgb_reference.pt",
        123: cache_root / "checkpoints" / "phase4" / "original_rgb__decoded_rgb_reference__seed123.pt",
        777: cache_root / "checkpoints" / "phase4" / "original_rgb__decoded_rgb_reference__seed777.pt",
    }
    models = {}
    for seed, path in model_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required frozen RGB checkpoint missing: {path}")
        model = RGBResNet18()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models[seed] = model.to(device).eval()
    return models


def _infer(models: dict[int, RGBResNet18], images: list[np.ndarray], device: torch.device) -> dict[int, np.ndarray]:
    if not images:
        return {seed: np.empty((0, NUM_CLASSES), dtype=np.float32) for seed in models}
    tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device)
    with torch.inference_mode():
        return {seed: model(tensor).float().cpu().numpy() for seed, model in models.items()}


def _score(logits: np.ndarray, labels: np.ndarray, prior_ce: float) -> dict[str, float | int]:
    if len(labels) == 0:
        return {"n_samples": 0, "accuracy": float("nan"), "cross_entropy_nats": float("nan"), "u_probe_bits": float("nan")}
    tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    cross_entropy = float(F.cross_entropy(tensor, label_tensor).item())
    accuracy = float((tensor.argmax(dim=1) == label_tensor).float().mean().item())
    return {
        "n_samples": int(len(labels)),
        "accuracy": accuracy,
        "cross_entropy_nats": cross_entropy,
        "class_prior_cross_entropy_nats": prior_ce,
        "u_probe_bits": float((prior_ce - cross_entropy) / math.log(2.0)),
    }


def _prior_cross_entropy(train_labels: np.ndarray, labels: np.ndarray) -> float:
    counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=NUM_CLASSES).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("the training split must contain all ten classes")
    probabilities = counts / counts.sum()
    return float(-np.log(probabilities[np.asarray(labels, dtype=np.int64)]).mean())


def _reference_rows(path: Path) -> dict[tuple[str, int], dict]:
    output = {}
    if not path.is_file():
        return output
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("representation") == "decoded_rgb_reference" and row.get("split") == "test":
                output[(row["condition"], int(row["seed"]))] = row
    return output


def _test_indices(rows: list[dict], requested: int) -> list[int]:
    indices = [index for index, row in enumerate(rows) if row["split"] == "test"]
    if not indices:
        raise ValueError("the shared manifest has no held-out test samples")
    count = min(requested, len(indices))
    offsets = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    return [indices[int(offset)] for offset in offsets]


def _read_payload(handle, reference: dict, expected_hash: str) -> bytes:
    handle.seek(int(reference["offset"]))
    payload = handle.read(int(reference["length"]))
    if len(payload) != int(reference["length"]):
        raise ValueError("short read from the verified codec stream")
    if hashlib.sha256(payload).hexdigest() != expected_hash or expected_hash != reference["sha256"]:
        raise ValueError("codec stream hash changed after manifest verification")
    return payload


def _check_resource_guard(disk_path: Path, starting: bool) -> dict:
    snapshot = collect_resource_snapshot(torch_module=torch, device=torch.device("cuda"), disk_path=disk_path, include_gpu=starting)
    disk_free = snapshot.get("disk_free_bytes")
    system = snapshot.get("system", {})
    available = system.get("memory_available_bytes")
    commit_free = None
    if system.get("commit_limit_bytes") is not None and system.get("commit_used_bytes") is not None:
        commit_free = system["commit_limit_bytes"] - system["commit_used_bytes"]
    gpu = snapshot.get("gpu", {})
    if disk_free is None or available is None or commit_free is None:
        raise RuntimeError("Phase 4.7 decoder extraction resource telemetry is incomplete")
    values = {
        "disk_free_bytes": int(disk_free),
        "memory_available_bytes": int(available),
        "commit_free_bytes": int(commit_free),
        "vram_free_mb": gpu.get("vram_free_mb"),
    }
    if disk_free < (20 if starting else 12) * 1024**3:
        raise RuntimeError("Phase 4.7 extraction stopped by the C: free-space guard")
    if available < (4 if starting else 2) * 1024**3:
        raise RuntimeError("Phase 4.7 extraction stopped by the host-memory guard")
    if commit_free < (5 if starting else 3) * 1024**3:
        raise RuntimeError("Phase 4.7 extraction stopped by the system-commit guard")
    if starting and (values["vram_free_mb"] is None or values["vram_free_mb"] < 1024):
        raise RuntimeError("Phase 4.7 extraction requires at least 1 GiB free VRAM")
    return values


def _condition_rows(condition: str, result_dir: Path, full_logits: dict[int, np.ndarray], test_labels: np.ndarray, train_labels: np.ndarray,
                    progression_logits: dict[int, dict[int, np.ndarray]], progression_info: dict[int, dict], test_position: dict[int, int]) -> list[dict]:
    output = []
    prior_ce = _prior_cross_entropy(train_labels, test_labels)
    for seed in RGB_SEEDS:
        metrics = _score(full_logits[seed], test_labels, prior_ce)
        output.append({
            "condition": condition,
            "stage": "decoded_rgb",
            "representation": "frozen RGBResNet18",
            "seed": seed,
            **metrics,
            "decoded_rgb_accuracy": metrics["accuracy"],
            "decoded_rgb_u_probe_bits": metrics["u_probe_bits"],
            "relative_u_probe_utility": 1.0,
            "relative_accuracy": 1.0,
            "accuracy_ratio": 1.0,
            "comparison_type": "same checkpoint on matched decoded RGB",
        })
    if not progression_info:
        return output
    stage_name = f"jxl_dc_progressive_{next(iter(progression_info.values()))['ratio']}x"
    for seed in RGB_SEEDS:
        by_position = progression_logits.get(seed, {})
        positions = sorted(by_position)
        if not positions:
            continue
        logits = np.stack([by_position[position] for position in positions])
        labels = test_labels[np.asarray(positions, dtype=np.int64)]
        paired_rgb = full_logits[seed][np.asarray(positions, dtype=np.int64)]
        stage_metrics = _score(logits, labels, _prior_cross_entropy(train_labels, labels))
        rgb_metrics = _score(paired_rgb, labels, _prior_cross_entropy(train_labels, labels))
        rgb_u = float(rgb_metrics["u_probe_bits"])
        rgb_acc = float(rgb_metrics["accuracy"])
        output.append({
            "condition": condition,
            "stage": stage_name,
            "representation": "libjxl progressive RGB buffer",
            "seed": seed,
            **stage_metrics,
            "decoded_rgb_accuracy": rgb_acc,
            "decoded_rgb_u_probe_bits": rgb_u,
            "relative_u_probe_utility": float(stage_metrics["u_probe_bits"]) / rgb_u if rgb_u > 0 else None,
            "relative_accuracy": (float(stage_metrics["accuracy"]) - 0.1) / (rgb_acc - 0.1) if rgb_acc > 0.1 else None,
            "accuracy_ratio": float(stage_metrics["accuracy"]) / rgb_acc if rgb_acc > 0 else None,
            "progression_coverage": len(positions) / len(test_labels),
            "mean_intended_downsampling_ratio": float(np.mean([progression_info[position]["ratio"] for position in positions])),
            "comparison_type": "same checkpoint; paired progressive subset versus matching full decode subset",
        })
    return output


def _run_timing(libraries: CodecLibraries, condition: str, stream_path: Path, rows: list[dict], stream_rows: list[dict],
                selected: list[int], repeats: int) -> list[dict]:
    output = []
    with stream_path.open("rb") as handle:
        warm_index = selected[0] if selected else next(index for index, row in enumerate(rows) if row["split"] == "test")
        warm_payload = _read_payload(handle, stream_rows[warm_index]["stream"], stream_rows[warm_index]["stream"]["sha256"])
        if condition.startswith("jxl"):
            libraries.decode_jxl(warm_payload, capture_progression=False)
        elif condition.startswith("webp"):
            libraries.decode_webp_coefficients(warm_payload)
            libraries.decode_webp_yuv(warm_payload)
            libraries.decode_webp_rgb(warm_payload)
        else:
            libraries.decode_avif_stage(warm_payload)
            libraries.decode_avif_yuv(warm_payload)
            libraries.decode_avif_rgb(warm_payload)
        for index in selected:
            row = rows[index]
            reference = stream_rows[index]["stream"]
            payload = _read_payload(handle, reference, reference["sha256"])
            for repeat in range(repeats):
                if condition.startswith("jxl"):
                    result = libraries.decode_jxl(payload, capture_progression=True)
                    if result.progressions:
                        early = libraries.decode_jxl(payload, stop_after_progression=1, capture_progression=True)
                        if not early.progressions:
                            raise RuntimeError("JXL progression disappeared on a repeated early-exit decode")
                        stage = early.progressions[0]
                        output.append({
                            "condition": condition, "cache_index": row["cache_index"], "image_id": row["image_id"], "split": row["split"],
                            "stage": f"jxl_dc_progressive_{stage.ratio}x", "repeat": repeat,
                            "decoder_ms": stage.decoder_ms, "capture_adapter_ms": 0.0, "model_adapter_ms": 0.0,
                            "cumulative_decoder_ms": stage.decoder_ms, "cumulative_decoder_adapter_ms": stage.decoder_ms,
                            "thread_count": 1, "timing_boundary": "JXL_DEC_FRAME_PROGRESSION + JxlDecoderFlushImage; stopped there",
                        })
                    output.append({
                        "condition": condition, "cache_index": row["cache_index"], "image_id": row["image_id"], "split": row["split"],
                        "stage": "decoded_rgb", "repeat": repeat, "decoder_ms": result.total_ms,
                        "capture_adapter_ms": 0.0, "model_adapter_ms": 0.0,
                        "cumulative_decoder_ms": result.total_ms, "cumulative_decoder_adapter_ms": result.total_ms,
                        "thread_count": 1, "timing_boundary": "JXL_DEC_SUCCESS after full RGB output",
                    })
                elif condition.startswith("webp"):
                    features, stage_ms, capture_ms = libraries.decode_webp_coefficients(payload)
                    adapter_start = time.perf_counter_ns()
                    adapt_common_tokens(features)
                    model_adapter_ms = (time.perf_counter_ns() - adapter_start) / 1_000_000
                    output.append({
                        "condition": condition, "cache_index": row["cache_index"], "image_id": row["image_id"], "split": row["split"],
                        "stage": "pre_reconstruction_coefficients", "repeat": repeat, "decoder_ms": stage_ms,
                        "capture_adapter_ms": capture_ms, "model_adapter_ms": model_adapter_ms,
                        "cumulative_decoder_ms": stage_ms,
                        "cumulative_decoder_adapter_ms": stage_ms + capture_ms + model_adapter_ms,
                        "thread_count": 1, "timing_boundary": "VP8ProcessRow after final ParseResiduals, before ReconstructRow/FinishRow",
                    })
                    _, _, yuv_ms = libraries.decode_webp_yuv(payload)
                    output.append({
                        "condition": condition, "cache_index": row["cache_index"], "image_id": row["image_id"], "split": row["split"],
                        "stage": "decoded_yuv", "repeat": repeat, "decoder_ms": yuv_ms,
                        "capture_adapter_ms": 0.0, "model_adapter_ms": 0.0,
                        "cumulative_decoder_ms": yuv_ms, "cumulative_decoder_adapter_ms": yuv_ms,
                        "thread_count": 1, "timing_boundary": "WebPDecodeYUV completed",
                    })
                    _, rgb_ms = libraries.decode_webp_rgb(payload)
                    output.append({
                        "condition": condition, "cache_index": row["cache_index"], "image_id": row["image_id"], "split": row["split"],
                        "stage": "decoded_rgb", "repeat": repeat, "decoder_ms": rgb_ms,
                        "capture_adapter_ms": 0.0, "model_adapter_ms": 0.0,
                        "cumulative_decoder_ms": rgb_ms, "cumulative_decoder_adapter_ms": rgb_ms,
                        "thread_count": 1, "timing_boundary": "WebPDecodeRGB completed",
                    })
                else:
                    features, stage_ms, capture_ms = libraries.decode_avif_stage(payload)
                    adapter_start = time.perf_counter_ns()
                    adapt_common_tokens(features)
                    model_adapter_ms = (time.perf_counter_ns() - adapter_start) / 1_000_000
                    output.append({
                        "condition": condition, "cache_index": row["cache_index"], "image_id": row["image_id"], "split": row["split"],
                        "stage": "pre_reconstruction_coefficients", "repeat": repeat, "decoder_ms": stage_ms,
                        "capture_adapter_ms": capture_ms, "model_adapter_ms": model_adapter_ms,
                        "cumulative_decoder_ms": stage_ms,
                        "cumulative_decoder_adapter_ms": stage_ms + capture_ms + model_adapter_ms,
                        "thread_count": 1, "timing_boundary": "libaom parse-only av1_decode_tg_tiles_and_wrapup before loop/restoration filters",
                    })
                    _, yuv_ms = libraries.decode_avif_yuv(payload)
                    output.append({
                        "condition": condition, "cache_index": row["cache_index"], "image_id": row["image_id"], "split": row["split"],
                        "stage": "decoded_yuv", "repeat": repeat, "decoder_ms": yuv_ms,
                        "capture_adapter_ms": 0.0, "model_adapter_ms": 0.0,
                        "cumulative_decoder_ms": yuv_ms, "cumulative_decoder_adapter_ms": yuv_ms,
                        "thread_count": 1, "timing_boundary": "avifDecoderReadMemory completed with full YUV planes",
                    })
                    _, rgb_decode_ms, rgb_convert_ms = libraries.decode_avif_rgb(payload)
                    rgb_ms = rgb_decode_ms + rgb_convert_ms
                    output.append({
                        "condition": condition, "cache_index": row["cache_index"], "image_id": row["image_id"], "split": row["split"],
                        "stage": "decoded_rgb", "repeat": repeat, "decoder_ms": rgb_ms,
                        "capture_adapter_ms": 0.0, "model_adapter_ms": 0.0,
                        "cumulative_decoder_ms": rgb_ms, "cumulative_decoder_adapter_ms": rgb_ms,
                        "thread_count": 1, "timing_boundary": "avifDecoderReadMemory + avifImageYUVToRGB completed",
                    })
    return output


def run(args, libraries: CodecLibraries, monitor: ResourceMonitor) -> dict:
    cache_root = args.cache_root.resolve()
    result_dir = args.results_dir.resolve()
    phase_stream_root = args.phase_stream_root.resolve()
    build_root = args.build_root.resolve()
    base_rows = _read_jsonl(cache_root / "manifest.jsonl")
    if len(base_rows) != 13_394 or [row["cache_index"] for row in base_rows] != list(range(len(base_rows))):
        raise ValueError("Phase 4.7 requires the exact contiguous 13,394-sample Phase 3 manifest")
    split_counts = {split: sum(row["split"] == split for row in base_rows) for split in ("train", "validation", "test")}
    if split_counts != {"train": 8522, "validation": 947, "test": 3925}:
        raise ValueError(f"unexpected shared split counts: {split_counts}")
    stream_sets = {}
    for condition in args.conditions:
        manifest_path = phase_stream_root / f"{condition}.jsonl"
        stream_path = phase_stream_root / f"{condition}.bin"
        condition_rows = _read_jsonl(manifest_path)
        verify_stream_alignment(base_rows, condition_rows, stream_path)
        stream_sets[condition] = (condition_rows, stream_path, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        print(f"verified {condition}: {len(condition_rows)} exact aligned streams", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 4.7 RGB inference requires CUDA; refusing to fall back to CPU")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    models = _load_rgb_models(cache_root, device)
    train_labels = np.asarray([int(row["label"]) for row in base_rows if row["split"] == "train"], dtype=np.int64)
    test_rows = [row for row in base_rows if row["split"] == "test"]
    test_labels = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)
    test_position = {row["cache_index"]: position for position, row in enumerate(test_rows)}
    feature_dir = cache_root / "phase47" / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    reference = _reference_rows(args.phase45b_results / "decoded_rgb_endpoints.csv")
    all_probe_rows = []
    all_timing_rows = []
    all_condition_checks = []
    all_manifest_rows = []
    all_jxl_progression = []

    for condition in args.conditions:
        condition_rows, stream_path, manifest_hash = stream_sets[condition]
        is_structured = condition in FEATURE_CONDITIONS
        feature_path = feature_dir / f"{condition}.npy"
        completion_path = feature_dir / f"{condition}.complete.json"
        feature_map = None
        extract_features = False
        if is_structured:
            if feature_path.exists() and not completion_path.exists():
                raise FileExistsError(f"incomplete feature cache exists; preserve and inspect before rerunning: {feature_path}")
            if completion_path.exists():
                metadata = json.loads(completion_path.read_text(encoding="utf-8"))
                if metadata.get("manifest_sha256") != manifest_hash or metadata.get("samples") != len(base_rows):
                    raise ValueError(f"completed feature cache does not match the exact stream manifest: {condition}")
                feature_map = np.load(feature_path, mmap_mode="r+", allow_pickle=False)
                if feature_map.shape != (len(base_rows), 256, 62) or feature_map.dtype != np.float32:
                    raise ValueError(f"feature cache has an unexpected shape or dtype: {condition}")
                print(f"reusing verified completed stage features: {condition}", flush=True)
            else:
                feature_map = np.lib.format.open_memmap(
                    feature_path, mode="w+", dtype=np.float32, shape=(len(base_rows), 256, 62)
                )
                extract_features = True

        full_logits = {seed: np.empty((len(test_rows), NUM_CLASSES), dtype=np.float32) for seed in RGB_SEEDS}
        progression_logits: dict[int, dict[int, np.ndarray]] = {seed: {} for seed in RGB_SEEDS}
        progression_info = {}
        full_images: list[np.ndarray] = []
        full_positions: list[int] = []
        progression_images: list[np.ndarray] = []
        progression_positions: list[int] = []
        progression_times: list[dict] = []
        check_hashes = {}

        def flush_full_batch() -> None:
            if not full_images:
                return
            outputs = _infer(models, full_images, device)
            for seed, logits in outputs.items():
                full_logits[seed][np.asarray(full_positions, dtype=np.int64)] = logits
            full_images.clear()
            full_positions.clear()

        def flush_progression_batch() -> None:
            if not progression_images:
                return
            outputs = _infer(models, progression_images, device)
            for seed, logits in outputs.items():
                for position, item_logits in zip(progression_positions, logits, strict=True):
                    progression_logits[seed][position] = item_logits
            progression_images.clear()
            progression_positions.clear()

        with stream_path.open("rb") as handle:
            for index, (row, encoded) in enumerate(zip(base_rows, condition_rows, strict=True)):
                stream_ref = encoded["stream"]
                payload = _read_payload(handle, stream_ref, stream_ref["sha256"])
                if extract_features:
                    if condition.startswith("webp"):
                        stage_features, _, _ = libraries.decode_webp_coefficients(payload)
                    else:
                        stage_features, _, _ = libraries.decode_avif_stage(payload)
                    metadata = {
                        "schema": SCHEMA,
                        "schema_version": SCHEMA_VERSION,
                        "stage": "pre_reconstruction_coefficients",
                        "codec": condition.split("_")[0],
                        "setting": condition,
                        "width": 256,
                        "height": 256,
                        "token_grid": list(TOKEN_GRID),
                        "feature_fields": 62,
                        "pre_reconstruction": True,
                    }
                    validate_ir(metadata, stage_features)
                    feature_map[index] = stage_features

                if row["split"] == "test":
                    test_pos = test_position[row["cache_index"]]
                    if condition.startswith("jxl"):
                        decoded = libraries.decode_jxl(payload, capture_progression=True, capture_progressive_pixels=True)
                        rgb = decoded.rgb
                        if decoded.progressions:
                            first = decoded.progressions[0]
                            if first.pixels is None:
                                raise RuntimeError("first JXL progression output was not captured")
                            progression_images.append(first.pixels)
                            progression_positions.append(test_pos)
                            progression_info[test_pos] = {"ratio": first.ratio, "decoder_ms": first.decoder_ms}
                            all_jxl_progression.append({
                                "condition": condition, "cache_index": row["cache_index"], "image_id": row["image_id"],
                                "intended_downsampling_ratio": first.ratio, "decoder_ms_to_progression": first.decoder_ms,
                            })
                            if len(progression_images) == args.inference_batch_size:
                                flush_progression_batch()
                    elif condition.startswith("webp"):
                        rgb, _ = libraries.decode_webp_rgb(payload)
                    else:
                        rgb, _, _ = libraries.decode_avif_rgb(payload)
                    full_images.append(rgb)
                    full_positions.append(test_pos)
                    if len(full_images) == args.inference_batch_size:
                        flush_full_batch()
                    if len(check_hashes) < 8:
                        check_hashes[row["cache_index"]] = hashlib.sha256(rgb.tobytes()).hexdigest()
                if index % 500 == 0:
                    _check_resource_guard(args.disk_path, starting=False)
                    print(f"{condition}: decoded {index + 1}/{len(base_rows)}", flush=True)
        flush_full_batch()
        flush_progression_batch()
        if feature_map is not None and extract_features:
            feature_map.flush()
            del feature_map
            _write_json(completion_path, {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "condition": condition,
                "samples": len(base_rows),
                "shape": [len(base_rows), 256, 62],
                "dtype": "float32",
                "manifest_sha256": manifest_hash,
                "stage": "pre_reconstruction_coefficients",
                "completed": True,
            })

        endpoint_rows = _condition_rows(
            condition, result_dir, full_logits, test_labels, train_labels, progression_logits, progression_info, test_position
        )
        all_probe_rows.extend(endpoint_rows)
        for seed in RGB_SEEDS:
            actual = next(row for row in endpoint_rows if row["seed"] == seed and row["stage"] == "decoded_rgb")
            expected = reference.get((condition, seed))
            if expected is not None:
                all_condition_checks.append({
                    "condition": condition,
                    "seed": seed,
                    "reference_samples": int(expected["n_samples"]),
                    "recomputed_samples": actual["n_samples"],
                    "reference_accuracy": float(expected["accuracy"]),
                    "recomputed_accuracy": actual["accuracy"],
                    "accuracy_delta": float(actual["accuracy"]) - float(expected["accuracy"]),
                    "reference_u_probe_bits": float(expected["u_probe_bits"]),
                    "recomputed_u_probe_bits": actual["u_probe_bits"],
                    "u_probe_delta_bits": float(actual["u_probe_bits"]) - float(expected["u_probe_bits"]),
                })
        for index, digest in check_hashes.items():
            row = base_rows[index]
            stream_ref = condition_rows[index]["stream"]
            with stream_path.open("rb") as handle:
                payload = _read_payload(handle, stream_ref, stream_ref["sha256"])
            if condition.startswith("jxl"):
                repeated = libraries.decode_jxl(payload, capture_progression=False).rgb
            elif condition.startswith("webp"):
                repeated, _ = libraries.decode_webp_rgb(payload)
            else:
                repeated, _, _ = libraries.decode_avif_rgb(payload)
            repeated_digest = hashlib.sha256(repeated.tobytes()).hexdigest()
            if repeated_digest != digest:
                raise ValueError(f"full decode path is nondeterministic for {condition} cache_index={index}")

        for row in endpoint_rows:
            all_manifest_rows.append({
                "condition": condition,
                "stage": row["stage"],
                "decoder": "libjxl" if condition.startswith("jxl") else "libwebp" if condition.startswith("webp") else "libavif + libaom",
                "native_representation": row["representation"],
                "test_samples": row["n_samples"],
                "pre_reconstruction": row["stage"] != "decoded_rgb",
            })
        if is_structured:
            all_manifest_rows.append({
                "condition": condition,
                "stage": "pre_reconstruction_coefficients",
                "decoder": "libwebp" if condition.startswith("webp") else "libaom + libavif",
                "native_representation": "canonical 16x16 token grid; coefficients, quantizer and prediction metadata",
                "test_samples": split_counts["test"],
                "pre_reconstruction": True,
            })
        timing_indices = _test_indices(base_rows, args.timing_samples)
        timing_rows = _run_timing(libraries, condition, stream_path, base_rows, condition_rows, timing_indices, args.timing_repeats)
        all_timing_rows.extend(timing_rows)
        _write_csv(result_dir / "decoder_timing_per_sample.csv", all_timing_rows)
        print(f"completed {condition}: RGB checkpoints evaluated; {len(timing_rows)} timing rows", flush=True)

    _write_csv(result_dir / "rgb_probe_per_seed.csv", all_probe_rows)
    _write_csv(result_dir / "decoder_stage_manifest.csv", all_manifest_rows)
    _write_csv(result_dir / "full_decode_rechecks.csv", all_condition_checks)
    _write_csv(result_dir / "jxl_progression_per_sample.csv", all_jxl_progression)
    metadata = {
        "phase": "4.7 Standard Codec Stages",
        "samples": len(base_rows),
        "split_counts": split_counts,
        "conditions": list(args.conditions),
        "streams_reencoded": False,
        "stream_manifest_sha256": {condition: stream_sets[condition][2] for condition in args.conditions},
        "rgb_probe_checkpoints": [42, 123, 777],
        "full_decode_sample_hash_rechecks": 8,
        "timing_sample_count_per_condition": min(args.timing_samples, split_counts["test"]),
        "timing_repeats": args.timing_repeats,
        "thread_count": 1,
        "cuda_device": torch.cuda.get_device_name(device),
        "jxl_progression_sample_count": len(all_jxl_progression),
        "resource_telemetry": monitor.stop(),
        "stage_cache_paths": {condition: str((feature_dir / f"{condition}.npy").relative_to(PROJECT_ROOT)) for condition in sorted(FEATURE_CONDITIONS.intersection(args.conditions))},
    }
    _write_json(result_dir / "extraction_metadata.json", metadata)
    del models
    torch.cuda.empty_cache()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "data/cache/neural_accessibility/imagenette_320")
    parser.add_argument("--phase-stream-root", type=Path, default=PROJECT_ROOT / "data/cache/neural_accessibility/imagenette_320/phase4_streams")
    parser.add_argument("--phase45b-results", type=Path, default=PROJECT_ROOT / "results/Phase 4.5B Analysis")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/Phase 4.7 Standard Codec Stages")
    parser.add_argument("--build-root", type=Path, default=os.environ.get("TAMTD_PHASE47_BUILD_DIR"))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--timing-samples", type=int, default=256)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--disk-path", type=Path, default=Path(PROJECT_ROOT.anchor or PROJECT_ROOT))
    args = parser.parse_args()
    if args.build_root is None:
        parser.error("--build-root or TAMTD_PHASE47_BUILD_DIR is required")
    if args.timing_samples < 100 or args.timing_repeats < 2:
        parser.error("timing requires at least 100 distinct test samples and 2 repeated measurements")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for frozen RGB checkpoint evaluation; CPU fallback is disabled")
    args.disk_path = args.disk_path.resolve()
    _check_resource_guard(args.disk_path, starting=True)
    libraries = CodecLibraries(args.build_root)
    monitor = ResourceMonitor(torch_module=torch, device=torch.device("cuda"), disk_path=args.disk_path, interval_seconds=1, gpu_interval_seconds=5).start()
    try:
        result = run(args, libraries, monitor)
        print(json.dumps({key: value for key, value in result.items() if key != "resource_telemetry"}, indent=2))
    except BaseException:
        diagnostics = monitor.stop()
        _write_json(args.results_dir / "extraction_resource_diagnostics.json", {"status": "stopped", "resources": diagnostics})
        raise


if __name__ == "__main__":
    main()
