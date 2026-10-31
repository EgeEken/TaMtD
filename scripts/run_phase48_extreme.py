from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version, features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from tamtd.analysis.phase48 import exact_calibration_duplicate, select_calibration_rows
from tamtd.codec_benchmark import image_metrics
from tamtd.codec_stage_ir import verify_stream_alignment
from tamtd.data.imagenette_accessibility import get_imagenette_source_root, load_prepared_rgb, read_manifest
from tamtd.runtime import cache_root, data_root

from scripts.build_phase4_raw_byte_cache import CODEC_FACTORIES
from scripts.run_phase4 import _python_executable, _run_job


NEW_CONDITIONS = ("jpeg_q1", "jpeg_q5", "webp_q5", "webp_q10", "webp_q20", "avif_q5", "avif_q10", "avif_q20")
CALIBRATION_PER_CLASS = 20
ENCODE_REPEATS = 3
DECODE_REPEATS = 3
REDUNDANCY_RULE = "Skip duplicate training only if encoded bytes match exactly for every calibration image; rate-distortion similarity alone does not establish raw-byte redundancy."


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _library_versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "pillow": pillow_version,
        "libjpeg": features.version("jpg"),
        "libwebp": features.version("webp"),
        "libavif": features.version("avif"),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def _calibration_summary(rows: list[dict]) -> list[dict]:
    output = []
    for condition in NEW_CONDITIONS:
        values = [row for row in rows if row["condition"] == condition]
        summary = {"condition": condition, "n_samples": len(values), "decode_valid_count": sum(row["decode_valid"] == "true" for row in values)}
        for field, source in (
            ("encoded_bytes", "encoded_bytes"), ("bpp", "bpp"), ("mse", "mse"),
            ("psnr_db", "psnr_db"), ("ssim", "ssim"),
            ("encode_ms", "encode_ms_median"), ("decode_ms", "decode_ms_median"),
        ):
            numbers = [float(row[source]) for row in values if row[source] is not None]
            summary[f"{field}_mean"] = statistics.fmean(numbers) if numbers else None
            summary[f"{field}_median"] = float(np.median(numbers)) if numbers else None
            summary[f"{field}_p95"] = float(np.percentile(numbers, 95)) if numbers else None
        summary["deterministic_stream_count"] = sum(row["encode_deterministic"] == "true" for row in values)
        summary["encoder_parameters"] = values[0]["encoder_parameters"] if values else "{}"
        output.append(summary)
    return output


def calibrate(args) -> dict:
    cache = args.cache_root.resolve()
    dataset = args.dataset_root.resolve()
    output = args.results_dir.resolve()
    per_sample_path = output / "calibration_per_sample.csv"
    summary_path = output / "calibration_summary.csv"
    metadata_path = output / "calibration_metadata.json"
    decisions_path = output / "calibration_decisions.csv"
    if any(path.exists() for path in (per_sample_path, summary_path, metadata_path, decisions_path)):
        raise FileExistsError(f"calibration outputs already exist under {output}; refusing to overwrite measurements")

    base_rows = read_manifest(cache / "manifest.jsonl")
    selected = select_calibration_rows(base_rows, CALIBRATION_PER_CLASS)
    source_root = get_imagenette_source_root(dataset)
    output.mkdir(parents=True, exist_ok=True)
    measurements = []
    parameters = {}
    for condition in NEW_CONDITIONS:
        codec = CODEC_FACTORIES[condition]()
        warm_rgb = load_prepared_rgb(source_root / selected[0]["image_id"], 256)
        warm_stream = codec.encode(Image.fromarray(warm_rgb)).data
        codec.decode(warm_stream)
        for row in selected:
            rgb = load_prepared_rgb(source_root / row["image_id"], 256)
            if _digest(rgb.tobytes()) != row["rgb_sha256"]:
                raise ValueError(f"prepared image hash mismatch: {row['image_id']}")
            image = Image.fromarray(rgb)
            encoded = []
            encode_ms = []
            for _ in range(ENCODE_REPEATS):
                started = time.perf_counter_ns()
                sample = codec.encode(image)
                encode_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
                encoded.append(sample.data)
                parameters[condition] = sample.metadata
            payload = encoded[0]
            decoded_images = []
            decode_ms = []
            decode_valid = True
            error = ""
            for _ in range(DECODE_REPEATS):
                try:
                    started = time.perf_counter_ns()
                    decoded = codec.decode(payload).convert("RGB")
                    decode_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
                    decoded_array = np.asarray(decoded, dtype=np.uint8)
                    if decoded_array.shape != (256, 256, 3):
                        raise ValueError(f"unexpected decoded shape {decoded_array.shape}")
                    decoded_images.append(decoded.copy())
                except Exception as exc:
                    decode_valid = False
                    error = f"{type(exc).__name__}: {exc}"
                    break
            if decode_valid:
                metrics = image_metrics(image, decoded_images[0])
                decoded_deterministic = len({_digest(np.asarray(value, dtype=np.uint8).tobytes()) for value in decoded_images}) == 1
            else:
                metrics = {"mse": None, "psnr_db": None, "ssim": None}
                decoded_deterministic = False
            measurements.append({
                "condition": condition,
                "image_id": row["image_id"],
                "cache_index": row["cache_index"],
                "label": row["label"],
                "split": row["split"],
                "rgb_sha256": row["rgb_sha256"],
                "stream_sha256": _digest(payload),
                "encoded_bytes": len(payload),
                "bpp": len(payload) * 8.0 / (256 * 256),
                **metrics,
                "decode_valid": str(decode_valid).lower(),
                "encode_deterministic": str(len(set(encoded)) == 1).lower(),
                "decoded_deterministic": str(decoded_deterministic).lower(),
                "encode_ms_median": float(np.median(encode_ms)),
                "encode_ms_runs": json.dumps(encode_ms),
                "decode_ms_median": float(np.median(decode_ms)) if decode_ms else None,
                "decode_ms_runs": json.dumps(decode_ms),
                "encoder_parameters": json.dumps(parameters[condition], sort_keys=True),
                "error": error,
            })
        print(f"calibrated {condition}: {sum(row['condition'] == condition for row in measurements)}/{len(selected)}", flush=True)

    summary = _calibration_summary(measurements)
    by_condition = {row["condition"]: [item for item in measurements if item["condition"] == row["condition"]] for row in summary}
    decisions = []
    training_conditions = set(NEW_CONDITIONS)
    for family, qualities in (("jpeg", (1, 5)), ("webp", (5, 10, 20)), ("avif", (5, 10, 20))):
        for left_quality, right_quality in zip(qualities, qualities[1:]):
            left, right = f"{family}_q{left_quality}", f"{family}_q{right_quality}"
            duplicate = exact_calibration_duplicate(by_condition[left], by_condition[right])
            decisions.append({
                "codec_family": family,
                "condition_a": left,
                "condition_b": right,
                "exact_same_streams_on_all_calibration_images": duplicate,
                "training_decision": f"skip {right}" if duplicate else "train both",
                "rule": REDUNDANCY_RULE,
            })
            if duplicate:
                training_conditions.discard(right)

    repo_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    metadata = {
        "schema": "tamtd.phase48.calibration.v1",
        "dataset": "Imagenette 320px distribution, exact Phase 3 prepared RGB, 256x256",
        "split_counts": {split: sum(row["split"] == split for row in base_rows) for split in ("train", "validation", "test")},
        "calibration_per_class": CALIBRATION_PER_CLASS,
        "calibration_samples": len(selected),
        "calibration_sample_ids": [row["image_id"] for row in selected],
        "calibration_selection": "lowest SHA256(tamtd-phase48-calibration-v1\\0 + image_id) values within each class; rows returned in cache_index order",
        "encode_repeats_per_sample": ENCODE_REPEATS,
        "decode_repeats_per_sample": DECODE_REPEATS,
        "timing_summary": "median and p95 across per-image medians from three repeats; one warm-up image per condition excluded",
        "source_manifest_sha256": _file_digest(cache / "manifest.jsonl"),
        "encoder_parameters_by_condition": parameters,
        "library_versions": _library_versions(),
        "git_head": repo_head,
        "git_working_tree_dirty": bool(git_status),
        "git_status_at_calibration": git_status,
        "redundancy_rule": REDUNDANCY_RULE,
        "conditions_to_train": [condition for condition in NEW_CONDITIONS if condition in training_conditions],
        "conditions_skipped_as_exact_calibration_duplicates": [condition for condition in NEW_CONDITIONS if condition not in training_conditions],
    }
    fields = list(measurements[0])
    _write_csv(per_sample_path, measurements, fields)
    _write_csv(summary_path, summary, list(summary[0]))
    _write_csv(decisions_path, decisions, list(decisions[0]))
    _write_json(metadata_path, metadata)
    invalid = [row for row in measurements if row["decode_valid"] != "true"]
    if invalid:
        raise RuntimeError(f"calibration had {len(invalid)} invalid decodes; see {per_sample_path}")
    return {"n_samples": len(selected), "conditions": len(NEW_CONDITIONS), "summary": str(summary_path), "conditions_to_train": metadata["conditions_to_train"]}


def _cache_signature(condition: str, metadata: dict, cache: Path) -> dict:
    family = condition.split("_", 1)[0]
    adapter_path = PROJECT_ROOT / "src/tamtd/codecs" / f"{family}.py"
    return {
        "schema": "tamtd.phase48.stream-cache.v1",
        "condition": condition,
        "encoder_parameters": metadata["encoder_parameters_by_condition"][condition],
        "library_versions": metadata["library_versions"],
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "dataset": metadata["dataset"],
        "sample_count": metadata["split_counts"]["train"] + metadata["split_counts"]["validation"] + metadata["split_counts"]["test"],
        "builder_sha256": _file_digest(PROJECT_ROOT / "scripts/build_phase4_raw_byte_cache.py"),
        "adapter_sha256": _file_digest(adapter_path),
    }


def build_cache(args) -> dict:
    cache = args.cache_root.resolve()
    dataset = args.dataset_root.resolve()
    results = args.results_dir.resolve()
    output_root = cache / "phase48_streams"
    metadata = json.loads((results / "calibration_metadata.json").read_text(encoding="utf-8"))
    base_rows = read_manifest(cache / "manifest.jsonl")
    python = _python_executable(None)
    build_script = PROJECT_ROOT / "scripts/build_phase4_raw_byte_cache.py"
    build_rows = []
    for condition in NEW_CONDITIONS:
        signature = _cache_signature(condition, metadata, cache)
        signature_path = output_root / f"{condition}.metadata.json"
        stream_path = output_root / f"{condition}.bin"
        manifest_path = output_root / f"{condition}.jsonl"
        if signature_path.exists():
            existing = json.loads(signature_path.read_text(encoding="utf-8"))
            if existing != signature:
                raise ValueError(f"Phase 4.8 cache signature mismatch for {condition}; preserving existing files")
        elif stream_path.exists() or manifest_path.exists():
            raise ValueError(f"cache files exist without a signature for {condition}; refusing to accept an unverified partial cache")
        else:
            _write_json(signature_path, signature)
        command = [
            str(python), str(build_script), "--condition", condition,
            "--cache-root", str(cache), "--dataset-root", str(dataset), "--output-root", str(output_root),
        ]
        record = _run_job(
            f"Phase 4.8 encode {condition}",
            command,
            results / "logs" / f"cache_{condition}.log",
            [stream_path, manifest_path],
        )
        additions = read_manifest(manifest_path)
        verify_stream_alignment(base_rows, additions, stream_path)
        if len(additions) != len(base_rows):
            raise ValueError(f"incomplete Phase 4.8 cache for {condition}: {len(additions)}/{len(base_rows)}")
        record.update({"condition": condition, "sample_count": len(additions), "stream_bytes": stream_path.stat().st_size, "manifest_sha256": _file_digest(manifest_path)})
        build_rows.append(record)
        _write_json(results / "cache_build_state.json", {"completed": [row["condition"] for row in build_rows], "conditions": list(NEW_CONDITIONS), "rows": build_rows})
    return {"completed_conditions": [row["condition"] for row in build_rows], "stream_bytes": sum(row["stream_bytes"] for row in build_rows)}


def train_conditions(args) -> dict:
    cache = args.cache_root.resolve()
    dataset = args.dataset_root.resolve()
    results = args.results_dir.resolve()
    cache_streams = cache / "phase48_streams"
    metadata = json.loads((results / "calibration_metadata.json").read_text(encoding="utf-8"))
    conditions = metadata["conditions_to_train"]
    if not conditions:
        raise RuntimeError("calibration did not select any conditions for training")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Phase 4.8 does not permit CPU fallback")
    python = _python_executable(None)
    trainer = PROJECT_ROOT / "scripts/run_phase4_condition.py"
    settings = PROJECT_ROOT / "results/Neural Accessibility/training_settings.json"
    parent_path = results / "training_resource_parent.json"
    state_path = results / "training_state.json"
    parent_rows = json.loads(parent_path.read_text(encoding="utf-8")) if parent_path.exists() else []
    done = []
    for condition in conditions:
        result_path = results / "per_seed/seed_42" / f"{condition}__codec_file_bytes.json"
        prediction_path = results / "predictions" / f"{condition}_seed42.csv.gz"
        checkpoint_dir = cache / "checkpoints/phase48"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not prediction_path.is_file() or not Path(result.get("checkpoint", "")).is_file() or result.get("test_metrics", {}).get("u_probe_bits") is None:
                raise ValueError(f"partial Phase 4.8 result exists for {condition}; preserving it for inspection")
            done.append(condition)
            continue
        if not (cache_streams / f"{condition}.bin").is_file() or not (cache_streams / f"{condition}.jsonl").is_file():
            raise FileNotFoundError(f"Phase 4.8 encoded stream is missing for {condition}")
        command = [
            str(python), str(trainer), "--condition", condition, "--representation", "codec_file_bytes",
            "--seed", "42", "--device", "cuda", "--cache-root", str(cache),
            "--phase4-cache-root", str(cache_streams), "--dataset-root", str(dataset),
            "--output-dir", str(results), "--checkpoint-dir", str(checkpoint_dir),
            "--training-settings", str(settings), "--byte-batch-mode", "token_budget",
            "--max-padded-tokens", "1048576", "--max-samples-per-batch", "32",
            "--effective-batch-size", "32", "--bucket-multiplier", "50",
            "--collect-probe-metrics", "--test-predictions-path", str(prediction_path), "--require-cuda",
        ]
        record = _run_job(
            f"Phase 4.8 train {condition} seed 42",
            command,
            results / "logs" / f"train_{condition}_seed42.log",
            [result_path, prediction_path],
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not Path(result["checkpoint"]).is_file():
            raise FileNotFoundError(f"checkpoint was not written for {condition}")
        record.update({"condition": condition, "result_path": str(result_path), "checkpoint_path": result["checkpoint"], "test_accuracy": result["test_accuracy"], "test_u_probe_bits": result["test_metrics"]["u_probe_bits"]})
        parent_rows = [row for row in parent_rows if row.get("condition") != condition]
        parent_rows.append(record)
        _write_json(parent_path, {"seed": 42, "rows": parent_rows})
        done.append(condition)
        _write_json(state_path, {"completed": done, "pending": [item for item in conditions if item not in done], "results": [row for row in parent_rows if row.get("condition") in done]})
    return {"completed_conditions": done, "pending_conditions": [item for item in conditions if item not in done]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibrate", "build-cache", "train", "all"), required=True)
    parser.add_argument("--cache-root", type=Path, default=cache_root() / "neural_accessibility/imagenette_320")
    parser.add_argument("--dataset-root", type=Path, default=data_root() / "raw/imagenette")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/Phase 4.8 Extreme Compression")
    args = parser.parse_args()
    if args.stage in ("calibrate", "all"):
        print(json.dumps(calibrate(args), indent=2), flush=True)
    if args.stage in ("build-cache", "all"):
        print(json.dumps(build_cache(args), indent=2), flush=True)
    if args.stage in ("train", "all"):
        print(json.dumps(train_conditions(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
