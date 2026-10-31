from __future__ import annotations

import csv
import gc
import gzip
import hashlib
import json
import mmap
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from tamtd.analysis.phase45b import prediction_metrics
from tamtd.codec_benchmark import image_metrics
from tamtd.data.imagenette_accessibility import get_imagenette_source_root, load_prepared_rgb, read_manifest
from tamtd.models.accessibility import RGBResNet18
from tamtd.runtime import cache_root, data_root
from tamtd.training.telemetry import ResourceMonitor

from scripts.build_phase4_raw_byte_cache import CODEC_FACTORIES
from scripts.run_phase48_extreme import NEW_CONDITIONS, _digest, _file_digest, _write_csv, _write_json


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_gz(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _model_paths(cache: Path) -> dict[int, Path]:
    return {
        42: cache / "checkpoints/original_rgb__decoded_rgb_reference.pt",
        123: cache / "checkpoints/phase4/original_rgb__decoded_rgb_reference__seed123.pt",
        777: cache / "checkpoints/phase4/original_rgb__decoded_rgb_reference__seed777.pt",
    }


def _load_models(cache: Path, device: torch.device) -> dict[int, torch.nn.Module]:
    models = {}
    for seed, path in _model_paths(cache).items():
        if not path.is_file():
            raise FileNotFoundError(f"missing existing RGB reference checkpoint for seed {seed}: {path}")
        model = RGBResNet18()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models[seed] = model.to(device).eval()
    return models


def evaluate_condition(condition: str, cache: Path, dataset: Path, results: Path, device: torch.device) -> dict:
    stream_root = cache / "phase48_streams"
    stream_path = stream_root / f"{condition}.bin"
    stream_rows = read_manifest(stream_root / f"{condition}.jsonl")
    base_rows = read_manifest(cache / "manifest.jsonl")
    if len(stream_rows) != len(base_rows):
        raise ValueError(f"stream count differs for {condition}: {len(stream_rows)}/{len(base_rows)}")
    for index, (base, stream) in enumerate(zip(base_rows, stream_rows, strict=True)):
        if any(base.get(key) != stream.get(key) for key in ("cache_index", "image_id", "label", "split", "rgb_sha256")):
            raise ValueError(f"stream alignment differs for {condition} at cache index {index}")
    if not stream_path.is_file():
        raise FileNotFoundError(stream_path)

    output_dir = results / "decoded_rgb" / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "test_predictions.csv.gz"
    rd_path = output_dir / "test_rate_distortion.csv.gz"
    if metrics_path.exists():
        existing = _read_json(metrics_path)
        for path in (predictions_path, rd_path):
            if not path.is_file():
                raise ValueError(f"partial decoded-RGB output exists for {condition}; preserving it")
        if existing.get("complete") is not True or existing.get("stream_manifest_sha256") != _file_digest(stream_root / f"{condition}.jsonl"):
            raise ValueError(f"decoded-RGB output does not match current stream manifest: {condition}")
        if existing.get("predictions_sha256") != _file_digest(predictions_path) or existing.get("rate_distortion_sha256") != _file_digest(rd_path):
            raise ValueError(f"decoded-RGB output checksum differs for {condition}")
        return existing
    if any(path.exists() for path in (predictions_path, rd_path)):
        raise ValueError(f"partial decoded-RGB artifacts exist for {condition}; preserving them")

    source_root = get_imagenette_source_root(dataset)
    train_labels = np.asarray([int(row["label"]) for row in base_rows if row["split"] == "train"], dtype=np.int64)
    test_rows = [row for row in base_rows if row["split"] == "test"]
    model_outputs = {seed: [] for seed in (42, 123, 777)}
    model_forward_ms = {seed: [] for seed in (42, 123, 777)}
    labels_all = []
    prediction_ids = []
    rd_rows = []
    codec = CODEC_FACTORIES[condition]()
    resource_monitor = ResourceMonitor(torch_module=torch, device=device, disk_path=cache, interval_seconds=1, gpu_interval_seconds=5).start()
    started = time.perf_counter()
    try:
        with stream_path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            models = _load_models(cache, device)
            batch_images = []
            batch_labels = []
            batch_ids = []

            def flush() -> None:
                if not batch_images:
                    return
                images = torch.from_numpy(np.stack(batch_images)).permute(0, 3, 1, 2).to(device)
                label_tensor = np.asarray(batch_labels, dtype=np.int64)
                labels_all.extend(batch_labels)
                prediction_ids.extend(batch_ids)
                for seed, model in models.items():
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    forward_started = time.perf_counter()
                    with torch.inference_mode():
                        logits = model(images=images).float()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    model_forward_ms[seed].append((time.perf_counter() - forward_started) * 1000.0)
                    model_outputs[seed].append(logits.cpu().numpy())
                batch_images.clear()
                batch_labels.clear()
                batch_ids.clear()

            for row in test_rows:
                reference = stream_rows[int(row["cache_index"])]["stream"]
                payload = mapped[int(reference["offset"]):int(reference["offset"]) + int(reference["length"])]
                if len(payload) != int(reference["length"]) or _digest(payload) != reference["sha256"]:
                    raise ValueError(f"corrupt encoded stream for {condition}/{row['image_id']}")
                decoded = codec.decode(payload).convert("RGB")
                pixels = np.asarray(decoded, dtype=np.uint8).copy()
                if pixels.shape != (256, 256, 3):
                    raise ValueError(f"decoded dimensions differ for {condition}/{row['image_id']}: {pixels.shape}")
                source = load_prepared_rgb(source_root / row["image_id"], 256)
                if _digest(source.tobytes()) != row["rgb_sha256"]:
                    raise ValueError(f"prepared RGB hash mismatch for {row['image_id']}")
                score = image_metrics(Image.fromarray(source), decoded)
                rd_rows.append({
                    "condition": condition, "cache_index": row["cache_index"], "image_id": row["image_id"],
                    "label": int(row["label"]), "encoded_bytes": int(reference["length"]),
                    "bpp": int(reference["length"]) * 8.0 / (256 * 256),
                    "mse": score["mse"], "psnr_db": score["psnr_db"], "ssim": score["ssim"],
                    "decode_valid": True, "rgb_sha256": row["rgb_sha256"], "stream_sha256": reference["sha256"],
                })
                batch_images.append(pixels)
                batch_labels.append(int(row["label"]))
                batch_ids.append(row["image_id"])
                if len(batch_images) == 16:
                    flush()
            flush()
            del models
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        resource_diagnostics = resource_monitor.stop()
    elapsed = time.perf_counter() - started
    labels = np.asarray(labels_all, dtype=np.int64)
    if len(labels) != len(test_rows) or prediction_ids != [row["image_id"] for row in test_rows]:
        raise ValueError(f"decoded RGB test order/count mismatch for {condition}")

    prediction_rows = []
    summary_rows = []
    checkpoint_paths = _model_paths(cache)
    for seed in (42, 123, 777):
        logits = np.concatenate(model_outputs[seed])
        metrics = prediction_metrics(logits, labels, train_labels)
        records = []
        for index, image_id in enumerate(prediction_ids):
            probabilities = metrics["probabilities"][index]
            record = {
                "condition": condition, "seed": seed, "image_id": image_id,
                "true_label": int(labels[index]), "predicted_label": int(metrics["predicted_labels"][index]),
                "correct": int(metrics["predicted_labels"][index] == labels[index]),
                "top1_probability": float(probabilities.max()),
                "true_class_probability": float(probabilities[labels[index]]),
                "cross_entropy_nats": float(metrics["per_image_cross_entropy_nats"][index]),
            }
            records.append(record)
        prediction_rows.extend(records)
        summary_rows.append({
            "condition": condition, "stage": "decoded_rgb", "representation": "decoded_rgb_reference",
            "codec_family": condition.split("_", 1)[0], "seed": seed, "n_samples": len(labels),
            "accuracy": float(metrics["accuracy"]), "cross_entropy_nats": float(metrics["cross_entropy_nats"]),
            "class_prior_cross_entropy_nats": float(metrics["class_prior_cross_entropy_nats"]),
            "u_probe_bits": float(metrics["u_probe_bits"]), "brier_score": float(metrics["brier_score"]),
            "joint_decode_and_three_model_eval_seconds": elapsed,
            "inference_model_forward_ms_per_sample": sum(model_forward_ms[seed]) / len(labels),
            "inference_batch_forward_ms_median": float(np.median(model_forward_ms[seed])),
            "inference_batch_forward_ms_p95": float(np.percentile(model_forward_ms[seed], 95)),
            "checkpoint": str(checkpoint_paths[seed]), "checkpoint_sha256": _file_digest(checkpoint_paths[seed]),
        })

    _csv_gz(predictions_path, prediction_rows, list(prediction_rows[0]))
    _csv_gz(rd_path, rd_rows, list(rd_rows[0]))
    result = {
        "schema": "tamtd.phase48.decoded-rgb-evaluation.v1", "condition": condition,
        "n_test_samples": len(test_rows), "complete": True,
        "stream_manifest_sha256": _file_digest(stream_root / f"{condition}.jsonl"),
        "stream_sha256": _file_digest(stream_path), "decoded_rgb_per_seed": summary_rows,
        "resource_diagnostics": resource_diagnostics,
        "prediction_rows": len(prediction_rows), "rate_distortion_rows": len(rd_rows),
        "predictions_sha256": _file_digest(predictions_path), "rate_distortion_sha256": _file_digest(rd_path),
    }
    _write_json(metrics_path, result)
    return result


def main() -> None:
    import argparse
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=cache_root() / "neural_accessibility/imagenette_320")
    parser.add_argument("--dataset-root", type=Path, default=data_root() / "raw/imagenette")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/Phase 4.8 Extreme Compression")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen RGB evaluation; CPU fallback is disabled")
    if shutil.disk_usage("C:/").free < 20_000_000_000:
        raise RuntimeError("fewer than 20 GB free on C:; refusing RGB-reference evaluation")
    device = torch.device("cuda")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    results = args.results_dir.resolve()
    cache = args.cache_root.resolve()
    dataset = args.dataset_root.resolve()
    summaries = []
    for condition in NEW_CONDITIONS:
        if shutil.disk_usage("C:/").free < 12_000_000_000:
            raise RuntimeError("C: fell below 12 GB; stopping RGB-reference evaluation")
        summaries.append(evaluate_condition(condition, cache, dataset, results, device))
        print(f"decoded RGB evaluated {condition}", flush=True)
    summary_rows = [row for summary in summaries for row in summary["decoded_rgb_per_seed"]]
    _write_csv(results / "decoded_rgb_reference_per_seed.csv", summary_rows, list(summary_rows[0]))
    prediction_files = [results / "decoded_rgb" / condition / "test_predictions.csv.gz" for condition in NEW_CONDITIONS]
    prediction_rows = []
    for path in prediction_files:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            prediction_rows.extend(csv.DictReader(handle))
    _csv_gz(results / "decoded_rgb_test_predictions.csv.gz", prediction_rows, list(prediction_rows[0]))
    rd_rows = []
    for condition in NEW_CONDITIONS:
        path = results / "decoded_rgb" / condition / "test_rate_distortion.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            rd_rows.extend(csv.DictReader(handle))
    _csv_gz(results / "new_test_rd_per_sample.csv.gz", rd_rows, list(rd_rows[0]))
    print(json.dumps({"conditions": len(summaries), "decoded_rgb_rows": len(summary_rows), "test_prediction_rows": len(prediction_rows), "test_rate_distortion_rows": len(rd_rows)}, indent=2))


if __name__ == "__main__":
    main()
