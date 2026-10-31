from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from tamtd.analysis.phase45b import prediction_metrics
from tamtd.data.imagenette_accessibility import read_manifest
from tamtd.models.accessibility import JPEGDCTNet, parameter_count
from tamtd.runtime import cache_root
from tamtd.training.accessibility import IndexedDictTensorDataset, make_loader, train_classifier
from tamtd.training.seed import seed_everything
from tamtd.training.telemetry import ResourceMonitor, collect_resource_snapshot

from scripts.run_phase48_extreme import _file_digest, _write_json


CONDITION = "jpeg_q1"
SEED = 42


def _guard(device: torch.device, disk_path: Path, starting: bool) -> dict:
    snapshot = collect_resource_snapshot(torch, device, disk_path=disk_path, include_gpu=True)
    system = snapshot.get("system", {})
    disk_free = snapshot.get("disk_free_bytes")
    available = system.get("memory_available_bytes")
    commit_limit = system.get("commit_limit_bytes")
    commit_used = system.get("commit_used_bytes")
    gpu = snapshot.get("gpu", {})
    commit_free = commit_limit - commit_used if commit_limit is not None and commit_used is not None else None
    result = {
        "disk_free_bytes": disk_free,
        "memory_available_bytes": available,
        "commit_free_bytes": commit_free,
        "vram_free_mb": gpu.get("vram_free_mb"),
    }
    if disk_free is None or available is None or commit_free is None or result["vram_free_mb"] is None:
        raise RuntimeError("JPEG DCT training resource telemetry is incomplete")
    floors = (20 * 1024**3, 4 * 1024**3, 5 * 1024**3, 1024) if starting else (12 * 1024**3, 2 * 1024**3, 3 * 1024**3, 512)
    values = (disk_free, available, commit_free, result["vram_free_mb"])
    if any(value < floor for value, floor in zip(values, floors, strict=True)):
        raise RuntimeError(f"JPEG DCT training stopped by resource safety guard: {result}")
    return result


def _write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args) -> dict:
    cache = args.cache_root.resolve()
    results = args.results_dir.resolve()
    feature_dir = cache / "phase48/features"
    checkpoint = cache / "checkpoints/phase48/jpeg_q1__jpeg_dct_ir__seed42.pt"
    prediction_path = results / "predictions/jpeg_q1_dct_ir_seed42.csv.gz"
    diagnostic_path = results / "per_seed/seed_42/jpeg_q1__jpeg_dct_ir.json"
    if any(path.exists() for path in (checkpoint, prediction_path, diagnostic_path)):
        raise FileExistsError("JPEG q1 DCT training artifacts already exist; refusing to overwrite them")
    feature_meta_path = results / "jpeg_q1_dct_feature_cache.json"
    feature_meta = json.loads(feature_meta_path.read_text(encoding="utf-8"))
    if feature_meta.get("complete") is not True or feature_meta.get("condition") != CONDITION:
        raise ValueError("JPEG q1 DCT feature cache is not complete")
    for name, path in feature_meta["feature_paths"].items():
        feature_path = Path(path)
        if not feature_path.is_file() or _file_digest(feature_path) != feature_meta["feature_sha256"][name]:
            raise ValueError(f"JPEG q1 DCT feature cache checksum mismatch: {name}")
    rows = read_manifest(cache / "manifest.jsonl")
    if len(rows) != 13_394 or sum(row["split"] == "test" for row in rows) != 3925:
        raise ValueError("JPEG DCT training requires the exact Phase 3 Imagenette split")
    train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
    validation_indices = [index for index, row in enumerate(rows) if row["split"] == "validation"]
    test_indices = [index for index, row in enumerate(rows) if row["split"] == "test"]
    labels_array = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    labels = torch.from_numpy(labels_array)
    inputs = {
        "y_coefficients": torch.from_numpy(np.load(feature_dir / "jpeg_q1_y.npy", mmap_mode="c", allow_pickle=False)),
        "cb_coefficients": torch.from_numpy(np.load(feature_dir / "jpeg_q1_cb.npy", mmap_mode="c", allow_pickle=False)),
        "cr_coefficients": torch.from_numpy(np.load(feature_dir / "jpeg_q1_cr.npy", mmap_mode="c", allow_pickle=False)),
        "quantization_tables": torch.from_numpy(np.load(feature_dir / "jpeg_q1_quantization.npy", mmap_mode="c", allow_pickle=False)),
    }
    train_set = IndexedDictTensorDataset(inputs, labels, train_indices)
    validation_set = IndexedDictTensorDataset(inputs, labels, validation_indices)
    test_set = IndexedDictTensorDataset(inputs, labels, test_indices)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase 4.8 JPEG DCT probe")
    device = torch.device("cuda")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    seed_everything(SEED)
    start_snapshot = _guard(device, cache, starting=True)
    loaders = (
        make_loader(train_set, 32, True, SEED, workers=0, pin_memory=False),
        make_loader(validation_set, 32, False, SEED, workers=0, pin_memory=False),
        make_loader(test_set, 32, False, SEED, workers=0, pin_memory=False),
    )
    model = JPEGDCTNet()
    model_size = parameter_count(model)
    evaluation_records = {}
    forward_ms = {"validation": [], "test": []}
    train_labels = labels_array[train_indices]
    prior = np.bincount(train_labels, minlength=10) / len(train_labels)

    def evaluate(model_instance, loader, evaluation_device):
        split = "validation" if "validation" not in evaluation_records else "test"
        logits = []
        targets = []
        model_instance.eval()
        with torch.inference_mode():
            for features, batch_labels, _ in loader:
                batch_features = {name: value.to(evaluation_device, non_blocking=False) for name, value in features.items()}
                batch_labels = batch_labels.to(evaluation_device)
                if evaluation_device.type == "cuda":
                    torch.cuda.synchronize(evaluation_device)
                started = time.perf_counter()
                values = model_instance(**batch_features).float()
                if evaluation_device.type == "cuda":
                    torch.cuda.synchronize(evaluation_device)
                forward_ms[split].append((time.perf_counter() - started) * 1000.0)
                logits.append(values.cpu().numpy())
                targets.append(batch_labels.cpu().numpy())
        scores = prediction_metrics(np.concatenate(logits), np.concatenate(targets), train_labels)
        evaluation_records[split] = scores
        return {key: value for key, value in scores.items() if np.isscalar(value)}

    monitor = ResourceMonitor(torch, device=device, disk_path=cache, interval_seconds=1, gpu_interval_seconds=5).start()
    try:
        training = train_classifier(
            model,
            *loaders,
            device,
            checkpoint,
            epochs=20,
            learning_rate=3e-4,
            weight_decay=1e-4,
            optimizer_name="adamw",
            patience=6,
            amp=True,
            evaluation_callback=evaluate,
        )
        finish_snapshot = _guard(device, cache, starting=False)
        resource_diagnostics = monitor.stop()
    finally:
        monitor.stop()
        del loaders
        del train_set, validation_set, test_set, inputs, labels, model
        gc.collect()
        torch.cuda.empty_cache()

    test_metrics = evaluation_records["test"]
    rgb_rows = [row for row in csv.DictReader((results / "decoded_rgb_reference_per_seed.csv").open(encoding="utf-8", newline="")) if row["condition"] == CONDITION and int(row["seed"]) == SEED and row["stage"] == "decoded_rgb"]
    if len(rgb_rows) != 1:
        raise ValueError("missing matched seed-42 decoded RGB reference")
    rgb_u = float(rgb_rows[0]["u_probe_bits"])
    rgb_accuracy = float(rgb_rows[0]["accuracy"])
    training.pop("validation_metrics", None)
    training.pop("test_metrics", None)
    prediction_rows = []
    probabilities = test_metrics["probabilities"]
    predictions = test_metrics["predicted_labels"]
    per_image_ce = test_metrics["per_image_cross_entropy_nats"]
    for position, index in enumerate(test_indices):
        prediction_rows.append({
            "condition": CONDITION,
            "representation": "jpeg_dct_ir",
            "model_family": "jpeg_dct_cnn",
            "seed": SEED,
            "image_id": rows[index]["image_id"],
            "true_label": int(labels_array[index]),
            "predicted_label": int(predictions[position]),
            "correct": int(predictions[position] == labels_array[index]),
            "top1_probability": float(probabilities[position].max()),
            "true_class_probability": float(probabilities[position, labels_array[index]]),
            "cross_entropy_nats": float(per_image_ce[position]),
        })
    _write_predictions(prediction_path, prediction_rows)
    scalar_metrics = {key: value for key, value in test_metrics.items() if np.isscalar(value)}
    result = {
        "schema": "tamtd.phase48.jpeg-dct-accessibility.v1",
        "condition": CONDITION,
        "representation": "jpeg_dct_ir",
        "model_name": "JPEGDCTNet",
        "parameter_count": model_size,
        "seed": SEED,
        "best_epoch": training["best_epoch"],
        "epochs_run": training["epochs_run"],
        "best_validation_accuracy": training["best_validation_accuracy"],
        "test_metrics": scalar_metrics,
        "relative_u_probe_utility_to_matched_rgb": float(test_metrics["u_probe_bits"]) / rgb_u if rgb_u > 0 else None,
        "relative_accuracy_above_chance_to_matched_rgb": (float(test_metrics["accuracy"]) - 0.1) / (rgb_accuracy - 0.1) if rgb_accuracy > 0.1 else None,
        "training_seconds": training["training_wall_seconds"],
        "examples_per_second": training["examples_per_second"],
        "peak_vram_mb": training["peak_vram_mb"],
        "inference_model_forward_ms_per_sample": sum(forward_ms["test"]) / 3925,
        "inference_batch_forward_ms_median": float(np.median(forward_ms["test"])),
        "inference_batch_forward_ms_p95": float(np.percentile(forward_ms["test"], 95)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_digest(checkpoint),
        "predictions": str(prediction_path),
        "predictions_sha256": _file_digest(prediction_path),
        "training_config": {"optimizer": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.0001, "max_epochs": 20, "patience": 6, "batch_size": 32, "workers": 0, "pin_memory": False, "amp": True},
        "device": torch.cuda.get_device_name(device),
        "resource_snapshots": {"start": start_snapshot, "end": finish_snapshot},
        "resource_diagnostics": resource_diagnostics,
        "feature_cache_sha256": feature_meta["feature_sha256"],
        "truncation": 0,
        "history": training["history"],
    }
    _write_json(diagnostic_path, result)
    print(json.dumps({"result": str(diagnostic_path), "test_accuracy": scalar_metrics["accuracy"], "u_probe_bits": scalar_metrics["u_probe_bits"]}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=cache_root() / "neural_accessibility/imagenette_320")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/Phase 4.8 Extreme Compression")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
