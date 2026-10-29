from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.analysis.phase46 import apply_jpeg_dct_ablation
from tamtd.models.accessibility import BytePatchTransformer, JPEGDCTNet, parameter_count
from tamtd.runtime import cache_root, data_root, results_root, temp_root
from tamtd.training.accessibility import ByteStreamFileDataset, IndexedDictTensorDataset, make_loader, run_tiny_overfit, train_classifier
from tamtd.training.seed import seed_everything
from tamtd.training.telemetry import ResourceMonitor
from tamtd.data.imagenette_accessibility import read_manifest


STANDARD_CODECS = ("jpeg_q10", "webp_q30", "avif_q40", "jxl_d6")
DCT_ABLATIONS = {
    "jpeg_dct_dc_only": "dc_only",
    "jpeg_dct_low4x4": "low4x4",
    "jpeg_dct_y_only": "y_only",
}
MODERN_CODECS = {"webp_q30", "avif_q40", "jxl_d6"}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_rows(cache_dir: Path, phase4_cache_dir: Path, condition: str) -> list[dict]:
    rows = read_manifest(cache_dir / "codec_manifest.jsonl")
    source_rows = read_manifest(cache_dir / "manifest.jsonl")
    if [(row["cache_index"], row["image_id"], row["label"], row["split"], row["rgb_sha256"]) for row in rows] != [
        (row["cache_index"], row["image_id"], row["label"], row["split"], row["rgb_sha256"]) for row in source_rows
    ]:
        raise ValueError("codec cache does not align with the shared source manifest")
    if condition not in MODERN_CODECS:
        return rows
    additions = read_manifest(phase4_cache_dir / f"{condition}.jsonl")
    if len(additions) != len(rows):
        raise ValueError(f"incomplete Phase 4 stream manifest for {condition}: {len(additions)}/{len(rows)}")
    merged = []
    for row, addition in zip(rows, additions, strict=True):
        if any(row[key] != addition[key] for key in ("cache_index", "image_id", "label", "split", "rgb_sha256")):
            raise ValueError(f"Phase 4 stream manifest alignment mismatch for {condition} at {row['cache_index']}")
        copy = dict(row)
        copy["streams"] = dict(row["streams"])
        copy["streams"][condition] = addition["stream"]
        merged.append(copy)
    return merged


def _stratified_indices(rows: list[dict], sample_count: int, seed: int) -> list[int]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    rng = np.random.default_rng(seed)
    per_class = sample_count // 10
    extras = sample_count % 10
    selected = []
    for class_id in range(10):
        candidates = np.flatnonzero(labels == class_id)
        count = per_class + int(class_id < extras)
        if len(candidates) < count:
            raise ValueError(f"not enough class-{class_id} training rows for the overfit gate")
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    return sorted(selected)


def _evaluation_metrics(train_labels: np.ndarray):
    counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=10).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("the training split must contain every Imagenette class")
    prior = torch.tensor(counts / counts.sum(), dtype=torch.float64)

    def evaluate(model, loader, device: torch.device) -> dict[str, float | int]:
        model.eval()
        device_prior = prior.to(device)
        correct = 0
        count = 0
        loss_sum = prior_loss_sum = brier_sum = 0.0
        with torch.inference_mode():
            for inputs, labels, _ in loader:
                inputs = {name: value.to(device, non_blocking=True) for name, value in inputs.items()}
                labels = labels.to(device, non_blocking=True)
                logits = model(**inputs).float()
                log_probabilities = F.log_softmax(logits, dim=1)
                probabilities = log_probabilities.exp()
                row_indices = torch.arange(len(labels), device=device)
                loss_sum += float(-log_probabilities[row_indices, labels].sum().item())
                prior_loss_sum += float(-device_prior[labels].log().sum().item())
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                one_hot = F.one_hot(labels, num_classes=10).float()
                brier_sum += float((probabilities - one_hot).square().sum(dim=1).sum().item())
                count += len(labels)
        cross_entropy = loss_sum / count
        prior_cross_entropy = prior_loss_sum / count
        return {
            "n": count,
            "accuracy": correct / count,
            "cross_entropy_nats": cross_entropy,
            "class_prior_cross_entropy_nats": prior_cross_entropy,
            "u_probe_bits": (prior_cross_entropy - cross_entropy) / math.log(2.0),
            "brier_score": brier_sum / count,
        }

    return evaluate


def _dct_inputs(cache_dir: Path, condition: str) -> dict[str, torch.Tensor]:
    feature_dir = cache_dir / "features"
    names = {
        "y_coefficients": f"{condition}_y.npy",
        "cb_coefficients": f"{condition}_cb.npy",
        "cr_coefficients": f"{condition}_cr.npy",
        "quantization_tables": f"{condition}_quantization.npy",
    }
    return {
        key: torch.from_numpy(np.load(feature_dir / filename, mmap_mode="r", allow_pickle=False))
        for key, filename in names.items()
    }


def _dct_loader(dataset, batch_size: int, shuffle: bool, seed: int, ablation: str) -> DataLoader:
    def collate(samples):
        inputs, labels = zip(*samples, strict=True)
        batch = default_collate(inputs)
        return apply_jpeg_dct_ablation(batch, ablation), default_collate(labels), None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )


def _checkpoint_path(args, condition: str, representation: str, seed: int) -> Path:
    base = args.checkpoint_dir / f"{condition}__{representation}__seed{seed}.pt"
    if not base.exists():
        return base
    retry = 1
    while True:
        candidate = base.with_name(f"{base.stem}__retry{retry}{base.suffix}")
        if not candidate.exists():
            return candidate
        retry += 1


def _train_byte_probe(args, rows: list[dict], device: torch.device, overfit_gate: bool) -> dict:
    condition = "jpeg_q10" if overfit_gate else args.condition
    if condition not in STANDARD_CODECS:
        raise ValueError(f"unsupported Phase 4.6 byte condition: {condition}")
    stream_path = args.phase4_cache_dir / f"{condition}.bin" if condition in MODERN_CODECS else args.cache_dir / "streams" / f"{condition}.bin"
    dataset = ByteStreamFileDataset(stream_path, rows, condition)
    full_train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
    if overfit_gate:
        selected_indices = _stratified_indices([rows[index] for index in full_train_indices], 128, args.seed)
        selected_global_indices = [full_train_indices[index] for index in selected_indices]
        dataset_for_training = torch.utils.data.Subset(
            torch.utils.data.Subset(dataset, full_train_indices), selected_indices
        )
    else:
        selected_global_indices = full_train_indices
        dataset_for_training = torch.utils.data.Subset(dataset, selected_global_indices)
    byte_batching = {
        "max_padded_tokens": args.max_padded_tokens,
        "max_samples_per_batch": args.max_samples_per_batch,
        "effective_batch_size": args.effective_batch_size,
        "bucket_multiplier": args.bucket_multiplier,
    }
    train_loader = make_loader(
        dataset_for_training,
        args.batch_size,
        True,
        args.seed,
        workers=0,
        pin_memory=False,
        byte_batching={
            "max_padded_tokens": byte_batching["max_padded_tokens"],
            "max_samples_per_batch": byte_batching["max_samples_per_batch"],
            "bucket_multiplier": byte_batching["bucket_multiplier"],
        },
    )
    model = BytePatchTransformer()
    if overfit_gate:
        result = run_tiny_overfit(
            model,
            train_loader,
            device,
            max_epochs=args.overfit_max_epochs,
            target_accuracy=args.overfit_target_accuracy,
            learning_rate=args.learning_rate,
            amp=True,
        )
        data_lengths = [dataset.stream_lengths[index] for index in selected_global_indices]
        return {
            "condition": condition,
            "representation": "byte_patch_transformer_overfit",
            "seed": args.seed,
            "status": "passed" if result["passed"] else "failed",
            "passed": result["passed"],
            "target_train_accuracy": args.overfit_target_accuracy,
            "stratified_sample_count": len(selected_global_indices),
            "stratified_class_counts": {str(class_id): int(sum(rows[index]["label"] == class_id for index in selected_global_indices)) for class_id in range(10)},
            "parameter_count": parameter_count(model),
            "device": device.type,
            "training_wall_seconds": result["wall_seconds"],
            "epochs_run": result["epochs_run"],
            "best_train_accuracy": result["best_train_accuracy"],
            "peak_vram_mb": result["peak_vram_mb"],
            "true_input_bytes_total": int(sum(data_lengths)),
            "input_byte_length_mean": float(np.mean(data_lengths)),
            "input_byte_length_max": int(max(data_lengths)),
            "batching": byte_batching,
        }
    train_indices = full_train_indices
    val_indices = [index for index, row in enumerate(rows) if row["split"] == "validation"]
    test_indices = [index for index, row in enumerate(rows) if row["split"] == "test"]
    validation_loader = make_loader(
        torch.utils.data.Subset(dataset, val_indices), args.batch_size, False, args.seed,
        workers=0, pin_memory=False, byte_batching={
            "max_padded_tokens": byte_batching["max_padded_tokens"],
            "max_samples_per_batch": byte_batching["max_samples_per_batch"],
            "bucket_multiplier": byte_batching["bucket_multiplier"],
        },
    )
    test_loader = make_loader(
        torch.utils.data.Subset(dataset, test_indices), args.batch_size, False, args.seed,
        workers=0, pin_memory=False, byte_batching={
            "max_padded_tokens": byte_batching["max_padded_tokens"],
            "max_samples_per_batch": byte_batching["max_samples_per_batch"],
            "bucket_multiplier": byte_batching["bucket_multiplier"],
        },
    )
    train_labels = np.asarray([rows[index]["label"] for index in train_indices], dtype=np.int64)
    checkpoint = _checkpoint_path(args, condition, args.representation, args.seed)
    result = train_classifier(
        model,
        train_loader,
        validation_loader,
        test_loader,
        device,
        checkpoint,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_name="adamw",
        patience=args.patience,
        amp=True,
        gradient_accumulation_samples=args.effective_batch_size,
        evaluation_callback=_evaluation_metrics(train_labels),
    )
    truncation = result["truncation"]
    if truncation["truncated_samples"] or truncation["truncated_bytes"]:
        raise ValueError(f"byte streams were truncated: {truncation}")
    train_byte_lengths = [dataset.stream_lengths[index] for index in selected_global_indices]
    output = {
        "condition": condition,
        "representation": args.representation,
        "model_family": "byte_patch_transformer",
        "model_name": "BytePatchTransformer",
        "parameter_count": parameter_count(model),
        "seed": args.seed,
        "device": device.type,
        "workers": 0,
        "pin_memory": False,
        "training_settings": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs_max": args.epochs,
            "patience": args.patience,
            "scheduler": "CosineAnnealingLR",
            "mixed_precision": "float16 autocast on CUDA",
            "augmentation": "none",
        },
        "batching": {
            **byte_batching,
            "mode": "length_bucket_token_budget",
            "gradient_accumulation_target_samples": args.effective_batch_size,
            "true_input_bytes_per_train_sample": {
                "count": len(train_byte_lengths),
                "total": int(sum(train_byte_lengths)),
                "mean": float(np.mean(train_byte_lengths)),
                "median": float(np.median(train_byte_lengths)),
                "max": int(max(train_byte_lengths)),
            },
        },
        **result,
    }
    return output


def _train_dct_ablation(args, rows: list[dict], device: torch.device) -> dict:
    if args.condition not in ("jpeg_q10", "jpeg_q60") or args.representation not in DCT_ABLATIONS:
        raise ValueError(f"unsupported JPEG DCT ablation: {args.condition}/{args.representation}")
    ablation = DCT_ABLATIONS[args.representation]
    inputs = _dct_inputs(args.cache_dir, args.condition)
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    split_indices = {
        split: [int(row["cache_index"]) for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    if any(len(values) != expected for values, expected in zip(split_indices.values(), (8522, 947, 3925), strict=True)):
        raise ValueError("Phase 4.6 requires the shared Phase 3/4 split counts")
    if any(len(value) != len(rows) for value in inputs.values()):
        raise ValueError("DCT cache arrays do not align with the source manifest")
    datasets = {
        split: IndexedDictTensorDataset(inputs, labels, indices)
        for split, indices in split_indices.items()
    }
    settings = args.dct_training_settings
    train_loader = _dct_loader(datasets["train"], settings["batch_size"], True, args.seed, ablation)
    validation_loader = _dct_loader(datasets["validation"], settings["batch_size"], False, args.seed, ablation)
    test_loader = _dct_loader(datasets["test"], settings["batch_size"], False, args.seed, ablation)
    train_labels = np.asarray([rows[index]["label"] for index in split_indices["train"]], dtype=np.int64)
    model = JPEGDCTNet()
    checkpoint = _checkpoint_path(args, args.condition, args.representation, args.seed)
    result = train_classifier(
        model,
        train_loader,
        validation_loader,
        test_loader,
        device,
        checkpoint,
        epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        optimizer_name=settings["optimizer"],
        patience=settings["patience"],
        amp=True,
        evaluation_callback=_evaluation_metrics(train_labels),
    )
    return {
        "condition": args.condition,
        "representation": args.representation,
        "ablation": ablation,
        "mask_convention": {
            "dc_only": "keep coefficient channel u*8+v=0; zero channels 1..63 in Y, Cb, and Cr",
            "low4x4": "keep channels u*8+v for 0<=u,v<4; zero all other frequencies in Y, Cb, and Cr",
            "y_only": "keep all Y coefficients; zero all Cb/Cr coefficients; leave quantization tables unchanged",
        }[ablation],
        "model_family": "jpeg_dct_cnn",
        "model_name": "JPEGDCTNet",
        "parameter_count": parameter_count(model),
        "seed": args.seed,
        "device": device.type,
        "workers": 0,
        "pin_memory": False,
        "training_settings": settings,
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one analysis-only Phase 4.6 CUDA condition in a fresh child process.")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=cache_root() / "neural_accessibility" / "imagenette_320")
    parser.add_argument("--phase4-cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=results_root() / "Phase 4.6 Mechanistic Controls")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--training-settings", type=Path, default=results_root() / "Neural Accessibility" / "training_settings.json")
    parser.add_argument("--temp-root", type=Path, default=temp_root())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-padded-tokens", type=int, default=1_048_576)
    parser.add_argument("--max-samples-per-batch", type=int, default=32)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--bucket-multiplier", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--overfit-max-epochs", type=int, default=60)
    parser.add_argument("--overfit-target-accuracy", type=float, default=0.90)
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 4.6 training requires CUDA; CPU fallback is disabled")
    if args.max_padded_tokens < 1 or args.max_samples_per_batch < 1 or args.effective_batch_size < 1:
        parser.error("byte batching limits must be positive")
    args.cache_dir = args.cache_dir.resolve()
    args.phase4_cache_dir = (args.phase4_cache_dir or args.cache_dir / "phase4_streams").resolve()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint_dir = (args.checkpoint_dir or args.cache_dir / "checkpoints" / "phase46").resolve()
    args.training_settings = args.training_settings.resolve()
    args.temp_root = args.temp_root.resolve()
    args.temp_root.mkdir(parents=True, exist_ok=True)
    for variable in ("TEMP", "TMP", "TMPDIR"):
        os.environ[variable] = str(args.temp_root)
    settings = json.loads(args.training_settings.read_text(encoding="utf-8"))
    args.dct_training_settings = dict(settings["jpeg_dct"])
    if args.representation.startswith("byte_patch_transformer"):
        byte_settings = settings.get("phase46_byte_patch_transformer", {})
        args.epochs = int(byte_settings.get("epochs", args.epochs))
        args.learning_rate = float(byte_settings.get("learning_rate", args.learning_rate))
        args.weight_decay = float(byte_settings.get("weight_decay", args.weight_decay))
        args.patience = int(byte_settings.get("patience", args.patience))
    torch.set_num_threads(min(8, torch.get_num_threads()))
    seed_everything(args.seed)
    if args.representation == "byte_patch_transformer_overfit":
        actual_condition = "jpeg_q10"
    else:
        actual_condition = args.condition
    rows = _load_rows(args.cache_dir, args.phase4_cache_dir, actual_condition)
    result_path = args.output_dir / "per_seed" / f"seed_{args.seed}" / f"{args.condition}__{args.representation}.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite Phase 4.6 result: {result_path}")
    if args.representation.startswith("byte_patch_transformer") and args.representation != "byte_patch_transformer_overfit":
        if args.condition not in STANDARD_CODECS:
            raise ValueError(f"unsupported Phase 4.6 standard codec: {args.condition}")
    resource_monitor = ResourceMonitor(torch, device=torch.device("cuda"), disk_path=args.cache_dir)
    resource_monitor.start()
    start = time.perf_counter()
    try:
        if args.representation.startswith("byte_patch_transformer"):
            result = _train_byte_probe(args, rows, torch.device("cuda"), args.representation == "byte_patch_transformer_overfit")
        else:
            result = _train_dct_ablation(args, rows, torch.device("cuda"))
    finally:
        resource_diagnostics = resource_monitor.stop()
        gc.collect()
        torch.cuda.empty_cache()
    result["resource_diagnostics"] = resource_diagnostics
    result["child_wall_seconds_including_setup"] = time.perf_counter() - start
    _write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "status": result.get("status", "completed"), "test_accuracy": result.get("test_accuracy"), "passed": result.get("passed")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
