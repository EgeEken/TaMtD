from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codec_stage_ir import adapt_common_tokens
from tamtd.models.block_stage import BlockStageTransformer
from tamtd.training.seed import seed_everything
from tamtd.training.telemetry import ResourceMonitor, collect_resource_snapshot


MAX_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 6
MIN_START_DISK_BYTES = 20 * 1024**3
MIN_RUNNING_DISK_BYTES = 12 * 1024**3
MIN_START_RAM_BYTES = 4 * 1024**3
MIN_RUNNING_RAM_BYTES = 2 * 1024**3
MIN_START_COMMIT_BYTES = 5 * 1024**3
MIN_RUNNING_COMMIT_BYTES = 3 * 1024**3
MIN_START_VRAM_MB = 1024
MIN_RUNNING_VRAM_MB = 512


class StageFeatureDataset(Dataset):
    def __init__(self, feature_path: Path, manifest_path: Path, split: str):
        self.features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        with manifest_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.indices = np.asarray([index for index, row in enumerate(rows) if row["split"] == split], dtype=np.int64)
        self.labels = np.asarray([int(rows[index]["label"]) for index in self.indices], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        values, modes, mask = adapt_common_tokens(self.features[index])
        return (
            torch.from_numpy(values),
            torch.from_numpy(modes.astype(np.int64, copy=False)),
            torch.from_numpy(mask),
            torch.tensor(self.labels[item], dtype=torch.long),
        )


def _snapshot(torch_module: torch, device: torch.device, disk_path: Path) -> dict:
    result = collect_resource_snapshot(torch_module, device, disk_path=disk_path)
    system = result.get("system", {})
    if result.get("disk_free_bytes") is None or system.get("memory_available_bytes") is None:
        raise RuntimeError("disk or host-memory telemetry is unavailable; stopping before training")
    commit_free = None
    if system.get("commit_limit_bytes") is not None and system.get("commit_used_bytes") is not None:
        commit_free = system["commit_limit_bytes"] - system["commit_used_bytes"]
    gpu = result.get("gpu", {})
    if gpu.get("vram_free_mb") is None:
        raise RuntimeError("GPU VRAM telemetry is unavailable; stopping before training")
    return {
        "disk_free_bytes": int(result["disk_free_bytes"]),
        "memory_available_bytes": int(system["memory_available_bytes"]),
        "commit_free_bytes": int(commit_free) if commit_free is not None else None,
        "vram_free_mb": int(gpu["vram_free_mb"]),
    }


def _guard(snapshot: dict, starting: bool) -> None:
    disk_floor = MIN_START_DISK_BYTES if starting else MIN_RUNNING_DISK_BYTES
    ram_floor = MIN_START_RAM_BYTES if starting else MIN_RUNNING_RAM_BYTES
    commit_floor = MIN_START_COMMIT_BYTES if starting else MIN_RUNNING_COMMIT_BYTES
    vram_floor = MIN_START_VRAM_MB if starting else MIN_RUNNING_VRAM_MB
    if snapshot["disk_free_bytes"] < disk_floor:
        raise RuntimeError(f"C: free space is below the {disk_floor / 1024**3:.1f} GiB safety guard")
    if snapshot["memory_available_bytes"] < ram_floor:
        raise RuntimeError(f"available host RAM is below the {ram_floor / 1024**3:.1f} GiB safety guard")
    if snapshot["commit_free_bytes"] is None or snapshot["commit_free_bytes"] < commit_floor:
        raise RuntimeError(f"system commit headroom is below the {commit_floor / 1024**3:.1f} GiB safety guard")
    if snapshot["vram_free_mb"] < vram_floor:
        raise RuntimeError(f"free GPU memory is below the {vram_floor} MiB safety guard")


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def _evaluate(model: BlockStageTransformer, loader: DataLoader, device: torch.device, prior: torch.Tensor | None = None) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.inference_mode():
        for values, modes, mask, labels in loader:
            values = values.to(device)
            modes = modes.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(values, modes, mask)
            total_loss += float(F.cross_entropy(logits.float(), labels, reduction="sum").item())
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += len(labels)
    if total == 0:
        raise ValueError("cannot evaluate an empty split")
    cross_entropy = total_loss / total
    accuracy = total_correct / total
    result = {"n_samples": total, "accuracy": accuracy, "cross_entropy_nats": cross_entropy}
    if prior is not None:
        result["class_prior_cross_entropy_nats"] = float(-np.log(prior.numpy()[loader.dataset.labels]).mean())
        result["u_probe_bits"] = float((result["class_prior_cross_entropy_nats"] - cross_entropy) / math.log(2.0))
    return result


def _class_prior(train_labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=10).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("training split must contain every Imagenette class")
    return counts / counts.sum()


def _best_checkpoint_path(directory: Path, condition: str, seed: int, gate: bool) -> Path:
    stem = "phase47_overfit_gate" if gate else f"{condition}__block_stage_transformer__seed{seed}"
    path = directory / f"{stem}.pt"
    retry = 1
    while path.exists():
        path = directory / f"{stem}__retry{retry}.pt"
        retry += 1
    return path


def _train(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 4.7 structured training; CPU fallback is disabled")
    device = torch.device("cuda")
    disk_path = Path(args.disk_path)
    initial_snapshot = _snapshot(torch, device, disk_path)
    _guard(initial_snapshot, starting=True)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    cache_root = args.cache_root.resolve()
    feature_path = cache_root / "phase47" / "features" / f"{args.condition}.npy"
    completion_path = cache_root / "phase47" / "features" / f"{args.condition}.complete.json"
    if not feature_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError(f"stage features are not complete for {args.condition}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("completed") is not True or completion.get("shape") != [13394, 256, 62]:
        raise ValueError(f"stage feature completion marker is invalid for {args.condition}")

    train_set = StageFeatureDataset(feature_path, cache_root / "manifest.jsonl", "train")
    validation_set = StageFeatureDataset(feature_path, cache_root / "manifest.jsonl", "validation")
    test_set = StageFeatureDataset(feature_path, cache_root / "manifest.jsonl", "test")
    train_loader = _loader(train_set, BATCH_SIZE, True, args.seed)
    validation_loader = _loader(validation_set, BATCH_SIZE, False, args.seed)
    test_loader = _loader(test_set, BATCH_SIZE, False, args.seed)
    prior = _class_prior(train_set.labels)
    model = BlockStageTransformer().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    monitor = ResourceMonitor(torch_module=torch, device=device, disk_path=disk_path, interval_seconds=1, gpu_interval_seconds=5).start()
    history = []
    best_accuracy = float("-inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    overfit_success = False
    start = time.perf_counter()
    try:
        if args.overfit_gate:
            rng = np.random.default_rng(args.seed)
            picked = []
            for label in range(10):
                candidates = np.flatnonzero(train_set.labels == label)
                picked.extend(rng.choice(candidates, size=12, replace=False).tolist())
            gate_set = Subset(train_set, sorted(picked))
            gate_loader = _loader(gate_set, 16, True, args.seed)
            max_gate_epochs = 120
            for epoch in range(1, max_gate_epochs + 1):
                model.train()
                total_loss = total_correct = total = 0
                for step, (values, modes, mask, labels) in enumerate(gate_loader, 1):
                    if step % 8 == 0:
                        _guard(_snapshot(torch, device, disk_path), starting=False)
                    values, modes, mask, labels = values.to(device), modes.to(device), mask.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(values, modes, mask)
                        loss = F.cross_entropy(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += float(loss.detach().float().item()) * len(labels)
                    total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
                    total += len(labels)
                gate_metrics = _evaluate(model, gate_loader, device)
                gate_accuracy = gate_metrics["accuracy"]
                history.append({"epoch": epoch, "train_loss": gate_metrics["cross_entropy_nats"], "train_accuracy": gate_accuracy})
                print(f"overfit gate epoch {epoch}: loss={gate_metrics['cross_entropy_nats']:.4f} accuracy={gate_accuracy:.3f}", flush=True)
                if gate_accuracy >= 0.95:
                    overfit_success = True
                    break
            if not overfit_success:
                raise RuntimeError("BlockStageTransformer did not reach 95% accuracy on the tiny overfit gate")
        else:
            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                total_loss = total_correct = total = 0
                for step, (values, modes, mask, labels) in enumerate(train_loader, 1):
                    if step % 20 == 0:
                        _guard(_snapshot(torch, device, disk_path), starting=False)
                    values, modes, mask, labels = values.to(device), modes.to(device), mask.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(values, modes, mask)
                        loss = F.cross_entropy(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += float(loss.detach().float().item()) * len(labels)
                    total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
                    total += len(labels)
                train_accuracy = total_correct / total
                train_loss = total_loss / total
                validation = _evaluate(model, validation_loader, device)
                history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation["cross_entropy_nats"],
                    "validation_accuracy": validation["accuracy"],
                })
                print(
                    f"{args.condition} seed {args.seed} epoch {epoch}: "
                    f"train={train_accuracy:.4f} validation={validation['accuracy']:.4f} val_ce={validation['cross_entropy_nats']:.4f}",
                    flush=True,
                )
                if validation["accuracy"] > best_accuracy:
                    best_accuracy = validation["accuracy"]
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                scheduler.step()
                if stale_epochs >= PATIENCE:
                    break
            if best_state is None:
                raise RuntimeError("validation did not produce a checkpoint")
            model.load_state_dict(best_state)
            test_metrics = _evaluate(model, test_loader, device, torch.from_numpy(prior))
            decoded_path = args.results_dir / "rgb_probe_per_seed.csv"
            with decoded_path.open(encoding="utf-8", newline="") as handle:
                reference_rows = [row for row in csv.DictReader(handle) if row["condition"] == args.condition and row["seed"] == str(args.seed) and row["stage"] == "decoded_rgb"]
            if len(reference_rows) != 1:
                raise ValueError(f"missing same-seed decoded-RGB endpoint for {args.condition}/{args.seed}")
            reference = reference_rows[0]
            decoded_accuracy = float(reference["accuracy"])
            decoded_u = float(reference["u_probe_bits"])
            test_metrics.update({
                "decoded_rgb_accuracy": decoded_accuracy,
                "decoded_rgb_u_probe_bits": decoded_u,
                "relative_u_probe_utility": test_metrics["u_probe_bits"] / decoded_u if decoded_u > 0 else None,
                "relative_accuracy": (test_metrics["accuracy"] - 0.1) / (decoded_accuracy - 0.1) if decoded_accuracy > 0.1 else None,
                "accuracy_ratio": test_metrics["accuracy"] / decoded_accuracy if decoded_accuracy > 0 else None,
            })

        duration_seconds = time.perf_counter() - start
        resource_snapshot = monitor.stop()
        monitor = None
        after_snapshot = _snapshot(torch, device, disk_path)
        if args.overfit_gate:
            result = {
                "status": "complete",
                "gate": "tiny_overfit",
                "condition": args.condition,
                "seed": args.seed,
                "n_samples": 120,
                "final_train_accuracy": history[-1]["train_accuracy"],
                "passed": overfit_success,
                "history": history,
                "training_seconds": duration_seconds,
                "resources": resource_snapshot,
                "post_training_snapshot": after_snapshot,
                "model_parameters": parameters,
                "device": torch.cuda.get_device_name(device),
                "training_config": {"batch_size": 16, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW", "max_epochs": max_gate_epochs, "workers": 0, "pin_memory": False, "amp": True},
            }
        else:
            checkpoint_dir = cache_root / "checkpoints" / "phase47"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = _best_checkpoint_path(checkpoint_dir, args.condition, args.seed, False)
            torch.save({
                "state_dict": best_state,
                "condition": args.condition,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "model_parameters": parameters,
                "training_config": {"optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE, "workers": 0, "pin_memory": False, "amp": True},
            }, checkpoint_path)
            result = {
                "status": "complete",
                "condition": args.condition,
                "stage": "pre_reconstruction_coefficients",
                "representation": "representation + shared structured probe",
                "seed": args.seed,
                "best_epoch": best_epoch,
                "epochs_completed": len(history),
                **test_metrics,
                "training_seconds": duration_seconds,
                "model_parameters": parameters,
                "device": torch.cuda.get_device_name(device),
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "history": history,
                "resources": resource_snapshot,
                "post_training_snapshot": after_snapshot,
                "training_config": {"optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE, "workers": 0, "pin_memory": False, "amp": True},
            }
        args.record_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.record_path.with_name(f".{args.record_path.name}.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.record_path)
        return result
    except BaseException as error:
        if monitor is not None:
            resource_snapshot = monitor.stop()
        else:
            resource_snapshot = None
        failure = {
            "status": "stopped",
            "condition": args.condition,
            "seed": args.seed,
            "overfit_gate": args.overfit_gate,
            "error": f"{type(error).__name__}: {error}",
            "history": history,
            "resources": resource_snapshot,
        }
        args.record_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.record_path.with_name(f".{args.record_path.name}.tmp")
        temporary.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.record_path)
        raise
    finally:
        if monitor is not None:
            monitor.stop()
        del train_loader, validation_loader, test_loader, model, optimizer, scheduler, scaler
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "data/cache/neural_accessibility/imagenette_320")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/Phase 4.7 Standard Codec Stages")
    parser.add_argument("--record-path", type=Path, required=True)
    parser.add_argument("--disk-path", default="C:\\")
    parser.add_argument("--overfit-gate", action="store_true")
    args = parser.parse_args()
    result = _train(args)
    print(json.dumps({key: result[key] for key in result if key not in ("history", "resources", "resource_after_cleanup")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
