import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from tamtd.data.cache import EncodedDataset, pad_collate
from tamtd.models.byte_cnn import ByteCNN, parameter_count
from tamtd.training.metrics import accuracy
from tamtd.training.seed import seed_everything


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def _git_sha() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _run_epoch(model, loader, optimizer, device, scaler, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_accuracy = 0.0
    total_items = 0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        tokens = batch["tokens"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None
        context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
        with context:
            logits = model(tokens, lengths)
            loss = criterion(logits, labels)
        if train:
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        count = labels.size(0)
        total_loss += loss.detach().float().item() * count
        total_accuracy += accuracy(logits.detach().float(), labels) * count
        total_items += count
    return total_loss / max(total_items, 1), total_accuracy / max(total_items, 1)


def train_bytecnn(
    manifest_path: str | Path,
    codec: str,
    output_dir: str | Path,
    train_split: str = "train",
    val_split: str = "test",
    num_classes: int = 10,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    max_length: int | None = 4096,
    workers: int = 0,
    device: str = "auto",
    test_split: str = "test",
    train_offset: int = 0,
    train_limit: int | None = None,
    val_offset: int = 0,
    val_limit: int | None = None,
    test_offset: int = 0,
    test_limit: int | None = None,
    shuffle_bytes: bool = False,
    shuffle_seed: int = 0,
) -> dict[str, Any]:
    seed_everything(seed)
    device_obj = resolve_device(device)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data = EncodedDataset(
        manifest_path,
        codec,
        train_split,
        max_length=max_length,
        offset=train_offset,
        limit=train_limit,
        shuffle_bytes=shuffle_bytes,
        shuffle_seed=shuffle_seed,
    )
    val_data = EncodedDataset(
        manifest_path,
        codec,
        val_split,
        max_length=max_length,
        offset=val_offset,
        limit=val_limit,
        shuffle_bytes=shuffle_bytes,
        shuffle_seed=shuffle_seed,
    )
    test_data = EncodedDataset(
        manifest_path,
        codec,
        test_split,
        max_length=max_length,
        offset=test_offset,
        limit=test_limit,
        shuffle_bytes=shuffle_bytes,
        shuffle_seed=shuffle_seed,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device_obj.type == "cuda",
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device_obj.type == "cuda",
        collate_fn=pad_collate,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device_obj.type == "cuda",
        collate_fn=pad_collate,
    )
    model = ByteCNN(num_classes=num_classes).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device_obj.type == "cuda") if device_obj.type == "cuda" else None
    start = time.perf_counter()
    history = []
    best_validation_accuracy = float("-inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, optimizer, device_obj, scaler, train=True)
        with torch.no_grad():
            val_loss, val_accuracy = _run_epoch(model, val_loader, optimizer, device_obj, scaler, train=False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy = val_accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    with torch.no_grad():
        test_loss, test_accuracy = _run_epoch(model, test_loader, optimizer, device_obj, scaler, train=False)
    if device_obj.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device_obj) / 1024**2
    else:
        peak_vram_mb = None
    result = {
        "model": "bytecnn",
        "codec": codec,
        "seed": seed,
        "shuffle_bytes": shuffle_bytes,
        "shuffle_seed": shuffle_seed,
        "git_sha": _git_sha(),
        "device": str(device_obj),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_length": max_length,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "parameter_count": parameter_count(model),
        "best_validation_accuracy": best_validation_accuracy,
        "selected_epoch": best_epoch,
        "final_validation_accuracy": history[-1]["val_accuracy"],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_evaluations": 1,
        "training_wall_time_seconds": elapsed,
        "examples_per_second": len(train_data) * epochs / max(elapsed, 1e-9),
        "peak_vram_mb": peak_vram_mb,
        "codec_metadata": train_data.rows[0].get("codec_metadata", {}),
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "train_truncation": train_data.truncation_stats(),
        "test_truncation": val_data.truncation_stats(),
        "evaluation_truncation": test_data.truncation_stats(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "manifest_path": str(manifest_path.resolve()),
                "codec": codec,
                "train_split": train_split,
                "val_split": val_split,
                "test_split": test_split,
                "num_classes": num_classes,
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "max_length": max_length,
                "workers": workers,
                "device": device,
                "train_offset": train_offset,
                "train_limit": train_limit,
                "val_offset": val_offset,
                "val_limit": val_limit,
                "test_offset": test_offset,
                "test_limit": test_limit,
                "shuffle_bytes": shuffle_bytes,
                "shuffle_seed": shuffle_seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    torch.save(model.state_dict(), output_dir / "last.pt")
    return result
