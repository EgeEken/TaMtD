import json
import subprocess
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from tamtd.data.cache import EncodedDataset, pad_collate
from tamtd.models.byte_cnn import parameter_count
from tamtd.models.invariant import DeepSets, HistogramMLP
from tamtd.training.metrics import accuracy
from tamtd.training.seed import seed_everything
from tamtd.training.train import resolve_device


def _git_sha() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _epoch(model, loader, optimizer, device, scaler, train):
    model.train(train)
    criterion = nn.CrossEntropyLoss()
    total_loss = total_accuracy = total_items = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
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


def train_invariant(
    manifest_path, codec, model_name, output_dir, train_offset=0, train_limit=8000,
    val_offset=8000, val_limit=2000, test_offset=0, test_limit=1000, seed=42,
    epochs=10, batch_size=64, learning_rate=1e-3, weight_decay=0.0,
    max_length=3072, device="auto",
):
    seed_everything(seed)
    device_obj = resolve_device(device)
    train_data = EncodedDataset(manifest_path, codec, "train", max_length, train_offset, train_limit)
    val_data = EncodedDataset(manifest_path, codec, "train", max_length, val_offset, val_limit)
    test_data = EncodedDataset(manifest_path, codec, "test", max_length, test_offset, test_limit)
    loaders = [
        DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate),
        DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=pad_collate),
        DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=pad_collate),
    ]
    model = HistogramMLP().to(device_obj) if model_name == "histogram_mlp" else DeepSets().to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device_obj.type == "cuda") if device_obj.type == "cuda" else None
    best_accuracy = float("-inf")
    best_state = None
    best_epoch = 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _epoch(model, loaders[0], optimizer, device_obj, scaler, True)
        with torch.no_grad():
            val_loss, val_accuracy = _epoch(model, loaders[1], optimizer, device_obj, scaler, False)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy})
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    with torch.no_grad():
        test_loss, test_accuracy = _epoch(model, loaders[2], optimizer, device_obj, scaler, False)
    result = {
        "model": model_name,
        "codec": codec,
        "seed": seed,
        "git_sha": _git_sha(),
        "device": str(device_obj),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_length": max_length,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "test_samples": len(test_data),
        "parameter_count": parameter_count(model),
        "best_validation_accuracy": best_accuracy,
        "selected_epoch": best_epoch,
        "test_accuracy": test_accuracy,
        "test_evaluations": 1,
        "training_wall_time_seconds": elapsed,
        "examples_per_second": len(train_data) * epochs / max(elapsed, 1e-9),
        "codec_metadata": train_data.rows[0].get("codec_metadata", {}),
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps({"manifest_path": str(Path(manifest_path).resolve()), "codec": codec, "model": model_name, "seed": seed, "epochs": epochs, "batch_size": batch_size, "max_length": max_length, "train_offset": train_offset, "train_limit": train_limit, "val_offset": val_offset, "val_limit": val_limit, "test_offset": test_offset, "test_limit": test_limit}, indent=2), encoding="utf-8")
    return result
