import csv
import json
import subprocess
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from tamtd.data.decoded_rgb import DecodedRGBDataset, rgb_collate
from tamtd.models.rgb_cnn import RGBCNN, parameter_count
from tamtd.training.metrics import accuracy
from tamtd.training.seed import seed_everything
from tamtd.training.train import resolve_device


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


def _run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_accuracy = 0.0
    total_items = 0
    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        count = labels.size(0)
        total_loss += loss.detach().item() * count
        total_accuracy += accuracy(logits.detach(), labels) * count
        total_items += count
    return total_loss / max(total_items, 1), total_accuracy / max(total_items, 1)


def train_decoded_rgb(
    manifest_path,
    codec,
    pbc_root,
    output_dir,
    train_split="train",
    val_split="train",
    test_split="test",
    train_offset=0,
    train_limit=None,
    val_offset=0,
    val_limit=None,
    test_offset=0,
    test_limit=None,
    seed=42,
    epochs=10,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=0.0,
    device="auto",
):
    seed_everything(seed)
    device_obj = resolve_device(device)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data = DecodedRGBDataset(manifest_path, codec, train_split, pbc_root, train_offset, train_limit)
    val_data = DecodedRGBDataset(manifest_path, codec, val_split, pbc_root, val_offset, val_limit)
    test_data = DecodedRGBDataset(manifest_path, codec, test_split, pbc_root, test_offset, test_limit)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=rgb_collate)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=rgb_collate)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=rgb_collate)
    model = RGBCNN().to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    best_validation_accuracy = float("-inf")
    best_epoch = 0
    best_state = None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, optimizer, device_obj, True)
        with torch.no_grad():
            val_loss, val_accuracy = _run_epoch(model, val_loader, optimizer, device_obj, False)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy})
        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy = val_accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    with torch.no_grad():
        test_loss, test_accuracy = _run_epoch(model, test_loader, optimizer, device_obj, False)
    elapsed = time.perf_counter() - start
    result = {
        "model": "decoded_rgb_cnn",
        "codec": codec,
        "seed": seed,
        "git_sha": _git_sha(),
        "device": str(device_obj),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "test_samples": len(test_data),
        "parameter_count": parameter_count(model),
        "best_validation_accuracy": best_validation_accuracy,
        "selected_epoch": best_epoch,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_evaluations": 1,
        "training_wall_time_seconds": elapsed,
        "examples_per_second": len(train_data) * epochs / max(elapsed, 1e-9),
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
    }
    (output_dir / "config.json").write_text(json.dumps({"manifest_path": str(manifest_path.resolve()), "codec": codec, "pbc_root": str(Path(pbc_root).resolve()) if pbc_root else None, "train_offset": train_offset, "train_limit": train_limit, "val_offset": val_offset, "val_limit": val_limit, "test_offset": test_offset, "test_limit": test_limit, "epochs": epochs, "batch_size": batch_size, "seed": seed}, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return result
