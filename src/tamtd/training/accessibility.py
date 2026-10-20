from __future__ import annotations

import time
import mmap
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, default_collate

from ..neural_accessibility import pad_byte_sequences


class DictTensorDataset(Dataset):
    def __init__(self, inputs: dict[str, torch.Tensor], labels: torch.Tensor) -> None:
        if not inputs or any(len(values) != len(labels) for values in inputs.values()):
            raise ValueError("all input tensors and labels must have the same nonzero length")
        if len(labels) == 0:
            raise ValueError("dataset cannot be empty")
        self.inputs = inputs
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return {name: values[index] for name, values in self.inputs.items()}, self.labels[index]


class ByteStreamDataset(Dataset):
    def __init__(self, streams: list[bytes], labels: torch.Tensor) -> None:
        if len(streams) != len(labels) or not streams:
            raise ValueError("byte streams and labels must have the same nonzero length")
        self.streams = streams
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return {"bytes": self.streams[index]}, self.labels[index]


class ByteStreamFileDataset(Dataset):
    def __init__(self, stream_path: str | Path, rows: list[dict], condition: str) -> None:
        self.stream_path = Path(stream_path)
        self.rows = rows
        self.condition = condition
        self.labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
        self._handle = self.stream_path.open("rb")
        self._mapping = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        reference = self.rows[index]["streams"][self.condition]
        value = self._mapping[reference["offset"] : reference["offset"] + reference["length"]]
        if len(value) != reference["length"]:
            raise ValueError(f"truncated stream cache entry for {self.condition}, row {index}")
        return {"bytes": value}, self.labels[index]

    def __del__(self):
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()


class IndexedDictTensorDataset(Dataset):
    def __init__(self, inputs: dict[str, torch.Tensor], labels: torch.Tensor, indices: list[int]) -> None:
        if not inputs or any(len(values) != len(labels) for values in inputs.values()):
            raise ValueError("all input tensors and labels must have the same nonzero length")
        if not indices:
            raise ValueError("dataset cannot be empty")
        self.inputs = inputs
        self.labels = labels.long()
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        return {name: values[source_index] for name, values in self.inputs.items()}, self.labels[source_index]


def collate_accessibility(samples: list[tuple[dict, torch.Tensor]], byte_max_length: int | None = None):
    inputs, labels = zip(*samples)
    labels = torch.stack(labels).long()
    if "bytes" in inputs[0]:
        tokens, lengths, truncation = pad_byte_sequences(
            [sample["bytes"] for sample in inputs], max_length=byte_max_length
        )
        return {"tokens": tokens, "lengths": lengths}, labels, truncation
    return default_collate(inputs), labels, None


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int = 0,
    byte_max_length: int | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        collate_fn=lambda samples: collate_accessibility(samples, byte_max_length),
        persistent_workers=workers > 0,
    )


def _move_inputs(inputs: dict, device: torch.device) -> dict:
    return {name: values.to(device, non_blocking=True) for name, values in inputs.items()}


@torch.inference_mode()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    count = 0
    for inputs, labels, _ in loader:
        inputs = _move_inputs(inputs, device)
        labels = labels.to(device, non_blocking=True)
        logits = model(**inputs)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        count += len(labels)
    if count == 0:
        raise ValueError("cannot evaluate an empty dataset")
    return correct / count


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
) -> tuple[float, dict]:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    truncation_count = 0
    truncation_bytes = 0
    for inputs, labels, truncation in loader:
        inputs = _move_inputs(inputs, device)
        labels = labels.to(device, non_blocking=True)
        if truncation:
            truncation_count += truncation["truncated_samples"]
            truncation_bytes += truncation["truncated_bytes"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(**inputs)
            loss = nn.functional.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach().item()) * len(labels)
        sample_count += len(labels)
    return loss_sum / sample_count, {
        "truncated_samples": truncation_count,
        "truncated_bytes": truncation_bytes,
    }


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_path: str | Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str = "adamw",
    momentum: float = 0.9,
    patience: int = 5,
    amp: bool = True,
) -> dict:
    model.to(device)
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_validation_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    truncation_counts = {"truncated_samples": 0, "truncated_bytes": 0}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, truncation = _train_epoch(
            model, train_loader, optimizer, scaler, device, amp
        )
        for key in truncation_counts:
            truncation_counts[key] += truncation[key]
        validation_accuracy = evaluate_accuracy(model, validation_loader, device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            break
    wall_seconds = time.perf_counter() - start
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    test_accuracy = evaluate_accuracy(model, test_loader, device)
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    trained_samples = len(train_loader.dataset) * len(history)
    del state, optimizer, scheduler, scaler
    return {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_accuracy": best_validation_accuracy,
        "test_accuracy": test_accuracy,
        "training_wall_seconds": wall_seconds,
        "examples_per_second": trained_samples / wall_seconds,
        "peak_vram_mb": peak_vram,
        "truncation": truncation_counts,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }


def run_tiny_overfit(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_epochs: int = 100,
    target_accuracy: float = 0.95,
    learning_rate: float = 0.001,
    amp: bool = True,
) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    best_accuracy = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        _train_epoch(model, loader, optimizer, scaler, device, amp)
        accuracy = evaluate_accuracy(model, loader, device)
        best_accuracy = max(best_accuracy, accuracy)
        epochs_run = epoch
        if accuracy >= target_accuracy:
            break
    return {
        "passed": best_accuracy >= target_accuracy,
        "target_accuracy": target_accuracy,
        "best_train_accuracy": best_accuracy,
        "epochs_run": epochs_run,
        "wall_seconds": time.perf_counter() - start,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None,
    }
