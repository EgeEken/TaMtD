from __future__ import annotations

import time
import mmap
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from ..neural_accessibility import pad_byte_sequences
from .byte_batching import LengthBucketBatchSampler, summarize_byte_batches


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
        self.stream_lengths = [len(stream) for stream in streams]
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
        self.stream_lengths = [int(row["streams"][condition]["length"]) for row in rows]
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
    byte_batching: dict | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
        "collate_fn": lambda samples: collate_accessibility(samples, byte_max_length),
        "persistent_workers": workers > 0,
    }
    if byte_batching is not None:
        lengths = byte_stream_lengths(dataset)
        sampler = LengthBucketBatchSampler(
            lengths,
            max_padded_tokens=int(byte_batching["max_padded_tokens"]),
            max_samples_per_batch=int(byte_batching["max_samples_per_batch"]),
            seed=seed,
            shuffle=shuffle,
            pad_multiple=int(byte_batching.get("pad_multiple", 256)),
            bucket_multiplier=int(byte_batching.get("bucket_multiplier", 50)),
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        **common,
    )


def byte_stream_lengths(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        base_lengths = byte_stream_lengths(dataset.dataset)
        return [base_lengths[int(index)] for index in dataset.indices]
    lengths = getattr(dataset, "stream_lengths", None)
    if lengths is None:
        raise TypeError("length-bucket batching requires a byte-stream dataset")
    return [int(length) for length in lengths]


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
    gradient_accumulation_samples: int | None = None,
) -> tuple[float, dict]:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    truncation_count = 0
    truncation_bytes = 0
    batch_lengths: list[list[int]] = []
    effective_update_sizes: list[int] = []
    accumulated_samples = 0
    if gradient_accumulation_samples is not None:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, (inputs, labels, truncation) in enumerate(loader):
        if "lengths" in inputs:
            batch_lengths.append([int(length) for length in inputs["lengths"].tolist()])
            if truncation and truncation["truncated_samples"]:
                raise ValueError("byte streams must not be truncated")
        inputs = _move_inputs(inputs, device)
        labels = labels.to(device, non_blocking=True)
        if truncation:
            truncation_count += truncation["truncated_samples"]
            truncation_bytes += truncation["truncated_bytes"]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(**inputs)
            loss_sum_batch = nn.functional.cross_entropy(logits, labels, reduction="sum")
        batch_size = len(labels)
        loss_sum += float(loss_sum_batch.detach().item())
        if gradient_accumulation_samples is None:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss_sum_batch / batch_size).backward()
            scaler.step(optimizer)
            scaler.update()
            effective_update_sizes.append(batch_size)
        else:
            scaler.scale(loss_sum_batch).backward()
            accumulated_samples += batch_size
            is_last_batch = batch_index + 1 == len(loader)
            if accumulated_samples >= gradient_accumulation_samples or is_last_batch:
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_samples)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                effective_update_sizes.append(accumulated_samples)
                accumulated_samples = 0
        sample_count += len(labels)
    summary = {
        "truncated_samples": truncation_count,
        "truncated_bytes": truncation_bytes,
    }
    if batch_lengths:
        summary["byte_batch_metrics"] = summarize_byte_batches(
            batch_lengths,
            pad_multiple=getattr(loader.batch_sampler, "pad_multiple", 256),
        )
        update_distribution: dict[str, int] = {}
        for size in effective_update_sizes:
            update_distribution[str(size)] = update_distribution.get(str(size), 0) + 1
        summary["optimizer_update_sample_distribution"] = update_distribution
    return loss_sum / sample_count, summary


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
    gradient_accumulation_samples: int | None = None,
    evaluation_callback: Callable[[nn.Module, DataLoader, torch.device], dict] | None = None,
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
    byte_epoch_metrics: list[dict] = []
    optimizer_update_distributions: list[dict] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch - 1)
        train_loss, truncation = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp,
            gradient_accumulation_samples=gradient_accumulation_samples,
        )
        for key in truncation_counts:
            truncation_counts[key] += truncation[key]
        if "byte_batch_metrics" in truncation:
            byte_epoch_metrics.append(
                {
                    "epoch": epoch,
                    **truncation["byte_batch_metrics"],
                    "optimizer_update_sample_distribution": truncation[
                        "optimizer_update_sample_distribution"
                    ],
                }
            )
            optimizer_update_distributions.append(
                truncation["optimizer_update_sample_distribution"]
            )
        validation_accuracy = evaluate_accuracy(model, validation_loader, device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **({"byte_batch_metrics": truncation["byte_batch_metrics"]} if "byte_batch_metrics" in truncation else {}),
            }
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
            torch.save(model.state_dict(), checkpoint_temporary)
            checkpoint_temporary.replace(checkpoint_path)
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            break
    wall_seconds = time.perf_counter() - start
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    validation_metrics = evaluation_callback(model, validation_loader, device) if evaluation_callback else None
    test_metrics = evaluation_callback(model, test_loader, device) if evaluation_callback else None
    test_accuracy = float(test_metrics["accuracy"]) if test_metrics is not None else evaluate_accuracy(model, test_loader, device)
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    trained_samples = len(train_loader.dataset) * len(history)
    byte_batching = None
    if byte_epoch_metrics:
        padded_total = sum(row["padded_total_bytes_or_tokens"] for row in byte_epoch_metrics)
        true_total = sum(row["true_total_input_bytes"] for row in byte_epoch_metrics)
        physical_batch_sizes: dict[str, int] = {}
        update_sizes: dict[str, int] = {}
        for row in byte_epoch_metrics:
            for size, count in row["effective_batch_size_distribution"].items():
                physical_batch_sizes[size] = physical_batch_sizes.get(size, 0) + count
        for distribution in optimizer_update_distributions:
            for size, count in distribution.items():
                update_sizes[size] = update_sizes.get(size, 0) + count
        byte_batching = {
            "policy": "length_bucket_token_budget" if isinstance(train_loader.batch_sampler, LengthBucketBatchSampler) else "fixed_batch",
            "gradient_accumulation_target_samples": gradient_accumulation_samples,
            "epochs": byte_epoch_metrics,
            "run": {
                "true_total_input_bytes": true_total,
                "padded_total_bytes_or_tokens": padded_total,
                "padding_ratio": 1.0 - true_total / padded_total,
                "padding_efficiency": true_total / padded_total,
                "max_sequence_length": max(row["max_sequence_length"] for row in byte_epoch_metrics),
                "mean_sequence_length": byte_epoch_metrics[0]["mean_sequence_length"],
                "max_padded_batch_width": max(row["max_padded_batch_width"] for row in byte_epoch_metrics),
                "max_samples_per_batch": max(row["max_samples_per_batch"] for row in byte_epoch_metrics),
                "effective_batch_size_distribution": physical_batch_sizes,
                "optimizer_update_sample_distribution": update_sizes,
            },
        }
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
        **({"byte_batching": byte_batching} if byte_batching is not None else {}),
        **({"validation_metrics": validation_metrics, "test_metrics": test_metrics} if evaluation_callback else {}),
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
