from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.models.accessibility import ByteStreamCNN
from tamtd.runtime import cache_root, default_device, temp_root
from tamtd.training.accessibility import ByteStreamFileDataset, byte_stream_lengths, make_loader
from tamtd.training.accessibility import _train_epoch
from tamtd.training.byte_batching import LengthBucketBatchSampler, summarize_byte_batches
from tamtd.training.seed import seed_everything
from tamtd.training.telemetry import ResourceMonitor


def _read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=cache_root() / "neural_accessibility" / "imagenette_320")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--temp-root", type=Path, default=temp_root())
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--max-padded-tokens", type=int, default=1048576)
    parser.add_argument("--max-samples-per-batch", type=int, default=32)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if args.sample_count < 1 or args.effective_batch_size < 1:
        raise ValueError("sample and effective batch sizes must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the bounded GPU stress diagnostic requires an available CUDA device")
    seed_everything(args.seed)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(args.temp_root.resolve())
    os.environ["TEMP"] = str(args.temp_root.resolve())
    os.environ["TMP"] = str(args.temp_root.resolve())

    manifest = _read_manifest(args.cache_root / "codec_manifest.jsonl")
    train_rows = [row for row in manifest if row["split"] == "train"]
    if args.sample_count > len(train_rows):
        raise ValueError(f"requested {args.sample_count} samples but only {len(train_rows)} are available")
    selection_generator = torch.Generator().manual_seed(args.seed)
    selected_indices = torch.randperm(len(train_rows), generator=selection_generator)[: args.sample_count].tolist()
    rows = [train_rows[index] for index in selected_indices]
    dataset = ByteStreamFileDataset(args.cache_root / "streams" / "pbc_quality_store.bin", rows, "pbc_quality_store")
    lengths = byte_stream_lengths(dataset)
    legacy_generator = torch.Generator().manual_seed(args.seed)
    fixed_order = torch.randperm(len(lengths), generator=legacy_generator).tolist()
    fixed_batches = [fixed_order[start : start + 32] for start in range(0, len(fixed_order), 32)]
    fixed_padding = summarize_byte_batches([[lengths[index] for index in batch] for batch in fixed_batches])
    token_sampler = LengthBucketBatchSampler(
        lengths,
        max_padded_tokens=args.max_padded_tokens,
        max_samples_per_batch=args.max_samples_per_batch,
        seed=args.seed,
        shuffle=True,
    )
    token_batches = list(token_sampler)
    token_padding = summarize_byte_batches([[lengths[index] for index in batch] for batch in token_batches])
    loader = make_loader(
        dataset,
        batch_size=args.max_samples_per_batch,
        shuffle=True,
        seed=args.seed,
        workers=0,
        pin_memory=False,
        byte_batching={
            "max_padded_tokens": args.max_padded_tokens,
            "max_samples_per_batch": args.max_samples_per_batch,
        },
    )
    model = ByteStreamCNN()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    amp = True
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    monitor = ResourceMonitor(
        torch_module=torch,
        device=device,
        disk_path=args.cache_root,
        interval_seconds=0.5,
        gpu_interval_seconds=1.0,
    ).start()
    start = time.perf_counter()
    loss, epoch = _train_epoch(
        model,
        loader,
        optimizer,
        scaler,
        device,
        amp,
        gradient_accumulation_samples=args.effective_batch_size,
    )
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - start
    resources = monitor.stop()
    actual_padding = epoch["byte_batch_metrics"]
    result = {
        "schema": "tamtd.byte-batching-diagnostic.v1",
        "diagnostic_only": True,
        "phase4_result_replacement": False,
        "condition": "pbc_quality/pbc_store_bytes",
        "seed": args.seed,
        "device": str(device),
        "sample_count": args.sample_count,
        "epochs": 1,
        "test_accuracy": None,
        "checkpoint_written": False,
        "training_loss": loss,
        "training_wall_seconds": wall_seconds,
        "examples_per_second": args.sample_count / wall_seconds,
        "microbatch_count": actual_padding["batch_count"],
        "optimizer_update_sample_distribution": epoch["optimizer_update_sample_distribution"],
        "fixed_batch_reference": {
            "batch_size": 32,
            "shuffle_seed": args.seed,
            "padding": fixed_padding,
            "memory_estimate": {
                "padded_token_slots": fixed_padding["max_padded_batch_tokens"],
                "padded_input_tensor_bytes_int64": fixed_padding["max_padded_batch_tokens"] * 8,
                "measurement": "derived from cache lengths; not recorded by the original Phase 4 runs",
            },
        },
        "token_budget_batching": {
            "max_padded_tokens": args.max_padded_tokens,
            "max_samples_per_batch": args.max_samples_per_batch,
            "effective_batch_size_target": args.effective_batch_size,
            "padding": actual_padding,
            "sampler_padding_check": token_padding,
            "padded_tokens_reduction_fraction": 1 - actual_padding["padded_total_bytes_or_tokens"] / fixed_padding["padded_total_bytes_or_tokens"],
        },
        "resource_diagnostics": resources,
    }
    _write_json(args.output.resolve(), result)
    print(json.dumps({"diagnostic": str(args.output.resolve()), "samples": args.sample_count, "device": str(device), "examples_per_second": result["examples_per_second"], "padding_reduction_fraction": result["token_budget_batching"]["padded_tokens_reduction_fraction"], "cuda_allocated_peak_bytes": resources["peaks"].get("cuda_allocated_peak_bytes"), "cuda_reserved_peak_bytes": resources["peaks"].get("cuda_reserved_peak_bytes"), "gpu_utilization_mean_percent": resources.get("gpu_utilization_mean_percent")}, indent=2), flush=True)
    del model, optimizer, scaler, loader, dataset


if __name__ == "__main__":
    main()
