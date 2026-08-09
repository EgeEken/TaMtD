import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from tamtd.data.reconstruction import ReconstructionDataset, reconstruction_collate
from tamtd.models.reconstruction import ByteReconstructor, HistogramReconstructor, parameter_count
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


def _metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    error = predictions.float() - targets.float()
    mse = float(error.square().mean().item())
    mae = float(error.abs().mean().item())
    psnr = float("inf") if mse == 0 else 10.0 * float(np.log10(1.0 / mse))
    return {"mse": mse, "mae": mae, "psnr_db": psnr}


def _run_epoch(model, loader, optimizer, device, scaler, train: bool) -> dict[str, float]:
    model.train(train)
    total_predictions = []
    total_targets = []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        lengths = batch["lengths"].to(device)
        targets = batch["targets"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            predictions = model(tokens, lengths)
            loss = nn.functional.mse_loss(predictions, targets)
        if train:
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_predictions.append(predictions.detach().float().cpu())
        total_targets.append(targets.detach().float().cpu())
    return _metrics(torch.cat(total_predictions), torch.cat(total_targets))


def _constant_metrics(loader, image: torch.Tensor) -> dict[str, float]:
    predictions = []
    targets = []
    for batch in loader:
        predictions.append(image.unsqueeze(0).expand(batch["targets"].size(0), -1, -1))
        targets.append(batch["targets"])
    return _metrics(torch.cat(predictions), torch.cat(targets))


def _save_preview(model, dataset, device, path: Path, count: int = 16, scale: int = 16) -> None:
    count = min(count, len(dataset))
    batch = reconstruction_collate([dataset[index] for index in range(count)])
    with torch.no_grad():
        predictions = model(batch["tokens"].to(device), batch["lengths"].to(device)).cpu()
    tile = dataset.target_size * scale
    canvas = Image.new("L", (2 * tile, count * tile), color=255)
    for index in range(count):
        target = Image.fromarray(np.uint8(np.clip(batch["targets"][index].numpy(), 0, 1) * 255), mode="L")
        prediction = Image.fromarray(np.uint8(np.clip(predictions[index].numpy(), 0, 1) * 255), mode="L")
        target = target.resize((tile, tile), Image.Resampling.NEAREST)
        prediction = prediction.resize((tile, tile), Image.Resampling.NEAREST)
        canvas.paste(target, (0, index * tile))
        canvas.paste(prediction, (tile, index * tile))
    canvas.save(path)


def train_reconstruction(
    manifest_path: str | Path,
    codec: str,
    model_name: str,
    output_dir: str | Path,
    pbc_root: str | Path | None = None,
    train_split: str = "train",
    val_split: str = "train",
    test_split: str = "test",
    train_offset: int = 0,
    train_limit: int | None = 8000,
    val_offset: int = 8000,
    val_limit: int | None = 2000,
    test_offset: int = 0,
    test_limit: int | None = 1000,
    seed: int = 42,
    shuffle_bytes: bool = False,
    shuffle_seed: int = 42,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    max_length: int = 3072,
    target_size: int = 8,
    workers: int = 0,
    device: str = "auto",
) -> dict:
    seed_everything(seed)
    device_obj = resolve_device(device)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_args = {
        "pbc_root": pbc_root,
        "target_size": target_size,
        "max_length": max_length,
        "shuffle_bytes": shuffle_bytes,
        "shuffle_seed": shuffle_seed,
    }
    train_data = ReconstructionDataset(manifest_path, codec, train_split, offset=train_offset, limit=train_limit, **dataset_args)
    val_data = ReconstructionDataset(manifest_path, codec, val_split, offset=val_offset, limit=val_limit, **dataset_args)
    test_data = ReconstructionDataset(manifest_path, codec, test_split, offset=test_offset, limit=test_limit, **dataset_args)
    loaders = [
        DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=reconstruction_collate),
        DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=reconstruction_collate),
        DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=reconstruction_collate),
    ]
    train_targets = torch.stack([train_data[index]["target"] for index in range(len(train_data))])
    mean_image = train_targets.mean(dim=0)
    baseline_val = _constant_metrics(loaders[1], mean_image)
    baseline_test = _constant_metrics(loaders[2], mean_image)
    if model_name == "sequence":
        model = ByteReconstructor(target_size=target_size, max_length=max_length).to(device_obj)
    elif model_name == "histogram":
        model = HistogramReconstructor(target_size=target_size, max_length=max_length).to(device_obj)
    else:
        raise ValueError(f"unknown reconstruction model: {model_name}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device_obj.type == "cuda") if device_obj.type == "cuda" else None
    best_validation_mse = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, loaders[0], optimizer, device_obj, scaler, train=True)
        with torch.no_grad():
            val_metrics = _run_epoch(model, loaders[1], optimizer, device_obj, scaler, train=False)
        history.append({"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"val_{key}": value for key, value in val_metrics.items()}})
        if val_metrics["mse"] < best_validation_mse:
            best_validation_mse = val_metrics["mse"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    with torch.no_grad():
        test_metrics = _run_epoch(model, loaders[2], optimizer, device_obj, scaler, train=False)
    if device_obj.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device_obj) / 1024**2
    else:
        peak_vram_mb = None
    result = {
        "model": model_name,
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
        "target_size": target_size,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "test_samples": len(test_data),
        "parameter_count": parameter_count(model),
        "best_validation_mse": best_validation_mse,
        "best_validation_mae": history[best_epoch - 1]["val_mae"],
        "best_validation_psnr_db": history[best_epoch - 1]["val_psnr_db"],
        "selected_epoch": best_epoch,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_psnr_db": test_metrics["psnr_db"],
        "test_evaluations": 1,
        "training_wall_time_seconds": elapsed,
        "examples_per_second": len(train_data) * epochs / max(elapsed, 1e-9),
        "peak_vram_mb": peak_vram_mb,
        "final_train_mse": history[-1]["train_mse"],
        "final_train_mae": history[-1]["train_mae"],
        "final_train_psnr_db": history[-1]["train_psnr_db"],
        "mean_baseline_validation": baseline_val,
        "mean_baseline_test": baseline_test,
        "relative_test_mse_reduction_vs_mean": 1.0 - test_metrics["mse"] / max(baseline_test["mse"], 1e-12),
        "train_truncation": train_data.truncation_stats(),
        "validation_truncation": val_data.truncation_stats(),
        "test_truncation": test_data.truncation_stats(),
        "codec_metadata": train_data.rows[0].get("codec_metadata", {}),
    }
    config = {
        "manifest_path": str(manifest_path.resolve()),
        "codec": codec,
        "model": model_name,
        "pbc_root": str(Path(pbc_root).resolve()) if pbc_root else None,
        "seed": seed,
        "shuffle_bytes": shuffle_bytes,
        "shuffle_seed": shuffle_seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "target_size": target_size,
        "train_offset": train_offset,
        "train_limit": train_limit,
        "val_offset": val_offset,
        "val_limit": val_limit,
        "test_offset": test_offset,
        "test_limit": test_limit,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    torch.save(best_state, output_dir / "best.pt")
    _save_preview(model, test_data, device_obj, output_dir / "preview.png")
    return result
