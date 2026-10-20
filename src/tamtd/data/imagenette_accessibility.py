from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torchvision.datasets import Imagenette

from .transforms import prepare_image


def make_imagenette_splits(
    train_samples: list[tuple[str, int]],
    official_validation_samples: list[tuple[str, int]],
    class_names: dict[int, str],
    seed: int = 42,
    validation_fraction: float = 0.1,
) -> list[dict]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    seen = set()
    grouped: dict[int, list[str]] = {}
    for image_id, label in train_samples:
        if image_id in seen:
            raise ValueError(f"duplicate Imagenette source ID: {image_id}")
        seen.add(image_id)
        grouped.setdefault(label, []).append(image_id)
    assignments = {}
    for label, image_ids in grouped.items():
        ordered = sorted(image_ids)
        shuffled = ordered.copy()
        random.Random(seed + label).shuffle(shuffled)
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation_ids = set(shuffled[:validation_count])
        for image_id in ordered:
            assignments[image_id] = "validation" if image_id in validation_ids else "train"
    rows = [
        {
            "image_id": image_id,
            "source_split": "train",
            "split": assignments[image_id],
            "label": int(label),
            "label_name": class_names[label],
        }
        for image_id, label in train_samples
    ]
    for image_id, label in official_validation_samples:
        if image_id in seen:
            raise ValueError(f"duplicate or cross-split Imagenette source ID: {image_id}")
        seen.add(image_id)
        rows.append(
            {
                "image_id": image_id,
                "source_split": "val",
                "split": "test",
                "label": int(label),
                "label_name": class_names[label],
            }
        )
    split_order = {"train": 0, "validation": 1, "test": 2}
    rows.sort(key=lambda row: (split_order[row["split"]], row["image_id"]))
    for index, row in enumerate(rows):
        row["cache_index"] = index
    return rows


def check_representation_alignment(expected_rows: list[dict], representation_rows: list[dict]) -> None:
    expected = [(row["image_id"], int(row["label"])) for row in expected_rows]
    actual = [(row["image_id"], int(row["label"])) for row in representation_rows]
    if actual != expected:
        raise ValueError("representation sample IDs or labels differ from the dataset manifest")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_imagenette_source_root(dataset_root: str | Path) -> Path:
    train = Imagenette(Path(dataset_root).resolve(), split="train", size="320px", download=False)
    return Path(train._size_root).resolve()


def load_prepared_rgb(source_path: str | Path, image_size: int = 256) -> np.ndarray:
    with Image.open(source_path) as image:
        prepared, _ = prepare_image(ImageOps.exif_transpose(image), image_size)
    rgb = np.asarray(prepared, dtype=np.uint8)
    if rgb.shape != (image_size, image_size, 3):
        raise ValueError(f"unexpected prepared RGB shape for {source_path}: {rgb.shape}")
    return rgb.copy()


def prepare_imagenette_cache(
    dataset_root: str | Path,
    cache_root: str | Path,
    seed: int = 42,
    validation_fraction: float = 0.1,
    image_size: int = 256,
) -> tuple[list[dict], Path]:
    dataset_root = Path(dataset_root).resolve()
    cache_root = Path(cache_root).resolve()
    train = Imagenette(dataset_root, split="train", size="320px", download=False)
    official_val = Imagenette(dataset_root, split="val", size="320px", download=False)
    source_root = Path(train._size_root).resolve()
    class_names = {index: names[0] for index, names in enumerate(train.classes)}
    train_samples = [
        (Path(path).relative_to(source_root).as_posix(), label) for path, label in train._samples
    ]
    val_samples = [
        (Path(path).relative_to(source_root).as_posix(), label) for path, label in official_val._samples
    ]
    rows = make_imagenette_splits(
        train_samples,
        val_samples,
        class_names,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.jsonl"
    if manifest_path.exists():
        cached_rows = read_manifest(manifest_path)
        if len(cached_rows) == len(rows) and all(
            (cached["image_id"], cached["label"], cached["split"], cached["cache_index"])
            == (expected["image_id"], expected["label"], expected["split"], expected["cache_index"])
            for cached, expected in zip(cached_rows, rows)
        ) and all("rgb_sha256" in row and "source_sha256" in row for row in cached_rows):
            return cached_rows, source_root
        raise ValueError(f"Imagenette cache does not match the requested split: {cache_root}")

    for row in rows:
        source = source_root / Path(row["image_id"])
        source_bytes = source.read_bytes()
        rgb = load_prepared_rgb(source, image_size)
        row["source_sha256"] = sha256(source_bytes)
        row["rgb_sha256"] = sha256(rgb.tobytes())
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows, source_root


def read_manifest(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
