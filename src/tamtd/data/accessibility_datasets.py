from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .imagenette_accessibility import load_prepared_rgb, sha256


class PixelImageDataset(Dataset):
    def __init__(self, source_root: str | Path, rows: list[dict], image_size: int = 256, verify_hashes: bool = False) -> None:
        self.source_root = Path(source_root)
        self.rows = rows
        self.image_size = image_size
        self.verify_hashes = verify_hashes
        self.labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        pixels = load_prepared_rgb(self.source_root / row["image_id"], self.image_size)
        if self.verify_hashes and sha256(pixels.tobytes()) != row["rgb_sha256"]:
            raise ValueError(f"prepared RGB hash mismatch for {row['image_id']}")
        image = torch.from_numpy(pixels).permute(2, 0, 1)
        return {"images": image}, self.labels[index]


class DecodedRGBImageDataset(Dataset):
    def __init__(self, stream_path: str | Path, rows: list[dict], condition: str, codec) -> None:
        self.stream_path = Path(stream_path)
        self.rows = rows
        self.condition = condition
        self.codec = codec
        self.labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        reference = self.rows[index]["streams"][self.condition]
        with self.stream_path.open("rb") as handle:
            handle.seek(reference["offset"])
            encoded = handle.read(reference["length"])
        if len(encoded) != reference["length"]:
            raise ValueError(f"truncated stream cache entry for {self.condition}, row {index}")
        image = self.codec.decode(encoded).convert("RGB")
        pixels = np.asarray(image, dtype=np.uint8).copy()
        return {"images": torch.from_numpy(pixels).permute(2, 0, 1)}, self.labels[index]
