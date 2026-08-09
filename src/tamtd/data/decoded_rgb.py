from pathlib import Path
from io import BytesIO

import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image

from tamtd.codecs import PBCCodecPair
from .cache import load_manifest


class DecodedRGBDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        codec: str,
        split: str,
        pbc_root: str | Path | None,
        offset: int = 0,
        limit: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.codec = codec
        self.root = self.manifest_path.parent
        self.rows = [
            row
            for row in load_manifest(self.manifest_path)
            if row.get("codec") == codec and row.get("split") == split
        ]
        self.rows = self.rows[offset:]
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"no rows for codec={codec!r}, split={split!r}")
        self.pair = None
        if codec.startswith("pbc_"):
            preset = self.rows[0]["codec_metadata"].get("pbc_preset", "balanced")
            self.pair = PBCCodecPair(pbc_root=pbc_root, preset=preset)
        self.cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        if index not in self.cache:
            row = self.rows[index]
            data = (self.root / row["encoded_path"]).read_bytes()
            if self.codec == "raw_rgb":
                image = np.frombuffer(data, dtype=np.uint8).reshape(32, 32, 3)
            elif self.pair is not None:
                image = self.pair.decode(data)
            else:
                with Image.open(BytesIO(data)) as source:
                    image = source.convert("RGB").copy()
            array = np.asarray(image, dtype=np.float32) / 255.0
            self.cache[index] = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return {"image": self.cache[index], "label": int(self.rows[index]["class_id"])}


def rgb_collate(batch):
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }
