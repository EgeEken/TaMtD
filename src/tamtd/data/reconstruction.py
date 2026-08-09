import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tamtd.codecs import PBCCodecPair

from .cache import load_manifest


class ReconstructionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        codec: str,
        split: str,
        pbc_root: str | Path | None = None,
        target_size: int = 8,
        max_length: int | None = 3072,
        offset: int = 0,
        limit: int | None = None,
        shuffle_bytes: bool = False,
        shuffle_seed: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.codec = codec
        self.root = self.manifest_path.parent
        self.target_size = target_size
        self.max_length = max_length
        self.shuffle_bytes = shuffle_bytes
        self.shuffle_seed = shuffle_seed
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
        self.target_cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _read(self, row: dict) -> bytes:
        path = self.root / row["encoded_path"]
        data = path.read_bytes()
        if len(data) != row["encoded_length"]:
            raise ValueError(f"encoded length mismatch for {row['sample_id']}")
        return data

    def _decode_target(self, index: int, data: bytes) -> torch.Tensor:
        if index not in self.target_cache:
            if self.codec.startswith("pbc_"):
                image = self.pair.decode(data)
            else:
                with Image.open(BytesIO(data)) as source:
                    image = source.convert("RGB").copy()
            image = image.convert("L").resize(
                (self.target_size, self.target_size), Image.Resampling.BOX
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
            self.target_cache[index] = torch.from_numpy(array.copy())
        return self.target_cache[index]

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        data = self._read(row)
        target = self._decode_target(index, data)
        if self.shuffle_bytes and len(data) > 1:
            seed_bytes = f"{self.shuffle_seed}:{row['sample_id']}".encode("utf-8")
            permutation_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little")
            generator = torch.Generator()
            generator.manual_seed(permutation_seed)
            permutation = torch.randperm(len(data), generator=generator).numpy()
            data = np.frombuffer(data, dtype=np.uint8)[permutation].tobytes()
        if self.max_length is not None:
            data = data[: self.max_length]
        return {
            "tokens": torch.tensor(list(data), dtype=torch.long),
            "length": len(data),
            "target": target,
            "row": row,
        }

    def truncation_stats(self) -> dict[str, float | int | None]:
        lengths = [int(row["encoded_length"]) for row in self.rows]
        if self.max_length is None:
            return {
                "max_length": None,
                "truncated_samples": 0,
                "truncation_percent": 0.0,
                "mean_retained_fraction": 1.0,
            }
        truncated = [length > self.max_length for length in lengths]
        retained = [min(length, self.max_length) / max(length, 1) for length in lengths]
        return {
            "max_length": self.max_length,
            "truncated_samples": sum(truncated),
            "truncation_percent": 100.0 * sum(truncated) / max(len(lengths), 1),
            "mean_retained_fraction": sum(retained) / max(len(retained), 1),
        }


def reconstruction_collate(batch: list[dict], pad_token: int = 256) -> dict:
    max_length = max(item["length"] for item in batch)
    tokens = torch.full((len(batch), max_length), pad_token, dtype=torch.long)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    for index, item in enumerate(batch):
        tokens[index, : item["length"]] = item["tokens"]
    return {
        "tokens": tokens,
        "lengths": lengths,
        "targets": torch.stack([item["target"] for item in batch]),
        "rows": [item["row"] for item in batch],
    }
