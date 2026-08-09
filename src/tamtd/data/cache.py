import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL manifest at line {line_number}") from error
    if not rows:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return rows


class EncodedDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        codec: str,
        split: str,
        max_length: int | None = None,
        offset: int = 0,
        limit: int | None = None,
        shuffle_bytes: bool = False,
        shuffle_seed: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.rows = [
            row
            for row in load_manifest(self.manifest_path)
            if row.get("codec") == codec and row.get("split") == split
        ]
        if not self.rows:
            raise ValueError(f"no rows for codec={codec!r}, split={split!r}")
        self.rows = self.rows[offset:] if offset else self.rows
        if limit is not None:
            self.rows = self.rows[:limit]
        self.max_length = max_length
        self.shuffle_bytes = shuffle_bytes
        self.shuffle_seed = shuffle_seed

    def __len__(self) -> int:
        return len(self.rows)

    def _read(self, row: dict[str, Any]) -> bytes:
        path = self.root / row["encoded_path"]
        if not path.is_file():
            raise FileNotFoundError(f"encoded bytes missing for {row['sample_id']}: {path}")
        data = path.read_bytes()
        if len(data) != row["encoded_length"]:
            raise ValueError(f"encoded length mismatch for {row['sample_id']}")
        return data

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        data = self._read(row)
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
            "label": int(row["class_id"]),
            "length": len(data),
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


def pad_collate(batch: Iterable[dict[str, Any]], pad_token: int = 256) -> dict[str, Any]:
    items = list(batch)
    max_length = max(item["length"] for item in items)
    tokens = torch.full((len(items), max_length), pad_token, dtype=torch.long)
    lengths = torch.tensor([item["length"] for item in items], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in items], dtype=torch.long)
    for index, item in enumerate(items):
        tokens[index, : item["length"]] = item["tokens"]
    return {"tokens": tokens, "lengths": lengths, "labels": labels, "rows": [item["row"] for item in items]}
