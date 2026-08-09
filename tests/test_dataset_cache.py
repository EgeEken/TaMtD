import json
import tempfile
import unittest
from pathlib import Path

from tamtd.data.cache import EncodedDataset, pad_collate


class DatasetCacheTests(unittest.TestCase):
    def test_manifest_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bytes" / "raw_rgb").mkdir(parents=True)
            (root / "bytes" / "raw_rgb" / "a.bin").write_bytes(b"abc")
            (root / "bytes" / "raw_rgb" / "b.bin").write_bytes(b"de")
            rows = [
                {"sample_id": "a", "codec": "raw_rgb", "split": "train", "encoded_path": "bytes/raw_rgb/a.bin", "encoded_length": 3, "class_id": 0},
                {"sample_id": "b", "codec": "raw_rgb", "split": "train", "encoded_path": "bytes/raw_rgb/b.bin", "encoded_length": 2, "class_id": 1},
            ]
            (root / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            dataset = EncodedDataset(root / "manifest.jsonl", "raw_rgb", "train")
            batch = pad_collate([dataset[0], dataset[1]])
            self.assertEqual(batch["tokens"].shape, (2, 3))
            self.assertEqual(batch["tokens"][1, 2].item(), 256)
            self.assertEqual(batch["lengths"].tolist(), [3, 2])

    def test_truncation_stats_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bytes").mkdir()
            (root / "bytes" / "a.bin").write_bytes(b"abcd")
            (root / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "a",
                        "codec": "raw_rgb",
                        "split": "train",
                        "encoded_path": "bytes/a.bin",
                        "encoded_length": 4,
                        "class_id": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dataset = EncodedDataset(root / "manifest.jsonl", "raw_rgb", "train", max_length=2)
            stats = dataset.truncation_stats()
            self.assertEqual(stats["truncated_samples"], 1)
            self.assertEqual(stats["truncation_percent"], 100.0)
            self.assertEqual(stats["mean_retained_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
