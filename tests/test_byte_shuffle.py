import json
import tempfile
import unittest
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tamtd.data.cache import EncodedDataset


class ByteShuffleTest(unittest.TestCase):
    def test_shuffle_preserves_bytes_and_is_sample_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bytes").mkdir()
            data = bytes(range(32))
            (root / "bytes" / "sample.bin").write_bytes(data)
            row = {
                "sample_id": "train_000000_v0",
                "codec": "test",
                "split": "train",
                "class_id": 3,
                "encoded_length": len(data),
                "encoded_path": "bytes/sample.bin",
            }
            (root / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            ordered = EncodedDataset(root / "manifest.jsonl", "test", "train")[0]["tokens"]
            shuffled_a = EncodedDataset(root / "manifest.jsonl", "test", "train", shuffle_bytes=True, shuffle_seed=42)[0]["tokens"]
            shuffled_b = EncodedDataset(root / "manifest.jsonl", "test", "train", shuffle_bytes=True, shuffle_seed=42)[0]["tokens"]
            self.assertTrue(torch.equal(shuffled_a, shuffled_b))
            self.assertEqual(sorted(shuffled_a.tolist()), ordered.tolist())
            self.assertNotEqual(shuffled_a.tolist(), ordered.tolist())


if __name__ == "__main__":
    unittest.main()
