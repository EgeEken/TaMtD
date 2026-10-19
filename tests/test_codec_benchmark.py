import csv
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, features

from tamtd.codec_benchmark import aggregate_rows, image_metrics, select_operating_points, write_sample_csv
from tamtd.codecs import AVIFCodec, JPEGCodec, JXLCodec, PBCStandardCodec, PNGCodec, WebPCodec


class CodecBenchmarkTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((256, 256))
        array = np.stack(((x * 3) % 256, (y * 5) % 256, ((x + y) * 7) % 256), axis=2).astype(np.uint8)
        self.image = Image.fromarray(array)

    def assert_roundtrip(self, codec):
        first = codec.encode(self.image)
        second = codec.encode(self.image)
        self.assertEqual(first.data, second.data)
        decoded = codec.decode(first.data)
        self.assertEqual(decoded.size, (256, 256))
        self.assertEqual(decoded.mode, "RGB")

    def test_jpeg_roundtrip_is_deterministic(self):
        self.assert_roundtrip(JPEGCodec(1))

    @unittest.skipUnless(features.check("webp"), "Pillow WebP support unavailable")
    def test_webp_roundtrip_is_deterministic(self):
        self.assert_roundtrip(WebPCodec(10))

    @unittest.skipUnless(features.check("avif"), "Pillow AVIF support unavailable")
    def test_avif_roundtrip_is_deterministic(self):
        self.assert_roundtrip(AVIFCodec(10))

    def test_png_roundtrip_is_lossless(self):
        codec = PNGCodec()
        self.assert_roundtrip(codec)
        decoded = codec.decode(codec.encode(self.image).data)
        np.testing.assert_array_equal(np.asarray(decoded), np.asarray(self.image))

    @unittest.skipUnless(shutil.which("cjxl") and shutil.which("djxl"), "JPEG XL tools unavailable")
    def test_jxl_roundtrip_is_deterministic(self):
        self.assert_roundtrip(JXLCodec(12))

    def test_metrics_use_uint8_rgb_range(self):
        original = Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8))
        decoded = Image.fromarray(np.full((16, 16, 3), 110, dtype=np.uint8))
        metrics = image_metrics(original, decoded)
        self.assertAlmostEqual(metrics["mse"], 100.0)
        self.assertAlmostEqual(metrics["psnr_db"], 10 * math.log10(255**2 / 100))
        self.assertGreater(metrics["ssim"], 0.99)
        exact = image_metrics(original, original)
        self.assertEqual(exact["mse"], 0.0)
        self.assertTrue(math.isinf(exact["psnr_db"]))
        self.assertAlmostEqual(exact["ssim"], 1.0)

    def test_csv_serialization_and_rate_match(self):
        rows = [
            {
                "image_id": "a",
                "codec_id": "pbc_compression",
                "codec_family": "pbc",
                "encoded_bytes": 500,
                "bpp": 500 * 8 / (256 * 256),
                "decode_valid": True,
                "mse": 100.0,
                "psnr_db": 28.0,
                "ssim": 0.9,
                "encode_ms_runs": [1.0, 2.0],
                "decode_ms_runs": [0.5, 0.6],
                "codec_metadata": {"pbc_outer_lzma_enabled": True},
            },
            {
                "image_id": "a",
                "codec_id": "jpeg_q1",
                "codec_family": "jpeg",
                "encoded_bytes": 510,
                "bpp": 510 * 8 / (256 * 256),
                "decode_valid": True,
                "mse": 110.0,
                "psnr_db": 27.5,
                "ssim": 0.89,
                "encode_ms_runs": [0.8, 0.9],
                "decode_ms_runs": [0.4, 0.5],
                "codec_metadata": {"quality": 1},
            },
        ]
        temp_root = Path(__file__).resolve().parents[1] / "data" / "cache"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            path = Path(directory) / "samples.csv"
            write_sample_csv(rows, path)
            with path.open(encoding="utf-8", newline="") as handle:
                loaded = next(csv.DictReader(handle))
            self.assertEqual(json.loads(loaded["encode_ms_runs"]), [1.0, 2.0])
            self.assertEqual(json.loads(loaded["codec_metadata"])["pbc_outer_lzma_enabled"], True)
        summary = aggregate_rows(rows)
        matches = select_operating_points(summary)
        self.assertEqual(matches["pbc_compression"]["codecs"]["jpeg"]["rate_match"]["codec_id"], "jpeg_q1")
        self.assertTrue(matches["pbc_compression"]["codecs"]["jpeg"]["rate_match"]["close"])

    def test_pbc_metadata_capture(self):
        class FakePBC:
            MAGIC = b"PBC3"
            ENTROPY_LZMA = 2

            @staticmethod
            def compress(image, config):
                return SimpleNamespace(data=b"PBC3\x00\x02payload", mse=25.0)

        class FakeConfig:
            def __init__(self, **values):
                self.values = values

        adapter = PBCStandardCodec.__new__(PBCStandardCodec)
        adapter.pbc_class = FakePBC
        adapter.config_class = FakeConfig
        adapter.config_values = {"use_lzma": True, "patch_count": 50}
        adapter.preset = "compression"
        adapter.name = "pbc_compression"
        adapter.git_state = {"pbc_git_sha": "abc123", "pbc_git_branch": "main", "pbc_worktree_dirty": False}
        sample = adapter.encode(self.image)
        self.assertTrue(sample.metadata["pbc_outer_lzma_enabled"])
        self.assertEqual(sample.metadata["pbc_entropy_mode"], "lzma")
        self.assertEqual(sample.metadata["pbc_git_sha"], "abc123")
        self.assertEqual(sample.metadata["pbc_config"]["patch_count"], 50)


if __name__ == "__main__":
    unittest.main()
