import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

from tamtd.data.accessibility_datasets import PixelImageDataset
from tamtd.data.imagenette_accessibility import check_representation_alignment, load_prepared_rgb, make_imagenette_splits, sha256
from tamtd.data.neural_accessibility import _recover_append_only_streams, _select_ir_verification_indices, jpeg_blocks_to_channels
from tamtd.models.accessibility import ByteStreamCNN, JPEGDCTNet, PBCInstructionNet, RGBResNet18
from tamtd.neural_accessibility import build_frontier_rows, pad_byte_sequences, semantic_accessibility
from tamtd.training.accessibility import DictTensorDataset, make_loader, run_tiny_overfit


class NeuralAccessibilityTests(unittest.TestCase):
    def test_pixel_dataset_recreates_rgb_from_source_and_checks_hash(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            source = root / "sample.jpg"
            Image.fromarray(np.full((280, 320, 3), [32, 96, 192], dtype=np.uint8)).save(source)
            pixels = load_prepared_rgb(source)
            row = {"image_id": source.name, "label": 3, "rgb_sha256": sha256(pixels.tobytes())}
            inputs, label = PixelImageDataset(root, [row], verify_hashes=True)[0]
        self.assertEqual(label.item(), 3)
        self.assertTrue(torch.equal(inputs["images"], torch.from_numpy(pixels.copy()).permute(2, 0, 1)))

    def test_resume_rolls_back_all_streams_after_interrupted_sample(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            stream_paths = {name: root / f"{name}.bin" for name in ("a", "b")}
            payloads = [{"a": b"A", "b": b"B"}, {"a": b"xyz", "b": b"DEF"}]
            rows = []
            offsets = {"a": 0, "b": 0}
            for index, sample in enumerate(payloads):
                references = {}
                for name, payload in sample.items():
                    references[name] = {"offset": offsets[name], "length": len(payload), "sha256": sha256(payload)}
                    offsets[name] += len(payload)
                rows.append({"cache_index": index, "streams": references})
            stream_paths["a"].write_bytes(b"Axy")
            stream_paths["b"].write_bytes(b"BDEF")
            manifest = root / "codec_manifest.jsonl"
            manifest.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))

            recovered = _recover_append_only_streams(manifest, stream_paths, ["a", "b"])

            self.assertEqual(len(recovered), 1)
            self.assertEqual(stream_paths["a"].read_bytes(), b"A")
            self.assertEqual(stream_paths["b"].read_bytes(), b"B")
            self.assertEqual(len(manifest.read_text().splitlines()), 1)

    def test_ir_verification_subset_is_deterministic_and_stratified(self):
        rows = [
            {"cache_index": index, "image_id": f"{split}/{label}/{item}.jpg", "split": split, "label": label}
            for index, (split, label, item) in enumerate(
                (split, label, item)
                for split in ("train", "validation", "test")
                for label in range(10)
                for item in range(4)
            )
        ]
        first = _select_ir_verification_indices(rows)
        second = _select_ir_verification_indices(rows)
        selected = [rows[index] for index in first]
        groups = {(row["split"], row["label"]) for row in selected}
        self.assertEqual(first, second)
        self.assertEqual(len(first), 95)
        self.assertEqual(len(groups), 30)
        self.assertTrue(all(sum((row["split"], row["label"]) == group for row in selected) >= 3 for group in groups))

    def test_imagenette_split_is_deterministic_and_stratified(self):
        train = [(f"train/{label}/{index}.jpg", label) for label in range(2) for index in range(20)]
        official_val = [(f"val/{label}/{index}.jpg", label) for label in range(2) for index in range(4)]
        names = {0: "class zero", 1: "class one"}
        first = make_imagenette_splits(train, official_val, names, seed=42)
        second = make_imagenette_splits(train, official_val, names, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(sum(row["split"] == "validation" for row in first), 4)
        self.assertEqual(sum(row["split"] == "test" for row in first), 8)
        self.assertTrue(all(row["split"] == "test" for row in first if row["source_split"] == "val"))

    def test_source_id_and_label_alignment(self):
        expected = [{"image_id": "a", "label": 0}, {"image_id": "b", "label": 1}]
        check_representation_alignment(expected, [dict(row) for row in expected])
        with self.assertRaises(ValueError):
            check_representation_alignment(expected, [{"image_id": "a", "label": 1}, expected[1]])

    def test_byte_batching_reports_truncation(self):
        tokens, lengths, stats = pad_byte_sequences([b"\x00\x01", b"\xff" * 300])
        self.assertEqual(tokens.shape, (2, 512))
        self.assertEqual(lengths.tolist(), [2, 300])
        self.assertEqual(stats["truncated_samples"], 0)
        self.assertEqual(int(tokens[0, 0]), 0)
        _, capped_lengths, capped = pad_byte_sequences([b"a" * 10, b"b" * 20], max_length=12)
        self.assertEqual(capped_lengths.tolist(), [10, 12])
        self.assertEqual(capped["truncated_bytes"], 8)

    def test_model_forward_passes_and_structured_batching(self):
        byte_model = ByteStreamCNN()
        byte_tokens, byte_lengths, _ = pad_byte_sequences([b"a" * 513, b"b" * 777])
        self.assertEqual(byte_model(byte_tokens, byte_lengths).shape, (2, 10))

        pbc_inputs = {
            "geometry": torch.rand(2, 5, 24),
            "content": torch.rand(2, 5, 144),
            "patch_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
        }
        pbc_model = PBCInstructionNet()
        pbc_batch, labels, _ = next(
            iter(make_loader(DictTensorDataset(pbc_inputs, torch.tensor([0, 1])), 2, False, 42))
        )
        self.assertEqual(pbc_model(**pbc_batch).shape, (2, 10))
        pbc_model.eval()
        permutation = torch.tensor([4, 2, 0, 3, 1])
        with torch.inference_mode():
            original = pbc_model(**pbc_inputs)
            permuted = pbc_model(
                geometry=pbc_inputs["geometry"][:, permutation],
                content=pbc_inputs["content"][:, permutation],
                patch_mask=pbc_inputs["patch_mask"][:, permutation],
            )
        self.assertTrue(torch.allclose(original, permuted, atol=1e-6))
        with self.assertRaises(ValueError):
            check_representation_alignment(
                [{"image_id": "a", "label": 0}], [{"image_id": "b", "label": 0}]
            )

        jpeg_model = JPEGDCTNet()
        jpeg_inputs = {
            "y_coefficients": torch.randint(-100, 100, (2, 64, 32, 32), dtype=torch.int16),
            "cb_coefficients": torch.randint(-100, 100, (2, 64, 16, 16), dtype=torch.int16),
            "cr_coefficients": torch.randint(-100, 100, (2, 64, 16, 16), dtype=torch.int16),
            "quantization_tables": torch.ones(2, 2, 8, 8, dtype=torch.int16) * 30,
        }
        self.assertEqual(jpeg_model(**jpeg_inputs).shape, (2, 10))
        blocks = torch.arange(2 * 3 * 8 * 8).reshape(2, 3, 8, 8).numpy()
        channels = jpeg_blocks_to_channels(blocks)
        self.assertEqual(channels.shape, (64, 2, 3))
        self.assertEqual(channels[0, 1, 2], blocks[1, 2, 0, 0])
        with self.assertRaises(ValueError):
            jpeg_blocks_to_channels(blocks[:, :, :7, :])
        self.assertEqual(RGBResNet18()(torch.zeros(2, 3, 256, 256)).shape, (2, 10))
        self.assertEqual(labels.tolist(), [0, 1])

    def test_pbc_model_accepts_samples_without_patches(self):
        model = PBCInstructionNet()
        logits = model(
            geometry=torch.zeros(2, 64, 24),
            content=torch.zeros(2, 64, 144),
            patch_mask=torch.zeros(2, 64, dtype=torch.bool),
        )
        self.assertEqual(logits.shape, (2, 10))
        self.assertTrue(torch.isfinite(logits).all())

    def test_semantic_accessibility_is_not_clamped(self):
        self.assertAlmostEqual(semantic_accessibility(0.6, 0.8), 5 / 7)
        self.assertGreater(semantic_accessibility(0.9, 0.8), 1.0)
        self.assertLess(semantic_accessibility(0.05, 0.8), 0.0)
        self.assertIsNone(semantic_accessibility(0.5, 0.1))

    def test_frontier_uses_cumulative_latency_and_reference_endpoint(self):
        preprocessing = {
            "pbc_compression": {
                "full_lzma_decode_ms": 5.0,
                "lzma_unpack_ms": 0.1,
                "store_body_parse_ms": 1.0,
                "canonical_ir_build_ms": 0.2,
            },
            "jpeg_q20": {"full_rgb_decode_ms": 1.0, "file_to_dct_ms": 0.9},
        }
        results = []
        for condition, representations in {
            "pbc_compression": [
                "pbc_lzma_bytes", "pbc_store_bytes", "pbc_parsed_fields", "pbc_canonical_ir"
            ],
            "jpeg_q20": ["jpeg_file_bytes", "jpeg_dct_ir"],
        }.items():
            for representation in representations:
                results.append({
                    "condition": condition,
                    "representation": representation,
                    "test_accuracy": 0.6,
                    "best_validation_accuracy": 0.5,
                    "model_family": "test",
                })
        sizes = {
            "pbc_compression": {
                "lzma_bytes": 100, "store_bytes": 120, "parsed_bytes": 400,
                "canonical_bytes": 500, "rgb_bytes": 196608, "bpp": 0.015,
            },
            "jpeg_q20": {"jpeg_bytes": 80, "dct_bytes": 400, "rgb_bytes": 196608, "bpp": 0.01},
        }
        rows = build_frontier_rows(
            results, {"pbc_compression": 0.8, "jpeg_q20": 0.8}, preprocessing, sizes
        )
        pbc = [row for row in rows if row["condition"] == "pbc_compression"]
        self.assertEqual([row["cumulative_preprocessing_ms"] for row in pbc], [0, 0.1, 1.1, 1.3, 5.0])
        self.assertAlmostEqual(pbc[-1]["test_accuracy"], 0.8)
        self.assertEqual(pbc[-1]["model_family"], "common_rgb_reference")

    def test_tiny_overfit_helper_returns_training_diagnostics(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(1, 2)

            def forward(self, values):
                return self.layer(values)

        dataset = DictTensorDataset({"values": torch.zeros(4, 1)}, torch.zeros(4, dtype=torch.long))
        loader = make_loader(dataset, batch_size=4, shuffle=False, seed=42)
        model = TinyModel()
        result = run_tiny_overfit(model, loader, torch.device("cpu"), max_epochs=1, target_accuracy=0.0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["epochs_run"], 1)


if __name__ == "__main__":
    unittest.main()
