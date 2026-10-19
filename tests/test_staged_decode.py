import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tamtd.codecs import PBCCodecPair
from tamtd.staged_decode import (
    build_jpeg_ir,
    build_pbc_ir,
    load_array_bundle,
    parse_pbc_store_body,
    read_jpeg_coefficients,
    rebuild_pbc_store_stream,
    render_pbc_ir,
    serialize_array_bundle,
)


PBC_ROOT = os.environ.get("PBC_ROOT")
HAS_JPEGIO = importlib.util.find_spec("jpegio") is not None


class StagedDecodeTests(unittest.TestCase):
    def test_array_bundle_is_deterministic_and_roundtrips(self):
        metadata = {"schema": "test", "values": [1, 2, 3]}
        arrays = {"coefficients": np.arange(48, dtype="<i2").reshape(2, 3, 8)}
        first = serialize_array_bundle(metadata, arrays)
        second = serialize_array_bundle(metadata, arrays)
        self.assertEqual(first, second)
        actual_metadata, actual_arrays = load_array_bundle(first)
        self.assertEqual(actual_metadata, metadata)
        np.testing.assert_array_equal(actual_arrays["coefficients"], arrays["coefficients"])

    @unittest.skipUnless(PBC_ROOT, "set PBC_ROOT to run PBC staged representation tests")
    def test_pbc_parse_ir_roundtrip_and_render(self):
        config = {
            "patch_count": 4,
            "search_depth": 4,
            "proposal_depth": 2,
            "exact_depth": 2,
            "learned_filler_enabled": False,
            "auto_downsample_init": False,
            "downsample_rate": 1.0,
            "min_patch_size": 2,
            "max_patch_size": 16,
        }
        pair = PBCCodecPair(PBC_ROOT, preset="compression", config=config)
        image = Image.fromarray(np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3))
        store, forced = pair.encode_pair(image)
        body = pair.unpack_body(store.data)
        self.assertEqual(body, pair.unpack_body(forced.data))

        parsed = parse_pbc_store_body(body, pair)
        metadata, arrays = build_pbc_ir(parsed, pair)
        serialized = serialize_array_bundle(metadata, arrays)
        self.assertEqual(serialized, serialize_array_bundle(metadata, arrays))
        loaded_metadata, loaded_arrays = load_array_bundle(serialized)
        self.assertEqual(rebuild_pbc_store_stream(loaded_metadata, loaded_arrays, pair), store.data)
        expected = pair.decode(store.data).convert("RGB")
        actual = render_pbc_ir(loaded_metadata, loaded_arrays, pair)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        np.testing.assert_array_equal(np.asarray(pair.decode(forced.data)), np.asarray(expected))
        self.assertIn("palette", loaded_metadata["patches"][0]["arrays"])
        self.assertIn("values", loaded_metadata["patches"][0]["arrays"])

    @unittest.skipUnless(HAS_JPEGIO, "install the staged-jpeg extra to run JPEG coefficient tests")
    def test_jpeg_coefficients_are_deterministic_and_metadata_matches(self):
        pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        image = Image.fromarray(pixels)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.jpg"
            image.save(path, format="JPEG", quality=60, subsampling=2, optimize=False, progressive=False)
            jpeg = read_jpeg_coefficients(path)
            metadata, arrays, native_array_bytes = build_jpeg_ir(jpeg)
            verify_metadata, verify_arrays, _ = build_jpeg_ir(read_jpeg_coefficients(path))
        self.assertEqual(metadata, verify_metadata)
        self.assertGreater(native_array_bytes, sum(array.nbytes for array in arrays.values()))
        self.assertEqual(jpeg.spatial_arrays, [])
        for component in metadata["components"]:
            coefficients = arrays[component["coefficient_array"]]
            self.assertEqual(coefficients.shape[:2], (component["height_in_blocks"], component["width_in_blocks"]))
            self.assertEqual(coefficients.shape[-2:], (8, 8))
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], verify_arrays[name])
        data = serialize_array_bundle(metadata, arrays)
        self.assertEqual(data, serialize_array_bundle(metadata, arrays))


if __name__ == "__main__":
    unittest.main()
