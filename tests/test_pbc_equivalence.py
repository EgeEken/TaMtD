import os
import unittest

import numpy as np
from PIL import Image

from tamtd.codecs import PBCCodecPair, PBCDependencyError


PBC_ROOT = os.environ.get("PBC_ROOT")


@unittest.skipUnless(PBC_ROOT, "set PBC_ROOT to run PBC equivalence tests")
class PBCEquivalenceTests(unittest.TestCase):
    def test_store_and_forced_lzma_share_body_and_decode(self):
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
        pair = PBCCodecPair(PBC_ROOT, config=config)
        images = [
            Image.fromarray(np.full((16, 16, 3), 30, dtype=np.uint8), "RGB"),
            Image.fromarray(np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3), "RGB"),
        ]
        for image in images:
            store, forced = pair.encode_pair(image)
            self.assertEqual(store.data[5], pair.pbc_class.ENTROPY_STORE)
            self.assertEqual(forced.data[5], pair.pbc_class.ENTROPY_LZMA)
            self.assertEqual(pair.unpack_body(store.data), pair.unpack_body(forced.data))
            np.testing.assert_array_equal(np.asarray(pair.decode(store.data)), np.asarray(pair.decode(forced.data)))


if __name__ == "__main__":
    unittest.main()
