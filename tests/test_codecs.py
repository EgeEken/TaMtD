import unittest

import numpy as np
from PIL import Image

from tamtd.codecs import JPEGCodec, RawRGBCodec


class CodecTests(unittest.TestCase):
    def setUp(self):
        array = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        self.image = Image.fromarray(array, "RGB")

    def test_raw_rgb_round_trip(self):
        codec = RawRGBCodec((16, 16))
        sample = codec.encode(self.image)
        decoded = codec.decode(sample.data)
        np.testing.assert_array_equal(np.asarray(decoded), np.asarray(self.image))

    def test_jpeg_is_deterministic_and_decodable(self):
        codec = JPEGCodec(75)
        first = codec.encode(self.image)
        second = codec.encode(self.image)
        self.assertEqual(first.data, second.data)
        decoded = codec.decode(first.data)
        self.assertEqual(decoded.size, self.image.size)
        self.assertEqual(decoded.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
