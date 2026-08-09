import numpy as np
from PIL import Image

from .base import CodecAdapter, EncodedSample


class RawRGBCodec(CodecAdapter):
    name = "raw_rgb"

    def __init__(self, image_size: tuple[int, int] | None = None) -> None:
        self.image_size = image_size

    def encode(self, image: Image.Image) -> EncodedSample:
        rgb = image.convert("RGB")
        if self.image_size is not None and rgb.size != self.image_size:
            raise ValueError(f"expected image size {self.image_size}, got {rgb.size}")
        array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        return EncodedSample(
            array.tobytes(),
            self.name,
            {"mode": "RGB", "width": rgb.width, "height": rgb.height, "channels": 3},
        )

    def decode(self, data: bytes) -> Image.Image:
        if self.image_size is None:
            raise ValueError("RawRGBCodec.decode requires image_size")
        width, height = self.image_size
        expected = width * height * 3
        if len(data) != expected:
            raise ValueError(f"expected {expected} raw RGB bytes, got {len(data)}")
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return Image.fromarray(array.copy(), "RGB")
