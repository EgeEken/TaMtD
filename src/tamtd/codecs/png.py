from io import BytesIO

from PIL import Image

from .base import CodecAdapter, EncodedSample


class PNGCodec(CodecAdapter):
    name = "png_lossless"

    def encode(self, image: Image.Image) -> EncodedSample:
        buffer = BytesIO()
        image.convert("RGB").save(
            buffer,
            format="PNG",
            compress_level=6,
            optimize=False,
        )
        return EncodedSample(
            buffer.getvalue(),
            self.name,
            {
                "codec_family": "png",
                "format": "PNG",
                "lossless": True,
                "compress_level": 6,
                "optimize": False,
            },
        )

    def decode(self, data: bytes) -> Image.Image:
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB").copy()
