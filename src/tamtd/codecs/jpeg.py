from io import BytesIO

from PIL import Image

from .base import CodecAdapter, EncodedSample


class JPEGCodec(CodecAdapter):
    def __init__(self, quality: int = 75) -> None:
        if not 1 <= quality <= 95:
            raise ValueError("JPEG quality must be between 1 and 95")
        self.quality = quality
        self.name = f"jpeg_q{quality}"

    def encode(self, image: Image.Image) -> EncodedSample:
        buffer = BytesIO()
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=self.quality,
            optimize=False,
            progressive=False,
            subsampling=2,
        )
        data = buffer.getvalue()
        return EncodedSample(
            data,
            self.name,
            {
                "format": "JPEG",
                "quality": self.quality,
                "progressive": False,
                "optimize": False,
                "subsampling": 2,
            },
        )

    def decode(self, data: bytes) -> Image.Image:
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB").copy()
