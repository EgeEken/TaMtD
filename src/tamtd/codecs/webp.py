from io import BytesIO

from PIL import Image

from .base import CodecAdapter, EncodedSample


class WebPCodec(CodecAdapter):
    def __init__(self, quality: int, method: int = 6) -> None:
        if not 0 <= quality <= 100:
            raise ValueError("WebP quality must be between 0 and 100")
        if not 0 <= method <= 6:
            raise ValueError("WebP method must be between 0 and 6")
        self.quality = quality
        self.method = method
        self.name = f"webp_q{quality}"

    def encode(self, image: Image.Image) -> EncodedSample:
        buffer = BytesIO()
        image.convert("RGB").save(
            buffer,
            format="WEBP",
            quality=self.quality,
            method=self.method,
            lossless=False,
            exact=False,
        )
        return EncodedSample(
            buffer.getvalue(),
            self.name,
            {
                "codec_family": "webp",
                "format": "WebP",
                "quality": self.quality,
                "method": self.method,
                "lossless": False,
                "exact": False,
                "subsampling": "libwebp default",
                "threading": "Pillow/libwebp default",
            },
        )

    def decode(self, data: bytes) -> Image.Image:
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB").copy()
