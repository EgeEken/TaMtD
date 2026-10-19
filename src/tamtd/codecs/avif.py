from io import BytesIO

from PIL import Image

from .base import CodecAdapter, EncodedSample


class AVIFCodec(CodecAdapter):
    def __init__(self, quality: int, speed: int = 6, max_threads: int = 1) -> None:
        if not 0 <= quality <= 100:
            raise ValueError("AVIF quality must be between 0 and 100")
        if not 0 <= speed <= 10:
            raise ValueError("AVIF speed must be between 0 and 10")
        if max_threads < 1:
            raise ValueError("AVIF max_threads must be at least 1")
        self.quality = quality
        self.speed = speed
        self.max_threads = max_threads
        self.name = f"avif_q{quality}"

    def encode(self, image: Image.Image) -> EncodedSample:
        buffer = BytesIO()
        image.convert("RGB").save(
            buffer,
            format="AVIF",
            quality=self.quality,
            speed=self.speed,
            subsampling="4:2:0",
            max_threads=self.max_threads,
            codec="auto",
            range="full",
        )
        return EncodedSample(
            buffer.getvalue(),
            self.name,
            {
                "codec_family": "avif",
                "format": "AVIF",
                "quality": self.quality,
                "speed": self.speed,
                "subsampling": "4:2:0",
                "max_threads": self.max_threads,
                "codec": "auto",
                "range": "full",
            },
        )

    def decode(self, data: bytes) -> Image.Image:
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB").copy()
