import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from ..runtime import cache_root

from .base import CodecAdapter, EncodedSample


class JXLCodec(CodecAdapter):
    def __init__(self, distance: float, effort: int = 7, num_threads: int = 1) -> None:
        if not 0 < distance <= 25:
            raise ValueError("JPEG XL distance must be in (0, 25]")
        if not 1 <= effort <= 10:
            raise ValueError("JPEG XL effort must be between 1 and 10")
        if num_threads < 1:
            raise ValueError("JPEG XL num_threads must be at least 1")
        cjxl = shutil.which("cjxl")
        djxl = shutil.which("djxl")
        if cjxl is None or djxl is None:
            raise RuntimeError("JPEG XL requires cjxl and djxl on PATH")
        self.cjxl = cjxl
        self.djxl = djxl
        self.distance = float(distance)
        self.effort = effort
        self.num_threads = num_threads
        self.name = f"jxl_d{self.distance:g}"
        self.temp_root = cache_root() / "codec_benchmark_temp"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        version = subprocess.run(
            [self.cjxl, "--version"], capture_output=True, text=True, check=True
        )
        self.version = (version.stdout or version.stderr).splitlines()[0].strip()

    def encode(self, image: Image.Image) -> EncodedSample:
        with tempfile.TemporaryDirectory(prefix="tamtd-jxl-", dir=self.temp_root) as directory:
            input_path = Path(directory) / "source.ppm"
            output_path = Path(directory) / "encoded.jxl"
            image.convert("RGB").save(input_path, format="PPM")
            subprocess.run(
                [
                    self.cjxl,
                    "--quiet",
                    "--distance",
                    str(self.distance),
                    "--effort",
                    str(self.effort),
                    f"--num_threads={self.num_threads}",
                    str(input_path),
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            data = output_path.read_bytes()
        return EncodedSample(
            data,
            self.name,
            {
                "codec_family": "jxl",
                "format": "JPEG XL",
                "distance": self.distance,
                "effort": self.effort,
                "num_threads": self.num_threads,
                "tool_version": self.version,
                "threading": self.num_threads,
            },
        )

    def decode(self, data: bytes) -> Image.Image:
        with tempfile.TemporaryDirectory(prefix="tamtd-jxl-", dir=self.temp_root) as directory:
            input_path = Path(directory) / "encoded.jxl"
            output_path = Path(directory) / "decoded.ppm"
            input_path.write_bytes(data)
            subprocess.run(
                [
                    self.djxl,
                    "--quiet",
                    f"--num_threads={self.num_threads}",
                    str(input_path),
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            with Image.open(output_path) as image:
                return image.convert("RGB").copy()
