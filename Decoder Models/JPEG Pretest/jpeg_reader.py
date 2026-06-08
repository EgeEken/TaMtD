from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
from PIL import Image


@dataclass
class JPEGSyntax:
    bytes: torch.Tensor
    scan_start: int
    scan_end: int
    width: int
    height: int
    quality: int
    markers: torch.Tensor


class JPEGReader:
    MARKERS = {
        0xC0: 0,
        0xC2: 1,
        0xC4: 2,
        0xDB: 3,
        0xDA: 4,
        0xDD: 5,
        0xE0: 6,
        0xE1: 7,
        0xFE: 8,
    }

    def __init__(self, jpeg_file_path=None, image_size=64, grayscale=True, quality=75, max_bytes=4096):
        self.jpeg_file_path = jpeg_file_path
        self.image_size = image_size
        self.grayscale = grayscale
        self.quality = quality
        self.max_bytes = max_bytes

    def prepare_image(self, image=None):
        if image is None:
            image = Image.open(self.jpeg_file_path)
        image = image.convert("L" if self.grayscale else "RGB")
        w, h = image.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

    def compress(self, image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality, optimize=False, progressive=False)
        return buffer.getvalue()

    def decode(self, jpeg_bytes):
        image = Image.open(BytesIO(jpeg_bytes))
        return image.convert("L" if self.grayscale else "RGB")

    def parse_syntax(self, jpeg_bytes):
        data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        markers = np.zeros(len(self.MARKERS), dtype=np.float32)
        scan_start = 0
        scan_end = max(len(data) - 2, 0)
        i = 0
        while i < len(data) - 1:
            if data[i] != 0xFF:
                i += 1
                continue
            while i < len(data) - 1 and data[i + 1] == 0xFF:
                i += 1
            marker = int(data[i + 1])
            if marker == 0x00 or 0xD0 <= marker <= 0xD9:
                i += 2
                continue
            if marker in self.MARKERS:
                markers[self.MARKERS[marker]] += 1.0
            if i + 4 > len(data):
                break
            length = int(data[i + 2]) * 256 + int(data[i + 3])
            if marker == 0xDA:
                scan_start = i + 2 + length
                break
            i += 2 + length

        for j in range(len(data) - 2, 1, -1):
            if data[j] == 0xFF and data[j + 1] == 0xD9:
                scan_end = j
                break

        scan = data[scan_start:scan_end].astype(np.int64)
        if len(scan) > self.max_bytes:
            scan = scan[: self.max_bytes]
        padded = np.full(self.max_bytes, 256, dtype=np.int64)
        padded[: len(scan)] = scan

        return JPEGSyntax(
            bytes=torch.from_numpy(padded),
            scan_start=scan_start,
            scan_end=scan_end,
            width=self.image_size,
            height=self.image_size,
            quality=self.quality,
            markers=torch.from_numpy(markers),
        )

    def encode_image(self, image=None):
        image = self.prepare_image(image)
        jpeg_bytes = self.compress(image)
        syntax = self.parse_syntax(jpeg_bytes)
        target_image = self.decode(jpeg_bytes)
        target = torch.from_numpy(np.asarray(target_image, dtype=np.float32) / 255.0)
        if self.grayscale:
            target = target.unsqueeze(0)
        else:
            target = target.permute(2, 0, 1)
        return syntax, target

    def get_bitstream(self):
        with open(self.jpeg_file_path, "rb") as f:
            return np.frombuffer(f.read(), dtype=np.uint8)

    def get_media_data(self):
        image = self.prepare_image()
        return np.array(image)