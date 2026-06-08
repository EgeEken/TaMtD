from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
from PIL import Image


ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]


@dataclass
class JPEGEntropySample:
    coefficients: torch.Tensor
    qtable: torch.Tensor
    target: torch.Tensor


class BitReader:
    def __init__(self, data):
        out = []
        i = 0
        while i < len(data):
            value = int(data[i])
            if value == 0xFF and i + 1 < len(data):
                nxt = int(data[i + 1])
                if nxt == 0x00:
                    out.append(0xFF)
                    i += 2
                    continue
                if 0xD0 <= nxt <= 0xD7:
                    i += 2
                    continue
                break
            out.append(value)
            i += 1
        self.data = out
        self.byte_pos = 0
        self.bit_pos = 0

    def read_bit(self):
        if self.byte_pos >= len(self.data):
            return 0
        bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return bit

    def read_bits(self, n):
        value = 0
        for _ in range(n):
            value = (value << 1) | self.read_bit()
        return value


def extend_value(value, size):
    if size == 0:
        return 0
    threshold = 1 << (size - 1)
    if value < threshold:
        value -= (1 << size) - 1
    return value


class JPEGEntropyReader:
    def __init__(self, image_size=64, quality=75, max_blocks=None):
        self.image_size = image_size
        self.quality = quality
        self.max_blocks = max_blocks or (image_size // 8) ** 2

    def prepare_image(self, image):
        image = image.convert("L")
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

    def decode_target(self, jpeg_bytes):
        image = Image.open(BytesIO(jpeg_bytes)).convert("L")
        return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)

    def parse(self, jpeg_bytes):
        data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        qtables = {}
        htables = {"dc": {}, "ac": {}}
        width = height = None
        scan_start = scan_end = None
        dc_table = ac_table = 0
        i = 0

        while i < len(data) - 1:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = int(data[i + 1])
            if marker == 0xD8 or marker == 0xD9:
                i += 2
                continue
            if marker == 0x00 or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            length = int(data[i + 2]) * 256 + int(data[i + 3])
            segment = data[i + 4 : i + 2 + length]

            if marker == 0xDB:
                p = 0
                while p < len(segment):
                    info = int(segment[p])
                    precision = info >> 4
                    table_id = info & 15
                    p += 1
                    if precision != 0:
                        raise ValueError("Only 8-bit quantization tables are supported")
                    qtables[table_id] = segment[p : p + 64].astype(np.float32)
                    p += 64
            elif marker == 0xC4:
                p = 0
                while p < len(segment):
                    info = int(segment[p])
                    table_class = info >> 4
                    table_id = info & 15
                    counts = segment[p + 1 : p + 17].astype(int)
                    p += 17
                    symbols = segment[p : p + counts.sum()].astype(int)
                    p += counts.sum()
                    htables["dc" if table_class == 0 else "ac"][table_id] = self._build_huffman(counts, symbols)
            elif marker == 0xC0:
                height = int(segment[1]) * 256 + int(segment[2])
                width = int(segment[3]) * 256 + int(segment[4])
                components = int(segment[5])
                if components != 1:
                    raise ValueError("Only grayscale JPEGs are supported by JPEGEntropyReader")
            elif marker == 0xDA:
                components = int(segment[0])
                if components != 1:
                    raise ValueError("Only single-component scans are supported")
                table_info = int(segment[2])
                dc_table = table_info >> 4
                ac_table = table_info & 15
                scan_start = i + 2 + length
                break

            i += 2 + length

        for j in range(len(data) - 2, 1, -1):
            if data[j] == 0xFF and data[j + 1] == 0xD9:
                scan_end = j
                break

        if width != self.image_size or height != self.image_size:
            raise ValueError("Unexpected JPEG size")
        if scan_start is None or scan_end is None:
            raise ValueError("Could not find scan data")

        qtable = qtables[0]
        blocks_w = width // 8
        blocks_h = height // 8
        block_count = blocks_w * blocks_h
        reader = BitReader(data[scan_start:scan_end])
        dc_huff = htables["dc"][dc_table]
        ac_huff = htables["ac"][ac_table]
        previous_dc = 0
        blocks = np.zeros((block_count, 64), dtype=np.float32)

        for block_idx in range(block_count):
            coeffs = np.zeros(64, dtype=np.float32)
            size = self._decode_symbol(reader, dc_huff)
            previous_dc += extend_value(reader.read_bits(size), size)
            coeffs[0] = previous_dc
            k = 1
            while k < 64:
                symbol = self._decode_symbol(reader, ac_huff)
                if symbol == 0:
                    break
                run = symbol >> 4
                size = symbol & 15
                if symbol == 0xF0:
                    k += 16
                    continue
                k += run
                if k >= 64:
                    break
                coeffs[ZIGZAG[k]] = extend_value(reader.read_bits(size), size)
                k += 1
            blocks[block_idx] = coeffs

        return blocks, qtable

    def encode_image(self, image):
        image = self.prepare_image(image)
        jpeg_bytes = self.compress(image)
        coeffs, qtable = self.parse(jpeg_bytes)
        coeffs = np.clip(coeffs, -1024, 1024) / 1024.0
        qtable = qtable / 255.0
        return JPEGEntropySample(
            coefficients=torch.from_numpy(coeffs),
            qtable=torch.from_numpy(qtable),
            target=self.decode_target(jpeg_bytes),
        )

    def _build_huffman(self, counts, symbols):
        table = {}
        code = 0
        p = 0
        for length, count in enumerate(counts, 1):
            for _ in range(int(count)):
                table[(code, length)] = int(symbols[p])
                code += 1
                p += 1
            code <<= 1
        return table

    def _decode_symbol(self, reader, table):
        code = 0
        for length in range(1, 17):
            code = (code << 1) | reader.read_bit()
            symbol = table.get((code, length))
            if symbol is not None:
                return symbol
        raise ValueError("Invalid Huffman code")