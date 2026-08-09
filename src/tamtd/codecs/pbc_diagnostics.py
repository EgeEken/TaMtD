import hashlib
from pathlib import Path

import numpy as np

from .pbc import PBCCodecPair


def _position(reader) -> int:
    return reader.i * 8 - reader.nbits


def _write_patch(writer, patch, channel_bits: int, zero_grid: bool = False) -> None:
    writer.write(patch["channel"], channel_bits)
    writer.write(patch["x"], 16)
    writer.write(patch["y"], 16)
    writer.write(patch["w"], 16)
    writer.write(patch["h"], 16)
    writer.write(0, 1)
    writer.write(patch["mask_len"], 10)
    for bit in patch["mask"]:
        writer.write(bit, 1)
    writer.write(patch["negative_max"], 8)
    writer.write(patch["positive_max"], 8)
    writer.write(patch["max_bitcount"], 4)
    writer.write(patch["cell_size"], 16)
    for value in patch["indices"]:
        writer.write(0 if zero_grid else value, patch["bitcount"])


def _read_patch(reader, channel_bits: int, positive_bias: bool) -> dict:
    start = _position(reader)
    channel = reader.read(channel_bits)
    x, y = reader.read(16), reader.read(16)
    w, h = reader.read(16), reader.read(16)
    if reader.read(1) != 0:
        raise ValueError("explicit PBC palette patches are unsupported")
    mask_len = reader.read(10)
    mask = [reader.read(1) for _ in range(mask_len)]
    negative_max, positive_max = reader.read(8), reader.read(8)
    max_bitcount = reader.read(4)
    cell_size = reader.read(16)
    import pbc3_ops as ops

    bitcount = ops.resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, positive_bias)
    grid_w = ops.ceil_div(w, cell_size)
    grid_h = ops.ceil_div(h, cell_size)
    indices = [reader.read(bitcount) for _ in range(grid_w * grid_h)]
    return {
        "start": start,
        "end": _position(reader),
        "channel": channel,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "mask_len": mask_len,
        "mask": mask,
        "negative_max": negative_max,
        "positive_max": positive_max,
        "max_bitcount": max_bitcount,
        "cell_size": cell_size,
        "bitcount": bitcount,
        "indices": indices,
    }


def _header_values(header, color_ids):
    (
        downsampled, original_w, original_h, w, h, color_space, channels,
        channel_bits, positive_bias, has_alpha, patch_count, base_values,
        warmup_on, warm_w, warm_h, warmup_split,
    ) = header
    return {
        "downsampled": downsampled,
        "original_w": original_w,
        "original_h": original_h,
        "w": w,
        "h": h,
        "color_id": color_ids[color_space],
        "color_space": color_space,
        "channels": channels,
        "channel_bits": channel_bits,
        "positive_bias": positive_bias,
        "has_alpha": has_alpha,
        "patch_count": patch_count,
        "base_values": base_values,
        "warmup_on": warmup_on,
        "warm_w": warm_w,
        "warm_h": warm_h,
        "warmup_split": warmup_split,
    }


def _write_header(writer, header, patch_count: int, base_values=None, warmup=None) -> None:
    writer.write(int(header["downsampled"]), 1)
    if header["downsampled"]:
        writer.write(header["original_w"], 16)
        writer.write(header["original_h"], 16)
    writer.write(header["w"], 16)
    writer.write(header["h"], 16)
    writer.write(header["color_id"], 2)
    writer.write(header["channels"], 8)
    writer.write(header["channel_bits"], 4)
    writer.write(int(header["positive_bias"]), 1)
    writer.write(int(header["has_alpha"]), 1)
    writer.write(patch_count, 32)
    for base in header["base_values"] if base_values is None else base_values:
        writer.write(base, 8)
    writer.write(int(warmup is not None), 1)
    if warmup is not None:
        writer.write(warmup[0], 16)
        writer.write(warmup[1], 16)
        writer.write(warmup[2], 32)


def _frame(pair: PBCCodecPair, body: bytes) -> bytes:
    return pair.pbc_class.MAGIC + bytes([pair.pbc_class.VERSION, pair.pbc_class.ENTROPY_STORE]) + body


class PBCDiagnosticStream:
    def __init__(self, data: bytes, pbc_root: str | Path | None = None, preset: str = "quality", pair: PBCCodecPair | None = None) -> None:
        self.pair = pair or PBCCodecPair(pbc_root=pbc_root, preset=preset)
        self.source_data = bytes(data)
        body = self.pair.unpack_body(self.source_data)
        reader = self.pair.pbc_class._read_header
        from pbc3_types import BitReader

        bit_reader = BitReader(body)
        raw_header = reader(bit_reader)
        self.header = _header_values(raw_header, {"RGB": 0, "YCbCr": 1})
        self.header_end = _position(bit_reader)
        self.body = body
        self.patches = []
        for _ in range(self.header["patch_count"]):
            self.patches.append(_read_patch(bit_reader, self.header["channel_bits"], self.header["positive_bias"]))
        if self.header["warmup_on"]:
            raise ValueError("diagnostic PBC variants do not support warmup streams")
        self.init_count = min(self.header["channels"], len(self.patches))

    def _build(self, variant: str, seed: int = 42) -> bytes:
        if variant == "full":
            return self.source_data
        patches = list(self.patches)
        if variant == "patch_shuffled":
            seed_bytes = f"{seed}:{hashlib.sha256(self.source_data).hexdigest()}".encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little"))
            order = list(range(self.init_count)) + list(rng.permutation(range(self.init_count, len(patches))))
            patches = [patches[index] for index in order]
        if variant == "init_only":
            patches = patches[: self.init_count]
        elif variant == "residual_only":
            patches = patches[self.init_count :]
        writer = self.pair.pbc_class.__dict__.get("_BitWriter")
        if writer is None:
            from pbc3_types import BitWriter

            writer = BitWriter
        bit_writer = writer()
        base_values = [0] * self.header["channels"] if variant == "residual_only" else None
        _write_header(bit_writer, self.header, len(patches), base_values=base_values)
        for index, patch in enumerate(patches):
            zero_grid = variant == "init_zeroed" and index < self.init_count
            _write_patch(bit_writer, patch, self.header["channel_bits"], zero_grid=zero_grid)
        return _frame(self.pair, bit_writer.finish())

    def variants(self, seed: int = 42) -> dict[str, bytes]:
        return {
            "pbc_init_only": self._build("init_only", seed),
            "pbc_residual_only": self._build("residual_only", seed),
            "pbc_init_zeroed": self._build("init_zeroed", seed),
            "pbc_patch_shuffled": self._build("patch_shuffled", seed),
        }

    def stats(self) -> dict:
        init = self.patches[: self.init_count]
        residual = self.patches[self.init_count :]

        def summarize(patches):
            if not patches:
                return {"count": 0}
            return {
                "count": len(patches),
                "width_mean": float(np.mean([p["w"] for p in patches])),
                "height_mean": float(np.mean([p["h"] for p in patches])),
                "cell_size_mean": float(np.mean([p["cell_size"] for p in patches])),
                "bitcount_mean": float(np.mean([p["bitcount"] for p in patches])),
                "mask_active_mean": float(np.mean([sum(p["mask"]) for p in patches])),
                "negative_max_mean": float(np.mean([p["negative_max"] for p in patches])),
                "positive_max_mean": float(np.mean([p["positive_max"] for p in patches])),
                "grid_index_mean": float(np.mean([np.mean(p["indices"]) for p in patches])),
                "grid_index_std_mean": float(np.mean([np.std(p["indices"]) for p in patches])),
            }

        return {
            "header": self.header,
            "body_length": len(self.body),
            "init_patch_count": self.init_count,
            "residual_patch_count": len(self.patches) - self.init_count,
            "init": summarize(init),
            "residual": summarize(residual),
        }
