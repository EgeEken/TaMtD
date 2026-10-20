from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class JPEGCoefficientDependencyError(RuntimeError):
    pass


def _metadata_bytes(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def serialize_array_bundle(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
    members = {"metadata_json": np.frombuffer(_metadata_bytes(metadata), dtype=np.uint8)}
    members.update(arrays)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, array in sorted(members.items()):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer,
                np.ascontiguousarray(array),
                version=(1, 0),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, array_buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def load_array_bundle(value: bytes | str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = io.BytesIO(value) if isinstance(value, bytes) else value
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
    return metadata, arrays


def parse_pbc_store_body(body: bytes, pair) -> dict[str, Any]:
    import pbc3_ops as ops
    import pbc3_stream as stream
    import pbc3_types

    reader = pbc3_types.BitReader(body)
    (
        downsampled, original_width, original_height, width, height, color_space,
        channels, channel_bits, positive_bias, has_alpha, patch_count, base_values,
        warmup_on, warmup_width, warmup_height, warmup_split,
    ) = pair.pbc_class._read_header(reader)
    header = {
        "downsampled": downsampled,
        "original_width": original_width,
        "original_height": original_height,
        "width": width,
        "height": height,
        "color_space": color_space,
        "channels": channels,
        "channel_bits": channel_bits,
        "positive_bias": positive_bias,
        "has_alpha": has_alpha,
        "patch_count": patch_count,
        "base_values": list(base_values),
        "warmup": {
            "width": warmup_width,
            "height": warmup_height,
            "split": warmup_split,
        } if warmup_on else None,
    }
    patches = []
    for _ in range(patch_count):
        channel = reader.read(channel_bits)
        x, y, patch_width, patch_height = (reader.read(16) for _ in range(4))
        palette_type = reader.read(1)
        if palette_type != pair.pbc_class.PALETTE_GENERATED:
            raise ValueError(f"unsupported PBC palette type {palette_type}")
        mask = [reader.read(1) for _ in range(reader.read(10))]
        negative_max, positive_max = reader.read(8), reader.read(8)
        max_bitcount = reader.read(4)
        cell_size = reader.read(16)
        bitcount = ops.resolve_palette_bitcount(
            mask, max_bitcount, negative_max, positive_max, positive_bias
        )
        grid_width = ops.ceil_div(patch_width, cell_size)
        grid_height = ops.ceil_div(patch_height, cell_size)
        indices = stream.read_grid(reader, grid_width * grid_height, bitcount).reshape(
            grid_height, grid_width
        )
        patches.append(
            {
                "channel": channel,
                "x": x,
                "y": y,
                "width": patch_width,
                "height": patch_height,
                "palette_type": palette_type,
                "mask": mask,
                "negative_max": negative_max,
                "positive_max": positive_max,
                "max_bitcount": max_bitcount,
                "cell_size": cell_size,
                "bitcount": bitcount,
                "indices": indices,
            }
        )
    remaining_bits = len(body) * 8 - (reader.i * 8 - reader.nbits)
    if not 0 <= remaining_bits <= 7 or reader.acc:
        raise ValueError("PBC body has unexpected trailing data or nonzero padding")
    return {"header": header, "patches": patches}


def build_pbc_parsed_ir(parsed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = {
        "schema": "tamtd.pbc-parsed-fields",
        "schema_version": 1,
        "header": parsed["header"],
        "patches": [],
    }
    arrays: dict[str, np.ndarray] = {}
    for index, patch in enumerate(parsed["patches"]):
        prefix = f"patch_{index:03d}"
        names = {"mask": f"{prefix}_mask", "indices": f"{prefix}_indices"}
        arrays[names["mask"]] = np.asarray(patch["mask"], dtype=np.uint8)
        arrays[names["indices"]] = np.asarray(patch["indices"], dtype="<u2")
        metadata["patches"].append(
            {
                key: patch[key]
                for key in (
                    "channel", "x", "y", "width", "height", "palette_type",
                    "negative_max", "positive_max", "max_bitcount", "cell_size",
                    "bitcount",
                )
            }
            | {"grid_shape": list(patch["indices"].shape), "arrays": names}
        )
    return metadata, arrays


def measure_parsed_pbc_size(parsed: dict[str, Any]) -> dict[str, int]:
    metadata, arrays = build_pbc_parsed_ir(parsed)
    return {
        "metadata_bytes": len(_metadata_bytes(metadata)),
        "array_bytes": sum(array.nbytes for array in arrays.values()),
        "serialized_bytes": len(serialize_array_bundle(metadata, arrays)),
    }


def build_pbc_ir(parsed: dict[str, Any], pair) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import pbc3_ops as ops

    header = parsed["header"]
    metadata = {
        "schema": "tamtd.pbc-ir",
        "schema_version": 1,
        "codec_version": pair.pbc_class.VERSION,
        "header": header,
        "patches": [],
    }
    arrays: dict[str, np.ndarray] = {}
    for index, patch in enumerate(parsed["patches"]):
        prefix = f"patch_{index:03d}"
        mask = np.asarray(patch["mask"], dtype=np.uint8)
        palette = ops.palette_generator(
            patch["mask"],
            patch["max_bitcount"],
            patch["negative_max"],
            patch["positive_max"],
            header["positive_bias"],
        ).astype("<i2", copy=False)
        indices = np.asarray(patch["indices"], dtype="<u2")
        values = np.asarray(palette[indices], dtype="<i2")
        array_names = {
            "mask": f"{prefix}_mask",
            "palette": f"{prefix}_palette",
            "indices": f"{prefix}_indices",
            "values": f"{prefix}_values",
        }
        arrays.update(
            {
                array_names["mask"]: mask,
                array_names["palette"]: palette,
                array_names["indices"]: indices,
                array_names["values"]: values,
            }
        )
        metadata["patches"].append(
            {
                "channel": patch["channel"],
                "x": patch["x"],
                "y": patch["y"],
                "width": patch["width"],
                "height": patch["height"],
                "palette_type": patch["palette_type"],
                "negative_max": patch["negative_max"],
                "positive_max": patch["positive_max"],
                "max_bitcount": patch["max_bitcount"],
                "bitcount": patch["bitcount"],
                "cell_size": patch["cell_size"],
                "grid_shape": list(indices.shape),
                "arrays": array_names,
            }
        )
    return metadata, arrays


def rebuild_pbc_store_stream(metadata: dict[str, Any], arrays: dict[str, np.ndarray], pair) -> bytes:
    import pbc3_stream as stream
    import pbc3_types

    header = metadata["header"]
    writer = pbc3_types.BitWriter()
    warmup = header["warmup"]
    stream.write_header(
        writer,
        header["width"],
        header["height"],
        header["original_width"],
        header["original_height"],
        header["downsampled"],
        pair.pbc_class.COLOR_SPACES[header["color_space"]],
        header["channels"],
        header["channel_bits"],
        header["positive_bias"],
        header["has_alpha"],
        len(metadata["patches"]),
        header["base_values"],
        warmup=(warmup["width"], warmup["height"], warmup["split"]) if warmup else None,
    )
    for patch in metadata["patches"]:
        names = patch["arrays"]
        stream.write_patch(
            writer,
            {
                "channel": patch["channel"],
                "x": patch["x"],
                "y": patch["y"],
                "w": patch["width"],
                "h": patch["height"],
                "mask": arrays[names["mask"]].tolist(),
                "neg": patch["negative_max"],
                "pos": patch["positive_max"],
                "max_bitcount": patch["max_bitcount"],
                "cell_size": patch["cell_size"],
                "bitcount": patch["bitcount"],
                "indices": arrays[names["indices"]],
            },
            header["channel_bits"],
        )
    body = writer.finish()
    return pair.pbc_class.MAGIC + bytes(
        [pair.pbc_class.VERSION, pair.pbc_class.ENTROPY_STORE]
    ) + body


def render_pbc_ir(metadata: dict[str, Any], arrays: dict[str, np.ndarray], pair) -> Image.Image:
    import pbc3_ops as ops

    header = metadata["header"]
    pbc_module = __import__("PBC3")
    canvas = pbc_module._canvas_from_bases(
        (header["height"], header["width"], header["channels"]), header["base_values"]
    )
    for index, patch in enumerate(metadata["patches"]):
        warmup = header["warmup"]
        if warmup and index == warmup["split"]:
            canvas = pair.pbc_class._resize_canvas(canvas, warmup["width"], warmup["height"])
        values = arrays[patch["arrays"]["values"]]
        delta = ops.signed_resample(values, patch["height"], patch["width"])
        canvas[
            patch["y"]:patch["y"] + patch["height"],
            patch["x"]:patch["x"] + patch["width"],
            patch["channel"],
        ] += delta
    image = pair.pbc_class._canvas_to_image(canvas, header["color_space"], header["has_alpha"])
    if header["downsampled"] and image.size != (
        header["original_width"], header["original_height"]
    ):
        image = image.resize(
            (header["original_width"], header["original_height"]),
            pair.pbc_class.RESAMPLE_FILTER,
            reducing_gap=pair.pbc_class.RESAMPLE_REDUCING_GAP,
        )
    return image.convert("RGB")


def read_jpeg_coefficients(path: str | Path):
    try:
        import jpegio
    except ImportError as error:
        raise JPEGCoefficientDependencyError(
            "JPEG coefficient extraction needs pyjpegio; install TaMtD with `pip install -e '.[staged-jpeg]'`."
        ) from error

    return jpegio.read(str(path), read_spatial=False)


def build_jpeg_ir(jpeg) -> tuple[dict[str, Any], dict[str, np.ndarray], int]:
    arrays: dict[str, np.ndarray] = {}
    components = []
    native_array_bytes = 0
    for index, (coefficients, info) in enumerate(zip(jpeg.coef_arrays, jpeg.comp_info)):
        coefficient_array = np.asarray(coefficients)
        native_array_bytes += coefficient_array.nbytes
        if coefficient_array.ndim != 2 or any(dimension % 8 for dimension in coefficient_array.shape):
            raise ValueError(f"unexpected JPEG coefficient array shape {coefficient_array.shape}")
        if coefficient_array.size and (
            coefficient_array.min() < np.iinfo(np.int16).min
            or coefficient_array.max() > np.iinfo(np.int16).max
        ):
            raise ValueError("JPEG coefficient does not fit the canonical int16 representation")
        block_height, block_width = coefficient_array.shape[0] // 8, coefficient_array.shape[1] // 8
        block_coefficients = coefficient_array.reshape(
            block_height, 8, block_width, 8
        ).transpose(0, 2, 1, 3).astype("<i2", copy=True)
        coefficient_name = f"component_{index}_coefficients"
        arrays[coefficient_name] = block_coefficients
        components.append(
            {
                "component_id": int(info.component_id),
                "h_sampling_factor": int(info.h_samp_factor),
                "v_sampling_factor": int(info.v_samp_factor),
                "width_in_blocks": int(info.width_in_blocks),
                "height_in_blocks": int(info.height_in_blocks),
                "downsampled_width": int(info.downsampled_width),
                "downsampled_height": int(info.downsampled_height),
                "quantization_table_id": int(info.quant_tbl_no),
                "dc_huffman_table_id": int(info.dc_tbl_no),
                "ac_huffman_table_id": int(info.ac_tbl_no),
                "coefficient_array": coefficient_name,
                "coefficient_shape_blocks_uv": list(block_coefficients.shape),
            }
        )
    quant_tables = []
    for index, table in enumerate(jpeg.quant_tables):
        if table is None:
            continue
        table_array = np.asarray(table)
        native_array_bytes += table_array.nbytes
        table_name = f"quantization_table_{index}"
        arrays[table_name] = table_array.astype("<u2", copy=True)
        quant_tables.append({"table_id": index, "array": table_name, "order": "natural row-major"})
    metadata = {
        "schema": "tamtd.jpeg-dct-ir",
        "schema_version": 1,
        "image_width": int(jpeg.image_width),
        "image_height": int(jpeg.image_height),
        "data_precision": int(jpeg.data_precision),
        "jpeg_color_space": int(jpeg.jpeg_color_space),
        "image_color_space": int(jpeg.image_color_space),
        "progressive": bool(jpeg.progressive_mode),
        "arithmetic_coding": bool(jpeg.arith_code),
        "restart_interval": int(jpeg.restart_interval),
        "adobe_transform": int(jpeg.adobe_transform),
        "has_adobe_marker": bool(jpeg.saw_adobe_marker),
        "has_jfif_marker": bool(jpeg.saw_jfif_marker),
        "coefficient_order": "natural 8x8 DCT order, arranged block_y, block_x, u, v",
        "coefficient_dtype": "int16",
        "components": components,
        "quantization_tables": quant_tables,
    }
    return metadata, arrays, native_array_bytes
