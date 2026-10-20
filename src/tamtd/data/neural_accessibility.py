from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..codecs.jpeg import JPEGCodec
from ..codecs.pbc import PBCCodecPair
from ..staged_decode import (
    build_jpeg_ir,
    build_pbc_ir,
    build_pbc_parsed_ir,
    parse_pbc_store_body,
    read_jpeg_coefficients,
    rebuild_pbc_store_stream,
    serialize_array_bundle,
)
from .imagenette_accessibility import check_representation_alignment, load_prepared_rgb, prepare_imagenette_cache, read_manifest, sha256


PBC_PRESETS = ("compression", "balanced", "quality")
JPEG_QUALITIES = (10, 20, 60, 80)
MAX_PBC_PATCHES = 64


def _metadata_size(metadata: dict) -> int:
    return len(json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _remap_metadata(metadata: dict, names: dict[str, str]) -> dict:
    result = copy.deepcopy(metadata)

    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            return names.get(value, value)
        return value

    return visit(result)


def _write_stream(handle, data: bytes) -> dict:
    offset = handle.tell()
    handle.write(data)
    return {"offset": offset, "length": len(data), "sha256": sha256(data)}


def _pool_1d(values: np.ndarray, bins: int = 64) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.asarray([part.mean() if len(part) else 0.0 for part in np.array_split(values, bins)], dtype=np.float32)


def _pool_grid(values: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
    return np.asarray(image.resize((8, 8), Image.Resampling.BOX), dtype=np.float32).reshape(-1)


def jpeg_blocks_to_channels(blocks: np.ndarray) -> np.ndarray:
    blocks = np.asarray(blocks)
    if blocks.ndim != 4 or blocks.shape[2:] != (8, 8):
        raise ValueError("JPEG DCT blocks must have shape [block_y, block_x, 8, 8]")
    height, width = blocks.shape[:2]
    return blocks.transpose(2, 3, 0, 1).reshape(64, height, width)


def _pbc_model_features(metadata: dict, arrays: dict[str, np.ndarray], canonical: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    header = metadata["header"]
    patches = metadata["patches"]
    if len(patches) > MAX_PBC_PATCHES:
        raise ValueError(f"PBC patch count exceeds the {MAX_PBC_PATCHES}-token model input: {len(patches)}")
    geometry = np.zeros((MAX_PBC_PATCHES, 24), dtype=np.float32)
    content = np.zeros((MAX_PBC_PATCHES, 144), dtype=np.float32)
    patch_mask = np.zeros(MAX_PBC_PATCHES, dtype=np.uint8)
    width = max(int(header["width"]), 1)
    height = max(int(header["height"]), 1)
    base_values = np.asarray(header["base_values"], dtype=np.float32)
    for index, patch in enumerate(patches):
        names = patch["arrays"]
        mask = arrays[names["mask"]].astype(np.float32, copy=False)
        grid = arrays[names["indices"]]
        grid_summary = _pool_grid(grid.astype(np.float32))
        mask_summary = _pool_1d(mask)
        geometry[index] = np.asarray(
            [
                patch["channel"] / max(header["channels"] - 1, 1),
                patch["x"] / width,
                patch["y"] / height,
                patch["width"] / width,
                patch["height"] / height,
                patch["cell_size"] / max(width, height),
                patch["bitcount"] / 16.0,
                patch["negative_max"] / 255.0,
                patch["positive_max"] / 255.0,
                patch["max_bitcount"] / 15.0,
                grid.shape[1] / 128.0,
                grid.shape[0] / 128.0,
                len(mask) / 1024.0,
                float(mask.mean()),
                float(grid_summary.mean()) / max((1 << int(patch["bitcount"])) - 1, 1),
                float(grid_summary.std()) / max((1 << int(patch["bitcount"])) - 1, 1),
                0.0,
                0.0,
                0.0,
                base_values[patch["channel"]] / 255.0,
                float(base_values.min()) / 255.0,
                float(base_values.max()) / 255.0,
                header["channel_bits"] / 16.0,
                float(header["positive_bias"]),
            ],
            dtype=np.float32,
        )
        patch_content = [mask_summary]
        if canonical:
            values = arrays[names["values"]].astype(np.float32, copy=False)
            palette = arrays[names["palette"]].astype(np.float32, copy=False)
            value_summary = _pool_grid(values) / 512.0
            palette_summary = np.interp(
                np.linspace(0, max(len(palette) - 1, 0), 16),
                np.arange(len(palette)),
                palette,
            ).astype(np.float32) / 512.0
            geometry[index, 16] = float(values.mean()) / 512.0
            geometry[index, 17] = float(values.std()) / 512.0
            geometry[index, 18] = len(palette) / 256.0
            patch_content.extend((value_summary, palette_summary))
        else:
            maximum = max((1 << int(patch["bitcount"])) - 1, 1)
            patch_content.extend((grid_summary / maximum, np.zeros(16, dtype=np.float32)))
        content[index] = np.concatenate(patch_content)
        patch_mask[index] = 1
    return geometry.astype(np.float16), content.astype(np.float16), patch_mask


def _create_memmap(path: Path, dtype, shape: tuple[int, ...]):
    if path.exists():
        values = np.load(path, mmap_mode="r+", allow_pickle=False)
        if values.shape != shape or values.dtype != np.dtype(dtype):
            raise ValueError(f"cache array has an incompatible shape or dtype: {path}")
        return values
    values = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    values[:] = 0
    values.flush()
    return values


def _create_full_feature_cache(cache_root: Path, rows: list[dict]):
    feature_dir = cache_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    features = {}
    for preset in PBC_PRESETS:
        for representation in ("parsed", "canonical"):
            name = f"pbc_{preset}_{representation}"
            features[name] = (
                _create_memmap(feature_dir / f"{name}_geometry.npy", np.float16, (len(rows), MAX_PBC_PATCHES, 24)),
                _create_memmap(feature_dir / f"{name}_content.npy", np.float16, (len(rows), MAX_PBC_PATCHES, 144)),
                _create_memmap(feature_dir / f"{name}_patch_mask.npy", np.uint8, (len(rows), MAX_PBC_PATCHES)),
            )
    dct = {}
    for quality in JPEG_QUALITIES:
        name = f"jpeg_q{quality}"
        dct[name] = {
            "y_coefficients": _create_memmap(feature_dir / f"{name}_y.npy", np.int16, (len(rows), 64, 32, 32)),
            "cb_coefficients": _create_memmap(feature_dir / f"{name}_cb.npy", np.int16, (len(rows), 64, 16, 16)),
            "cr_coefficients": _create_memmap(feature_dir / f"{name}_cr.npy", np.int16, (len(rows), 64, 16, 16)),
            "quantization_tables": _create_memmap(feature_dir / f"{name}_quantization.npy", np.uint16, (len(rows), 2, 8, 8)),
        }
    return features, dct


def _select_ir_verification_indices(rows: list[dict], count: int = 95, seed: int = 42) -> list[int]:
    count = min(count, len(rows))
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["split"], int(row["label"])), []).append(row)
    ordered_groups = sorted(groups)
    base, remainder = divmod(count, len(ordered_groups))
    extra_groups = set(
        sorted(
            ordered_groups,
            key=lambda key: hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).digest(),
        )[:remainder]
    )
    selected = []
    for key in ordered_groups:
        quota = base + (key in extra_groups)
        candidates = sorted(
            groups[key],
            key=lambda row: hashlib.sha256(f"{seed}:{row['image_id']}".encode()).digest(),
        )
        selected.extend(row["cache_index"] for row in candidates[:quota])
    if len(selected) < count:
        remaining = [row for row in rows if row["cache_index"] not in set(selected)]
        remaining.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['image_id']}".encode()).digest())
        selected.extend(row["cache_index"] for row in remaining[: count - len(selected)])
    return sorted(selected)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(data)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _recover_append_only_streams(
    manifest_path: Path,
    stream_paths: dict[str, Path],
    stream_names: list[str],
    ir_dir: Path | None = None,
    required_ir_indices: set[int] | None = None,
) -> list[dict]:
    manifest_bytes = manifest_path.read_bytes() if manifest_path.exists() else b""
    decoded_rows = []
    for line in manifest_bytes.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        try:
            decoded_rows.append(json.loads(line))
        except json.JSONDecodeError:
            break

    endpoints = {name: 0 for name in stream_names}
    valid_rows = []
    for row in decoded_rows:
        index = len(valid_rows)
        if row.get("cache_index") != index or set(row.get("streams", {})) != set(stream_names):
            break
        if required_ir_indices and index in required_ir_indices:
            bundle_path = ir_dir / f"{index:05d}.npz" if ir_dir is not None else None
            if bundle_path is None or not bundle_path.is_file():
                break
        next_endpoints = endpoints.copy()
        valid = True
        for name in stream_names:
            reference = row["streams"][name]
            offset = int(reference["offset"])
            length = int(reference["length"])
            if offset != endpoints[name] or length < 0:
                valid = False
                break
            path = stream_paths[name]
            if not path.is_file():
                valid = False
                break
            with path.open("rb") as handle:
                handle.seek(offset)
                payload = handle.read(length)
            if len(payload) != length or sha256(payload) != reference["sha256"]:
                valid = False
                break
            next_endpoints[name] = offset + length
        if not valid:
            break
        endpoints = next_endpoints
        valid_rows.append(row)

    committed_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in valid_rows
    )
    if manifest_bytes != committed_bytes:
        _atomic_write(manifest_path, committed_bytes)
    for name, path in stream_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("r+b") if path.exists() else path.open("w+b") as handle:
            handle.truncate(endpoints[name])
            handle.flush()
            os.fsync(handle.fileno())
    return valid_rows


def _clean_ir_bundles(ir_dir: Path, keep_indices: set[int], committed_count: int) -> None:
    for path in ir_dir.glob("*.npz"):
        try:
            index = int(path.stem)
        except ValueError:
            continue
        if index >= committed_count or index not in keep_indices:
            path.unlink()


def build_neural_accessibility_cache(
    dataset_root: str | Path,
    cache_root: str | Path,
    pbc_root: str | Path | None = None,
    seed: int = 42,
    validation_fraction: float = 0.1,
    limit: int | None = None,
    max_new_samples: int | None = None,
) -> dict:
    cache_root = Path(cache_root).resolve()
    rows, source_root = prepare_imagenette_cache(
        dataset_root, cache_root, seed=seed, validation_fraction=validation_fraction, image_size=256
    )
    if limit is not None:
        rows = rows[:limit]
        for index, row in enumerate(rows):
            row["cache_index"] = index

    representation_dir = cache_root / "ir"
    stream_dir = cache_root / "streams"
    temp_dir = cache_root / "jpeg_temp"
    representation_dir.mkdir(parents=True, exist_ok=True)
    stream_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / ("codec_manifest.jsonl" if limit is None else f"codec_manifest_first_{limit}.jsonl")
    ir_pattern = "{index:05d}.npz"
    stream_names = [
        *(f"pbc_{preset}_{entropy}" for preset in PBC_PRESETS for entropy in ("lzma", "store")),
        *(f"jpeg_q{quality}" for quality in JPEG_QUALITIES),
    ]
    keep_ir_indices = set(_select_ir_verification_indices(rows, seed=seed))
    subset_path = representation_dir / (
        "verification_subset.json" if limit is None else f"verification_subset_first_{limit}.json"
    )
    subset = {
        "count": len(keep_ir_indices),
        "selection": "deterministic hash-ranked samples stratified across split and class",
        "samples": [
            {"cache_index": index, "image_id": rows[index]["image_id"], "split": rows[index]["split"], "label": int(rows[index]["label"])}
            for index in sorted(keep_ir_indices)
        ],
    }
    subset_bytes = (json.dumps(subset, indent=2) + "\n").encode("utf-8")
    if subset_path.exists() and subset_path.read_bytes() != subset_bytes:
        raise ValueError(f"IR verification subset differs from the saved selection: {subset_path}")
    _atomic_write(subset_path, subset_bytes)

    stream_paths = {name: stream_dir / f"{name}.bin" for name in stream_names}
    completed = _recover_append_only_streams(
        manifest_path,
        stream_paths,
        stream_names,
        representation_dir,
        keep_ir_indices,
    )
    check_representation_alignment(rows[: len(completed)], completed)
    _clean_ir_bundles(representation_dir, keep_ir_indices, len(completed))
    if len(completed) == len(rows):
        return {"rows": len(rows), "cache_root": str(cache_root), "manifest": str(manifest_path), "resumed_from": len(completed)}

    streams = {}
    for name in stream_names:
        path = stream_paths[name]
        expected_length = path.stat().st_size
        handle = path.open("r+b") if path.exists() else path.open("w+b")
        handle.seek(expected_length)
        streams[name] = handle
    pairs = {preset: PBCCodecPair(pbc_root=pbc_root, preset=preset) for preset in PBC_PRESETS}
    jpeg_codecs = {quality: JPEGCodec(quality) for quality in JPEG_QUALITIES}
    feature_cache, dct_cache = _create_full_feature_cache(cache_root, rows) if limit is None else ({}, {})
    started = time.perf_counter()
    rows_to_build = rows[len(completed) :]
    if max_new_samples is not None:
        rows_to_build = rows_to_build[:max_new_samples]
    try:
        for row in rows_to_build:
            index = row["cache_index"]
            rgb = load_prepared_rgb(source_root / row["image_id"], 256)
            if sha256(rgb.tobytes()) != row["rgb_sha256"]:
                raise ValueError(f"prepared RGB hash mismatch for {row['image_id']}")
            image = Image.fromarray(rgb, mode="RGB")
            metadata = {"pbc": {}, "jpeg": {}}
            arrays: dict[str, np.ndarray] = {}
            references = {}
            representation_sizes = {}
            body_hashes = {}

            for preset, pair in pairs.items():
                store, forced = pair.encode_pair(image)
                store_body = pair.unpack_body(store.data)
                lzma_body = pair.unpack_body(forced.data)
                if store_body != lzma_body:
                    raise ValueError(f"PBC entropy wrappers changed the body for {row['image_id']} ({preset})")
                body_hashes[preset] = sha256(store_body)
                references[f"pbc_{preset}_lzma"] = _write_stream(streams[f"pbc_{preset}_lzma"], forced.data)
                references[f"pbc_{preset}_store"] = _write_stream(streams[f"pbc_{preset}_store"], store.data)

                parsed = parse_pbc_store_body(store_body, pair)
                parsed_metadata, parsed_arrays = build_pbc_parsed_ir(parsed)
                if rebuild_pbc_store_stream(parsed_metadata, parsed_arrays, pair) != store.data:
                    raise ValueError(f"parsed PBC fields do not rebuild the source stream for {row['image_id']} ({preset})")
                canonical_metadata, canonical_arrays = build_pbc_ir(parsed, pair)
                for patch_index, patch in enumerate(canonical_metadata["patches"]):
                    names = patch["arrays"]
                    if not np.array_equal(
                        canonical_arrays[names["values"]],
                        canonical_arrays[names["palette"]][canonical_arrays[names["indices"]]],
                    ):
                        raise ValueError(f"canonical PBC residuals do not match their indices for {row['image_id']}")
                if limit is None:
                    for representation, model_metadata, model_arrays, canonical in (
                        ("parsed", parsed_metadata, parsed_arrays, False),
                        ("canonical", canonical_metadata, canonical_arrays, True),
                    ):
                        feature_values = _pbc_model_features(model_metadata, model_arrays, canonical)
                        targets = feature_cache[f"pbc_{preset}_{representation}"]
                        for target, values in zip(targets, feature_values):
                            target[index] = values
                p2_names = {name: f"pbc_{preset}_p2_{name}" for name in parsed_arrays}
                p3_names = {name: f"pbc_{preset}_p3_{name}" for name in canonical_arrays}
                arrays.update({p2_names[name]: value for name, value in parsed_arrays.items()})
                arrays.update({p3_names[name]: value for name, value in canonical_arrays.items()})
                parsed_metadata = _remap_metadata(parsed_metadata, p2_names)
                canonical_metadata = _remap_metadata(canonical_metadata, p3_names)
                metadata["pbc"][preset] = {"parsed": parsed_metadata, "canonical": canonical_metadata}
                representation_sizes[f"pbc_{preset}_parsed"] = _metadata_size(parsed_metadata) + sum(
                    array.nbytes for array in parsed_arrays.values()
                )
                representation_sizes[f"pbc_{preset}_canonical"] = _metadata_size(canonical_metadata) + sum(
                    array.nbytes for array in canonical_arrays.values()
                )

            for quality, codec in jpeg_codecs.items():
                condition = f"jpeg_q{quality}"
                encoded = codec.encode(image).data
                references[condition] = _write_stream(streams[condition], encoded)
                with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".jpg", delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                    temporary.write(encoded)
                try:
                    jpeg = read_jpeg_coefficients(temporary_path)
                    jpeg_metadata, jpeg_arrays, _ = build_jpeg_ir(jpeg)
                finally:
                    temporary_path.unlink(missing_ok=True)
                expected_shapes = [(32, 32, 8, 8), (16, 16, 8, 8), (16, 16, 8, 8)]
                actual_shapes = [tuple(jpeg_arrays[item["coefficient_array"]].shape) for item in jpeg_metadata["components"]]
                if actual_shapes != expected_shapes or len(jpeg_metadata["components"]) != 3:
                    raise ValueError(f"unexpected {condition} DCT component layout: {actual_shapes}")
                if limit is None:
                    condition_cache = dct_cache[condition]
                    for component, feature_name in zip(
                        jpeg_metadata["components"], ("y_coefficients", "cb_coefficients", "cr_coefficients")
                    ):
                        block_array = jpeg_arrays[component["coefficient_array"]]
                        condition_cache[feature_name][index] = jpeg_blocks_to_channels(block_array)
                    quantization = condition_cache["quantization_tables"]
                    for table in jpeg_metadata["quantization_tables"]:
                        quantization[index, table["table_id"]] = jpeg_arrays[table["array"]]
                names = {name: f"{condition}_{name}" for name in jpeg_arrays}
                arrays.update({names[name]: value for name, value in jpeg_arrays.items()})
                jpeg_metadata = _remap_metadata(jpeg_metadata, names)
                metadata["jpeg"][condition] = jpeg_metadata
                representation_sizes[f"{condition}_dct"] = _metadata_size(jpeg_metadata) + sum(
                    array.nbytes for array in jpeg_arrays.values()
                )

            row_result = {
                "image_id": row["image_id"],
                "label": int(row["label"]),
                "split": row["split"],
                "cache_index": index,
                "rgb_sha256": row["rgb_sha256"],
                "body_sha256": body_hashes,
                "streams": references,
                "representation_sizes": representation_sizes,
            }
            if index in keep_ir_indices:
                bundle = serialize_array_bundle(metadata, arrays)
                _atomic_write(representation_dir / ir_pattern.format(index=index), bundle)
            for handle in streams.values():
                handle.flush()
            for name, reference in references.items():
                handle = streams[name]
                handle.seek(reference["offset"])
                payload = handle.read(reference["length"])
                if len(payload) != reference["length"] or sha256(payload) != reference["sha256"]:
                    raise IOError(f"stream verification failed for {name}, sample {index}")
                handle.seek(0, os.SEEK_END)
            for feature_arrays in feature_cache.values():
                for array in feature_arrays:
                    array.flush()
            for condition_cache in dct_cache.values():
                for array in condition_cache.values():
                    array.flush()
            with manifest_path.open("ab") as handle_manifest:
                handle_manifest.write((json.dumps(row_result, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
                handle_manifest.flush()
                os.fsync(handle_manifest.fileno())
            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                elapsed = time.perf_counter() - started
                print(f"prepared {index + 1}/{len(rows)} samples ({elapsed / (index + 1):.2f}s/sample)", flush=True)
    finally:
        for handle in streams.values():
            handle.close()

    generated_rows = read_manifest(manifest_path)
    check_representation_alignment(rows[: len(generated_rows)], generated_rows)
    return {
        "rows": len(generated_rows),
        "complete": len(generated_rows) == len(rows),
        "ir_verification_bundles": len(keep_ir_indices),
        "cache_root": str(cache_root),
        "manifest": str(manifest_path),
        "resumed_from": len(completed),
    }
