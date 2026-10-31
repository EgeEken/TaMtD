from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from tamtd.data.imagenette_accessibility import read_manifest
from tamtd.data.neural_accessibility import jpeg_blocks_to_channels
from tamtd.runtime import cache_root
from tamtd.staged_decode import build_jpeg_ir, read_jpeg_coefficients

from scripts.run_phase48_extreme import _file_digest, _write_json


CONDITION = "jpeg_q1"
FEATURE_NAMES = ("y", "cb", "cr", "quantization")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _feature_shapes(samples: int) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    return {
        "y": ((samples, 64, 32, 32), np.dtype(np.int16)),
        "cb": ((samples, 64, 16, 16), np.dtype(np.int16)),
        "cr": ((samples, 64, 16, 16), np.dtype(np.int16)),
        "quantization": ((samples, 2, 8, 8), np.dtype(np.uint16)),
    }


def _json_feature_layout(samples: int) -> dict[str, dict]:
    return {
        name: {"shape": list(shape), "dtype": str(dtype)}
        for name, (shape, dtype) in _feature_shapes(samples).items()
    }


def build(args) -> dict:
    cache = args.cache_root.resolve()
    results = args.results_dir.resolve()
    cache_root_path = cache / "phase48/features"
    cache_root_path.mkdir(parents=True, exist_ok=True)
    stream_root = cache / "phase48_streams"
    stream_path = stream_root / f"{CONDITION}.bin"
    stream_manifest = stream_root / f"{CONDITION}.jsonl"
    base_manifest = cache / "manifest.jsonl"
    rows = read_manifest(base_manifest)
    stream_rows = read_manifest(stream_manifest)
    if len(rows) != 13_394 or len(stream_rows) != len(rows):
        raise ValueError("JPEG DCT cache requires all 13,394 exact Phase 3 streams")
    for index, (base, entry) in enumerate(zip(rows, stream_rows, strict=True)):
        if any(base.get(field) != entry.get(field) for field in ("cache_index", "image_id", "label", "split", "rgb_sha256")):
            raise ValueError(f"stream alignment mismatch at cache index {index}")

    output_meta = results / "jpeg_q1_dct_feature_cache.json"
    progress_path = cache_root_path / ".jpeg_q1_progress.json"
    names = {name: cache_root_path / f"jpeg_q1_{name}.npy" for name in FEATURE_NAMES}
    signature = {
        "schema": "tamtd.phase48.jpeg-dct-feature-cache.v1",
        "condition": CONDITION,
        "samples": len(rows),
        "source_manifest_sha256": _file_digest(base_manifest),
        "stream_manifest_sha256": _file_digest(stream_manifest),
        "stream_sha256": _file_digest(stream_path),
        "feature_layout": _json_feature_layout(len(rows)),
    }
    if output_meta.is_file():
        saved = json.loads(output_meta.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in signature.items()):
            raise ValueError("completed JPEG DCT feature cache does not match the current source streams")
        for name, path in names.items():
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            shape, dtype = signature["feature_layout"][name]["shape"], signature["feature_layout"][name]["dtype"]
            if values.shape != tuple(shape) or values.dtype != np.dtype(dtype) or saved["feature_sha256"][name] != _file_digest(path):
                raise ValueError(f"completed JPEG DCT feature cache failed validation: {name}")
        return saved

    start_index = 0
    if progress_path.is_file():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in signature.items()):
            raise ValueError("partial JPEG DCT cache belongs to different source streams; preserving it")
        start_index = int(state["next_index"])
        if not 0 <= start_index <= len(rows):
            raise ValueError("JPEG DCT cache resume index is invalid")
    elif any(path.exists() for path in names.values()):
        raise FileExistsError("partial JPEG DCT feature arrays exist without a resumable progress record")

    maps = {}
    for name, path in names.items():
        shape, dtype = _feature_shapes(len(rows))[name]
        maps[name] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=dtype, shape=shape)
    started = time.perf_counter()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", dir=results, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        with stream_path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            for index in range(start_index, len(rows)):
                base, entry = rows[index], stream_rows[index]
                reference = entry["stream"]
                payload = mapped[int(reference["offset"]):int(reference["offset"]) + int(reference["length"])]
                if len(payload) != int(reference["length"]) or _digest(payload) != reference["sha256"]:
                    raise ValueError(f"corrupt JPEG q1 stream at cache index {index}")
                temporary_path.write_bytes(payload)
                read_start = time.perf_counter_ns()
                jpeg = read_jpeg_coefficients(temporary_path)
                coefficient_read_ms = (time.perf_counter_ns() - read_start) / 1_000_000.0
                ir_start = time.perf_counter_ns()
                metadata, arrays, _ = build_jpeg_ir(jpeg)
                ir_build_ms = (time.perf_counter_ns() - ir_start) / 1_000_000.0
                expected_shapes = ((32, 32, 8, 8), (16, 16, 8, 8), (16, 16, 8, 8))
                components = metadata["components"]
                actual_shapes = tuple(tuple(component["coefficient_shape_blocks_uv"]) for component in components)
                if [metadata["image_width"], metadata["image_height"]] != [256, 256] or actual_shapes != expected_shapes:
                    raise ValueError(f"unexpected q1 DCT layout for {base['image_id']}: {actual_shapes}")
                adapter_start = time.perf_counter_ns()
                for component, feature in zip(components, FEATURE_NAMES[:3], strict=True):
                    block_array = arrays[component["coefficient_array"]]
                    maps[feature][index] = jpeg_blocks_to_channels(block_array)
                tables = maps["quantization"][index]
                for table in metadata["quantization_tables"]:
                    tables[int(table["table_id"])] = arrays[table["array"]]
                adapter_ms = (time.perf_counter_ns() - adapter_start) / 1_000_000.0
                if (index + 1) % 250 == 0 or index + 1 == len(rows):
                    for values in maps.values():
                        values.flush()
                    _write_json(progress_path, {**signature, "next_index": index + 1})
                    print(f"JPEG q1 DCT feature cache: {index + 1}/{len(rows)}", flush=True)
        for values in maps.values():
            values.flush()
        maps.clear()
        test_rows = [row for row in rows if row["split"] == "test"]
        selected_positions = np.linspace(0, len(test_rows) - 1, min(256, len(test_rows)), dtype=np.int64)
        timing_rows = []
        with tempfile.NamedTemporaryFile(suffix=".jpg", dir=results, delete=False) as temporary:
            timing_temp_path = Path(temporary.name)
        try:
            with stream_path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                for position in selected_positions:
                    base = test_rows[int(position)]
                    index = int(base["cache_index"])
                    reference = stream_rows[index]["stream"]
                    payload = mapped[int(reference["offset"]):int(reference["offset"]) + int(reference["length"])]
                    if _digest(payload) != reference["sha256"]:
                        raise ValueError(f"corrupt JPEG q1 timing stream at cache index {index}")
                    for repeat in range(4):
                        timing_temp_path.write_bytes(payload)
                        read_start = time.perf_counter_ns()
                        jpeg = read_jpeg_coefficients(timing_temp_path)
                        coefficient_read_ms = (time.perf_counter_ns() - read_start) / 1_000_000.0
                        ir_start = time.perf_counter_ns()
                        metadata, arrays, _ = build_jpeg_ir(jpeg)
                        ir_build_ms = (time.perf_counter_ns() - ir_start) / 1_000_000.0
                        adapter_start = time.perf_counter_ns()
                        for component in metadata["components"]:
                            jpeg_blocks_to_channels(arrays[component["coefficient_array"]])
                        for table in metadata["quantization_tables"]:
                            np.asarray(arrays[table["array"]], dtype=np.uint16)
                        adapter_ms = (time.perf_counter_ns() - adapter_start) / 1_000_000.0
                        if repeat:
                            timing_rows.append({
                                "condition": CONDITION,
                                "image_id": base["image_id"],
                                "repeat": repeat - 1,
                                "coefficient_read_ms": coefficient_read_ms,
                                "canonical_ir_build_ms": ir_build_ms,
                                "model_array_adapter_ms": adapter_ms,
                                "pre_reconstruction_decoder_plus_adapter_ms": coefficient_read_ms + ir_build_ms + adapter_ms,
                            })
        finally:
            timing_temp_path.unlink(missing_ok=True)

        feature_hashes = {name: _file_digest(path) for name, path in names.items()}
        summaries = {}
        for field in ("coefficient_read_ms", "canonical_ir_build_ms", "model_array_adapter_ms", "pre_reconstruction_decoder_plus_adapter_ms"):
            values = np.asarray([row[field] for row in timing_rows], dtype=np.float64)
            summaries[field] = {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "n": int(len(values))}
        result = {
            **signature,
            "feature_paths": {name: str(path) for name, path in names.items()},
            "feature_sha256": feature_hashes,
            "feature_total_bytes": sum(path.stat().st_size for path in names.values()),
            "feature_layout": signature["feature_layout"],
            "timing_summary": summaries,
            "timing_boundary": "pyjpegio entropy decode -> canonical int16 DCT IR -> channel-grid and quantization-table adaptation",
            "elapsed_seconds": time.perf_counter() - started,
            "complete": True,
        }
        timing_path = results / "jpeg_q1_dct_extraction_timing_per_sample.csv"
        with timing_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0]))
            writer.writeheader()
            writer.writerows(timing_rows)
        _write_json(output_meta, result)
        return result
    finally:
        for values in maps.values():
            values.flush()
            del values
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=cache_root() / "neural_accessibility/imagenette_320")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/Phase 4.8 Extreme Compression")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
