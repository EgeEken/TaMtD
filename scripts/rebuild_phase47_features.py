from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codec_stage_ir import FEATURE_FIELDS, SCHEMA, SCHEMA_VERSION, TOKEN_GRID, validate_ir, verify_stream_alignment
from tamtd.phase47.decoder_bindings import CodecLibraries
from tamtd.training.telemetry import collect_resource_snapshot
from scripts.extract_phase47_stages import _read_jsonl, _read_payload, _test_indices


CONDITIONS = ("webp_q30", "webp_q75")
SAMPLES = 13_394


def _check_resources() -> None:
    snapshot = collect_resource_snapshot(disk_path=Path("C:\\"))
    if snapshot.get("disk_free_bytes", 0) < 20 * 1024**3:
        raise RuntimeError("feature cache rebuild stopped by the C: free-space guard")
    available = snapshot.get("system", {}).get("memory_available_bytes", 0)
    if available < 4 * 1024**3:
        raise RuntimeError("feature cache rebuild stopped by the host-memory guard")


def _rebuild(condition: str, libraries: CodecLibraries, base_rows: list[dict], cache: Path, stream_root: Path) -> dict:
    manifest_path = stream_root / f"{condition}.jsonl"
    stream_path = stream_root / f"{condition}.bin"
    stream_rows = _read_jsonl(manifest_path)
    verify_stream_alignment(base_rows, stream_rows, stream_path)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    feature_dir = cache / "phase47" / "features"
    feature_path = feature_dir / f"{condition}.npy"
    completion_path = feature_dir / f"{condition}.complete.json"
    backup_feature = feature_dir / f"{condition}.pre_skip_init.npy"
    backup_completion = feature_dir / f"{condition}.complete.pre_skip_init.json"
    temporary_feature = feature_dir / f"{condition}.rebuild.npy"
    temporary_completion = feature_dir / f"{condition}.rebuild.complete.json"
    if not feature_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError(f"completed feature cache is missing: {condition}")
    if backup_feature.exists() or backup_completion.exists() or temporary_feature.exists() or temporary_completion.exists():
        raise FileExistsError(f"preserved or incomplete rebuild artifact already exists: {condition}")

    feature_map = np.lib.format.open_memmap(
        temporary_feature,
        mode="w+",
        dtype=np.float32,
        shape=(len(base_rows), TOKEN_GRID[0] * TOKEN_GRID[1], len(FEATURE_FIELDS)),
    )
    decode = libraries.decode_webp_coefficients if condition.startswith("webp") else libraries.decode_avif_stage
    try:
        with stream_path.open("rb") as handle:
            for index, row in enumerate(stream_rows):
                reference = row["stream"]
                payload = _read_payload(handle, reference, reference["sha256"])
                features, _, _ = decode(payload)
                validate_ir({"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "pre_reconstruction": True}, features)
                feature_map[index] = features
                if index and index % 500 == 0:
                    _check_resources()
                    print(f"{condition}: rebuilt {index}/{SAMPLES}", flush=True)
        feature_map.flush()
    finally:
        del feature_map

    rebuilt = np.load(temporary_feature, mmap_mode="r", allow_pickle=False)
    sample_indices = _test_indices(base_rows, 8)
    with stream_path.open("rb") as handle:
        for index in sample_indices:
            reference = stream_rows[index]["stream"]
            payload = _read_payload(handle, reference, reference["sha256"])
            features, _, _ = decode(payload)
            if not np.array_equal(features, rebuilt[index]):
                raise ValueError(f"rebuilt feature cache is not deterministic: {condition} sample {index}")
    del rebuilt

    versions = json.loads((PROJECT_ROOT / "patches/phase47/versions.json").read_text(encoding="utf-8"))
    patch_path = PROJECT_ROOT / "patches/phase47/libwebp-pre-reconstruction.patch"
    metadata = {
        "completed": True,
        "condition": condition,
        "dtype": "float32",
        "manifest_sha256": manifest_hash,
        "samples": len(base_rows),
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "shape": [len(base_rows), TOKEN_GRID[0] * TOKEN_GRID[1], len(FEATURE_FIELDS)],
        "stage": "pre_reconstruction_coefficients",
        "feature_rebuild": "zero-initialized VP8 skip metadata; exact cached streams",
        "libwebp_commit": versions["sources"]["libwebp"]["commit"],
        "libwebp_patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
        "decoder_bindings_sha256": hashlib.sha256((PROJECT_ROOT / "src/tamtd/phase47/decoder_bindings.py").read_bytes()).hexdigest(),
        "determinism_samples": len(sample_indices),
    }
    temporary_completion.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(feature_path, backup_feature)
    os.replace(completion_path, backup_completion)
    os.replace(temporary_feature, feature_path)
    os.replace(temporary_completion, completion_path)
    return {"condition": condition, "samples": len(base_rows), "determinism_samples": len(sample_indices), "manifest_sha256": manifest_hash}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=Path, default=os.environ.get("TAMTD_PHASE47_BUILD_DIR"), required=os.environ.get("TAMTD_PHASE47_BUILD_DIR") is None)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, required=True)
    args = parser.parse_args()
    cache = PROJECT_ROOT / "data/cache/neural_accessibility/imagenette_320"
    base_rows = _read_jsonl(cache / "manifest.jsonl")
    if len(base_rows) != SAMPLES or [row["cache_index"] for row in base_rows] != list(range(SAMPLES)):
        raise ValueError("feature rebuild requires the exact 13,394-sample manifest")
    stream_root = cache / "phase4_streams"
    libraries = CodecLibraries(args.build_root)
    _check_resources()
    rebuilt = [_rebuild(condition, libraries, base_rows, cache, stream_root) for condition in args.conditions]
    print(json.dumps(rebuilt, indent=2))


if __name__ == "__main__":
    main()
