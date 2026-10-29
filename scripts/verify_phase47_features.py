from __future__ import annotations

import hashlib
import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codec_stage_ir import verify_stream_alignment
from tamtd.phase47.decoder_bindings import CodecLibraries
from scripts.extract_phase47_stages import _read_jsonl, _read_payload, _test_indices


CONDITIONS = ("webp_q30", "webp_q75", "avif_q40", "avif_q60")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=Path, default=os.environ.get("TAMTD_PHASE47_BUILD_DIR"), required=os.environ.get("TAMTD_PHASE47_BUILD_DIR") is None)
    args = parser.parse_args()
    cache = PROJECT_ROOT / "data/cache/neural_accessibility/imagenette_320"
    stream_root = cache / "phase4_streams"
    feature_root = cache / "phase47/features"
    base_rows = _read_jsonl(cache / "manifest.jsonl")
    indices = _test_indices(base_rows, 8)
    libraries = CodecLibraries(args.build_root)
    verified = {}
    for condition in CONDITIONS:
        manifest_path = stream_root / f"{condition}.jsonl"
        stream_path = stream_root / f"{condition}.bin"
        stream_rows = _read_jsonl(manifest_path)
        verify_stream_alignment(base_rows, stream_rows, stream_path)
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        complete = json.loads((feature_root / f"{condition}.complete.json").read_text(encoding="utf-8"))
        expected_shape = [len(base_rows), 256, 62]
        if (
            complete.get("completed") is not True
            or complete.get("manifest_sha256") != manifest_hash
            or complete.get("samples") != len(base_rows)
            or complete.get("shape") != expected_shape
        ):
            raise ValueError(f"completed cache provenance mismatch: {condition}")
        features = np.load(feature_root / f"{condition}.npy", mmap_mode="r", allow_pickle=False)
        if features.shape != (len(base_rows), 256, 62) or features.dtype != np.float32:
            raise ValueError(f"unexpected cached feature schema: {condition}")
        decode = libraries.decode_webp_coefficients if condition.startswith("webp") else libraries.decode_avif_stage
        with stream_path.open("rb") as handle:
            for index in indices:
                reference = stream_rows[index]["stream"]
                payload = _read_payload(handle, reference, reference["sha256"])
                reconstructed, _, _ = decode(payload)
                if not np.array_equal(reconstructed, features[index]):
                    difference = np.abs(reconstructed - features[index])
                    first_changed = np.argwhere(difference > 0)[0]
                    token, field = (int(value) for value in first_changed)
                    raise ValueError(
                        f"deterministic feature mismatch: {condition} sample {index}; "
                        f"changed={np.count_nonzero(difference)}, max_abs={difference.max()}, "
                        f"first=({token},{field}) decoded={reconstructed[token, field]}, "
                        f"cached={features[index, token, field]}"
                    )
        verified[condition] = {"samples_checked": len(indices), "manifest_sha256": manifest_hash}
        print(f"verified {condition}: {len(indices)} deterministic held-out feature rows", flush=True)
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()
