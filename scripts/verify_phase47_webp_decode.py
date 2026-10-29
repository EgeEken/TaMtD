from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codec_stage_ir import verify_stream_alignment
from scripts.extract_phase47_stages import _read_jsonl, _read_payload, _test_indices


CONDITIONS = ("webp_q30", "webp_q75")


def _load_decoder(path: Path):
    library = ctypes.CDLL(str(path.resolve()))
    decode = library.WebPDecodeRGB
    decode.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    decode.restype = ctypes.POINTER(ctypes.c_uint8)
    free = library.WebPFree
    free.argtypes = [ctypes.c_void_p]
    return library, decode, free


def _decode(payload: bytes, decode, free) -> str:
    source = ctypes.create_string_buffer(payload)
    width, height = ctypes.c_int(), ctypes.c_int()
    pointer = decode(ctypes.cast(source, ctypes.POINTER(ctypes.c_uint8)), len(payload), ctypes.byref(width), ctypes.byref(height))
    if not pointer or (width.value, height.value) != (256, 256):
        raise RuntimeError("WebP full RGB decode failed or returned unexpected dimensions")
    try:
        pixels = np.ctypeslib.as_array(pointer, shape=(256 * 256 * 3,)).copy()
        return hashlib.sha256(pixels.tobytes()).hexdigest()
    finally:
        free(pointer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, default=os.environ.get("TAMTD_PHASE47_BUILD_DIR"), required=os.environ.get("TAMTD_PHASE47_BUILD_DIR") is None)
    args = parser.parse_args()
    dll_directory = os.add_dll_directory(str((args.build_root / "build-libwebp").resolve()))
    cache = PROJECT_ROOT / "data/cache/neural_accessibility/imagenette_320"
    base_rows = _read_jsonl(cache / "manifest.jsonl")
    indices = _test_indices(base_rows, 8)
    stream_root = cache / "phase4_streams"
    before_lib, before_decode, before_free = _load_decoder(args.before)
    after_lib, after_decode, after_free = _load_decoder(args.after)
    checked = 0
    for condition in CONDITIONS:
        manifest = _read_jsonl(stream_root / f"{condition}.jsonl")
        stream_path = stream_root / f"{condition}.bin"
        verify_stream_alignment(base_rows, manifest, stream_path)
        with stream_path.open("rb") as handle:
            for index in indices:
                reference = manifest[index]["stream"]
                payload = _read_payload(handle, reference, reference["sha256"])
                before = _decode(payload, before_decode, before_free)
                after = _decode(payload, after_decode, after_free)
                if before != after:
                    raise ValueError(f"full RGB decode changed after instrumentation fix: {condition} sample {index}")
                checked += 1
        print(f"verified {condition}: {len(indices)} byte-identical full RGB outputs", flush=True)
    print(json.dumps({"samples_checked": checked, "pixel_hashes_match": True}, indent=2))
    del before_lib, after_lib
    dll_directory.close()


if __name__ == "__main__":
    main()
