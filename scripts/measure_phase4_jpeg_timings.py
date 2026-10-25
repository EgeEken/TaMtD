import argparse
import importlib.metadata
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version, features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codecs.jpeg import JPEGCodec
from tamtd.runtime import results_root, temp_root
from tamtd.staged_decode import build_jpeg_ir, read_jpeg_coefficients


def _percentiles(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _pyjpegio_version() -> str | None:
    distributions = importlib.metadata.packages_distributions().get("jpegio", [])
    for name in distributions:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def _measure(args) -> dict:
    source_dir = args.source_dir.resolve()
    manifest_path = source_dir / "manifest.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(manifest) != 95:
        raise ValueError(f"expected the fixed 95-image Phase 2 calibration set, found {len(manifest)}")
    sources = [source_dir / row["source_png"] for row in manifest]
    if any(not path.is_file() for path in sources):
        raise FileNotFoundError("Phase 1 prepared source PNG is missing")
    samples = {}
    temporary_root = temp_root()
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root, prefix="phase4-jpeg-timing-") as temporary:
        temporary = Path(temporary)
        for quality in (10, 20, 60, 80):
            codec = JPEGCodec(quality)
            extraction_ms = []
            ir_build_ms = []
            file_to_ir_ms = []
            rgb_decode_ms = []
            for index, source in enumerate(sources):
                with Image.open(source) as image:
                    rgb = image.convert("RGB").copy()
                if rgb.size != (256, 256):
                    raise ValueError(f"prepared source is not 256x256: {source}")
                encoded = codec.encode(rgb).data
                jpeg_path = temporary / f"q{quality}_{index:03d}.jpg"
                jpeg_path.write_bytes(encoded)
                for _ in range(args.warmups):
                    coefficients = read_jpeg_coefficients(jpeg_path)
                    build_jpeg_ir(coefficients)
                    with Image.open(jpeg_path) as decoded:
                        decoded.convert("RGB").copy()
                extraction_repeats = []
                build_repeats = []
                decode_repeats = []
                for _ in range(args.repeats):
                    start = time.perf_counter_ns()
                    coefficients = read_jpeg_coefficients(jpeg_path)
                    extraction_repeats.append((time.perf_counter_ns() - start) / 1_000_000)
                    start = time.perf_counter_ns()
                    build_jpeg_ir(coefficients)
                    build_repeats.append((time.perf_counter_ns() - start) / 1_000_000)
                    start = time.perf_counter_ns()
                    with Image.open(jpeg_path) as decoded:
                        decoded.convert("RGB").copy()
                    decode_repeats.append((time.perf_counter_ns() - start) / 1_000_000)
                extraction_ms.append(float(statistics.median(extraction_repeats)))
                ir_build_ms.append(float(statistics.median(build_repeats)))
                file_to_ir_ms.append(float(statistics.median(extraction_repeats) + statistics.median(build_repeats)))
                rgb_decode_ms.append(float(statistics.median(decode_repeats)))
            samples[f"jpeg_q{quality}"] = {
                "file_to_coefficients_ms": _percentiles(extraction_ms),
                "canonical_ir_build_ms": _percentiles(ir_build_ms),
                "file_to_dct_ir_ms": _percentiles(file_to_ir_ms),
                "full_rgb_decode_ms": _percentiles(rgb_decode_ms),
                "ratio_dct_to_full_decode": float(np.median(file_to_ir_ms) / np.median(rgb_decode_ms)),
                "sample_median_file_to_dct_ir_ms": file_to_ir_ms,
                "sample_median_full_rgb_decode_ms": rgb_decode_ms,
            }
            print(
                f"jpeg_q{quality}: file-to-DCT={samples[f'jpeg_q{quality}']['file_to_dct_ir_ms']['median']:.4f} ms; "
                f"RGB={samples[f'jpeg_q{quality}']['full_rgb_decode_ms']['median']:.4f} ms; "
                f"ratio={samples[f'jpeg_q{quality}']['ratio_dct_to_full_decode']:.3f}",
                flush=True,
            )
    try:
        jpeg_lib_version = features.version_feature("libjpeg_turbo")
    except ValueError:
        jpeg_lib_version = None
    try:
        jpeg_abi_version = features.version("jpg")
    except ValueError:
        jpeg_abi_version = None
    return {
        "dataset": "Phase 1/2 fixed 95-image prepared calibration set",
        "source_manifest": str(manifest_path),
        "source_count": len(sources),
        "image_size": [256, 256],
        "warmups_per_image": args.warmups,
        "repeats_per_image": args.repeats,
        "encoder": "Pillow JPEG; quality 10/20/60/80; subsampling=4:2:0; optimize=false; progressive=false",
        "stage": "file-to-coefficients via pyjpegio/jpegio + canonical tamtd.jpeg-dct-ir array construction; full decode via Pillow/libjpeg-turbo",
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "pillow": pillow_version,
            "libjpeg_turbo": jpeg_lib_version,
            "jpeg_abi": jpeg_abi_version,
            "pyjpegio": _pyjpegio_version(),
            "numpy": np.__version__,
            "cpu": platform.processor(),
        },
        "conditions": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=results_root() / "Codec Benchmark/sources")
    parser.add_argument("--output", type=Path, default=results_root() / "Phase 4 Confirmation/jpeg_timing_refresh.json")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    result = _measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "environment": result["environment"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
