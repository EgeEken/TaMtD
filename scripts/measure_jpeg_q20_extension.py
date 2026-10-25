import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codecs.jpeg import JPEGCodec
from tamtd.data.transforms import prepare_image
from tamtd.runtime import results_root, temp_root
from tamtd.staged_decode import build_jpeg_ir, read_jpeg_coefficients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path,
        default=results_root() / "Codec Benchmark" / "sources",
    )
    parser.add_argument(
        "--output", type=Path,
        default=results_root() / "Neural Accessibility" / "jpeg_q20_timing_extension.json",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    sources = sorted(args.source_dir.glob("*"), key=lambda path: (path.name.casefold(), path.name))
    sources = [path for path in sources if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    if not sources:
        raise FileNotFoundError(f"no Phase 1 source images found in {args.source_dir}")
    codec = JPEGCodec(20)
    extract_samples_ms = []
    build_samples_ms = []
    decode_samples_ms = []
    temporary_root = temp_root()
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root, prefix="jpeg-q20-") as temp_dir:
        temp_dir = Path(temp_dir)
        for source in sources:
            with Image.open(source) as image:
                prepared, _ = prepare_image(ImageOps.exif_transpose(image), 256)
            encoded = codec.encode(prepared).data
            jpeg_path = temp_dir / f"{source.stem}.jpg"
            jpeg_path.write_bytes(encoded)
            for _ in range(args.warmups):
                coefficients = read_jpeg_coefficients(jpeg_path)
            extract_elapsed = []
            for _ in range(args.repeats):
                start = time.perf_counter_ns()
                coefficients = read_jpeg_coefficients(jpeg_path)
                extract_elapsed.append((time.perf_counter_ns() - start) / 1_000_000)
            extract_samples_ms.append(statistics.median(extract_elapsed))
            for _ in range(args.warmups):
                build_jpeg_ir(coefficients)
            build_elapsed = []
            for _ in range(args.repeats):
                start = time.perf_counter_ns()
                build_jpeg_ir(coefficients)
                build_elapsed.append((time.perf_counter_ns() - start) / 1_000_000)
            build_samples_ms.append(statistics.median(build_elapsed))
            for _ in range(args.warmups):
                with Image.open(jpeg_path) as image:
                    image.convert("RGB").copy()
            decode_elapsed = []
            for _ in range(args.repeats):
                start = time.perf_counter_ns()
                with Image.open(jpeg_path) as image:
                    image.convert("RGB").copy()
                decode_elapsed.append((time.perf_counter_ns() - start) / 1_000_000)
            decode_samples_ms.append(statistics.median(decode_elapsed))
    samples_ms = [extract + build for extract, build in zip(extract_samples_ms, build_samples_ms)]
    payload = {
        "name": "targeted q20 JPEG file-to-DCT timing extension",
        "phase2_matrix_repeated": False,
        "dataset": "same Phase 1 local PBC training_set natural-image calibration sources",
        "source_count": len(sources),
        "image_size": [256, 256],
        "jpeg_quality": 20,
        "jpeg_encoder": "Pillow JPEG / libjpeg-turbo, optimize=false, progressive=false, subsampling=4:2:0",
        "stage": "same Phase 2 boundaries: pyjpegio file-to-coefficients plus separately timed canonical tamtd.jpeg-dct-ir array construction",
        "warmups_per_source": args.warmups,
        "repeats_per_source": args.repeats,
        "sample_median_ms": [float(value) for value in samples_ms],
        "summary_ms": {
            "mean": float(np.mean(samples_ms)),
            "median": float(np.median(samples_ms)),
            "p95": float(np.percentile(samples_ms, 95)),
        },
        "file_to_coefficients_summary_ms": {
            "mean": float(np.mean(extract_samples_ms)),
            "median": float(np.median(extract_samples_ms)),
            "p95": float(np.percentile(extract_samples_ms, 95)),
        },
        "canonical_ir_build_summary_ms": {
            "mean": float(np.mean(build_samples_ms)),
            "median": float(np.median(build_samples_ms)),
            "p95": float(np.percentile(build_samples_ms, 95)),
        },
        "full_rgb_decode_summary_ms": {
            "mean": float(np.mean(decode_samples_ms)),
            "median": float(np.median(decode_samples_ms)),
            "p95": float(np.percentile(decode_samples_ms, 95)),
        },
        "file_to_coefficients_sample_median_ms": [float(value) for value in extract_samples_ms],
        "canonical_ir_build_sample_median_ms": [float(value) for value in build_samples_ms],
        "full_rgb_decode_sample_median_ms": [float(value) for value in decode_samples_ms],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"sample_median_ms"}}, indent=2))


if __name__ == "__main__":
    main()
