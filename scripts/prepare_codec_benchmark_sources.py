import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps, __version__ as pillow_version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.data.transforms import prepare_image


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--exclude-filename", action="append", default=[])
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"source output directory is not empty: {output}")

    files = [
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    files.sort(key=lambda path: (path.relative_to(source_root).as_posix().casefold(), path.relative_to(source_root).as_posix()))
    excluded = {name.casefold() for name in args.exclude_filename}
    files = [path for path in files if path.name.casefold() not in excluded]
    selected = files[: args.limit]
    if not selected:
        raise ValueError(f"no JPEG/PNG images found in {source_root}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(selected):
        source_bytes = path.read_bytes()
        with Image.open(path) as original:
            original_size = list(original.size)
            original_mode = original.mode
            oriented = ImageOps.exif_transpose(original)
            image, transform = prepare_image(oriented, 256)
        relative_path = f"image_{index:06d}.png"
        destination = output / relative_path
        image.save(destination, format="PNG", compress_level=9, optimize=False)
        pixels = image.tobytes()
        rows.append(
            {
                "image_id": f"image_{index:06d}",
                "source_relative_path": path.relative_to(source_root).as_posix(),
                "source_sha256": sha256(source_bytes),
                "source_size": original_size,
                "source_mode": original_mode,
                "source_png": relative_path,
                "source_png_sha256": sha256(destination.read_bytes()),
                "rgb_pixels_sha256": sha256(pixels),
                "transform": {
                    **transform,
                    "exif_transpose": True,
                    "resampling": "Pillow BICUBIC",
                },
            }
        )
    with (output / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "dataset": "local PBC training_set natural-image calibration set",
        "source_root": str(source_root),
        "selection": "lexicographically sorted relative paths (casefold, then exact); exclude listed filenames, then take first --limit files",
        "excluded_filenames": sorted(excluded),
        "requested_limit": args.limit,
        "supported_extensions": [".jpg", ".jpeg", ".png"],
        "image_count": len(rows),
        "output_size": [256, 256],
        "output_mode": "RGB",
        "crop_resize": "EXIF transpose, center crop, Pillow BICUBIC fit, centering=(0.5, 0.5)",
        "pillow_version": pillow_version,
        "source_manifest": "manifest.jsonl",
        "source_images_are_lossless_png_copies_of_preprocessed_pixels": True,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "images": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
