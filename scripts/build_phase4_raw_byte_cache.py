import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image, __version__ as pillow_version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codecs.avif import AVIFCodec
from tamtd.codecs.jxl import JXLCodec
from tamtd.codecs.webp import WebPCodec
from tamtd.data.imagenette_accessibility import get_imagenette_source_root, load_prepared_rgb, read_manifest
from tamtd.runtime import cache_root, data_root


CODEC_FACTORIES = {
    "webp_q30": lambda: WebPCodec(30, method=6),
    "webp_q75": lambda: WebPCodec(75, method=6),
    "avif_q40": lambda: AVIFCodec(40, speed=6, max_threads=1),
    "avif_q60": lambda: AVIFCodec(60, speed=6, max_threads=1),
    "jxl_d6": lambda: JXLCodec(6, effort=7, num_threads=1),
    "jxl_d3": lambda: JXLCodec(3, effort=7, num_threads=1),
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _recover(stream_path: Path, manifest_path: Path, rows: list[dict]) -> list[dict]:
    data = manifest_path.read_bytes() if manifest_path.exists() else b""
    valid = []
    endpoint = 0
    with stream_path.open("rb") if stream_path.exists() else open(os.devnull, "rb") as stream:
        for line in data.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                break
            index = len(valid)
            if index >= len(rows) or item.get("cache_index") != index or item.get("image_id") != rows[index]["image_id"]:
                break
            reference = item.get("stream", {})
            offset = int(reference.get("offset", -1))
            length = int(reference.get("length", -1))
            if offset != endpoint or length < 0 or not stream_path.is_file():
                break
            stream.seek(offset)
            payload = stream.read(length)
            if len(payload) != length or _digest(payload) != reference.get("sha256"):
                break
            endpoint += length
            valid.append(item)
    committed = b"".join((json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode() for item in valid)
    if data != committed:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(committed)
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    with stream_path.open("r+b") if stream_path.exists() else stream_path.open("w+b") as stream:
        stream.truncate(endpoint)
        stream.flush()
        os.fsync(stream.fileno())
    return valid


def build_condition(condition: str, cache_root: Path, dataset_root: Path, output_root: Path, limit: int | None = None) -> dict:
    if condition not in CODEC_FACTORIES:
        raise ValueError(f"unknown Phase 4 codec setting: {condition}")
    rows = read_manifest(cache_root / "manifest.jsonl")
    if limit is not None:
        rows = rows[:limit]
    source_root = get_imagenette_source_root(dataset_root)
    codec = CODEC_FACTORIES[condition]()
    output_root.mkdir(parents=True, exist_ok=True)
    stream_path = output_root / f"{condition}.bin"
    manifest_path = output_root / f"{condition}.jsonl"
    completed = _recover(stream_path, manifest_path, rows)
    if len(completed) == len(rows):
        return {"condition": condition, "samples": len(rows), "bytes": stream_path.stat().st_size, "resumed_from": len(completed)}

    with stream_path.open("ab") as stream, manifest_path.open("ab") as manifest:
        for row in rows[len(completed) :]:
            rgb = load_prepared_rgb(source_root / row["image_id"], 256)
            if _digest(rgb.tobytes()) != row["rgb_sha256"]:
                raise ValueError(f"prepared RGB hash mismatch for {row['image_id']}")
            encoded = codec.encode(Image.fromarray(rgb)).data
            reference = {"offset": stream.tell(), "length": len(encoded), "sha256": _digest(encoded)}
            stream.write(encoded)
            item = {
                "cache_index": row["cache_index"],
                "image_id": row["image_id"],
                "label": int(row["label"]),
                "split": row["split"],
                "rgb_sha256": row["rgb_sha256"],
                "stream": reference,
            }
            manifest.write((json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode())
            if (int(row["cache_index"]) + 1) % 1000 == 0 or int(row["cache_index"]) + 1 == len(rows):
                stream.flush()
                os.fsync(stream.fileno())
                manifest.flush()
                os.fsync(manifest.fileno())
                print(f"{condition}: encoded {int(row['cache_index']) + 1}/{len(rows)}", flush=True)

    return {
        "condition": condition,
        "samples": len(rows),
        "bytes": stream_path.stat().st_size,
        "resumed_from": len(completed),
        "pillow_version": pillow_version,
        "encoder_version": getattr(codec, "version", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=sorted(CODEC_FACTORIES))
    neural_cache = cache_root() / "neural_accessibility" / "imagenette_320"
    parser.add_argument("--cache-root", type=Path, default=neural_cache)
    parser.add_argument("--dataset-root", type=Path, default=data_root() / "raw" / "imagenette")
    parser.add_argument("--output-root", type=Path, default=neural_cache / "phase4_streams")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = build_condition(args.condition, args.cache_root.resolve(), args.dataset_root.resolve(), args.output_root.resolve(), args.limit)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
