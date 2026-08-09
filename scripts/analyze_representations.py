import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codecs import PBCCodecPair
from tamtd.data.cache import load_manifest


def decode(row, data, pair):
    if row["codec"].startswith("jpeg_"):
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if row["codec"].startswith("pbc_"):
        return np.asarray(pair.decode(data), dtype=np.uint8)
    return np.frombuffer(data, dtype=np.uint8).reshape(32, 32, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--codecs", required=True)
    parser.add_argument("--pbc-root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    wanted = [codec.strip() for codec in args.codecs.split(",") if codec.strip()]
    raw_rows = {row["source_image_id"]: row for row in rows if row["codec"] == "raw_rgb"}
    root = Path(args.manifest).parent
    references = {
        source_id: np.frombuffer((root / row["encoded_path"]).read_bytes(), dtype=np.uint8).reshape(32, 32, 3).astype(np.float32)
        for source_id, row in raw_rows.items()
    }
    output = {}
    pairs = {}
    for codec in wanted:
        selected = [row for row in rows if row["codec"] == codec]
        if not selected:
            raise ValueError(f"manifest has no rows for {codec}")
        lengths = []
        mses = []
        valid_jpeg = 0
        for row in selected:
            data = (root / row["encoded_path"]).read_bytes()
            reference_array = references[row["source_image_id"]]
            if row["codec"].startswith("pbc_") and row["codec_metadata"].get("pbc_mse") is not None:
                mse = float(row["codec_metadata"]["pbc_mse"])
            else:
                pair = None
                if row["codec"].startswith("pbc_"):
                    preset = row["codec_metadata"].get("pbc_preset", "balanced")
                    if preset not in pairs:
                        pairs[preset] = PBCCodecPair(pbc_root=args.pbc_root, preset=preset)
                    pair = pairs[preset]
                decoded = decode(row, data, pair).astype(np.float32)
                mse = float(np.mean((decoded - reference_array) ** 2))
            lengths.append(len(data))
            mses.append(mse)
            if codec.startswith("jpeg_"):
                valid_jpeg += 1
        lengths_array = np.asarray(lengths, dtype=np.float64)
        mses_array = np.asarray(mses, dtype=np.float64)
        mean_mse = float(mses_array.mean())
        output[codec] = {
            "count": len(selected),
            "mean_bytes": float(lengths_array.mean()),
            "median_bytes": float(np.median(lengths_array)),
            "p95_bytes": float(np.percentile(lengths_array, 95)),
            "p99_bytes": float(np.percentile(lengths_array, 99)),
            "max_bytes": int(lengths_array.max()),
            "mean_compression_ratio": float(np.mean(3072.0 / lengths_array)),
            "ratio_of_mean_bytes": float(3072.0 / lengths_array.mean()),
            "mean_mse": mean_mse,
            "mean_psnr_db": float(10.0 * math.log10((255.0**2) / mean_mse)) if mean_mse > 0 else float("inf"),
            "valid_jpeg_count": valid_jpeg if codec.startswith("jpeg_") else None,
            "codec_metadata": selected[0].get("codec_metadata", {}),
        }
    text = json.dumps(output, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
