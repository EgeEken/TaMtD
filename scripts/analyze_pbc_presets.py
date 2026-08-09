import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.data.cache import load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--source-bytes", type=int, default=32 * 32 * 3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = {}
    for manifest_value in args.manifest:
        manifest = Path(manifest_value)
        rows = [row for row in load_manifest(manifest) if row["codec"] == "pbc_store"]
        if not rows:
            raise ValueError(f"manifest has no pbc_store rows: {manifest}")
        lengths = np.asarray([row["encoded_length"] for row in rows], dtype=np.float64)
        bodies = np.asarray(
            [row["codec_metadata"]["raw_pbc_body_length"] for row in rows], dtype=np.float64
        )
        mses = np.asarray([row["codec_metadata"]["pbc_mse"] for row in rows], dtype=np.float64)
        psnrs = np.asarray(
            [row["codec_metadata"]["pbc_psnr_db"] for row in rows if math.isfinite(row["codec_metadata"]["pbc_psnr_db"])],
            dtype=np.float64,
        )
        preset = rows[0]["codec_metadata"].get("pbc_preset")
        output[preset or manifest.parent.name] = {
            "manifest": str(manifest.resolve()),
            "preset": preset,
            "count": int(lengths.size),
            "mean_bytes": float(lengths.mean()),
            "median_bytes": float(np.median(lengths)),
            "p95_bytes": float(np.percentile(lengths, 95)),
            "max_bytes": int(lengths.max()),
            "mean_raw_body_bytes": float(bodies.mean()),
            "mean_mse": float(mses.mean()),
            "mean_psnr_db": float(psnrs.mean()) if psnrs.size else float("inf"),
            "psnr_from_mean_mse_db": float(10.0 * math.log10((255.0**2) / mses.mean())) if mses.mean() > 0 else float("inf"),
            "mean_compression_ratio": float(np.mean(args.source_bytes / lengths)),
            "ratio_of_mean_bytes": float(args.source_bytes / lengths.mean()),
            "pbc_config": rows[0]["codec_metadata"]["pbc_config"],
        }
    text = json.dumps(output, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
