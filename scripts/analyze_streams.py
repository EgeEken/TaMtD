import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.data.cache import load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    rows = load_manifest(manifest)
    output = {}
    for codec in sorted({row["codec"] for row in rows}):
        lengths = np.asarray([row["encoded_length"] for row in rows if row["codec"] == codec], dtype=np.float64)
        output[codec] = {
            "count": int(lengths.size),
            "min": int(lengths.min()),
            "p25": float(np.percentile(lengths, 25)),
            "median": float(np.percentile(lengths, 50)),
            "p75": float(np.percentile(lengths, 75)),
            "p90": float(np.percentile(lengths, 90)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
        }
        if args.max_length is not None:
            retained = np.minimum(lengths, args.max_length) / np.maximum(lengths, 1)
            output[codec].update(
                {
                    "max_length": args.max_length,
                    "truncated_samples": int(np.sum(lengths > args.max_length)),
                    "truncation_percent": float(np.mean(lengths > args.max_length) * 100.0),
                    "mean_retained_fraction": float(retained.mean()),
                }
            )
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
