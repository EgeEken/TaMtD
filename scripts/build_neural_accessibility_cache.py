import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.data.neural_accessibility import build_neural_accessibility_cache
from tamtd.runtime import cache_root, data_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=data_root() / "raw" / "imagenette")
    parser.add_argument(
        "--cache-root", type=Path,
        default=cache_root() / "neural_accessibility" / "imagenette_320",
    )
    parser.add_argument("--pbc-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-samples", type=int, default=None)
    args = parser.parse_args()
    result = build_neural_accessibility_cache(
        args.dataset_root,
        args.cache_root,
        args.pbc_root,
        args.seed,
        args.validation_fraction,
        args.limit,
        args.max_new_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
