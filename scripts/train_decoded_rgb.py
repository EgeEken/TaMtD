import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.training.decoded_rgb import train_decoded_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--codec", default="pbc_store")
    parser.add_argument("--pbc-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--val-offset", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--test-offset", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = train_decoded_rgb(
        manifest_path=args.manifest,
        codec=args.codec,
        pbc_root=args.pbc_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_offset=args.train_offset,
        train_limit=args.train_samples,
        val_offset=args.val_offset,
        val_limit=args.val_samples,
        test_offset=args.test_offset,
        test_limit=args.test_samples,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
