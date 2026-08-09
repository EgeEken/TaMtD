import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.training.train import train_bytecnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="results/raw/overfit_sanity")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = train_bytecnn(
        manifest_path=args.manifest,
        codec="raw_rgb",
        output_dir=args.output_dir,
        train_split="train",
        val_split="train",
        test_split="train",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        device=args.device,
        train_limit=args.samples,
        val_limit=args.samples,
        test_limit=args.samples,
    )
    print(json.dumps(result, indent=2))
    if result["best_validation_accuracy"] < 0.95:
        raise SystemExit("overfit sanity check failed: best validation accuracy is below 0.95")


if __name__ == "__main__":
    main()
