import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.training.baselines import run_baseline
from tamtd.training.train import train_bytecnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--codec", default=None)
    parser.add_argument("--model", choices=["majority", "length", "histogram", "bytecnn"], default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--train-offset", type=int, default=None)
    parser.add_argument("--val-offset", type=int, default=None)
    parser.add_argument("--test-offset", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--shuffle-bytes", action="store_true", default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    args = vars(parser.parse_args())
    config = {}
    if args["config"]:
        config = yaml.safe_load(Path(args["config"]).read_text(encoding="utf-8")) or {}
    for key, value in config.items():
        if key in args and args[key] is None:
            args[key] = value
    args.update(
        {
            "train_split": args["train_split"] or "train",
            "val_split": args["val_split"] or "test",
            "test_split": args["test_split"] or "test",
            "seed": 42 if args["seed"] is None else args["seed"],
            "epochs": 3 if args["epochs"] is None else args["epochs"],
            "batch_size": 64 if args["batch_size"] is None else args["batch_size"],
            "device": "auto" if args["device"] is None else args["device"],
            "shuffle_bytes": False if args["shuffle_bytes"] is None else args["shuffle_bytes"],
            "shuffle_seed": args["seed"] if args["shuffle_seed"] is None else args["shuffle_seed"],
        }
    )
    if not args["manifest"] or not args["codec"] or not args["model"]:
        parser.error("--manifest, --codec, and --model are required directly or in --config")
    output_dir = Path(args["output_dir"] or f"results/raw/{args['model']}_{args['codec']}_seed{args['seed']}")
    if args["model"] == "bytecnn":
        result = train_bytecnn(
            manifest_path=args["manifest"],
            codec=args["codec"],
            output_dir=output_dir,
            train_split=args["train_split"],
            val_split=args["val_split"],
            test_split=args["test_split"],
            seed=args["seed"],
            epochs=args["epochs"],
            batch_size=args["batch_size"],
            learning_rate=1e-3 if args["learning_rate"] is None else args["learning_rate"],
            weight_decay=0.0 if args["weight_decay"] is None else args["weight_decay"],
            max_length=args["max_length"],
            device=args["device"],
            train_offset=0 if args["train_offset"] is None else args["train_offset"],
            train_limit=args["train_samples"],
            val_offset=0 if args["val_offset"] is None else args["val_offset"],
            val_limit=args["val_samples"],
            test_offset=0 if args["test_offset"] is None else args["test_offset"],
            test_limit=args["test_samples"],
            shuffle_bytes=args["shuffle_bytes"],
            shuffle_seed=args["shuffle_seed"],
        )
    else:
        result = run_baseline(
            args["manifest"],
            args["codec"],
            args["train_split"],
            args["val_split"],
            args["model"],
            args["seed"],
            train_limit=args["train_samples"],
            val_limit=args["val_samples"],
            train_offset=0 if args["train_offset"] is None else args["train_offset"],
            val_offset=0 if args["val_offset"] is None else args["val_offset"],
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(
                {
                    "manifest_path": str(Path(args["manifest"]).resolve()),
                    "codec": args["codec"],
                    "model": args["model"],
                    "train_split": args["train_split"],
                    "val_split": args["val_split"],
                    "seed": args["seed"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
