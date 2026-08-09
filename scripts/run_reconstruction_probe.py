import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.data.reconstruction import ReconstructionDataset
from tamtd.training.reconstruction import train_reconstruction


def _target_identity(manifest: Path, pbc_root: str, count: int) -> dict:
    store = ReconstructionDataset(manifest, "pbc_store", "test", pbc_root=pbc_root, limit=count)
    lzma = ReconstructionDataset(manifest, "pbc_lzma_forced", "test", pbc_root=pbc_root, limit=count)
    if [row["sample_id"] for row in store.rows] != [row["sample_id"] for row in lzma.rows]:
        raise ValueError("PBC STORE/LZMA test sample ordering differs")
    equal = []
    for index in range(count):
        equal.append(bool(torch.equal(store[index]["target"], lzma[index]["target"])))
    return {"samples_checked": count, "pixel_identical_targets": all(equal), "per_sample": equal}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbc-manifest", required=True)
    parser.add_argument("--jpeg-manifest", required=True)
    parser.add_argument("--pbc-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=8000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pbc_manifest = Path(args.pbc_manifest)
    jpeg_manifest = Path(args.jpeg_manifest)
    results = []
    common = {
        "train_limit": args.train_samples,
        "val_limit": args.validation_samples,
        "test_limit": args.test_samples,
        "seed": args.seed,
        "shuffle_seed": args.shuffle_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": args.device,
    }
    specs = [
        ("pbc_quality", pbc_manifest, "pbc_store", True),
        ("jpeg_q80", jpeg_manifest, "jpeg_q80", False),
        ("pbc_quality_lzma", pbc_manifest, "pbc_lzma_forced", True),
    ]
    for name, manifest, codec, is_pbc in specs:
        ordered_dir = output_root / f"{name}_ordered"
        sequence = train_reconstruction(
            manifest_path=manifest,
            codec=codec,
            model_name="sequence",
            output_dir=ordered_dir,
            pbc_root=args.pbc_root if is_pbc else None,
            **common,
        )
        results.append({"condition": f"{name}_ordered", **sequence})
        histogram = train_reconstruction(
            manifest_path=manifest,
            codec=codec,
            model_name="histogram",
            output_dir=output_root / f"{name}_histogram",
            pbc_root=args.pbc_root if is_pbc else None,
            **common,
        )
        results.append({"condition": f"{name}_histogram", **histogram})
        if name != "pbc_quality_lzma":
            shuffled = train_reconstruction(
                manifest_path=manifest,
                codec=codec,
                model_name="sequence",
                output_dir=output_root / f"{name}_shuffled",
                pbc_root=args.pbc_root if is_pbc else None,
                shuffle_bytes=True,
                **common,
            )
            results.append({"condition": f"{name}_shuffled", **shuffled})
        else:
            shuffled = train_reconstruction(
                manifest_path=manifest,
                codec=codec,
                model_name="sequence",
                output_dir=output_root / f"{name}_shuffled",
                pbc_root=args.pbc_root if is_pbc else None,
                shuffle_bytes=True,
                **common,
            )
            results.append({"condition": f"{name}_shuffled", **shuffled})
    summary = {
        "protocol": {
            "target": "codec decode -> grayscale -> 8x8 BOX downsample",
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "test_samples": args.test_samples,
            "seed": args.seed,
            "shuffle_seed": args.shuffle_seed,
        },
        "pbc_store_lzma_target_identity": _target_identity(pbc_manifest, args.pbc_root, min(32, args.test_samples)),
        "results": results,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
