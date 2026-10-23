import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codecs.jpeg import JPEGCodec
from tamtd.codecs.pbc import PBCCodecPair
from tamtd.data.accessibility_datasets import DecodedRGBImageDataset, PixelImageDataset
from tamtd.data.imagenette_accessibility import get_imagenette_source_root, read_manifest
from tamtd.models.accessibility import ByteStreamCNN, JPEGDCTNet, PBCInstructionNet, RGBResNet18, parameter_count
from tamtd.training.accessibility import ByteStreamFileDataset, IndexedDictTensorDataset, evaluate_accuracy, make_loader, train_classifier
from tamtd.training.seed import seed_everything


PBC_PRESETS = ("compression", "balanced", "quality")
JPEG_QUALITIES = (10, 60)
RGB_REFERENCE_CONDITIONS = (
    "pbc_compression", "pbc_balanced", "pbc_quality", "jpeg_q10", "jpeg_q60"
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _feature_inputs(feature_dir: Path, condition: str, representation: str) -> dict[str, torch.Tensor]:
    if condition.startswith("pbc_"):
        return {
            name: torch.from_numpy(np.load(feature_dir / f"{condition}_{representation}_{suffix}.npy", mmap_mode="r", allow_pickle=False))
            for name, suffix in (("geometry", "geometry"), ("content", "content"), ("patch_mask", "patch_mask"))
        }
    filenames = {
        "y_coefficients": f"{condition}_y.npy",
        "cb_coefficients": f"{condition}_cb.npy",
        "cr_coefficients": f"{condition}_cr.npy",
        "quantization_tables": f"{condition}_quantization.npy",
    }
    return {name: torch.from_numpy(np.load(feature_dir / filename, mmap_mode="r", allow_pickle=False)) for name, filename in filenames.items()}


def _load_phase4_stream_rows(output_root: Path, condition: str, rows: list[dict]) -> list[dict]:
    additions = read_manifest(output_root / f"{condition}.jsonl")
    if len(additions) != len(rows):
        raise ValueError(f"incomplete cross-codec cache for {condition}: {len(additions)}/{len(rows)}")
    merged = []
    for row, addition in zip(rows, additions):
        if addition["cache_index"] != row["cache_index"] or addition["image_id"] != row["image_id"] or addition["rgb_sha256"] != row["rgb_sha256"]:
            raise ValueError(f"cross-codec cache alignment mismatch for {condition} at index {row['cache_index']}")
        copy = dict(row)
        copy["streams"] = dict(row["streams"])
        copy["streams"][condition] = addition["stream"]
        merged.append(copy)
    return merged


def _settings_for(family: str, training_settings: dict) -> dict:
    return training_settings[family]


def _run_training(args, condition: str, representation: str, rows: list[dict], train_rows: list[dict], validation_rows: list[dict], test_rows: list[dict], device: torch.device) -> tuple[dict, torch.nn.Module]:
    feature_dir = args.cache_root / "features"
    stream_dir = args.cache_root / "streams"
    seed_everything(args.seed)

    if representation in {"pbc_lzma_bytes", "pbc_store_bytes", "jpeg_file_bytes", "codec_file_bytes"}:
        if representation == "pbc_lzma_bytes":
            stream_name = f"{condition}_lzma"
        elif representation == "pbc_store_bytes":
            stream_name = f"{condition}_store"
        elif representation == "jpeg_file_bytes":
            stream_name = condition
        else:
            stream_name = condition
        path = (args.phase4_cache_root / f"{condition}.bin") if representation == "codec_file_bytes" else (stream_dir / f"{stream_name}.bin")
        dataset = ByteStreamFileDataset(path, rows, stream_name)
        indices = {split: [i for i, row in enumerate(rows) if row["split"] == split] for split in ("train", "validation", "test")}
        datasets = [torch.utils.data.Subset(dataset, indices[split]) for split in ("train", "validation", "test")]
        model = ByteStreamCNN()
        family = "byte"
        model_name = "ByteStreamCNN"
    elif representation in {"pbc_canonical_ir", "jpeg_dct_ir"}:
        feature_stage = "canonical" if representation == "pbc_canonical_ir" else "dct"
        inputs = _feature_inputs(feature_dir, condition, feature_stage)
        labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
        datasets = [
            IndexedDictTensorDataset(inputs, labels, [row["cache_index"] for row in rows if row["split"] == split])
            for split in ("train", "validation", "test")
        ]
        model = PBCInstructionNet() if representation == "pbc_canonical_ir" else JPEGDCTNet()
        family = "pbc" if representation == "pbc_canonical_ir" else "jpeg_dct"
        model_name = "PBCInstructionNet" if family == "pbc" else "JPEGDCTNet"
    elif representation == "decoded_rgb_reference":
        datasets = [PixelImageDataset(args.source_root, split) for split in (train_rows, validation_rows, test_rows)]
        model = RGBResNet18()
        family = "rgb"
        model_name = "RGBResNet18(width=32, random initialization)"
    else:
        raise ValueError(f"unsupported Phase 4 representation: {representation}")

    settings = _settings_for(family, args.training_settings)
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    batch_size = settings["batch_size"]
    loaders = [make_loader(dataset, batch_size, split == "train", args.seed, workers=0, pin_memory=False) for dataset, split in zip(datasets, ("train", "validation", "test"))]
    checkpoint_base = args.checkpoint_dir / f"{condition}__{representation}__seed{args.seed}.pt"
    checkpoint = checkpoint_base
    retry = 1
    while checkpoint.exists():
        checkpoint = checkpoint_base.with_name(f"{checkpoint_base.stem}__retry{retry}{checkpoint_base.suffix}")
        retry += 1
    result = train_classifier(
        model,
        loaders[0],
        loaders[1],
        loaders[2],
        device,
        checkpoint,
        epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        optimizer_name=settings["optimizer"],
        patience=settings["patience"],
        amp=device.type == "cuda" and bool(args.training_settings.get("mixed_precision", True)),
    )
    result.pop("checkpoint", None)
    result_row = {
        "condition": condition,
        "representation": representation,
        "model_family": family if family != "rgb" else "rgb_resnet18_style",
        "model_name": model_name,
        "parameter_count": parameter_count(model),
        "seed": args.seed,
        **result,
        "checkpoint": str(checkpoint),
        "workers": 0,
        "pin_memory": False,
        "device": device.type,
        "training_settings": settings,
    }
    if representation == "decoded_rgb_reference":
        references = {}
        for codec_condition in RGB_REFERENCE_CONDITIONS:
            codec = (
                PBCCodecPair(pbc_root=args.pbc_root, preset=codec_condition.removeprefix("pbc_"))
                if codec_condition.startswith("pbc_")
                else JPEGCodec(int(codec_condition.removeprefix("jpeg_q")))
            )
            stream_name = f"{codec_condition}_lzma" if codec_condition.startswith("pbc_") else codec_condition
            references[codec_condition] = {}
            for split, split_rows in (("validation", validation_rows), ("test", test_rows)):
                decoded = DecodedRGBImageDataset(stream_dir / f"{stream_name}.bin", split_rows, stream_name, codec)
                loader = make_loader(decoded, batch_size, False, args.seed, workers=0, pin_memory=False)
                references[codec_condition][f"{split}_accuracy"] = evaluate_accuracy(model, loader, device)
                del loader, decoded
                gc.collect()
            print(f"decoded RGB {codec_condition}: val={references[codec_condition]['validation_accuracy']:.4f} test={references[codec_condition]['test_accuracy']:.4f}", flush=True)
        result_row["decoded_rgb_references"] = references

    return result_row, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "data/cache/neural_accessibility/imagenette_320")
    parser.add_argument("--phase4-cache-root", type=Path, default=PROJECT_ROOT / "data/cache/neural_accessibility/imagenette_320/phase4_streams")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/raw/imagenette")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/Phase 4 Confirmation")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--pbc-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--result-suffix")
    parser.add_argument("--eval-existing-rgb", action="store_true")
    args = parser.parse_args()
    args.cache_root = args.cache_root.resolve()
    args.phase4_cache_root = args.phase4_cache_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint_dir = (args.checkpoint_dir or args.cache_root / "checkpoints" / "phase4").resolve()
    args.source_root = get_imagenette_source_root(args.dataset_root)
    args.training_settings = json.loads((PROJECT_ROOT / "results/Neural Accessibility/training_settings.json").read_text(encoding="utf-8"))
    args.training_settings["mixed_precision"] = True

    rows = read_manifest(args.cache_root / "codec_manifest.jsonl")
    source_rows = read_manifest(args.cache_root / "manifest.jsonl")
    if [(row["image_id"], row["label"], row["split"]) for row in rows] != [(row["image_id"], row["label"], row["split"]) for row in source_rows]:
        raise ValueError("Phase 3 codec cache does not align with its source manifest")
    if args.condition in {"webp_q30", "webp_q75", "avif_q40", "avif_q60", "jxl_d6", "jxl_d3"}:
        rows = _load_phase4_stream_rows(args.phase4_cache_root, args.condition, rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    result_suffix = f"__{args.result_suffix}" if args.result_suffix else ""
    output_path = args.output_dir / "per_seed" / f"seed_{args.seed}" / f"{args.condition}__{args.representation}{result_suffix}.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite completed Phase 4 result: {output_path}")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if args.eval_existing_rgb:
        if args.condition != "original_rgb" or args.representation != "decoded_rgb_reference":
            raise ValueError("--eval-existing-rgb requires the original RGB condition")
        model = RGBResNet18()
        phase3_checkpoint = args.cache_root / "checkpoints" / "original_rgb__decoded_rgb_reference.pt"
        model.load_state_dict(torch.load(phase3_checkpoint, map_location="cpu", weights_only=True))
        model.to(device).eval()
        train_rows = [row for row in rows if row["split"] == "train"]
        references = {}
        for codec_condition in RGB_REFERENCE_CONDITIONS:
            codec = (
                PBCCodecPair(pbc_root=args.pbc_root, preset=codec_condition.removeprefix("pbc_"))
                if codec_condition.startswith("pbc_")
                else JPEGCodec(int(codec_condition.removeprefix("jpeg_q")))
            )
            stream_name = f"{codec_condition}_lzma" if codec_condition.startswith("pbc_") else codec_condition
            decoded = DecodedRGBImageDataset(args.cache_root / "streams" / f"{stream_name}.bin", validation_rows, stream_name, codec)
            loader = make_loader(decoded, args.training_settings["rgb"]["batch_size"], False, args.seed, workers=0, pin_memory=False)
            references[codec_condition] = {"validation_accuracy": evaluate_accuracy(model, loader, device)}
            del loader, decoded
            gc.collect()
        _write_json(args.output_dir / "per_seed" / "seed_42" / "decoded_rgb_validation.json", references)
        print(json.dumps({"validation_references": references}, indent=2), flush=True)
        return

    result_row, model = _run_training(args, args.condition, args.representation, rows, train_rows, validation_rows, test_rows, device)
    _write_json(output_path, result_row)
    print(json.dumps({"result": str(output_path), "test_accuracy": result_row["test_accuracy"], "best_validation_accuracy": result_row["best_validation_accuracy"]}, indent=2), flush=True)
    del model, rows, train_rows, validation_rows, test_rows, result_row
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
