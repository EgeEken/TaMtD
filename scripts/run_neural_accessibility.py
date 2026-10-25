import argparse
import gc
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codecs.jpeg import JPEGCodec
from tamtd.codecs.pbc import PBCCodecPair
from tamtd.data.accessibility_datasets import DecodedRGBImageDataset, PixelImageDataset
from tamtd.data.imagenette_accessibility import check_representation_alignment, get_imagenette_source_root, read_manifest
from tamtd.models.accessibility import ByteStreamCNN, JPEGDCTNet, PBCInstructionNet, RGBResNet18, parameter_count
from tamtd.neural_accessibility import build_frontier_rows
from tamtd.runtime import cache_root, data_root, default_device, results_root
from tamtd.training.accessibility import (
    ByteStreamFileDataset,
    IndexedDictTensorDataset,
    evaluate_accuracy,
    make_loader,
    run_tiny_overfit,
    train_classifier,
)
from tamtd.training.seed import seed_everything


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _feature_inputs(feature_dir: Path, prefix: str, representation: str | None = None, mmap_mode: str = "c") -> dict[str, torch.Tensor]:
    if prefix.startswith("pbc_"):
        name = f"{prefix}_{representation}"
        return {
            "geometry": torch.from_numpy(np.load(feature_dir / f"{name}_geometry.npy", mmap_mode=mmap_mode, allow_pickle=False)),
            "content": torch.from_numpy(np.load(feature_dir / f"{name}_content.npy", mmap_mode=mmap_mode, allow_pickle=False)),
            "patch_mask": torch.from_numpy(np.load(feature_dir / f"{name}_patch_mask.npy", mmap_mode=mmap_mode, allow_pickle=False)),
        }
    feature_names = {
        "y_coefficients": f"{prefix}_y.npy",
        "cb_coefficients": f"{prefix}_cb.npy",
        "cr_coefficients": f"{prefix}_cr.npy",
        "quantization_tables": f"{prefix}_quantization.npy",
    }
    return {
        name: torch.from_numpy(np.load(feature_dir / filename, mmap_mode=mmap_mode, allow_pickle=False))
        for name, filename in feature_names.items()
    }


def _condition_splits(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    if not train or not validation or not test:
        raise ValueError("all three train, validation, and test splits must be nonempty")
    train_ids = {row["image_id"] for row in train}
    validation_ids = {row["image_id"] for row in validation}
    test_ids = {row["image_id"] for row in test}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("source IDs leak across dataset splits")
    return train, validation, test


def _byte_length_stats(rows: list[dict], stream_name: str) -> dict:
    lengths = np.asarray([row["streams"][stream_name]["length"] for row in rows], dtype=np.int64)
    return {
        "sample_count": int(len(lengths)),
        "minimum_bytes": int(lengths.min()),
        "median_bytes": float(np.median(lengths)),
        "p95_bytes": float(np.percentile(lengths, 95)),
        "maximum_bytes": int(lengths.max()),
        "truncation_cap_bytes": None,
        "expected_truncated_samples": 0,
    }


def _tiny_stratified_subset(rows: list[dict], seed: int, size: int = 128) -> list[dict]:
    by_class = {}
    for row in rows:
        by_class.setdefault(int(row["label"]), []).append(row)
    classes = sorted(by_class)
    if size < len(classes) or any(len(values) < size // len(classes) for values in by_class.values()):
        raise ValueError("not enough train examples for a stratified tiny-set overfit check")
    base, remainder = divmod(size, len(classes))
    selected = []
    for index, label in enumerate(classes):
        values = sorted(by_class[label], key=lambda row: row["image_id"])
        shuffled = values.copy()
        random.Random(seed + label).shuffle(shuffled)
        selected.extend(shuffled[: base + (index < remainder)])
    return sorted(selected, key=lambda row: row["cache_index"])


def _run_overfit_checks(cache_rows: list[dict], train_rows: list[dict], cache_root: Path, output_dir: Path, args, device) -> dict:
    result_path = output_dir / "overfit_sanity.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if all(item["passed"] for item in result["models"].values()):
            return result
    selected = _tiny_stratified_subset(train_rows, args.seed)
    selected_set = {row["cache_index"] for row in selected}
    source_indices = [index for index, row in enumerate(cache_rows) if row["cache_index"] in selected_set]
    labels = torch.tensor([int(row["label"]) for row in cache_rows], dtype=torch.long)
    feature_dir = cache_root / "features"
    stream_dir = cache_root / "streams"
    settings = {"batch": 16, "max_epochs": 60, "target_accuracy": 0.9, "learning_rate": 0.001}
    results = {}

    def run(name, model, dataset, batch_size):
        loader = make_loader(dataset, batch_size, True, args.seed, workers=0)
        print(f"starting {name} tiny-set overfit ({len(dataset)} balanced train samples)", flush=True)
        outcome = run_tiny_overfit(
            model,
            loader,
            device,
            max_epochs=settings["max_epochs"],
            target_accuracy=settings["target_accuracy"],
            learning_rate=settings["learning_rate"],
            amp=device.type == "cuda",
        )
        outcome["parameter_count"] = parameter_count(model)
        outcome["samples"] = len(dataset)
        results[name] = outcome
        _write_json(result_path, {"seed": args.seed, "stratified_subset": True, "settings": settings, "models": results})
        print(f"{name} tiny-set overfit: {outcome['best_train_accuracy']:.4f} ({'PASS' if outcome['passed'] else 'FAIL'})", flush=True)

    seed_everything(args.seed)
    byte_dataset = ByteStreamFileDataset(stream_dir / "pbc_compression_lzma.bin", cache_rows, "pbc_compression_lzma")
    seed_everything(args.seed)
    run(
        "byte_cnn",
        ByteStreamCNN(),
        torch.utils.data.Subset(byte_dataset, source_indices),
        settings["batch"],
    )

    pbc_inputs = _feature_inputs(feature_dir, "pbc_compression", "parsed")
    seed_everything(args.seed)
    run(
        "pbc_instruction_set",
        PBCInstructionNet(),
        IndexedDictTensorDataset(pbc_inputs, labels, [row["cache_index"] for row in selected]),
        32,
    )

    dct_inputs = _feature_inputs(feature_dir, "jpeg_q60")
    seed_everything(args.seed)
    run(
        "jpeg_dct_cnn",
        JPEGDCTNet(),
        IndexedDictTensorDataset(dct_inputs, labels, [row["cache_index"] for row in selected]),
        settings["batch"],
    )

    rgb_dataset = PixelImageDataset(args.source_root, selected)
    seed_everything(args.seed)
    run("rgb_resnet18_style", RGBResNet18(), rgb_dataset, settings["batch"])
    if not all(item["passed"] for item in results.values()):
        raise RuntimeError("tiny-set overfit sanity check failed; full training matrix was not started")
    return {"seed": args.seed, "stratified_subset": True, "settings": settings, "models": results}


def _prepare_measurement_results(cache_rows: list[dict], train_rows: list[dict], validation_rows: list[dict], test_rows: list[dict], output_dir: Path, cache_root: Path, args, device) -> tuple[list[dict], dict]:
    feature_dir = cache_root / "features"
    stream_dir = cache_root / "streams"
    labels = torch.tensor([int(row["label"]) for row in cache_rows], dtype=torch.long)
    all_results_path = output_dir / "model_results.json"
    existing = json.loads(all_results_path.read_text(encoding="utf-8")) if all_results_path.exists() else []
    result_lookup = {(row["condition"], row["representation"]): row for row in existing}
    only_condition = tuple(args.only_condition) if args.only_condition else None
    if only_condition is not None and only_condition in result_lookup:
        raise RuntimeError(f"refusing to rerun completed result {only_condition[0]} / {only_condition[1]}")
    model_results = list(existing)
    training_settings = {
        "byte": {"epochs": 15, "batch_size": 32, "learning_rate": 0.001, "weight_decay": 0.0001, "optimizer": "adamw", "patience": 5},
        "pbc": {"epochs": 20, "batch_size": 64, "learning_rate": 0.0003, "weight_decay": 0.0001, "optimizer": "adamw", "patience": 6},
        "jpeg_dct": {"epochs": 20, "batch_size": 32, "learning_rate": 0.0003, "weight_decay": 0.0001, "optimizer": "adamw", "patience": 6},
        "rgb": {"epochs": 30, "batch_size": 32, "learning_rate": 0.001, "weight_decay": 0.0001, "optimizer": "adamw", "patience": 8},
        "augmentation": "none",
        "checkpoint_selection": "highest validation accuracy; official validation test set evaluated once after selection",
        "mixed_precision": bool(device.type == "cuda"),
        "workers": 0,
        "byte_truncation": "disabled; each batch pads to its longest sequence rounded to 256 bytes",
    }

    def save_result(result: dict) -> None:
        model_results.append(result)
        result_lookup[(result["condition"], result["representation"])] = result
        _write_json(all_results_path, model_results)
        print(
            f"finished {result['condition']} / {result['representation']}: "
            f"test={result['test_accuracy']:.4f}, val={result['best_validation_accuracy']:.4f}, "
            f"{result['training_wall_seconds']:.1f}s, {result['examples_per_second']:.1f} examples/s",
            flush=True,
        )

    def run_family(condition: str, representation: str, family: str, model, train_dataset, val_dataset, test_dataset, settings: dict, model_name: str) -> None:
        key = (condition, representation)
        if key in result_lookup:
            return
        seed_everything(args.seed)
        pin_memory = only_condition is None and torch.cuda.is_available()
        train_loader = make_loader(train_dataset, settings["batch_size"], True, args.seed, workers=0, pin_memory=pin_memory)
        validation_loader = make_loader(val_dataset, settings["batch_size"], False, args.seed, workers=0, pin_memory=pin_memory)
        test_loader = make_loader(test_dataset, settings["batch_size"], False, args.seed, workers=0, pin_memory=pin_memory)
        checkpoint = cache_root / "checkpoints" / f"{condition}__{representation}.pt"
        training = train_classifier(
            model,
            train_loader,
            validation_loader,
            test_loader,
            device,
            checkpoint,
            epochs=settings["epochs"],
            learning_rate=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
            optimizer_name=settings["optimizer"],
            patience=settings["patience"],
            amp=training_settings["mixed_precision"],
        )
        training.pop("checkpoint", None)
        save_result(
            {
                "condition": condition,
                "representation": representation,
                "model_family": family,
                "model_name": model_name,
                "parameter_count": parameter_count(model),
                "seed": args.seed,
                **training,
            }
        )
        del train_loader, validation_loader, test_loader, model, train_dataset, val_dataset, test_dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if only_condition is None:
        byte_stream_names = [
            *(f"pbc_{preset}_{entropy}" for preset in ("compression", "balanced", "quality") for entropy in ("lzma", "store")),
            *(f"jpeg_q{quality}" for quality in (10, 20, 60, 80)),
        ]
        byte_lengths = {name: _byte_length_stats(cache_rows, name) for name in byte_stream_names}
        _write_json(output_dir / "raw_stream_lengths.json", byte_lengths)
        _run_overfit_checks(cache_rows, train_rows, cache_root, output_dir, args, device)
    else:
        byte_lengths = {}

    if only_condition is None or only_condition[0].startswith("pbc_"):
        for preset in ("compression", "balanced", "quality"):
            condition = f"pbc_{preset}"
            for representation, stream_kind in (("pbc_lzma_bytes", "lzma"), ("pbc_store_bytes", "store")):
                if (condition, representation) in result_lookup or (only_condition is not None and only_condition != (condition, representation)):
                    continue
                stream_name = f"{condition}_{stream_kind}"
                dataset = ByteStreamFileDataset(stream_dir / f"{stream_name}.bin", cache_rows, stream_name)
                splits = [torch.utils.data.Subset(dataset, [i for i, row in enumerate(cache_rows) if row["split"] == name]) for name in ("train", "validation", "test")]
                seed_everything(args.seed)
                run_family(condition, representation, "byte_cnn", ByteStreamCNN(), *splits, training_settings["byte"], "ByteStreamCNN")
                if only_condition is not None:
                    del splits, dataset
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            for representation, feature_stage in (("pbc_parsed_fields", "parsed"), ("pbc_canonical_ir", "canonical")):
                if (condition, representation) in result_lookup or (only_condition is not None and only_condition != (condition, representation)):
                    continue
                inputs = _feature_inputs(feature_dir, condition, feature_stage, "r" if only_condition is not None else "c")
                datasets = [
                    IndexedDictTensorDataset(inputs, labels, [row["cache_index"] for row in cache_rows if row["split"] == split])
                    for split in ("train", "validation", "test")
                ]
                seed_everything(args.seed)
                run_family(condition, representation, "pbc_instruction_set", PBCInstructionNet(), *datasets, training_settings["pbc"], "PBCInstructionNet")
                if only_condition is not None:
                    del inputs, datasets
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    for quality in (10, 20, 60, 80):
        condition = f"jpeg_q{quality}"
        if (condition, "jpeg_file_bytes") not in result_lookup and (only_condition is None or only_condition == (condition, "jpeg_file_bytes")):
            byte_dataset = ByteStreamFileDataset(stream_dir / f"{condition}.bin", cache_rows, condition)
            byte_splits = [torch.utils.data.Subset(byte_dataset, [i for i, row in enumerate(cache_rows) if row["split"] == name]) for name in ("train", "validation", "test")]
            seed_everything(args.seed)
            run_family(condition, "jpeg_file_bytes", "byte_cnn", ByteStreamCNN(), *byte_splits, training_settings["byte"], "ByteStreamCNN")
            if only_condition is not None:
                del byte_splits, byte_dataset
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if (condition, "jpeg_dct_ir") not in result_lookup and (only_condition is None or only_condition == (condition, "jpeg_dct_ir")):
            inputs = _feature_inputs(feature_dir, condition, mmap_mode="r" if only_condition is not None else "c")
            datasets = [
                IndexedDictTensorDataset(inputs, labels, [row["cache_index"] for row in cache_rows if row["split"] == split])
                for split in ("train", "validation", "test")
            ]
            seed_everything(args.seed)
            run_family(condition, "jpeg_dct_ir", "jpeg_dct_cnn", JPEGDCTNet(), *datasets, training_settings["jpeg_dct"], "JPEGDCTNet")
            if only_condition is not None:
                del inputs, datasets
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    rgb_key = ("original_rgb", "decoded_rgb_reference")
    if only_condition is not None and only_condition != rgb_key:
        return model_results, {}, byte_lengths, training_settings
    rgb_result = result_lookup.get(rgb_key)
    seed_everything(args.seed)
    rgb_model = RGBResNet18()
    if rgb_result is None:
        rgb_datasets = [PixelImageDataset(args.source_root, split) for split in (train_rows, validation_rows, test_rows)]
        seed_everything(args.seed)
        pin_memory = only_condition is None and torch.cuda.is_available()
        rgb_train_loader = make_loader(rgb_datasets[0], training_settings["rgb"]["batch_size"], True, args.seed, pin_memory=pin_memory)
        rgb_val_loader = make_loader(rgb_datasets[1], training_settings["rgb"]["batch_size"], False, args.seed, pin_memory=pin_memory)
        rgb_test_loader = make_loader(rgb_datasets[2], training_settings["rgb"]["batch_size"], False, args.seed, pin_memory=pin_memory)
        rgb_checkpoint = cache_root / "checkpoints" / "original_rgb__decoded_rgb_reference.pt"
        training = train_classifier(
            rgb_model,
            rgb_train_loader,
            rgb_val_loader,
            rgb_test_loader,
            device,
            rgb_checkpoint,
            epochs=training_settings["rgb"]["epochs"],
            learning_rate=training_settings["rgb"]["learning_rate"],
            weight_decay=training_settings["rgb"]["weight_decay"],
            optimizer_name=training_settings["rgb"]["optimizer"],
            patience=training_settings["rgb"]["patience"],
            amp=training_settings["mixed_precision"],
        )
        training.pop("checkpoint", None)
        rgb_result = {
            "condition": "original_rgb",
            "representation": "decoded_rgb_reference",
            "model_family": "rgb_resnet18_style",
            "model_name": "RGBResNet18(width=32, random initialization)",
            "parameter_count": parameter_count(rgb_model),
            "seed": args.seed,
            **training,
        }
        save_result(rgb_result)
        if only_condition is not None:
            del rgb_train_loader, rgb_val_loader, rgb_test_loader, rgb_datasets, rgb_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return model_results, {}, byte_lengths, training_settings
    else:
        checkpoint = cache_root / "checkpoints" / "original_rgb__decoded_rgb_reference.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"RGB reference checkpoint is missing: {checkpoint}")
        rgb_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        rgb_model.to(device)
    rgb_model.eval()
    reference_accuracies = {"original_rgb": rgb_result["test_accuracy"]}
    decode_rows = []
    for preset in ("compression", "balanced", "quality"):
        condition = f"pbc_{preset}"
        decoded_dataset = DecodedRGBImageDataset(
            stream_dir / f"{condition}_lzma.bin", test_rows, f"{condition}_lzma", PBCCodecPair(pbc_root=args.pbc_root, preset=preset)
        )
        decoded_loader = make_loader(decoded_dataset, training_settings["rgb"]["batch_size"], False, args.seed)
        reference_accuracies[condition] = evaluate_accuracy(rgb_model, decoded_loader, device)
        decode_rows.append({"condition": condition, "test_accuracy": reference_accuracies[condition]})
        print(f"decoded RGB reference {condition}: {reference_accuracies[condition]:.4f}", flush=True)
    for quality in (10, 20, 60, 80):
        condition = f"jpeg_q{quality}"
        decoded_dataset = DecodedRGBImageDataset(
            stream_dir / f"{condition}.bin", test_rows, condition, JPEGCodec(quality)
        )
        decoded_loader = make_loader(decoded_dataset, training_settings["rgb"]["batch_size"], False, args.seed)
        reference_accuracies[condition] = evaluate_accuracy(rgb_model, decoded_loader, device)
        decode_rows.append({"condition": condition, "test_accuracy": reference_accuracies[condition]})
        print(f"decoded RGB reference {condition}: {reference_accuracies[condition]:.4f}", flush=True)
    _write_json(output_dir / "decoded_rgb_references.json", reference_accuracies)
    _write_json(output_dir / "decoded_rgb_reference_metrics.json", decode_rows)
    _write_json(output_dir / "training_settings.json", training_settings)
    return model_results, reference_accuracies, byte_lengths, training_settings


def _condition_sizes(rows: list[dict]) -> dict:
    sizes = {}
    for preset in ("compression", "balanced", "quality"):
        condition = f"pbc_{preset}"
        fields = {
            "lzma_bytes": f"{condition}_lzma",
            "store_bytes": f"{condition}_store",
            "parsed_bytes": f"{condition}_parsed",
            "canonical_bytes": f"{condition}_canonical",
        }
        values = {}
        for field, name in fields.items():
            if field.endswith("bytes") and field in {"lzma_bytes", "store_bytes"}:
                sizes_for_condition = [row["streams"][name]["length"] for row in rows]
            else:
                sizes_for_condition = [row["representation_sizes"][name] for row in rows]
            values[field] = int(statistics.median(sizes_for_condition))
        mean_encoded = statistics.mean(row["streams"][f"{condition}_lzma"]["length"] for row in rows)
        values["encoded_bpp"] = mean_encoded * 8.0 / (256 * 256)
        values["rgb_bytes"] = 256 * 256 * 3
        values["bpp"] = values["encoded_bpp"]
        sizes[condition] = values
    for quality in (10, 20, 60, 80):
        condition = f"jpeg_q{quality}"
        jpeg_bytes = [row["streams"][condition]["length"] for row in rows]
        dct_bytes = [row["representation_sizes"][f"{condition}_dct"] for row in rows]
        mean_encoded = statistics.mean(jpeg_bytes)
        sizes[condition] = {
            "jpeg_bytes": int(statistics.median(jpeg_bytes)),
            "dct_bytes": int(statistics.median(dct_bytes)),
            "rgb_bytes": 256 * 256 * 3,
            "encoded_bpp": mean_encoded * 8.0 / (256 * 256),
            "bpp": mean_encoded * 8.0 / (256 * 256),
        }
    return sizes


def _phase2_timings(results_dir: Path) -> dict:
    phase2 = json.loads((results_dir.parent / "Staged Decoding" / "summary.json").read_text(encoding="utf-8"))
    conditions = {item["condition"]: item for item in phase2["conditions"]}
    output = {}
    for preset in ("compression", "balanced", "quality"):
        forced = conditions[f"pbc_{preset}_forced_lzma"]
        store = conditions[f"pbc_{preset}_store"]
        output[f"pbc_{preset}"] = {
            "lzma_unpack_ms": forced["stage_timings_ms"]["forced_lzma_unpack"]["median"],
            "store_body_parse_ms": store["stage_timings_ms"]["store_body_parse"]["median"],
            "canonical_ir_build_ms": store["stage_timings_ms"]["canonical_ir_build"]["median"],
            "full_lzma_decode_ms": forced["full_decode_ms"]["median"],
            "full_store_decode_ms": store["full_decode_ms"]["median"],
            "stage_source": "Phase 2; original 95-sample timing set",
        }
    q20_path = results_dir / "jpeg_q20_timing_extension.json"
    q20 = json.loads(q20_path.read_text(encoding="utf-8"))
    for quality in (10, 20, 60, 80):
        condition = f"jpeg_q{quality}"
        if quality == 20:
            output[condition] = {
                "file_to_coefficients_ms": None,
                "canonical_ir_build_ms": None,
                "file_to_dct_ms": q20["summary_ms"]["median"],
                "full_rgb_decode_ms": q20["full_rgb_decode_summary_ms"]["median"],
                "stage_source": f"{q20['name']}; combined file-to-DCT timing",
            }
        else:
            item = conditions[condition]
            file_to_coefficients_ms = item["stage_timings_ms"]["jpeg_file_to_coefficients"]["median"]
            canonical_ir_build_ms = item["stage_timings_ms"]["jpeg_canonical_ir_build"]["median"]
            output[condition] = {
                "file_to_coefficients_ms": file_to_coefficients_ms,
                "canonical_ir_build_ms": canonical_ir_build_ms,
                "file_to_dct_ms": file_to_coefficients_ms + canonical_ir_build_ms,
                "full_rgb_decode_ms": item["full_decode_ms"]["median"],
                "stage_source": "Phase 2; original 95-sample timing set",
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=cache_root() / "neural_accessibility" / "imagenette_320")
    parser.add_argument("--dataset-root", type=Path, default=data_root() / "raw" / "imagenette")
    parser.add_argument("--results-dir", type=Path, default=results_root() / "Neural Accessibility")
    parser.add_argument("--pbc-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--only-condition", nargs=2, metavar=("CONDITION", "REPRESENTATION"))
    args = parser.parse_args()
    if args.only_condition is not None:
        valid_conditions = {
            *( (f"pbc_{preset}", representation) for preset in ("compression", "balanced", "quality") for representation in ("pbc_lzma_bytes", "pbc_store_bytes", "pbc_parsed_fields", "pbc_canonical_ir") ),
            *( (f"jpeg_q{quality}", representation) for quality in (10, 20, 60, 80) for representation in ("jpeg_file_bytes", "jpeg_dct_ir") ),
            ("original_rgb", "decoded_rgb_reference"),
        }
        if tuple(args.only_condition) not in valid_conditions:
            parser.error("unknown condition/representation pair")
    args.cache_root = args.cache_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.source_root = get_imagenette_source_root(args.dataset_root)
    args.results_dir = args.results_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_root / "codec_manifest.jsonl"
    cache_rows = read_manifest(manifest_path)
    source_rows = read_manifest(args.cache_root / "manifest.jsonl")
    check_representation_alignment(source_rows, cache_rows)
    train_rows, validation_rows, test_rows = _condition_splits(cache_rows)
    for row in cache_rows:
        for preset in ("compression", "balanced", "quality"):
            if row["body_sha256"][preset] == "":
                raise ValueError(f"empty PBC body hash for {row['image_id']}")
        if set(row["streams"]) != {
            *(f"pbc_{preset}_{entropy}" for preset in ("compression", "balanced", "quality") for entropy in ("lzma", "store")),
            *(f"jpeg_q{quality}" for quality in (10, 20, 60, 80)),
        }:
            raise ValueError(f"incomplete codec representation row: {row['image_id']}")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model_results, rgb_references, byte_lengths, training_settings = _prepare_measurement_results(
        cache_rows, train_rows, validation_rows, test_rows, args.results_dir, args.cache_root, args, device
    )
    if args.only_condition is not None:
        print(json.dumps({"completed_condition": args.only_condition, "models": len(model_results)}, indent=2))
        return
    preprocessing = _phase2_timings(args.results_dir)
    condition_sizes = _condition_sizes(cache_rows)
    frontier = build_frontier_rows(model_results, rgb_references, preprocessing, condition_sizes)
    summary = {
        "dataset": {
            "name": "Imagenette 320px distribution",
            "archive_url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
            "prepared_image_size": [256, 256],
            "preprocessing": "EXIF transpose, RGB conversion, Pillow BICUBIC center crop/fit with centering=(0.5, 0.5)",
            "seed": args.seed,
            "train_samples": len(train_rows),
            "validation_samples": len(validation_rows),
            "test_samples": len(test_rows),
            "official_validation_used_as_test": True,
            "test_images_in_train_or_validation": 0,
            "class_counts": {
                split: {
                    str(label): sum(row["label"] == label for row in cache_rows if row["split"] == split)
                    for label in sorted({row["label"] for row in cache_rows})
                }
                for split in ("train", "validation", "test")
            },
        },
        "device": str(device),
        "model_results": model_results,
        "rgb_reference_accuracies": rgb_references,
        "preprocessing": preprocessing,
        "condition_sizes": condition_sizes,
        "frontier": frontier,
        "raw_stream_length_statistics": byte_lengths,
        "training_settings": training_settings,
        "interpretation": {
            "latency_scope": "implementation-level systems measurements on this machine; PBC Python/NumPy/Numba research code and JPEG libjpeg-turbo are not pure algorithmic comparisons",
            "pbc_fraction_scope": "fractions use forced-LZMA full decode for each PBC preset; report absolute milliseconds alongside fractions",
            "semantic_accessibility": "(model accuracy - 0.1) / (common decoded-RGB accuracy - 0.1); values are not clamped",
            "training_data": "prepared original RGB for the reference; one seed; validation-selected checkpoints; official validation test set used once per final evaluation",
        },
    }
    _write_json(args.results_dir / "summary.json", summary)
    print(json.dumps({"summary": str(args.results_dir / "summary.json"), "models": len(model_results), "splits": {"train": len(train_rows), "validation": len(validation_rows), "test": len(test_rows)}}, indent=2))


if __name__ == "__main__":
    main()
