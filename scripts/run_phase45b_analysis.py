from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.analysis.phase45b import prediction_metrics, validate_manifest_alignment
from tamtd.codec_benchmark import image_metrics
from tamtd.codecs.jpeg import JPEGCodec
from tamtd.codecs.jxl import JXLCodec
from tamtd.codecs.pbc import PBCCodecPair
from tamtd.data.accessibility_datasets import PixelImageDataset
from tamtd.data.imagenette_accessibility import get_imagenette_source_root, load_prepared_rgb, read_manifest, sha256
from tamtd.data.neural_accessibility import jpeg_blocks_to_channels
from tamtd.models.accessibility import ByteStreamCNN, JPEGDCTNet, PBCInstructionNet, RGBResNet18
from tamtd.runtime import cache_root, data_root, default_device, results_root
from tamtd.training.accessibility import ByteStreamFileDataset, IndexedDictTensorDataset, make_loader


RD_CONDITIONS = [
    *(f"pbc_{preset}" for preset in ("compression", "balanced", "quality")),
    "jpeg_q10", "jpeg_q20", "jpeg_q60", "jpeg_q80",
    "webp_q30", "webp_q75", "avif_q40", "avif_q60", "jxl_d6", "jxl_d3",
]
ENDPOINT_CONDITIONS = [
    *(f"pbc_{preset}" for preset in ("compression", "balanced", "quality")),
    "jpeg_q10", "jpeg_q20", "jpeg_q60", "jpeg_q80",
    "webp_q30", "webp_q75", "avif_q40", "avif_q60", "jxl_d6", "jxl_d3",
]
RD_FIELDS = [
    "cache_index", "image_id", "split", "label", "condition", "codec_family", "operating_point",
    "representation", "encoded_bytes", "bpp", "mse", "psnr_db", "ssim", "decode_valid",
    "rgb_sha256", "stream_sha256",
]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _portable_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _read_cached_stream(handle, reference: dict, expected_id: str) -> bytes:
    handle.seek(int(reference["offset"]))
    payload = handle.read(int(reference["length"]))
    if len(payload) != int(reference["length"]):
        raise ValueError(f"short stream at {expected_id}")
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError(f"stream hash mismatch at {expected_id}")
    return payload


def _decode(condition: str, payload: bytes, pbc_pairs: dict[str, PBCCodecPair], jxl_codecs: dict[str, JXLCodec]) -> Image.Image:
    if condition.startswith("pbc_"):
        return pbc_pairs[condition].decode(payload)
    if condition.startswith("jxl_"):
        return jxl_codecs[condition].decode(payload)
    with Image.open(io.BytesIO(payload)) as decoded:
        return decoded.convert("RGB").copy()


def _condition_stream_key(condition: str, representation: str | None = None) -> str:
    if condition.startswith("pbc_"):
        if representation == "pbc_store_bytes":
            return f"{condition}_store"
        return f"{condition}_lzma"
    return condition


def _load_manifests(cache_dir: Path) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    rows = read_manifest(cache_dir / "manifest.jsonl")
    codec_rows = read_manifest(cache_dir / "codec_manifest.jsonl")
    validate_manifest_alignment(rows, codec_rows)
    if any(row["cache_index"] != index for index, row in enumerate(rows)):
        raise ValueError("dataset manifest cache_index values are not contiguous")
    phase4_rows = {}
    for condition in ("webp_q30", "webp_q75", "avif_q40", "avif_q60", "jxl_d6", "jxl_d3"):
        additions = read_manifest(cache_dir / "phase4_streams" / f"{condition}.jsonl")
        validate_manifest_alignment(rows, additions)
        phase4_rows[condition] = additions
    return rows, codec_rows, phase4_rows


def _stream_handles(cache_dir: Path, codec_rows: list[dict], conditions: list[str]):
    handles = {}
    for condition in conditions:
        if condition.startswith("pbc_"):
            keys = (f"{condition}_lzma", f"{condition}_store")
            for key in keys:
                handles[key] = (cache_dir / "streams" / f"{key}.bin").open("rb")
        elif condition.startswith("jpeg_"):
            handles[condition] = (cache_dir / "streams" / f"{condition}.bin").open("rb")
        else:
            handles[condition] = (cache_dir / "phase4_streams" / f"{condition}.bin").open("rb")
    return handles


def generate_rd(cache_dir: Path, source_root: Path, out_dir: Path, pbc_root: Path) -> dict:
    rows, codec_rows, phase4_rows = _load_manifests(cache_dir)
    pairs = {condition: PBCCodecPair(pbc_root=pbc_root, preset=condition.removeprefix("pbc_")) for condition in RD_CONDITIONS if condition.startswith("pbc_")}
    jxl = {condition: JXLCodec(float(condition.removeprefix("jxl_d"))) for condition in RD_CONDITIONS if condition.startswith("jxl_")}
    handles = _stream_handles(cache_dir, codec_rows, RD_CONDITIONS)
    path = out_dir / "imagenette_rd_per_sample.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    failures = []
    try:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RD_FIELDS)
            writer.writeheader()
            for index, (row, codec_row) in enumerate(zip(rows, codec_rows, strict=True)):
                source_path = source_root / row["image_id"]
                source = load_prepared_rgb(source_path)
                source_hash = sha256(source.tobytes())
                if source_hash != row["rgb_sha256"] or codec_row["rgb_sha256"] != source_hash:
                    raise ValueError(f"prepared RGB hash mismatch at {row['image_id']}")
                metrics_by_pbc = {}
                for preset in ("compression", "balanced", "quality"):
                    condition = f"pbc_{preset}"
                    lzma_key = f"{condition}_lzma"
                    store_key = f"{condition}_store"
                    lzma_bytes = _read_cached_stream(handles[lzma_key], codec_row["streams"][lzma_key], row["image_id"])
                    store_bytes = _read_cached_stream(handles[store_key], codec_row["streams"][store_key], row["image_id"])
                    decoded_lzma = np.asarray(pairs[condition].decode(lzma_bytes), dtype=np.uint8)
                    decoded_store = np.asarray(pairs[condition].decode(store_bytes), dtype=np.uint8)
                    if decoded_lzma.shape != source.shape or not np.array_equal(decoded_lzma, decoded_store):
                        raise ValueError(f"PBC LZMA/STORE decode mismatch at {row['image_id']} for {condition}")
                    score = image_metrics(Image.fromarray(source), Image.fromarray(decoded_lzma))
                    metrics_by_pbc[condition] = score
                    for representation, stream_key, payload in (
                        ("pbc_lzma_bytes", lzma_key, lzma_bytes),
                        ("pbc_store_bytes", store_key, store_bytes),
                    ):
                        writer.writerow({
                            "cache_index": index, "image_id": row["image_id"], "split": row["split"], "label": row["label"],
                            "condition": condition, "codec_family": "pbc", "operating_point": preset,
                            "representation": representation, "encoded_bytes": len(payload),
                            "bpp": len(payload) * 8.0 / (256 * 256), "mse": score["mse"], "psnr_db": score["psnr_db"],
                            "ssim": score["ssim"], "decode_valid": True, "rgb_sha256": source_hash,
                            "stream_sha256": hashlib.sha256(payload).hexdigest(),
                        })
                        count += 1
                for condition in ("jpeg_q10", "jpeg_q20", "jpeg_q60", "jpeg_q80", "webp_q30", "webp_q75", "avif_q40", "avif_q60", "jxl_d6", "jxl_d3"):
                    if condition.startswith("jpeg_"):
                        reference = codec_row["streams"][condition]
                        handle = handles[condition]
                    else:
                        addition = phase4_rows[condition][index]
                        reference = addition["stream"]
                        handle = handles[condition]
                    payload = _read_cached_stream(handle, reference, row["image_id"])
                    decoded = _decode(condition, payload, pairs, jxl)
                    decoded_rgb = np.asarray(decoded, dtype=np.uint8)
                    if decoded_rgb.shape != source.shape:
                        raise ValueError(f"decoded image has wrong dimensions at {row['image_id']} for {condition}")
                    score = image_metrics(Image.fromarray(source), decoded)
                    writer.writerow({
                        "cache_index": index, "image_id": row["image_id"], "split": row["split"], "label": row["label"],
                        "condition": condition, "codec_family": condition.split("_")[0],
                        "operating_point": condition.removeprefix("jpeg_q").removeprefix("webp_q").removeprefix("avif_q").removeprefix("jxl_d"),
                        "representation": "jpeg_file_bytes" if condition.startswith("jpeg_") else "codec_file_bytes",
                        "encoded_bytes": len(payload),
                        "bpp": len(payload) * 8.0 / (256 * 256), "mse": score["mse"], "psnr_db": score["psnr_db"],
                        "ssim": score["ssim"], "decode_valid": True, "rgb_sha256": source_hash,
                        "stream_sha256": hashlib.sha256(payload).hexdigest(),
                    })
                    count += 1
                if (index + 1) % 500 == 0 or index + 1 == len(rows):
                    print(f"Imagenette rate-distortion: {index + 1}/{len(rows)} samples", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    return {"rows": count, "samples": len(rows), "stream_forms": 16, "path": str(path), "source_hashes_checked": len(rows), "pbc_store_lzma_pixel_identity_checked": len(rows) * 3}


def _load_rgb_models(cache_dir: Path, model_specs: list[dict], device: torch.device) -> dict[int, torch.nn.Module]:
    result = {}
    for spec in model_specs:
        if spec["condition"] != "original_rgb" or spec["representation"] != "decoded_rgb_reference":
            continue
        model = RGBResNet18()
        model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True))
        model.to(device).eval()
        result[int(spec["seed"])] = model
    if not result:
        raise FileNotFoundError(f"no original RGB reference checkpoints under {cache_dir / 'checkpoints'}")
    return result


def _discover_model_specs(out_dir: Path, cache_dir: Path) -> list[dict]:
    phase4_root = out_dir / "Phase 4 Confirmation" / "per_seed"
    candidates = {}
    for path in sorted(phase4_root.rglob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not row.get("checkpoint"):
            continue
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        if checkpoint.is_file():
            key = (row.get("condition"), row.get("representation"), int(row.get("seed", -1)))
            candidates.setdefault(key, []).append((row, checkpoint))
    specs = []
    summary_path = out_dir / "Phase 4 Confirmation" / "confirmation_per_seed.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for summary in csv.DictReader(handle):
            condition = summary["condition"]
            representation = summary["representation"]
            seed = int(summary["seed"])
            if representation == "decoded_rgb_reference" and condition != "original_rgb":
                continue
            key = (condition, representation, seed)
            target = float(summary["test_accuracy"])
            options = [item for item in candidates.get(key, []) if abs(float(item[0].get("test_accuracy", -1)) - target) < 1e-9]
            if not options:
                if seed != 42:
                    raise FileNotFoundError(f"no saved checkpoint metadata matches Phase 4 result {key}={target}")
                phase3 = next((item for item in json.loads((out_dir / "Neural Accessibility" / "model_results.json").read_text(encoding="utf-8")) if (item["condition"], item["representation"], int(item["seed"])) == key and abs(float(item["test_accuracy"]) - target) < 1e-9), None)
                if phase3 is None:
                    raise FileNotFoundError(f"Phase 4 seed-42 row has no matching Phase 3 checkpoint result: {key}={target}")
                checkpoint = cache_dir / "checkpoints" / f"{condition}__{representation}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"reused Phase 3 checkpoint is missing: {checkpoint}")
                row = phase3
                specs.append({**row, "condition": condition, "representation": representation, "seed": seed, "checkpoint": str(checkpoint), "expected_test_accuracy": target, "history_source": str(out_dir / "Neural Accessibility" / "model_results.json")})
                continue
            row, checkpoint = options[-1]
            specs.append({**row, "condition": condition, "representation": representation, "seed": seed, "checkpoint": str(checkpoint), "expected_test_accuracy": target, "history_source": str(phase4_root)})
    for path in sorted(phase4_root.glob("seed_42/*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if row.get("representation") != "codec_file_bytes" or not row.get("checkpoint"):
            continue
        condition = row["condition"]
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        key = (condition, "codec_file_bytes", int(row["seed"]))
        if checkpoint.is_file() and not any((s["condition"], s["representation"], int(s["seed"])) == key for s in specs):
            specs.append({**row, "checkpoint": str(checkpoint), "expected_test_accuracy": float(row["test_accuracy"]), "history_source": str(path)})
    included = {(spec["condition"], spec["representation"]) for spec in specs}
    for row in json.loads((out_dir / "Neural Accessibility" / "model_results.json").read_text(encoding="utf-8")):
        condition, representation = row["condition"], row["representation"]
        if (condition, representation) in included:
            continue
        checkpoint = cache_dir / "checkpoints" / f"{condition}__{representation}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"legacy Phase 3 checkpoint is missing: {checkpoint}")
        specs.append({**row, "checkpoint": str(checkpoint), "expected_test_accuracy": float(row["test_accuracy"]), "history_source": str(out_dir / "Neural Accessibility" / "model_results.json")})
        included.add((condition, representation))
    unique = {}
    for spec in specs:
        key = (spec["condition"], spec["representation"], int(spec["seed"]))
        if key not in unique:
            unique[key] = spec
    return [unique[key] for key in sorted(unique)]


class _FeatureDataset(Dataset):
    def __init__(self, inputs: dict[str, np.ndarray], labels: np.ndarray, indices: list[int]) -> None:
        self.inputs = inputs
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        inputs = {name: torch.from_numpy(np.array(values[source_index], copy=True)) for name, values in self.inputs.items()}
        return inputs, torch.tensor(int(self.labels[source_index]), dtype=torch.long)


def _feature_inputs(cache_dir: Path, condition: str, representation: str, arrays: dict[Path, np.ndarray]) -> dict[str, np.ndarray]:
    feature_dir = cache_dir / "features"
    if representation.startswith("pbc_"):
        stage = "canonical" if representation == "pbc_canonical_ir" else "parsed"
        suffixes = {"geometry": "geometry", "content": "content", "patch_mask": "patch_mask"}
        return {key: arrays.setdefault(feature_dir / f"{condition}_{stage}_{suffix}.npy", np.load(feature_dir / f"{condition}_{stage}_{suffix}.npy", mmap_mode="r", allow_pickle=False)) for key, suffix in suffixes.items()}
    suffixes = {
        "y_coefficients": "y", "cb_coefficients": "cb", "cr_coefficients": "cr", "quantization_tables": "quantization",
    }
    return {key: arrays.setdefault(feature_dir / f"{condition}_{suffix}.npy", np.load(feature_dir / f"{condition}_{suffix}.npy", mmap_mode="r", allow_pickle=False)) for key, suffix in suffixes.items()}


def _model_for_spec(spec: dict) -> torch.nn.Module:
    representation = spec["representation"]
    if representation in {"pbc_lzma_bytes", "pbc_store_bytes", "jpeg_file_bytes", "codec_file_bytes"}:
        return ByteStreamCNN()
    if representation in {"pbc_parsed_fields", "pbc_canonical_ir"}:
        return PBCInstructionNet()
    if representation == "jpeg_dct_ir":
        return JPEGDCTNet()
    if representation == "decoded_rgb_reference":
        return RGBResNet18()
    raise ValueError(f"unsupported checkpoint representation: {representation}")


def _split_rows(rows: list[dict], split: str) -> list[dict]:
    return [row for row in rows if row["split"] == split]


def _merged_stream_rows(rows: list[dict], codec_rows: list[dict], phase4_rows: dict[str, list[dict]], condition: str, stream_key: str) -> list[dict]:
    result = []
    for row, codec_row in zip(rows, codec_rows, strict=True):
        merged = dict(row)
        merged["streams"] = dict(codec_row["streams"])
        if condition in phase4_rows:
            merged["streams"][stream_key] = phase4_rows[condition][row["cache_index"]]["stream"]
        result.append(merged)
    return result


def _make_test_loader(spec: dict, rows: list[dict], codec_rows: list[dict], phase4_rows: dict[str, list[dict]], cache_dir: Path, source_root: Path, arrays: dict[Path, np.ndarray], batch_size: int):
    condition, representation = spec["condition"], spec["representation"]
    test_rows = _split_rows(rows, "test")
    if representation in {"pbc_lzma_bytes", "pbc_store_bytes", "jpeg_file_bytes", "codec_file_bytes"}:
        stream_key = _condition_stream_key(condition, representation)
        merged = _merged_stream_rows(rows, codec_rows, phase4_rows, condition, stream_key)
        selected = [row for row in merged if row["split"] == "test"]
        if condition.startswith("pbc_") or condition.startswith("jpeg_"):
            stream_path = cache_dir / "streams" / f"{stream_key}.bin"
        else:
            stream_path = cache_dir / "phase4_streams" / f"{condition}.bin"
        dataset = ByteStreamFileDataset(stream_path, selected, stream_key)
        return make_loader(dataset, batch_size, False, int(spec["seed"]), workers=0, pin_memory=False), len(test_rows)
    if representation == "decoded_rgb_reference":
        dataset = PixelImageDataset(source_root, test_rows, verify_hashes=True)
    else:
        inputs = _feature_inputs(cache_dir, condition, representation, arrays)
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        indices = [row["cache_index"] for row in test_rows]
        dataset = _FeatureDataset(inputs, labels, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return loader, len(test_rows)


def _predict_loader(model: torch.nn.Module, loader, device: torch.device) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.inference_mode():
        for inputs, labels, *rest in loader:
            moved = {name: tensor.to(device) for name, tensor in inputs.items()}
            chunks.append(model(**moved).float().cpu().numpy())
    if not chunks:
        raise ValueError("empty inference loader")
    return np.concatenate(chunks)


def _generate_trained_predictions(cache_dir: Path, out_dir: Path, source_root: Path, rows: list[dict], codec_rows: list[dict], phase4_rows: dict[str, list[dict]], device: torch.device, batch_size: int) -> tuple[dict, list[dict]]:
    specs = _discover_model_specs(out_dir, cache_dir)
    arrays: dict[Path, np.ndarray] = {}
    outputs = {}
    checks = []
    labels = np.asarray([row["label"] for row in _split_rows(rows, "test")], dtype=np.int64)
    test_rows = _split_rows(rows, "test")
    for index, spec in enumerate(specs, 1):
        model = _model_for_spec(spec)
        model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True))
        model.to(device).eval()
        loader, expected_count = _make_test_loader(spec, rows, codec_rows, phase4_rows, cache_dir, source_root, arrays, batch_size)
        logits = _predict_loader(model, loader, device)
        if len(logits) != expected_count:
            raise ValueError(f"inference count mismatch for {spec['condition']}/{spec['representation']}/{spec['seed']}")
        result = prediction_metrics(logits, labels, np.asarray([row["label"] for row in rows if row["split"] == "train"]))
        expected = float(spec["expected_test_accuracy"])
        delta = float(result["accuracy"] - expected)
        checks.append({"condition": spec["condition"], "representation": spec["representation"], "seed": int(spec["seed"]), "expected_test_accuracy": expected, "recomputed_test_accuracy": float(result["accuracy"]), "delta": delta, "checkpoint": _portable_path(spec["checkpoint"])})
        if abs(delta) > max(0.001, 2.0 / len(labels)):
            raise ValueError(f"recomputed test accuracy disagrees with recorded result for {spec['condition']}/{spec['representation']}/{spec['seed']}: {delta:+.5f}")
        key = (spec["condition"], spec["representation"], int(spec["seed"]), "test")
        outputs[key] = {"logits": logits.astype(np.float32), "labels": labels.copy(), "image_ids": [row["image_id"] for row in test_rows], "model_family": spec.get("model_family", ""), "checkpoint": spec["checkpoint"]}
        del loader, model, logits
        arrays.clear()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if index % 10 == 0 or index == len(specs):
            print(f"Existing model checkpoints evaluated: {index}/{len(specs)}", flush=True)
    return outputs, checks


def _endpoint_stream(condition: str, row: dict, codec_row: dict, phase4_rows: dict[str, list[dict]]) -> tuple[str, dict]:
    key = _condition_stream_key(condition)
    if condition in phase4_rows:
        return key, phase4_rows[condition][row["cache_index"]]["stream"]
    return key, codec_row["streams"][key]


def _generate_endpoint_predictions(cache_dir: Path, out_dir: Path, source_root: Path, rows: list[dict], codec_rows: list[dict], phase4_rows: dict[str, list[dict]], pbc_root: Path, device: torch.device, batch_size: int, model_specs: list[dict]) -> tuple[dict, list[dict]]:
    rgb_models = _load_rgb_models(cache_dir, model_specs, device)
    pairs = {condition: PBCCodecPair(pbc_root=pbc_root, preset=condition.removeprefix("pbc_")) for condition in ENDPOINT_CONDITIONS if condition.startswith("pbc_")}
    jxl = {condition: JXLCodec(float(condition.removeprefix("jxl_d"))) for condition in ENDPOINT_CONDITIONS if condition.startswith("jxl_")}
    handles = _stream_handles(cache_dir, codec_rows, ENDPOINT_CONDITIONS)
    train_labels = np.asarray([row["label"] for row in rows if row["split"] == "train"], dtype=np.int64)
    outputs = {}
    metrics_rows = []
    try:
        for condition in ENDPOINT_CONDITIONS:
            stream_key = _condition_stream_key(condition)
            for split in ("validation", "test"):
                selected = [(row, codec_rows[row["cache_index"]]) for row in rows if row["split"] == split]
                chunks = {seed: [] for seed in rgb_models}
                labels = []
                ids = []
                batch_images = []
                batch_labels = []
                batch_ids = []

                def flush_batch():
                    if not batch_images:
                        return
                    tensor = torch.from_numpy(np.stack(batch_images)).permute(0, 3, 1, 2).to(device)
                    with torch.inference_mode():
                        for seed, model in rgb_models.items():
                            chunks[seed].append(model(images=tensor).float().cpu().numpy())
                    labels.extend(batch_labels)
                    ids.extend(batch_ids)
                    batch_images.clear()
                    batch_labels.clear()
                    batch_ids.clear()

                for row, codec_row in selected:
                    key, reference = _endpoint_stream(condition, row, codec_row, phase4_rows)
                    payload = _read_cached_stream(handles[key], reference, row["image_id"])
                    decoded = np.asarray(_decode(condition, payload, pairs, jxl), dtype=np.uint8)
                    if decoded.shape != (256, 256, 3):
                        raise ValueError(f"decoded RGB endpoint dimensions differ for {condition}: {row['image_id']}")
                    batch_images.append(decoded)
                    batch_labels.append(int(row["label"]))
                    batch_ids.append(row["image_id"])
                    if len(batch_images) == batch_size:
                        flush_batch()
                flush_batch()
                label_array = np.asarray(labels, dtype=np.int64)
                for seed, pieces in chunks.items():
                    logits = np.concatenate(pieces)
                    result = prediction_metrics(logits, label_array, train_labels)
                    original_key = ("original_rgb", "decoded_rgb_reference", int(seed), split)
                    original = outputs.get(original_key)
                    metrics_rows.append({
                        "condition": condition, "codec_family": condition.split("_")[0], "representation": "decoded_rgb_reference",
                        "seed": int(seed), "split": split, "n_samples": len(label_array), "accuracy": float(result["accuracy"]),
                        "cross_entropy_nats": float(result["cross_entropy_nats"]), "class_prior_cross_entropy_nats": float(result["class_prior_cross_entropy_nats"]),
                        "u_probe_bits": float(result["u_probe_bits"]), "brier_score": float(result["brier_score"]),
                    })
                    outputs[(condition, "decoded_rgb_reference", int(seed), split)] = {
                        "logits": logits.astype(np.float32), "labels": label_array.copy(), "image_ids": ids.copy(),
                        "model_family": "RGBResNet18", "checkpoint": model_specs_by_seed(model_specs, int(seed))["checkpoint"],
                    }
                print(f"Decoded RGB endpoint evaluated: {condition}/{split}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    for model in rgb_models.values():
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs, metrics_rows


def model_specs_by_seed(model_specs: list[dict], seed: int) -> dict:
    for spec in model_specs:
        if spec["condition"] == "original_rgb" and spec["representation"] == "decoded_rgb_reference" and int(spec["seed"]) == seed:
            return spec
    raise KeyError(f"missing original RGB checkpoint for seed {seed}")


def _generate_original_rgb_validation(cache_dir: Path, source_root: Path, rows: list[dict], model_specs: list[dict], device: torch.device, batch_size: int) -> dict:
    validation_rows = _split_rows(rows, "validation")
    dataset = PixelImageDataset(source_root, validation_rows, verify_hashes=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    labels = np.asarray([row["label"] for row in validation_rows], dtype=np.int64)
    outputs = {}
    for spec in model_specs:
        if spec["condition"] != "original_rgb" or spec["representation"] != "decoded_rgb_reference":
            continue
        model = RGBResNet18()
        model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True))
        model.to(device).eval()
        logits = _predict_loader(model, loader, device)
        outputs[("original_rgb", "decoded_rgb_reference", int(spec["seed"]), "validation")] = {
            "logits": logits.astype(np.float32), "labels": labels.copy(), "image_ids": [row["image_id"] for row in validation_rows],
            "model_family": "RGBResNet18", "checkpoint": spec["checkpoint"],
        }
        del model, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return outputs


def _write_prediction_artifacts(outputs: dict, cache_dir: Path, out_dir: Path, rows: list[dict]) -> dict:
    train_labels = np.asarray([row["label"] for row in rows if row["split"] == "train"], dtype=np.int64)
    test_labels = np.asarray([row["label"] for row in rows if row["split"] == "test"], dtype=np.int64)
    test_rows = _split_rows(rows, "test")
    result_dir = out_dir / "Phase 4.5B Analysis"
    cache_output = cache_dir / "phase45b"
    cache_output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    index_rows = []
    summary_path = result_dir / "test_predictions_summary.csv.gz"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "model_condition", "representation", "model_family", "seed", "true_label", "predicted_label", "correct", "top1_probability", "true_class_probability", "cross_entropy_nats", "brier_score"]
    with gzip.open(summary_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for serial, key in enumerate(sorted(outputs), 1):
            condition, representation, seed, split = key
            output = outputs[key]
            logits = output["logits"]
            labels = output["labels"]
            metrics = prediction_metrics(logits, labels, train_labels)
            probabilities = metrics["probabilities"]
            predictions = metrics["predicted_labels"]
            ce_values = metrics["per_image_cross_entropy_nats"]
            true_probabilities = probabilities[np.arange(len(labels)), labels]
            brier_values = np.sum(probabilities**2, axis=1) - 2.0 * true_probabilities + 1.0
            ids = output["image_ids"]
            logits_key = f"logits_{serial:04d}"
            arrays[logits_key] = logits.astype(np.float16)
            index_rows.append({"logits_key": logits_key, "condition": condition, "representation": representation, "seed": int(seed), "split": split, "model_family": output["model_family"], "n": len(labels)})
            if split == "test":
                for position, image_id in enumerate(ids):
                    writer.writerow({
                        "image_id": image_id, "model_condition": condition, "representation": representation,
                        "model_family": output["model_family"], "seed": int(seed), "true_label": int(labels[position]),
                        "predicted_label": int(predictions[position]), "correct": int(predictions[position] == labels[position]),
                        "top1_probability": float(probabilities[position].max()), "true_class_probability": float(probabilities[position, labels[position]]),
                        "cross_entropy_nats": float(ce_values[position]), "brier_score": float(brier_values[position]),
                    })
    np.savez_compressed(cache_output / "prediction_logits.npz", **arrays)
    _write_json(cache_output / "prediction_logits_index.json", {"dtype": "float16 logits", "keys": index_rows})
    return {"model_splits": len(index_rows), "test_prediction_rows": sum(int(row["n"]) for row in index_rows if row["split"] == "test"), "summary_path": str(summary_path), "logits_cache": str(cache_output / "prediction_logits.npz")}


def generate_inputs(args) -> dict:
    cache_dir = args.cache_dir.resolve()
    out_dir = args.results_root.resolve()
    pbc_root = args.pbc_root.resolve()
    source_root = get_imagenette_source_root(args.dataset_root.resolve())
    rows, codec_rows, phase4_rows = _load_manifests(cache_dir)
    if len(rows) != 13394:
        raise ValueError(f"expected the Phase 3/4 Imagenette cache (13,394 rows), found {len(rows)}")
    result_dir = out_dir / "Phase 4.5B Analysis"
    result_dir.mkdir(parents=True, exist_ok=True)
    rd_summary = generate_rd(cache_dir, source_root, result_dir, pbc_root)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_specs = _discover_model_specs(out_dir, cache_dir)
    print(f"Inference device: {device}; checkpoints: {len(model_specs)}", flush=True)
    trained_outputs, accuracy_checks = _generate_trained_predictions(cache_dir, out_dir, source_root, rows, codec_rows, phase4_rows, device, args.batch_size)
    endpoint_outputs, endpoint_metrics = _generate_endpoint_predictions(cache_dir, out_dir, source_root, rows, codec_rows, phase4_rows, pbc_root, device, args.batch_size, model_specs)
    original_validation = _generate_original_rgb_validation(cache_dir, source_root, rows, model_specs, device, args.batch_size)
    outputs = {**trained_outputs, **endpoint_outputs, **original_validation}
    train_labels = np.asarray([row["label"] for row in rows if row["split"] == "train"], dtype=np.int64)
    endpoint_keys = {(row["condition"], int(row["seed"]), row["split"]) for row in endpoint_metrics}
    for key, output in outputs.items():
        condition, representation, seed, split = key
        if condition != "original_rgb" or representation != "decoded_rgb_reference" or (condition, int(seed), split) in endpoint_keys:
            continue
        result = prediction_metrics(output["logits"], output["labels"], train_labels)
        endpoint_metrics.append({
            "condition": "original_rgb", "codec_family": "original", "representation": "decoded_rgb_reference",
            "seed": int(seed), "split": split, "n_samples": len(output["labels"]), "accuracy": float(result["accuracy"]),
            "cross_entropy_nats": float(result["cross_entropy_nats"]), "class_prior_cross_entropy_nats": float(result["class_prior_cross_entropy_nats"]),
            "u_probe_bits": float(result["u_probe_bits"]), "brier_score": float(result["brier_score"]),
        })
    predictions_summary = _write_prediction_artifacts(outputs, cache_dir, out_dir, rows)
    endpoint_path = result_dir / "decoded_rgb_endpoints.csv"
    endpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with endpoint_path.open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in endpoint_metrics for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(endpoint_metrics)
    _write_json(result_dir / "checkpoint_accuracy_recheck.json", {"checks": accuracy_checks, "max_abs_accuracy_delta": max(abs(row["delta"]) for row in accuracy_checks), "models": len(accuracy_checks), "new_training": False})
    metadata = {
        "dataset": "Imagenette 320px distribution prepared as 256x256 RGB",
        "samples": len(rows), "split_counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")},
        "manifest": _portable_path(cache_dir / "manifest.jsonl"), "codec_manifest": _portable_path(cache_dir / "codec_manifest.jsonl"),
        "rate_distortion": {**rd_summary, "path": _portable_path(rd_summary["path"])},
        "checkpoint_evaluations": len(model_specs), "checkpoint_accuracy_recheck": _portable_path(result_dir / "checkpoint_accuracy_recheck.json"),
        "decoded_endpoint_rows": len(endpoint_metrics),
        "predictions": {**predictions_summary, "summary_path": _portable_path(predictions_summary["summary_path"]), "logits_cache": _portable_path(predictions_summary["logits_cache"])},
        "device": str(device),
        "source_rgb_hashes_verified": len(rows), "training_performed": False,
    }
    _write_json(result_dir / "generation_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis-only Phase 4.5B RD and existing-checkpoint prediction artifacts.")
    parser.add_argument("--cache-dir", type=Path, default=cache_root() / "neural_accessibility" / "imagenette_320")
    parser.add_argument("--dataset-root", type=Path, default=data_root() / "raw" / "imagenette")
    parser.add_argument("--results-root", type=Path, default=results_root())
    parser.add_argument("--pbc-root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=default_device())
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    metadata = generate_inputs(args)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
