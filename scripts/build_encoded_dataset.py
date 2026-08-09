import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from PIL import Image
from torchvision.datasets import CIFAR10, ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codecs import JPEGCodec, PBCCodecPair, PBCForcedLZMACodec, PBCStoreCodec, RawRGBCodec
from tamtd.data.transforms import prepare_image


def parse_codecs(value: str) -> list[str]:
    codecs = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"raw_rgb", "pbc_store", "pbc_lzma_forced"}
    for codec in codecs:
        if codec in allowed:
            continue
        if codec.startswith("jpeg_q"):
            try:
                quality = int(codec.removeprefix("jpeg_q"))
            except ValueError as error:
                raise ValueError(f"invalid JPEG codec: {codec}") from error
            if 1 <= quality <= 95:
                continue
        raise ValueError(f"unknown codec: {codec}")
    return codecs


def make_adapters(names, image_size, pbc_root, pbc_preset):
    adapters = {}
    if "raw_rgb" in names:
        adapters["raw_rgb"] = RawRGBCodec((image_size, image_size))
    for name in names:
        if name.startswith("jpeg_q"):
            adapters[name] = JPEGCodec(int(name.removeprefix("jpeg_q")))
    if "pbc_store" in names or "pbc_lzma_forced" in names:
        pair = PBCCodecPair(pbc_root=pbc_root, preset=pbc_preset)
        adapters["pbc_pair"] = pair
        adapters["pbc_store"] = PBCStoreCodec(pbc_root=pbc_root, preset=pbc_preset)
        adapters["pbc_lzma_forced"] = PBCForcedLZMACodec(pbc_root=pbc_root, preset=pbc_preset)
    return adapters


class DatasetView:
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else list(indices)
        self.classes = dataset.classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def load_datasets(dataset_name, data_root, seed, train_fraction):
    data_root = Path(data_root)
    if dataset_name == "cifar10":
        try:
            train = CIFAR10(data_root, train=True, download=False)
            test = CIFAR10(data_root, train=False, download=False)
        except RuntimeError as error:
            raise RuntimeError(
                "CIFAR-10 was not found locally. Download/extract the official dataset so "
                f"{data_root / 'cifar-10-batches-py'} contains data_batch_1..5, test_batch, and batches.meta, "
                "then rerun with --dataset cifar10 --data-root "
                f"{data_root}. No download is attempted by this command."
            ) from error
        return train, test, {"dataset": "cifar10", "data_root": str(data_root.resolve())}
    if dataset_name != "imagefolder":
        raise ValueError(f"unknown dataset: {dataset_name}")
    train_root = data_root / "train"
    test_root = data_root / "test"
    if train_root.is_dir() and test_root.is_dir():
        train = ImageFolder(train_root)
        test = ImageFolder(test_root)
        if train.classes != test.classes:
            raise ValueError("ImageFolder train/test class directories do not match")
        return train, test, {"dataset": "imagefolder", "data_root": str(data_root.resolve()), "split_layout": "train_test"}
    source = ImageFolder(data_root)
    rng = random.Random(seed)
    by_class = {class_id: [] for class_id in range(len(source.classes))}
    for index, (_, class_id) in enumerate(source.samples):
        by_class[class_id].append(index)
    train_indices, test_indices = [], []
    for indices in by_class.values():
        rng.shuffle(indices)
        split = max(1, min(len(indices) - 1, round(len(indices) * train_fraction))) if len(indices) > 1 else len(indices)
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])
    train_indices.sort()
    test_indices.sort()
    if not test_indices:
        raise ValueError("ImageFolder needs at least two images in one or more classes for a train/test split")
    return (
        DatasetView(source, train_indices),
        DatasetView(source, test_indices),
        {
            "dataset": "imagefolder",
            "data_root": str(data_root.resolve()),
            "split_layout": "deterministic_class_stratified_split",
            "train_fraction": train_fraction,
        },
    )


def encode_split(dataset, split, limit, names, adapters, image_size, output_dir, view_count):
    rows = []
    count = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(count):
        image, label = dataset[index]
        source_id = f"{split}:{index}"
        for view_id in range(view_count):
            transformed, transform_metadata = prepare_image(image, image_size)
            encoded = {}
            if "pbc_pair" in adapters and {"pbc_store", "pbc_lzma_forced"}.issubset(names):
                store, forced = adapters["pbc_pair"].encode_pair(transformed)
                encoded[store.codec] = store
                encoded[forced.codec] = forced
            for name in names:
                if name not in encoded:
                    encoded[name] = adapters[name].encode(transformed)
            for name in names:
                sample = encoded[name]
                relative_path = Path("bytes") / name / f"{split}_{index:06d}_v{view_id}.bin"
                path = output_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(sample.data)
                rows.append(
                    {
                        "sample_id": f"{split}_{index:06d}_v{view_id}",
                        "source_image_id": source_id,
                        "split": split,
                        "class_id": int(label),
                        "class_name": dataset.classes[int(label)],
                        "view_id": view_id,
                        "codec": sample.codec,
                        "encoded_length": len(sample.data),
                        "encoded_sha256": hashlib.sha256(sample.data).hexdigest(),
                        "encoded_path": str(relative_path).replace("\\", "/"),
                        "transform": transform_metadata,
                        "codec_metadata": sample.metadata,
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "imagefolder"], default="cifar10")
    parser.add_argument("--data-root", "--dataset-root", dest="data_root", default="data/raw")
    parser.add_argument("--output", required=True)
    parser.add_argument("--codecs", default="raw_rgb,jpeg_q75,pbc_store,pbc_lzma_forced")
    parser.add_argument("--pbc-root", default=None)
    parser.add_argument(
        "--pbc-preset",
        choices=["compression", "balanced", "quality", "high_quality"],
        default="balanced",
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument("--views", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    names = parse_codecs(args.codecs)
    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    adapters = make_adapters(names, args.image_size, args.pbc_root, args.pbc_preset)
    train, test, dataset_metadata = load_datasets(args.dataset, args.data_root, args.seed, args.train_fraction)
    rows = encode_split(train, "train", args.train_limit, names, adapters, args.image_size, output_dir, args.views)
    rows.extend(encode_split(test, "test", args.val_limit, names, adapters, args.image_size, output_dir, 1))
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        **dataset_metadata,
        "train_limit": args.train_limit,
        "validation_limit": args.val_limit,
        "codecs": names,
        "image_size": args.image_size,
        "views": args.views,
        "seed": args.seed,
        "data_root": str(Path(args.data_root).resolve()),
        "pbc_preset": args.pbc_preset if "pbc_store" in names or "pbc_lzma_forced" in names else None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
