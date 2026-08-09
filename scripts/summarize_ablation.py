import argparse
import json
from pathlib import Path


RUNS = [
    ("pbc_compression", "pbc", "compression"),
    ("pbc_balanced", "pbc", "balanced"),
    ("pbc_quality", "pbc", "quality"),
    ("pbc_high_quality", "pbc", "high_quality"),
    ("jpeg_q1", "jpeg", "jpeg_q1"),
    ("jpeg_q60", "jpeg", "jpeg_q60"),
    ("jpeg_q75", "jpeg", "jpeg_q75"),
    ("jpeg_q80", "jpeg", "jpeg_q80"),
    ("jpeg_q95", "jpeg", "jpeg_q95"),
]

MANIFESTS = {
    "pbc_compression": "data/cache/pbc_compression_ablation/manifest.jsonl",
    "pbc_balanced": "data/cache/pbc_balanced_ablation/manifest.jsonl",
    "pbc_quality": "data/cache/pbc_quality_ablation/manifest.jsonl",
    "pbc_high_quality": "data/cache/pbc_high_quality_ablation/manifest.jsonl",
    "jpeg_q1": "data/cache/jpeg_quality_ablation/manifest.jsonl",
    "jpeg_q60": "data/cache/jpeg_quality_ablation/manifest.jsonl",
    "jpeg_q75": "data/cache/cifar10_smoke/manifest.jsonl",
    "jpeg_q80": "data/cache/jpeg_quality_ablation/manifest.jsonl",
    "jpeg_q95": "data/cache/jpeg_quality_ablation/manifest.jsonl",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def empirical_chance(manifest_path, codec):
    rows = [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_labels = [row["class_id"] for row in rows if row["codec"] == codec and row["split"] == "train"][:8000]
    test_labels = [row["class_id"] for row in rows if row["codec"] == codec and row["split"] == "test"][:1000]
    majority_label = max(set(train_labels), key=train_labels.count)
    return sum(label == majority_label for label in test_labels) / len(test_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/raw")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.results_root)
    pbc_stats = read_json(root / "pbc_preset_stats.json")
    jpeg_stats = read_json(root / "jpeg_quality_stats.json")
    jpeg_stats.update(read_json(root / "jpeg_q75_stats.json"))
    rows = []
    for name, kind, key in RUNS:
        stats = pbc_stats[key] if kind == "pbc" else jpeg_stats[key]
        bytecnn = read_json(root / f"ablation_bytecnn_{name}/metrics.json")
        decoded = read_json(root / f"ablation_decoded_rgb_{name}/metrics.json")
        length = read_json(root / f"ablation_baseline_length_{name}/metrics.json")
        histogram = read_json(root / f"ablation_baseline_histogram_{name}/metrics.json")
        chance = empirical_chance(MANIFESTS[name], key if kind == "jpeg" else "pbc_store")
        rows.append(
            {
                "representation": name,
                "length_baseline": length["accuracy"],
                "histogram_baseline": histogram["accuracy"],
                "bytecnn_test_accuracy": bytecnn["test_accuracy"],
                "best_validation_accuracy": bytecnn["best_validation_accuracy"],
                "decoded_rgb_test_accuracy": decoded["test_accuracy"],
                "representation_gap": decoded["test_accuracy"] - bytecnn["test_accuracy"],
                "chance_accuracy": chance,
                "semantic_accessibility": (bytecnn["test_accuracy"] - chance) / (decoded["test_accuracy"] - chance),
                "mean_bytes": stats["mean_bytes"],
                "p95_bytes": stats["p95_bytes"],
                "mean_compression_ratio": stats["mean_compression_ratio"],
                "mean_mse": stats["mean_mse"],
                "mean_psnr_db": stats["mean_psnr_db"],
                "pbc_config": stats.get("pbc_config"),
                "jpeg_metadata": stats.get("codec_metadata"),
            }
        )
    output = {row["representation"]: row for row in rows}
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print("| representation | length | histogram | ByteCNN test | best validation | decoded RGB | gap | accessibility | mean bytes | p95 bytes | PSNR dB |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['representation']} | {row['length_baseline']:.3f} | {row['histogram_baseline']:.3f} | "
            f"{row['bytecnn_test_accuracy']:.3f} | {row['best_validation_accuracy']:.3f} | "
            f"{row['decoded_rgb_test_accuracy']:.3f} | {row['representation_gap']:.3f} | "
            f"{row['semantic_accessibility']:.3f} | "
            f"{row['mean_bytes']:.1f} | {row['p95_bytes']:.1f} | {row['mean_psnr_db']:.2f} |"
        )


if __name__ == "__main__":
    main()
