import json
import statistics
from pathlib import Path


ROOT = Path("results/raw")
CORE = {
    "pbc_quality": "pbc_quality",
    "pbc_high_quality": "pbc_high_quality",
    "jpeg_q80": "jpeg_q80",
    "jpeg_q95": "jpeg_q95",
    "jpeg_q1": "jpeg_q1",
    "pbc_quality_lzma": "pbc_quality_lzma",
}


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def values(prefix, name):
    return [read(f"{prefix}_{name}_seed{seed}/metrics.json") for seed in [42, 123, 777]]


def mean_std(items, key):
    values = [item[key] for item in items]
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def main():
    invariant = {}
    for display, name in CORE.items():
        hist = values("attribution_histogram_mlp", name)
        deep = read(f"attribution_deepsets_{name}_seed42/metrics.json")
        ordered = [read(f"ablation_bytecnn_{name}/metrics.json")]
        repeat_paths = [ROOT / f"repeat_bytecnn_{name}_seed{seed}/metrics.json" for seed in [123, 777]]
        if all(path.exists() for path in repeat_paths):
            ordered.extend([json.loads(path.read_text(encoding="utf-8")) for path in repeat_paths])
        shuffled = read(f"shuffled_bytecnn_{name}_seed42/metrics.json")
        linear = read(f"ablation_baseline_histogram_{name}/metrics.json")
        invariant[display] = {
            "linear_histogram": linear["accuracy"],
            "histogram_mlp": mean_std(hist, "test_accuracy"),
            "deepsets_seed42": deep["test_accuracy"],
            "shuffled_bytecnn_seed42": shuffled["test_accuracy"],
            "ordered_bytecnn": mean_std(ordered, "test_accuracy") if len(ordered) > 1 else ordered[0]["test_accuracy"],
        }

    diagnostic = {}
    for name in ["pbc_init_only", "pbc_residual_only", "pbc_init_zeroed", "pbc_patch_shuffled"]:
        bytecnn = read(f"attribution_bytecnn_{name}_seed42/metrics.json")
        diagnostic[name] = {
            "linear_histogram": read(f"attribution_baseline_histogram_{name}/metrics.json")["accuracy"],
            "length": read(f"attribution_baseline_length_{name}/metrics.json")["accuracy"],
            "histogram_mlp_seed42": read(f"attribution_histogram_mlp_{name}_seed42/metrics.json")["test_accuracy"] if (ROOT / f"attribution_histogram_mlp_{name}_seed42/metrics.json").exists() else None,
            "deepsets_seed42": read(f"attribution_deepsets_{name}_seed42/metrics.json")["test_accuracy"] if (ROOT / f"attribution_deepsets_{name}_seed42/metrics.json").exists() else None,
            "bytecnn_seed42": bytecnn["test_accuracy"],
        }
    diagnostic["pbc_full"] = {
        "linear_histogram": read("ablation_baseline_histogram_pbc_quality/metrics.json")["accuracy"],
        "histogram_mlp_seed42": read("attribution_histogram_mlp_pbc_quality_seed42/metrics.json")["test_accuracy"],
        "deepsets_seed42": read("attribution_deepsets_pbc_quality_seed42/metrics.json")["test_accuracy"],
        "bytecnn_seed42": read("ablation_bytecnn_pbc_quality/metrics.json")["test_accuracy"],
    }
    output = {
        "protocol": {"train": 8000, "validation": 2000, "test": 1000, "seeds": [42, 123, 777], "test_evaluations": 1},
        "invariant_baselines": invariant,
        "pbc_diagnostics": diagnostic,
        "rgb_reference": {
            "original": read("ablation_decoded_rgb_original_rgb/metrics.json"),
            "full_pbc_quality": read("ablation_decoded_rgb_pbc_quality/metrics.json"),
            "init_only": read("attribution_decoded_rgb_pbc_init_only/metrics.json"),
        },
        "patch_order": {
            "ordered": read("ablation_bytecnn_pbc_quality/metrics.json")["test_accuracy"],
            "patch_shuffled": read("attribution_bytecnn_pbc_patch_shuffled_seed42/metrics.json")["test_accuracy"],
            "byte_shuffled": read("shuffled_bytecnn_pbc_quality_seed42/metrics.json")["test_accuracy"],
        },
    }
    path = ROOT / "attribution_summary.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
