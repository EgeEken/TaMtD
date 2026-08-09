import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codecs.pbc_diagnostics import PBCDiagnosticStream
from tamtd.codecs.pbc import PBCCodecPair
from tamtd.data.cache import load_manifest
def pack_ranges(body, patches):
    from pbc3_types import BitWriter

    writer = BitWriter()
    for patch in patches:
        for bit_position in range(patch["start"], patch["end"]):
            value = (body[bit_position // 8] >> (7 - bit_position % 8)) & 1
            writer.write(value, 1)
    return writer.finish()


def entropy(data):
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0] / len(data)
    return float(-(probabilities * np.log2(probabilities)).sum())


def top_histogram(counts, total):
    return [
        {"byte": int(index), "fraction": float(counts[index] / max(total, 1))}
        for index in np.argsort(counts)[-8:][::-1]
    ]


def summary(values):
    if not values:
        return {}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def main():
    pbc_manifest = Path("data/cache/pbc_quality_ablation/manifest.jsonl")
    jpeg_manifest = Path("data/cache/jpeg_quality_ablation/manifest.jsonl")
    pbc_root = r"C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC"
    pair = PBCCodecPair(pbc_root=pbc_root, preset="quality")
    pbc_rows = [row for row in load_manifest(pbc_manifest) if row["codec"] == "pbc_store"]
    pbc_classes = defaultdict(lambda: {"length": [], "body_entropy": [], "init_length": [], "residual_length": [], "patch_count": [], "residual_patch_count": [], "width": [], "height": [], "cell_size": [], "negative_max": [], "positive_max": [], "base_values": [[], [], []], "init_hist": np.zeros(256, dtype=np.int64), "residual_hist": np.zeros(256, dtype=np.int64), "init_bytes": 0, "residual_bytes": 0})
    for row in pbc_rows:
        data = (pbc_manifest.parent / row["encoded_path"]).read_bytes()
        diagnostic = PBCDiagnosticStream(data, pair=pair)
        body = diagnostic.body
        init = diagnostic.patches[:diagnostic.init_count]
        residual = diagnostic.patches[diagnostic.init_count:]
        init_data = pack_ranges(body, init)
        residual_data = pack_ranges(body, residual)
        values = pbc_classes[int(row["class_id"])]
        values["length"].append(len(data))
        values["body_entropy"].append(entropy(body))
        values["init_length"].append(len(init_data))
        values["residual_length"].append(len(residual_data))
        values["patch_count"].append(len(diagnostic.patches))
        values["residual_patch_count"].append(len(residual))
        values["width"].extend([patch["w"] for patch in residual])
        values["height"].extend([patch["h"] for patch in residual])
        values["cell_size"].extend([patch["cell_size"] for patch in residual])
        values["negative_max"].extend([patch["negative_max"] for patch in residual])
        values["positive_max"].extend([patch["positive_max"] for patch in residual])
        for channel, value in enumerate(diagnostic.header["base_values"]):
            values["base_values"][channel].append(value)
        values["init_hist"] += np.bincount(np.frombuffer(init_data, dtype=np.uint8), minlength=256)
        values["residual_hist"] += np.bincount(np.frombuffer(residual_data, dtype=np.uint8), minlength=256)
        values["init_bytes"] += len(init_data)
        values["residual_bytes"] += len(residual_data)

    output = {"pbc_quality": {}}
    for class_id, values in sorted(pbc_classes.items()):
        output["pbc_quality"][str(class_id)] = {
            "stream_length": summary(values["length"]),
            "body_entropy": summary(values["body_entropy"]),
            "init_length": summary(values["init_length"]),
            "residual_length": summary(values["residual_length"]),
            "patch_count": summary(values["patch_count"]),
            "residual_patch_count": summary(values["residual_patch_count"]),
            "residual_width": summary(values["width"]),
            "residual_height": summary(values["height"]),
            "residual_cell_size": summary(values["cell_size"]),
            "residual_negative_max": summary(values["negative_max"]),
            "residual_positive_max": summary(values["positive_max"]),
            "base_value_means": [summary(values) for values in values["base_values"]],
            "init_top_bytes": top_histogram(values["init_hist"], values["init_bytes"]),
            "residual_top_bytes": top_histogram(values["residual_hist"], values["residual_bytes"]),
        }

    jpeg_rows = load_manifest(jpeg_manifest)
    for codec in ["jpeg_q1", "jpeg_q80", "jpeg_q95"]:
        by_class = defaultdict(lambda: {"length": [], "entropy": [], "hist": np.zeros(256, dtype=np.int64), "bytes": 0})
        for row in jpeg_rows:
            if row["codec"] != codec:
                continue
            data = (jpeg_manifest.parent / row["encoded_path"]).read_bytes()
            values = by_class[int(row["class_id"])]
            values["length"].append(len(data))
            values["entropy"].append(entropy(data))
            values["hist"] += np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
            values["bytes"] += len(data)
        output[codec] = {
            str(class_id): {
                "stream_length": summary(values["length"]),
                "byte_entropy": summary(values["entropy"]),
                "top_bytes": top_histogram(values["hist"], values["bytes"]),
            }
            for class_id, values in sorted(by_class.items())
        }
    path = Path("results/raw/pbc_quality_statistics.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "pbc_samples": len(pbc_rows)}, indent=2))


if __name__ == "__main__":
    main()
