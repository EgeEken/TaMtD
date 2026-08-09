import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codecs.pbc import PBCCodecPair
from tamtd.codecs.pbc_diagnostics import PBCDiagnosticStream
from tamtd.data.cache import load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/cache/pbc_quality_ablation/manifest.jsonl")
    parser.add_argument("--output", default="data/cache/pbc_quality_diagnostics")
    parser.add_argument("--pbc-root", required=True)
    parser.add_argument("--preset", default="quality")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-limit", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for row in load_manifest(args.manifest) if row["codec"] == "pbc_store"]
    pair = PBCCodecPair(pbc_root=args.pbc_root, preset=args.preset)
    variants = ["pbc_init_only", "pbc_residual_only", "pbc_init_zeroed", "pbc_patch_shuffled"]
    definitions = {
        "pbc_init_only": "original header and first three per-channel full-image initialization patches only",
        "pbc_residual_only": "canonical header with zero base values and original residual patches only",
        "pbc_init_zeroed": "full original stream with every initialization grid index replaced by zero; mask, palette bounds, cell size, and field widths retained",
        "pbc_patch_shuffled": "original header and initialization patches followed by deterministically reordered complete residual patch records",
    }
    output_rows = []
    stats = []
    verified = 0
    for row in rows:
        source_path = Path(args.manifest).parent / row["encoded_path"]
        source_data = source_path.read_bytes()
        diagnostic = PBCDiagnosticStream(source_data, pair=pair)
        source_hash = hashlib.sha256(source_data).hexdigest()
        stats.append({"sample_id": row["sample_id"], **diagnostic.stats()})
        generated = diagnostic.variants(args.seed)
        if verified < args.verify_limit:
            original = pair.decode(source_data)
            shuffled = pair.decode(generated["pbc_patch_shuffled"])
            if original.tobytes() != shuffled.tobytes():
                raise AssertionError(f"patch shuffle changed decoded image: {row['sample_id']}")
            verified += 1
        for codec in variants:
            data = generated[codec]
            relative_path = Path("bytes") / codec / f"{row['split']}_{row['sample_id'].split('_')[1]}_v{row['view_id']}.bin"
            path = output / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            output_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "source_image_id": row["source_image_id"],
                    "source_pbc_sample_id": row["sample_id"],
                    "source_pbc_sha256": source_hash,
                    "split": row["split"],
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "view_id": row["view_id"],
                    "codec": codec,
                    "encoded_length": len(data),
                    "encoded_sha256": hashlib.sha256(data).hexdigest(),
                    "encoded_path": str(relative_path).replace("\\", "/"),
                    "transform": row["transform"],
                    "codec_metadata": {
                        "pbc_preset": args.preset,
                        "pbc_diagnostic_source": "same serialized PBC quality stream",
                        "pbc_diagnostic_definition": definitions[codec],
                        "pbc_diagnostic_seed": args.seed,
                        "pbc_init_patch_count": diagnostic.init_count,
                        "pbc_original_patch_count": len(diagnostic.patches),
                        "pbc_git_sha": row.get("codec_metadata", {}).get("pbc_git_sha"),
                    },
                }
            )
    with (output / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "source_manifest": str(Path(args.manifest).resolve()),
                "pbc_root": str(Path(args.pbc_root).resolve()),
                "preset": args.preset,
                "seed": args.seed,
                "variants": variants,
                "definitions": definitions,
                "verified_patch_shuffle_samples": verified,
                "record_count": len(output_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "pbc_patch_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(output_rows), "verified": verified}, indent=2))


if __name__ == "__main__":
    main()
