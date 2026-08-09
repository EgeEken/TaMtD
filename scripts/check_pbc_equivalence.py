import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codecs import PBCCodecPair
from tamtd.data.cache import load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pbc-root", default=None)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    rows = load_manifest(manifest)
    by_sample = {}
    for row in rows:
        if row["codec"] in {"pbc_store", "pbc_lzma_forced"}:
            by_sample.setdefault(row["sample_id"], {})[row["codec"]] = row
    pair = PBCCodecPair(args.pbc_root)
    checked = 0
    for sample_id, sample_rows in sorted(by_sample.items()):
        if checked >= args.limit:
            break
        if set(sample_rows) != {"pbc_store", "pbc_lzma_forced"}:
            raise ValueError(f"missing PBC pair for {sample_id}")
        store_row = sample_rows["pbc_store"]
        forced_row = sample_rows["pbc_lzma_forced"]
        store_data = (manifest.parent / store_row["encoded_path"]).read_bytes()
        forced_data = (manifest.parent / forced_row["encoded_path"]).read_bytes()
        store_body = pair.unpack_body(store_data)
        forced_body = pair.unpack_body(forced_data)
        if store_body != forced_body:
            raise AssertionError(f"raw PBC bodies differ for {sample_id}")
        store_image = np.asarray(pair.decode(store_data))
        forced_image = np.asarray(pair.decode(forced_data))
        if not np.array_equal(store_image, forced_image):
            raise AssertionError(f"decoded images differ for {sample_id}")
        if store_row["codec_metadata"]["raw_pbc_body_sha256"] != forced_row["codec_metadata"]["raw_pbc_body_sha256"]:
            raise AssertionError(f"body hashes differ for {sample_id}")
        checked += 1
    if checked == 0:
        raise ValueError("no paired PBC samples found")
    print(json.dumps({"checked": checked, "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
