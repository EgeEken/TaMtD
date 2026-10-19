import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


SAMPLE_COLUMNS = [
    "image_id",
    "codec_id",
    "codec_family",
    "encoded_bytes",
    "bpp",
    "decode_valid",
    "mse",
    "psnr_db",
    "psnr_infinite",
    "ssim",
    "encode_ms_runs",
    "decode_ms_runs",
    "encode_deterministic",
    "codec_metadata",
    "error",
]


def image_metrics(reference: Image.Image, decoded: Image.Image) -> dict[str, float]:
    source = np.asarray(reference.convert("RGB"), dtype=np.float64)
    output = np.asarray(decoded.convert("RGB"), dtype=np.float64)
    if source.shape != output.shape:
        raise ValueError(f"image dimensions differ: {source.shape} != {output.shape}")
    mse = float(np.mean((source - output) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10((255.0**2) / mse))
    mu_source = cv2.GaussianBlur(source, (11, 11), 1.5)
    mu_output = cv2.GaussianBlur(output, (11, 11), 1.5)
    sigma_source = cv2.GaussianBlur(source * source, (11, 11), 1.5) - mu_source * mu_source
    sigma_output = cv2.GaussianBlur(output * output, (11, 11), 1.5) - mu_output * mu_output
    sigma_cross = cv2.GaussianBlur(source * output, (11, 11), 1.5) - mu_source * mu_output
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ssim = ((2 * mu_source * mu_output + c1) * (2 * sigma_cross + c2)) / (
        (mu_source * mu_source + mu_output * mu_output + c1)
        * (sigma_source + sigma_output + c2)
    )
    return {"mse": mse, "psnr_db": psnr, "ssim": float(np.mean(ssim))}


def write_sample_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["encode_ms_runs"] = json.dumps(row.get("encode_ms_runs", []))
            serialized["decode_ms_runs"] = json.dumps(row.get("decode_ms_runs", []))
            serialized["codec_metadata"] = json.dumps(
                row.get("codec_metadata", {}), sort_keys=True
            )
            writer.writerow({column: serialized.get(column) for column in SAMPLE_COLUMNS})


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["codec_id"])].append(row)
    summary = []
    for codec_id, group in grouped.items():
        rate_rows = [row for row in group if row.get("encoded_bytes") is not None]
        valid_rows = [row for row in rate_rows if row.get("decode_valid")]
        encode_times = [value for row in group for value in row.get("encode_ms_runs", [])]
        decode_times = [value for row in group for value in row.get("decode_ms_runs", [])]
        metadata = next((row.get("codec_metadata") for row in group if row.get("codec_metadata")), {})
        entropy_modes = Counter(
            row.get("codec_metadata", {}).get("pbc_entropy_mode")
            for row in group
            if row.get("codec_metadata", {}).get("pbc_entropy_mode")
        )

        def stat(values: list[float], kind: str) -> float | None:
            if not values:
                return None
            if kind == "mean":
                return float(np.mean(values))
            if kind == "median":
                return float(np.median(values))
            return float(np.percentile(values, 95))

        sizes = [float(row["encoded_bytes"]) for row in rate_rows]
        bpps = [float(row["bpp"]) for row in rate_rows]
        psnrs = [float(row["psnr_db"]) for row in valid_rows if row.get("psnr_db") is not None]
        ssims = [float(row["ssim"]) for row in valid_rows if row.get("ssim") is not None]
        summary.append(
            {
                "codec_id": codec_id,
                "codec_family": group[0]["codec_family"],
                "settings": metadata,
                "pbc_entropy_modes": dict(entropy_modes),
                "samples": len(group),
                "valid_samples": len(valid_rows),
                "mean_bytes": stat(sizes, "mean"),
                "median_bytes": stat(sizes, "median"),
                "mean_bpp": stat(bpps, "mean"),
                "mean_psnr_db": stat(psnrs, "mean"),
                "mean_ssim": stat(ssims, "mean"),
                "mean_encode_ms": stat(encode_times, "mean"),
                "median_encode_ms": stat(encode_times, "median"),
                "p95_encode_ms": stat(encode_times, "p95"),
                "median_decode_ms": stat(decode_times, "median"),
                "p95_decode_ms": stat(decode_times, "p95"),
            }
        )
    return sorted(summary, key=lambda row: (row["codec_family"], row["mean_bpp"] or 0))


def select_operating_points(summary: list[dict[str, Any]]) -> dict[str, Any]:
    pbc_rows = [row for row in summary if row["codec_family"] == "pbc"]
    comparisons: dict[str, Any] = {}
    for pbc in pbc_rows:
        by_codec = {}
        for family in ("jpeg", "webp", "avif", "jxl"):
            candidates = [
                row
                for row in summary
                if row["codec_family"] == family and row["mean_bpp"] is not None
            ]
            if not candidates:
                by_codec[family] = {"available": False}
                continue
            nearest_rate = min(candidates, key=lambda row: abs(row["mean_bpp"] - pbc["mean_bpp"]))
            rate_delta = nearest_rate["mean_bpp"] - pbc["mean_bpp"]
            rate_tolerance = max(0.025, 0.10 * pbc["mean_bpp"])
            psnr_candidates = [row for row in candidates if row["mean_psnr_db"] is not None]
            nearest_psnr = min(
                psnr_candidates,
                key=lambda row: abs(row["mean_psnr_db"] - pbc["mean_psnr_db"]),
            ) if psnr_candidates else None
            ssim_candidates = [row for row in candidates if row["mean_ssim"] is not None]
            nearest_ssim = min(
                ssim_candidates,
                key=lambda row: abs(row["mean_ssim"] - pbc["mean_ssim"]),
            ) if ssim_candidates else None
            by_codec[family] = {
                "available": True,
                "rate_match": {
                    "codec_id": nearest_rate["codec_id"],
                    "bpp": nearest_rate["mean_bpp"],
                    "delta_bpp": rate_delta,
                    "relative_delta": rate_delta / pbc["mean_bpp"] if pbc["mean_bpp"] else None,
                    "close": abs(rate_delta) <= rate_tolerance,
                    "tolerance_bpp": rate_tolerance,
                },
                "psnr_match": None if nearest_psnr is None else {
                    "codec_id": nearest_psnr["codec_id"],
                    "psnr_db": nearest_psnr["mean_psnr_db"],
                    "delta_psnr_db": nearest_psnr["mean_psnr_db"] - pbc["mean_psnr_db"],
                    "ssim": nearest_psnr["mean_ssim"],
                    "delta_ssim": nearest_psnr["mean_ssim"] - pbc["mean_ssim"],
                    "close": abs(nearest_psnr["mean_psnr_db"] - pbc["mean_psnr_db"]) <= 1.0,
                },
                "ssim_match": None if nearest_ssim is None else {
                    "codec_id": nearest_ssim["codec_id"],
                    "ssim": nearest_ssim["mean_ssim"],
                    "delta_ssim": nearest_ssim["mean_ssim"] - pbc["mean_ssim"],
                    "psnr_db": nearest_ssim["mean_psnr_db"],
                    "delta_psnr_db": nearest_ssim["mean_psnr_db"] - pbc["mean_psnr_db"],
                    "close": abs(nearest_ssim["mean_ssim"] - pbc["mean_ssim"]) <= 0.02,
                },
            }
        comparisons[pbc["codec_id"]] = {
            "pbc_bpp": pbc["mean_bpp"],
            "pbc_psnr_db": pbc["mean_psnr_db"],
            "pbc_ssim": pbc["mean_ssim"],
            "codecs": by_codec,
        }
    return comparisons
