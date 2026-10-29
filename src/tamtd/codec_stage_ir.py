from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "tamtd.codec-stage-ir"
SCHEMA_VERSION = 1
TOKEN_GRID = (16, 16)
FEATURE_FIELDS = (
    "x_norm", "y_norm",
    *(f"coeff_{plane}_{index}" for plane in ("y", "u", "v") for index in range(8)),
    *(f"quant_{index}" for index in range(8)),
    *(f"mode_hist_{index}" for index in range(8)),
    "chroma_mode", "is_small_intra", "skip", "segment",
    "nonzero_y", "nonzero_chroma", "transform_log2_w", "transform_log2_h",
    "coeff_valid_y", "coeff_valid_u", "coeff_valid_v",
    *(f"quant_valid_{index}" for index in range(8)),
    "mode_valid",
)


def validate_ir(metadata: dict, features: np.ndarray) -> None:
    if metadata.get("schema") != SCHEMA or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported codec-stage IR schema")
    if metadata.get("pre_reconstruction") is True and any(
        key in metadata for key in ("rgb", "pixels", "yuv_pixels", "reconstructed_image")
    ):
        raise ValueError("pre-reconstruction IR cannot contain reconstructed pixels")
    if features.ndim != 2 or features.shape != (TOKEN_GRID[0] * TOKEN_GRID[1], len(FEATURE_FIELDS)):
        raise ValueError("IR feature grid does not match the canonical token schema")
    if not np.isfinite(features).all():
        raise ValueError("IR features must be finite")


def serialize_ir_metadata(metadata: dict) -> bytes:
    return (json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def verify_stream_alignment(base_rows: list[dict], stream_rows: list[dict], stream_path: str | Path) -> None:
    if len(base_rows) != len(stream_rows):
        raise ValueError("stream manifest sample count differs from the base manifest")
    expected_offset = 0
    with Path(stream_path).open("rb") as stream:
        for index, (base, encoded) in enumerate(zip(base_rows, stream_rows, strict=True)):
            for key in ("cache_index", "image_id", "label", "split", "rgb_sha256"):
                if encoded.get(key) != base.get(key):
                    raise ValueError(f"stream manifest {key} mismatch at row {index}")
            reference = encoded["stream"]
            if int(reference["offset"]) != expected_offset:
                raise ValueError(f"stream offset is not contiguous at row {index}")
            stream.seek(int(reference["offset"]))
            data = stream.read(int(reference["length"]))
            if len(data) != int(reference["length"]):
                raise ValueError(f"stream length mismatch at row {index}")
            if hashlib.sha256(data).hexdigest() != reference["sha256"]:
                raise ValueError(f"stream hash mismatch at row {index}")
            expected_offset += len(data)
        if stream.seek(0, 2) != expected_offset:
            raise ValueError("stream file has unreferenced trailing bytes")


def prediction_mode_ids(features: np.ndarray) -> np.ndarray:
    histogram = features[:, 34:42]
    primary = histogram.argmax(axis=1).astype(np.int64)
    chroma = np.clip(np.rint(features[:, 42]), 0, 7).astype(np.int64)
    return np.stack((primary, chroma), axis=1)


def model_continuous_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32).copy()
    values[:, 2:26] = np.sign(values[:, 2:26]) * np.log1p(np.abs(values[:, 2:26])) / 8.0
    values[:, 26:34] = np.log1p(np.maximum(values[:, 26:34], 0)) / 8.0
    values[:, 34:42] /= 16.0
    values[:, 42:43] /= 7.0
    values[:, 45:46] /= 15.0
    values[:, 48:50] /= 6.0
    return values


def adapt_common_tokens(features: np.ndarray, token_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_ir(
        {"schema": SCHEMA, "schema_version": SCHEMA_VERSION},
        np.asarray(features),
    )
    mask = np.ones(256, dtype=np.bool_) if token_mask is None else np.asarray(token_mask, dtype=np.bool_)
    if mask.shape != (256,):
        raise ValueError("token mask must have shape [256]")
    return model_continuous_features(features), prediction_mode_ids(features), mask


def adapt_webp_tokens(features: np.ndarray, token_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return adapt_common_tokens(features, token_mask)


def adapt_av1_tokens(features: np.ndarray, token_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return adapt_common_tokens(features, token_mask)
