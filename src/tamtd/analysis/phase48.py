from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr


def select_calibration_rows(rows: list[dict], per_class: int = 20) -> list[dict]:
    if per_class < 1:
        raise ValueError("per_class must be positive")
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["label"])].append(row)
    selected = []
    for label, values in sorted(groups.items()):
        ordered = sorted(
            values,
            key=lambda row: hashlib.sha256(
                f"tamtd-phase48-calibration-v1\0{row['image_id']}".encode("utf-8")
            ).digest(),
        )
        if len(ordered) < per_class:
            raise ValueError(f"class {label} has only {len(ordered)} samples; need {per_class}")
        selected.extend(ordered[:per_class])
    return sorted(selected, key=lambda row: int(row["cache_index"]))


def exact_calibration_duplicate(left: list[dict], right: list[dict]) -> bool:
    if not left or len(left) != len(right):
        return False
    by_id = {row["image_id"]: row["stream_sha256"] for row in right}
    return all(by_id.get(row["image_id"]) == row["stream_sha256"] for row in left)


def within_image_correlation(
    rows: list[dict],
    x_field: str,
    y_field: str,
    cluster_field: str = "image_id",
    replicates: int = 2000,
    seed: int = 4801,
) -> dict:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x, y = row.get(x_field), row.get(y_field)
        if x is None or y is None:
            continue
        x, y = float(x), float(y)
        if math.isfinite(x) and math.isfinite(y):
            groups[str(row[cluster_field])].append((x, y))
    paired = {key: values for key, values in groups.items() if len(values) >= 2}
    if len(paired) < 3 or replicates < 1:
        return {"n_images": len(paired), "n_observations": sum(map(len, paired.values())), "pearson_r": None, "spearman_rho": None, "pearson_ci95_low": None, "pearson_ci95_high": None}

    def centered(values: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
        points = np.asarray(values, dtype=np.float64)
        return (points[:, 0] - points[:, 0].mean()).tolist(), (points[:, 1] - points[:, 1].mean()).tolist()

    centered_groups = {key: centered(values) for key, values in paired.items()}

    def correlation(keys: list[str], rank: bool = False) -> float | None:
        x = np.asarray([value for key in keys for value in centered_groups[key][0]], dtype=np.float64)
        y = np.asarray([value for key in keys for value in centered_groups[key][1]], dtype=np.float64)
        if np.std(x) == 0 or np.std(y) == 0:
            return None
        return float((spearmanr if rank else pearsonr)(x, y).statistic)

    keys = list(paired)
    pearson = correlation(keys)
    spearman = correlation(keys, rank=True)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sampled = rng.choice(keys, size=len(keys), replace=True).tolist()
        value = correlation(sampled)
        if value is not None:
            draws.append(value)
    low, high = np.quantile(draws, (0.025, 0.975)) if draws else (None, None)
    return {
        "n_images": len(paired),
        "n_observations": sum(map(len, paired.values())),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "pearson_ci95_low": float(low) if low is not None else None,
        "pearson_ci95_high": float(high) if high is not None else None,
        "bootstrap_replicates": len(draws),
        "centering": "within image identity",
    }
