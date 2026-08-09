from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from tamtd.data.cache import load_manifest


def _features(rows: list[dict[str, Any]], root: Path, kind: str) -> np.ndarray:
    values = []
    for row in rows:
        data = (root / row["encoded_path"]).read_bytes()
        if kind == "length":
            values.append([len(data)])
        elif kind == "histogram":
            histogram = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float32)
            values.append(histogram / max(len(data), 1))
        else:
            raise ValueError(f"unknown baseline: {kind}")
    return np.asarray(values, dtype=np.float32)


def run_baseline(
    manifest_path: str | Path,
    codec: str,
    train_split: str,
    val_split: str,
    kind: str,
    seed: int = 42,
    train_offset: int = 0,
    train_limit: int | None = None,
    val_offset: int = 0,
    val_limit: int | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    rows = load_manifest(manifest_path)
    train_rows = [row for row in rows if row["codec"] == codec and row["split"] == train_split]
    val_rows = [row for row in rows if row["codec"] == codec and row["split"] == val_split]
    train_rows = train_rows[train_offset:]
    val_rows = val_rows[val_offset:]
    if train_limit is not None:
        train_rows = train_rows[:train_limit]
    if val_limit is not None:
        val_rows = val_rows[:val_limit]
    if not train_rows or not val_rows:
        raise ValueError(f"manifest needs train and validation rows for {codec}")
    labels = [int(row["class_id"]) for row in train_rows]
    majority = Counter(labels).most_common(1)[0][0]
    if kind == "majority":
        predictions = np.full(len(val_rows), majority)
        return {
            "model": "majority",
            "codec": codec,
            "accuracy": float(np.mean(predictions == np.asarray([row["class_id"] for row in val_rows]))),
            "parameter_count": 0,
        }
    x_train = _features(train_rows, manifest_path.parent, kind)
    x_val = _features(val_rows, manifest_path.parent, kind)
    classifier = LogisticRegression(max_iter=300, random_state=seed)
    classifier.fit(x_train, labels)
    return {
        "model": kind,
        "codec": codec,
        "accuracy": float(classifier.score(x_val, [int(row["class_id"]) for row in val_rows])),
        "parameter_count": int(classifier.coef_.size + classifier.intercept_.size),
    }
