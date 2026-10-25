from __future__ import annotations

import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_environment(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    path = Path(value).expanduser() if value else default
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def data_root() -> Path:
    return _path_from_environment("DATA_ROOT", PROJECT_ROOT / "data")


def cache_root() -> Path:
    return _path_from_environment("CACHE_ROOT", data_root() / "cache")


def results_root() -> Path:
    return _path_from_environment("RESULTS_ROOT", PROJECT_ROOT / "results")


def pbc_root() -> Path | None:
    value = os.environ.get("PBC_ROOT")
    return _path_from_environment("PBC_ROOT", Path(value)) if value else None


def temp_root() -> Path:
    value = os.environ.get("TEMP_ROOT") or os.environ.get("TMPDIR") or os.environ.get("TEMP") or tempfile.gettempdir()
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def default_device() -> str:
    return os.environ.get("CUDA_DEVICE") or os.environ.get("DEVICE") or "auto"
