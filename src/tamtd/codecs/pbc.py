import hashlib
import importlib
import lzma
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from .base import CodecAdapter, EncodedSample


class PBCDependencyError(RuntimeError):
    pass


def resolve_pbc_root(pbc_root: str | os.PathLike[str] | None = None) -> Path:
    value = pbc_root or os.environ.get("PBC_ROOT")
    if not value:
        raise PBCDependencyError(
            "PBC experiments require the EgeEken/PBC repository. "
            "Set PBC_ROOT or pass --pbc-root /path/to/PBC."
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir() or not (root / "PBC3.py").is_file():
        raise PBCDependencyError(f"PBC_ROOT does not contain PBC3.py: {root}")
    return root


def _load_pbc(pbc_root: Path):
    if "NUMBA_CACHE_DIR" not in os.environ:
        cache_dir = Path(tempfile.gettempdir()) / "tamtd-numba-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    if str(pbc_root) not in sys.path:
        sys.path.insert(0, str(pbc_root))
    importlib.invalidate_caches()
    try:
        pbc3 = importlib.import_module("PBC3")
        types = importlib.import_module("pbc3_types")
    except ModuleNotFoundError as error:
        raise PBCDependencyError(
            f"Could not import PBC from {pbc_root}; missing dependency/module {error.name}. "
            "Install the PBC repository dependencies and retry."
        ) from error
    return pbc3.PBC3, types.PBC3Config


def _git_state(root: Path) -> dict[str, Any]:
    state = {
        "pbc_git_sha": None,
        "pbc_git_branch": None,
        "pbc_worktree_dirty": None,
        "pbc_status_check_timed_out": False,
    }
    try:
        sha = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(root), "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )
        status = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        state["pbc_worktree_dirty"] = True
        state["pbc_status_check_timed_out"] = True
        return state
    except (OSError, subprocess.CalledProcessError):
        return state
    state.update(
        {
            "pbc_git_sha": sha.stdout.strip() or None,
            "pbc_git_branch": branch.stdout.strip() or None,
            "pbc_worktree_dirty": bool(status.stdout.strip()),
        }
    )
    return state


class PBCCodecPair:
    """Encode one image once, then expose STORE and forced-LZMA wrappers."""

    def __init__(
        self,
        pbc_root: str | os.PathLike[str] | None = None,
        preset: str = "balanced",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.pbc_root = resolve_pbc_root(pbc_root)
        self.pbc_class, self.config_class = _load_pbc(self.pbc_root)
        preset_factory = getattr(self.config_class, preset, None)
        if not callable(preset_factory):
            raise ValueError(f"unknown PBC preset: {preset}")
        self.preset = preset
        defaults = dict(preset_factory().__dict__)
        defaults["use_lzma"] = False
        if defaults.get("learned_filler_model_path"):
            defaults["learned_filler_model_path"] = str(
                (self.pbc_root / defaults["learned_filler_model_path"]).resolve()
            )
        self.config_values = {**defaults, **(config or {}), "use_lzma": False}
        if self.config_values.get("learned_filler_model_path"):
            model_path = Path(self.config_values["learned_filler_model_path"])
            if not model_path.is_absolute():
                self.config_values["learned_filler_model_path"] = str(
                    (self.pbc_root / model_path).resolve()
                )
        self.git_state = _git_state(self.pbc_root)

    def _config(self):
        return self.config_class(**self.config_values)

    def _compress_store(self, image: Image.Image) -> tuple[bytes, bytes, dict[str, Any]]:
        result = self.pbc_class.compress(image, config=self._config())
        data = bytes(result.data)
        if data[:4] != self.pbc_class.MAGIC or len(data) < 6:
            raise ValueError("PBC3 returned an invalid framed stream")
        if data[5] != self.pbc_class.ENTROPY_STORE:
            raise ValueError("PBC STORE encoding unexpectedly returned a compressed stream")
        body = data[6:]
        mse = None if result.mse is None else float(result.mse)
        psnr = None if mse is None else (float("inf") if mse == 0 else 10.0 * math.log10((255.0**2) / mse))
        return data, body, {
            **self.git_state,
            "pbc_preset": self.preset,
            "pbc_config": dict(self.config_values),
            "pbc_encoder_use_lzma": bool(self.config_values["use_lzma"]),
            "pbc_mse": mse,
            "pbc_psnr_db": psnr,
            "raw_pbc_body_length": len(body),
            "raw_pbc_body_sha256": hashlib.sha256(body).hexdigest(),
        }

    def _forced_lzma(self, store_data: bytes, raw_body: bytes) -> bytes:
        filters = getattr(self.pbc_class, "_LZMA_FILTERS", None)
        if filters is None:
            raise PBCDependencyError("This PBC checkout does not expose its LZMA2 filter configuration")
        wrapped = lzma.compress(raw_body, format=lzma.FORMAT_RAW, filters=filters)
        return store_data[:4] + bytes([store_data[4], self.pbc_class.ENTROPY_LZMA]) + wrapped

    def encode_pair(self, image: Image.Image) -> tuple[EncodedSample, EncodedSample]:
        store_data, raw_body, metadata = self._compress_store(image)
        forced_data = self._forced_lzma(store_data, raw_body)
        shared = {**metadata, "entropy_mode": self.pbc_class.ENTROPY_STORE}
        store = EncodedSample(
            store_data,
            "pbc_store",
            {**shared, "wrapped_length": len(store_data) - 6},
        )
        forced = EncodedSample(
            forced_data,
            "pbc_lzma_forced",
            {
                **metadata,
                "entropy_mode": self.pbc_class.ENTROPY_LZMA,
                "wrapped_length": len(forced_data) - 6,
            },
        )
        return store, forced

    def unpack_body(self, data: bytes) -> bytes:
        if data[:4] != self.pbc_class.MAGIC or len(data) < 6:
            raise ValueError("not a PBC3 stream")
        method = data[5]
        if method == self.pbc_class.ENTROPY_STORE:
            return data[6:]
        if method == self.pbc_class.ENTROPY_LZMA:
            return lzma.decompress(
                data[6:],
                format=lzma.FORMAT_RAW,
                filters=self.pbc_class._LZMA_FILTERS,
            )
        raise ValueError(f"unknown PBC entropy method {method}")

    def decode(self, data: bytes) -> Image.Image:
        return self.pbc_class.decompress(data).image.convert("RGB")


class _PBCAdapter(CodecAdapter):
    def __init__(self, pbc_root=None, preset="balanced", config=None) -> None:
        self.pair = PBCCodecPair(pbc_root=pbc_root, preset=preset, config=config)

    def decode(self, data: bytes) -> Image.Image:
        return self.pair.decode(data)


class PBCStoreCodec(_PBCAdapter):
    name = "pbc_store"

    def encode(self, image: Image.Image) -> EncodedSample:
        return self.pair.encode_pair(image)[0]


class PBCForcedLZMACodec(_PBCAdapter):
    name = "pbc_lzma_forced"

    def encode(self, image: Image.Image) -> EncodedSample:
        return self.pair.encode_pair(image)[1]
