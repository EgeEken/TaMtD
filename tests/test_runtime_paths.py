import os
import tempfile
import unittest
from pathlib import Path

from tamtd.codecs.pbc import _resolve_model_path
from tamtd.runtime import cache_root, data_root, results_root, temp_root


class RuntimePathTests(unittest.TestCase):
    def test_environment_roots_resolve_relative_to_project(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory).resolve()
            old = {name: os.environ.get(name) for name in ("DATA_ROOT", "CACHE_ROOT", "RESULTS_ROOT", "TEMP_ROOT")}
            try:
                os.environ.update({
                    "DATA_ROOT": str(root / "data"),
                    "CACHE_ROOT": "",
                    "RESULTS_ROOT": str(root / "results"),
                    "TEMP_ROOT": str(root / "tmp"),
                })
                self.assertEqual(data_root(), root / "data")
                self.assertEqual(cache_root(), root / "data" / "cache")
                self.assertEqual(results_root(), root / "results")
                self.assertEqual(temp_root(), root / "tmp")
            finally:
                for name, value in old.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def test_windows_pbc_checkpoint_path_maps_under_linux_checkout(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            (root / "patch_policy.npz").write_bytes(b"test")
            resolved = _resolve_model_path(
                root,
                r"C:\Users\research\PBC\patch_policy.npz",
            )
            self.assertEqual(resolved, str((root / "patch_policy.npz").resolve()))


if __name__ == "__main__":
    unittest.main()
