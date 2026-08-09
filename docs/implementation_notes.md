# Initial implementation notes

The repository currently contains a PDM project with a minimal `pyproject.toml` and legacy JPEG/AV1 prototype files under `Decoder Models/`. Those files are preserved; the new experiment code lives under `src/tamtd/`.

The current environment has PyTorch, Pillow, NumPy, scikit-learn, torchvision, pandas, PyYAML, and PyArrow available. The repository does not contain a local PBC checkout. PBC support therefore uses `PBC_ROOT` or `--pbc-root` and fails with an actionable error when the dependency is missing.

The first cache format is deterministic JSONL plus one byte file per encoded sample. The cache builder uses the same transformed image for every codec and encodes PBC STORE and forced-LZMA from one shared PBC3 body when both are requested.

## Local dataset layout

For CIFAR-10, place the extracted torchvision directory at `data/raw/cifar-10-batches-py/` with `data_batch_1` through `data_batch_5`, `test_batch`, and `batches.meta`. The builder uses `download=False`:

```bash
python scripts/build_encoded_dataset.py --dataset cifar10 --data-root data/raw --output data/cache/cifar10_smoke --codecs raw_rgb,jpeg_q75,pbc_store,pbc_lzma_forced --pbc-root /path/to/PBC
```

For a local image directory, use either `data/imagefolder/train/<class>/*.jpg` and `data/imagefolder/test/<class>/*.jpg`, or a single `data/imagefolder/<class>/*.jpg` tree, which is split deterministically:

```bash
python scripts/build_encoded_dataset.py --dataset imagefolder --data-root data/imagefolder --output data/cache/imagefolder_smoke --codecs raw_rgb,jpeg_q75
```
