import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_encoded_dataset import load_datasets


class DatasetSourceTests(unittest.TestCase):
    def test_imagefolder_is_split_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for class_name in ("class_a", "class_b"):
                class_dir = root / class_name
                class_dir.mkdir(parents=True)
                for index in range(4):
                    image = Image.fromarray(np.full((8, 8, 3), index, dtype=np.uint8), "RGB")
                    image.save(class_dir / f"{index}.png")
            train_a, test_a, metadata_a = load_datasets("imagefolder", root, seed=42, train_fraction=0.5)
            train_b, test_b, metadata_b = load_datasets("imagefolder", root, seed=42, train_fraction=0.5)
            self.assertEqual(train_a.indices, train_b.indices)
            self.assertEqual(test_a.indices, test_b.indices)
            self.assertEqual(len(train_a) + len(test_a), 8)
            self.assertEqual(metadata_a, metadata_b)


if __name__ == "__main__":
    unittest.main()
