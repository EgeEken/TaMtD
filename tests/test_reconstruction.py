import unittest

import torch

from tamtd.models.reconstruction import ByteReconstructor, HistogramReconstructor


class ReconstructionModelTests(unittest.TestCase):
    def test_models_return_8x8_outputs(self):
        tokens = torch.full((3, 17), 256, dtype=torch.long)
        tokens[0, :17] = torch.arange(17)
        tokens[1, :5] = torch.tensor([10, 20, 30, 40, 50])
        tokens[2, :1] = torch.tensor([60])
        lengths = torch.tensor([17, 5, 1])
        for model in (ByteReconstructor(max_length=32), HistogramReconstructor(max_length=32)):
            output = model(tokens, lengths)
            self.assertEqual(output.shape, (3, 8, 8))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(((output >= 0) & (output <= 1)).all())


if __name__ == "__main__":
    unittest.main()
