import unittest

import torch

from tamtd.models import ByteCNN


class ModelTests(unittest.TestCase):
    def test_bytecnn_handles_odd_and_short_sequences(self):
        model = ByteCNN(num_classes=10)
        tokens = torch.full((3, 7), 256, dtype=torch.long)
        tokens[0, :7] = torch.arange(7)
        tokens[1, :3] = torch.tensor([10, 20, 30])
        tokens[2, :1] = torch.tensor([40])
        logits = model(tokens, torch.tensor([7, 3, 1]))
        self.assertEqual(logits.shape, (3, 10))
        self.assertTrue(torch.isfinite(logits).all())

    def test_right_padding_does_not_change_logits(self):
        torch.manual_seed(0)
        model = ByteCNN(num_classes=3).eval()
        short = torch.tensor([[1, 2, 256, 256]])
        long = torch.tensor([[1, 2, 256, 256, 256, 256]])
        with torch.no_grad():
            short_logits = model(short, torch.tensor([2]))
            long_logits = model(long, torch.tensor([2]))
        torch.testing.assert_close(short_logits, long_logits)


if __name__ == "__main__":
    unittest.main()
