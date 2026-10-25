import unittest

import torch

from tamtd.training.accessibility import ByteStreamDataset, byte_stream_lengths, make_loader
from tamtd.training.byte_batching import LengthBucketBatchSampler, padded_width, summarize_byte_batches


class ByteBatchingTests(unittest.TestCase):
    def test_training_bucket_order_is_seeded_and_epoch_specific(self):
        lengths = [((index * 7919) % 12000) + 1 for index in range(160)]
        first = LengthBucketBatchSampler(lengths, 65536, 8, 91, True, bucket_multiplier=3)
        second = LengthBucketBatchSampler(lengths, 65536, 8, 91, True, bucket_multiplier=3)
        first_epoch = list(first)
        self.assertEqual(first_epoch, list(second))
        first.set_epoch(1)
        self.assertNotEqual(first_epoch, list(first))

    def test_all_samples_appear_once_and_budget_is_respected(self):
        lengths = [257 + index * 37 for index in range(73)]
        sampler = LengthBucketBatchSampler(lengths, 16384, 12, 12, True, bucket_multiplier=4)
        batches = list(sampler)
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), list(range(len(lengths))))
        for batch in batches:
            width = padded_width(max(lengths[index] for index in batch))
            self.assertLessEqual(width * len(batch), 16384)
            self.assertLessEqual(len(batch), 12)

    def test_stream_loader_does_not_truncate_and_preserves_validation_order(self):
        streams = [bytes([index]) * length for index, length in enumerate((300, 700, 500, 1100, 900))]
        dataset = ByteStreamDataset(streams, torch.tensor([0, 1, 0, 1, 0]))
        subset = torch.utils.data.Subset(dataset, [0, 1, 2, 3, 4])
        self.assertEqual(byte_stream_lengths(subset), [300, 700, 500, 1100, 900])
        options = {"max_padded_tokens": 4096, "max_samples_per_batch": 4}
        first = make_loader(subset, 4, False, 2, byte_batching=options)
        second = make_loader(subset, 4, False, 2, byte_batching=options)
        observed_a = []
        observed_b = []
        for loader, output in ((first, observed_a), (second, observed_b)):
            for inputs, _, truncation in loader:
                output.extend(inputs["lengths"].tolist())
                self.assertEqual(truncation["truncated_samples"], 0)
                self.assertEqual(inputs["tokens"].shape[1], padded_width(int(inputs["lengths"].max())))
        self.assertEqual(observed_a, observed_b)
        self.assertEqual(sorted(observed_a), sorted(len(stream) for stream in streams))

    def test_validation_batches_are_stably_sorted_by_length(self):
        lengths = [800, 1, 510, 300, 1000, 280]
        sampler = LengthBucketBatchSampler(lengths, 4096, 3, 777, False)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        batch_maxima = [max(lengths[index] for index in batch) for batch in first]
        self.assertEqual(batch_maxima, sorted(batch_maxima))

    def test_padding_metrics_count_real_and_padded_bytes(self):
        metrics = summarize_byte_batches([[1, 255, 256], [257]])
        self.assertEqual(metrics["true_total_input_bytes"], 769)
        self.assertEqual(metrics["padded_total_bytes_or_tokens"], 1280)
        self.assertAlmostEqual(metrics["padding_ratio"], 1 - 769 / 1280)
        self.assertAlmostEqual(metrics["padding_efficiency"], 769 / 1280)
        self.assertEqual(metrics["max_sequence_length"], 257)
        self.assertEqual(metrics["mean_sequence_length"], 769 / 4)
        self.assertEqual(metrics["effective_batch_size_distribution"], {"1": 1, "3": 1})


if __name__ == "__main__":
    unittest.main()
