from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import torch
from torch.utils.data import Sampler


def padded_width(length: int, pad_multiple: int = 256) -> int:
    if length < 1 or pad_multiple < 1:
        raise ValueError("length and pad_multiple must be positive")
    return ((length + pad_multiple - 1) // pad_multiple) * pad_multiple


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        max_padded_tokens: int,
        max_samples_per_batch: int,
        seed: int,
        shuffle: bool,
        pad_multiple: int = 256,
        bucket_multiplier: int = 50,
    ) -> None:
        self.lengths = [int(length) for length in lengths]
        if not self.lengths or min(self.lengths) < 1:
            raise ValueError("lengths must contain positive values")
        if max_padded_tokens < padded_width(max(self.lengths), pad_multiple):
            raise ValueError("max_padded_tokens must fit the longest padded sample")
        if max_samples_per_batch < 1 or bucket_multiplier < 1:
            raise ValueError("batch and bucket limits must be positive")
        self.max_padded_tokens = int(max_padded_tokens)
        self.max_samples_per_batch = int(max_samples_per_batch)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.pad_multiple = int(pad_multiple)
        self.bucket_multiplier = int(bucket_multiplier)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _pack(self, indices: list[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        batch: list[int] = []
        max_length = 0
        for index in indices:
            next_max = max(max_length, self.lengths[index])
            next_count = len(batch) + 1
            padded_tokens = padded_width(next_max, self.pad_multiple) * next_count
            if batch and (
                next_count > self.max_samples_per_batch
                or padded_tokens > self.max_padded_tokens
            ):
                batches.append(batch)
                batch = []
                max_length = 0
                next_count = 1
                next_max = self.lengths[index]
                padded_tokens = padded_width(next_max, self.pad_multiple)
            if padded_tokens > self.max_padded_tokens:
                raise ValueError(f"sample {index} exceeds the padded-token budget")
            batch.append(index)
            max_length = next_max
        if batch:
            batches.append(batch)
        return batches

    def _batches(self) -> list[list[int]]:
        if not self.shuffle:
            return self._pack(sorted(range(len(self.lengths)), key=lambda i: (self.lengths[i], i)))

        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shuffled = torch.randperm(len(self.lengths), generator=generator).tolist()
        bucket_size = self.max_samples_per_batch * self.bucket_multiplier
        batches: list[list[int]] = []
        for start in range(0, len(shuffled), bucket_size):
            bucket = shuffled[start : start + bucket_size]
            bucket.sort(key=lambda i: (self.lengths[i], i))
            batches.extend(self._pack(bucket))
        if len(batches) > 1:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]
        return batches

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


def summarize_byte_batches(
    batch_lengths: Iterable[Sequence[int]], pad_multiple: int = 256
) -> dict:
    true_total = 0
    padded_total = 0
    sample_count = 0
    max_length = 0
    max_width = 0
    max_padded_tokens = 0
    max_batch_size = 0
    batch_sizes: Counter[int] = Counter()
    for lengths in batch_lengths:
        if not lengths:
            continue
        count = len(lengths)
        width = padded_width(max(int(length) for length in lengths), pad_multiple)
        true_total += sum(int(length) for length in lengths)
        padded_total += width * count
        sample_count += count
        max_length = max(max_length, max(int(length) for length in lengths))
        max_width = max(max_width, width)
        max_padded_tokens = max(max_padded_tokens, width * count)
        max_batch_size = max(max_batch_size, count)
        batch_sizes[count] += 1
    if sample_count == 0:
        raise ValueError("at least one nonempty batch is required")
    padding_ratio = 1.0 - true_total / padded_total
    return {
        "sample_count": sample_count,
        "batch_count": sum(batch_sizes.values()),
        "true_total_input_bytes": true_total,
        "padded_total_bytes_or_tokens": padded_total,
        "padding_ratio": padding_ratio,
        "padding_efficiency": 1.0 - padding_ratio,
        "max_sequence_length": max_length,
        "mean_sequence_length": true_total / sample_count,
        "max_padded_batch_width": max_width,
        "max_padded_batch_tokens": max_padded_tokens,
        "max_samples_per_batch": max_batch_size,
        "effective_batch_size_distribution": {
            str(size): count for size, count in sorted(batch_sizes.items())
        },
    }
