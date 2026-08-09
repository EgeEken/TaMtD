import math

import torch
from torch import nn

from .invariant import histogram_features


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class SequenceBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, kernel_size=5, padding=2, groups=width)
        self.pointwise = nn.Conv1d(width, width * 2, kernel_size=1)
        self.output = nn.Conv1d(width * 2, width, kernel_size=1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        y = torch.nn.functional.gelu(self.pointwise(y))
        y = self.output(y).transpose(1, 2)
        return x + y * valid.unsqueeze(2)


class ByteReconstructor(nn.Module):
    def __init__(
        self,
        target_size: int = 8,
        max_length: int = 3072,
        width: int = 64,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        self.pad_token = 256
        self.max_length = max_length
        self.target_size = target_size
        self.embedding = nn.Embedding(self.pad_token + 1, width, padding_idx=self.pad_token)
        self.stem = nn.Conv1d(width, width, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList([SequenceBlock(width) for _ in range(blocks)])
        self.sequence_norm = nn.LayerNorm(width)
        self.queries = nn.Parameter(torch.randn(target_size * target_size, width) * 0.02)
        self.cross_attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)
        self.register_buffer("position_encoding", self._position_encoding(max_length, width), persistent=False)

    @staticmethod
    def _position_encoding(length: int, width: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
        encoding = torch.zeros(length, width)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        return encoding

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        if tokens.size(1) > self.max_length:
            raise ValueError(f"sequence length exceeds max_length={self.max_length}")
        valid = torch.arange(tokens.size(1), device=tokens.device)[None, :] < lengths[:, None]
        x = self.embedding(tokens) + self.position_encoding[: tokens.size(1)].to(tokens.device)
        x = torch.nn.functional.gelu(self.stem(x.transpose(1, 2))).transpose(1, 2)
        x = x * valid.unsqueeze(2)
        for block in self.blocks:
            x = block(x, valid)
        x = self.sequence_norm(x)
        queries = self.queries.unsqueeze(0).expand(tokens.size(0), -1, -1)
        attended, _ = self.cross_attention(
            queries,
            x,
            x,
            key_padding_mask=~valid,
            need_weights=False,
        )
        output = self.output(self.output_norm(attended)).squeeze(-1)
        return torch.sigmoid(output).view(-1, self.target_size, self.target_size)


class HistogramReconstructor(nn.Module):
    def __init__(self, target_size: int = 8, max_length: int = 3072) -> None:
        super().__init__()
        self.max_length = max_length
        self.target_size = target_size
        self.network = nn.Sequential(
            nn.Linear(257, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, target_size * target_size),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        output = self.network(histogram_features(tokens, lengths, self.max_length))
        return torch.sigmoid(output).view(-1, self.target_size, self.target_size)
