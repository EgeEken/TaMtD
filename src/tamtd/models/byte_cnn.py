import torch
from torch import nn


class ByteCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 64,
        channels: tuple[int, int, int] = (96, 128, 160),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_token = 256
        self.embedding = nn.Embedding(self.pad_token + 1, embedding_dim, padding_idx=self.pad_token)
        self.convolutions = nn.Sequential(
            nn.Conv1d(embedding_dim, channels[0], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels[0], channels[1], kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(channels[1], channels[2], kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels[2], num_classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        mask = tokens.ne(self.pad_token)
        if lengths is not None:
            valid_lengths = lengths.to(device=tokens.device).clamp_min(1)
        else:
            valid_lengths = mask.sum(dim=1).clamp_min(1)
        x = self.embedding(tokens).transpose(1, 2)
        x = self.convolutions[0](x)
        x = self.convolutions[1](x)
        x = self.convolutions[2](x)
        valid_lengths = (valid_lengths + 1) // 2
        x = self.convolutions[3](x)
        x = self.convolutions[4](x)
        valid_lengths = (valid_lengths + 1) // 2
        x = self.convolutions[5](x)
        positions = torch.arange(x.size(-1), device=x.device).unsqueeze(0)
        mask = (positions < valid_lengths.unsqueeze(1)).to(dtype=x.dtype)
        pooled = (x * mask.unsqueeze(1)).sum(dim=-1) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
