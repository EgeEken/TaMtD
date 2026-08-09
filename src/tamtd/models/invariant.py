import torch
from torch import nn


def histogram_features(tokens: torch.Tensor, lengths: torch.Tensor, max_length: int = 3072) -> torch.Tensor:
    mask = torch.arange(tokens.size(1), device=tokens.device)[None, :] < lengths[:, None]
    values = tokens.masked_fill(~mask, 0)
    hist = torch.zeros(tokens.size(0), 256, device=tokens.device)
    hist.scatter_add_(1, values, mask.float())
    hist = hist / lengths.clamp_min(1).float().unsqueeze(1)
    return torch.cat([hist, (lengths.float() / max_length).unsqueeze(1)], dim=1)


class HistogramMLP(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(257, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, tokens, lengths):
        return self.network(histogram_features(tokens, lengths, 3072))


class DeepSets(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, 64)
        self.per_byte = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(257, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, tokens, lengths):
        mask = torch.arange(tokens.size(1), device=tokens.device)[None, :] < lengths[:, None]
        byte_lookup = self.per_byte(self.embedding.weight)
        values = byte_lookup[tokens]
        pooled = (values * mask.unsqueeze(2)).sum(dim=1) / lengths.clamp_min(1).float().unsqueeze(1)
        features = torch.cat([pooled, (lengths.float() / 3072).unsqueeze(1)], dim=1)
        return self.classifier(features)
