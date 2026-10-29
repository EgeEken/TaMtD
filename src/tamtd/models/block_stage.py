from __future__ import annotations

import torch
from torch import nn


class BlockStageTransformer(nn.Module):
    def __init__(self, continuous_features: int = 62, model_dim: int = 128, layers: int = 3, heads: int = 4) -> None:
        super().__init__()
        self.continuous_projection = nn.Linear(continuous_features, model_dim)
        self.mode_embedding = nn.Embedding(8, model_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, 256, model_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 3,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.output = nn.Linear(model_dim, 10)

    def forward(self, values: torch.Tensor, mode_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (256, 62):
            raise ValueError("continuous block features must have shape [batch, 256, 62]")
        if mode_ids.shape != (*values.shape[:2], 2) or mask.shape != values.shape[:2]:
            raise ValueError("mode IDs and token mask have incompatible shapes")
        positions = self.position_embedding[:, : values.size(1)]
        modes = self.mode_embedding(mode_ids.long().clamp(0, 7)).sum(dim=2)
        tokens = self.continuous_projection(values.float()) + modes + positions
        encoded = self.encoder(tokens, src_key_padding_mask=~mask.bool())
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output(pooled)
