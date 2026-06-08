import torch
from torch import nn


class JPEGCoefficientDecoderModel(nn.Module):
    def __init__(self, image_size=64, dim=192, depth=6, heads=6):
        super().__init__()
        self.image_size = image_size
        self.grid = image_size // 8
        self.num_blocks = self.grid * self.grid

        self.coeff_projection = nn.Linear(64, dim)
        self.qtable_projection = nn.Linear(64, dim)
        self.pos = nn.Parameter(torch.randn(self.num_blocks, dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.block_decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, 64),
            nn.Sigmoid(),
        )

    def forward(self, coefficients, qtable):
        x = self.coeff_projection(coefficients.float())
        q = self.qtable_projection(qtable.float()).unsqueeze(1)
        x = x + q + self.pos.unsqueeze(0)
        x = self.transformer(x)
        x = self.block_decoder(x)
        bsz = x.size(0)
        x = x.view(bsz, self.grid, self.grid, 1, 8, 8)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(bsz, 1, self.image_size, self.image_size)