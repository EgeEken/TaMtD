import torch
from torch import nn


class JPEGDecoderModel(nn.Module):
    def __init__(
        self,
        image_size=64,
        max_bytes=4096,
        patch_size=16,
        image_patch_size=4,
        dim=192,
        depth=6,
        heads=6,
        grayscale=True,
    ):
        super().__init__()
        self.image_size = image_size
        self.max_bytes = max_bytes
        self.patch_size = patch_size
        self.image_patch_size = image_patch_size
        self.channels = 1 if grayscale else 3
        self.num_byte_patches = max_bytes // patch_size
        self.image_grid = image_size // image_patch_size
        self.num_image_patches = self.image_grid * self.image_grid

        if max_bytes % patch_size != 0:
            raise ValueError("max_bytes must be divisible by patch_size")
        if image_size % image_patch_size != 0:
            raise ValueError("image_size must be divisible by image_patch_size")

        self.byte_embedding = nn.Embedding(257, dim, padding_idx=256)
        self.byte_pos = nn.Parameter(torch.randn(patch_size, dim) * 0.02)
        self.byte_patch_projection = nn.Sequential(
            nn.LayerNorm(patch_size * dim),
            nn.Linear(patch_size * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.header_projection = nn.Linear(9, dim)
        self.byte_patch_pos = nn.Parameter(torch.randn(self.num_byte_patches, dim) * 0.02)
        self.image_queries = nn.Parameter(torch.randn(self.num_image_patches, dim) * 0.02)
        self.image_pos = nn.Parameter(torch.randn(self.num_image_patches, dim) * 0.02)

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
        self.patch_decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, self.channels * image_patch_size * image_patch_size),
            nn.Sigmoid(),
        )

    def forward(self, byte_tokens, markers):
        bsz = byte_tokens.size(0)
        byte_tokens = byte_tokens[:, : self.max_bytes]
        if byte_tokens.size(1) < self.max_bytes:
            pad = byte_tokens.new_full((bsz, self.max_bytes - byte_tokens.size(1)), 256)
            byte_tokens = torch.cat([byte_tokens, pad], dim=1)

        x = byte_tokens.view(bsz, self.num_byte_patches, self.patch_size)
        x = self.byte_embedding(x) + self.byte_pos.view(1, 1, self.patch_size, -1)
        x = x.reshape(bsz, self.num_byte_patches, self.patch_size * x.size(-1))
        x = self.byte_patch_projection(x) + self.byte_patch_pos.unsqueeze(0)

        header = self.header_projection(markers.float()).unsqueeze(1)
        queries = self.image_queries.unsqueeze(0).expand(bsz, -1, -1) + self.image_pos.unsqueeze(0)
        x = torch.cat([header, x, queries], dim=1)
        x = self.transformer(x)
        x = x[:, -self.num_image_patches :]
        x = self.patch_decoder(x)

        p = self.image_patch_size
        x = x.view(bsz, self.image_grid, self.image_grid, self.channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(bsz, self.channels, self.image_size, self.image_size)


class JPEGDecoder:
    def __init__(self, model=None, device=None, **model_kwargs):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or JPEGDecoderModel(**model_kwargs)
        self.model.to(self.device)

    @torch.no_grad()
    def decode(self, syntax):
        self.model.eval()
        byte_tokens = syntax.bytes.unsqueeze(0).to(self.device)
        markers = syntax.markers.unsqueeze(0).to(self.device)
        return self.model(byte_tokens, markers).squeeze(0).cpu()