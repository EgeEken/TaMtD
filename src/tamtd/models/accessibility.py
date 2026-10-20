from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _sequence_mask(lengths: torch.Tensor, width: int, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(width, device=lengths.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).to(dtype=dtype).unsqueeze(1)


class MaskedResidual1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = _sequence_mask(lengths, values.size(-1), values.dtype)
        update = self.depthwise(values * mask)
        update = self.activation(update) * mask
        update = self.pointwise(update) * mask
        return self.activation(values + update) * mask


class ByteStreamCNN(nn.Module):
    pad_token = 256

    def __init__(self, num_classes: int = 10, embedding_dim: int = 32, dropout: float = 0.15) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, embedding_dim, padding_idx=self.pad_token)
        self.stem = nn.Sequential(nn.Conv1d(embedding_dim, 64, 9, stride=8, padding=4), nn.GELU())
        self.block1 = MaskedResidual1d(64)
        self.downsample1 = nn.Sequential(nn.Conv1d(64, 128, 5, stride=4, padding=2), nn.GELU())
        self.block2 = MaskedResidual1d(128)
        self.downsample2 = nn.Sequential(nn.Conv1d(128, 160, 5, stride=2, padding=2), nn.GELU())
        self.block3 = MaskedResidual1d(160)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(160, num_classes)

    @staticmethod
    def _downsample_lengths(lengths: torch.Tensor, stride: int) -> torch.Tensor:
        return torch.div(lengths + stride - 1, stride, rounding_mode="floor").clamp_min(1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or lengths.ndim != 1 or tokens.size(0) != lengths.size(0):
            raise ValueError("tokens and lengths must have shapes [batch, sequence] and [batch]")
        if torch.any(lengths <= 0) or torch.any(lengths > tokens.size(1)):
            raise ValueError("byte lengths must be positive and no greater than the padded width")
        valid_lengths = lengths.to(device=tokens.device)
        values = self.embedding(tokens).transpose(1, 2)
        values = self.stem(values)
        valid_lengths = self._downsample_lengths(valid_lengths, 8)
        values = values * _sequence_mask(valid_lengths, values.size(-1), values.dtype)
        values = self.block1(values, valid_lengths)
        values = self.downsample1(values)
        valid_lengths = self._downsample_lengths(valid_lengths, 4)
        values = values * _sequence_mask(valid_lengths, values.size(-1), values.dtype)
        values = self.block2(values, valid_lengths)
        values = self.downsample2(values)
        valid_lengths = self._downsample_lengths(valid_lengths, 2)
        values = values * _sequence_mask(valid_lengths, values.size(-1), values.dtype)
        values = self.block3(values, valid_lengths)
        mask = _sequence_mask(valid_lengths, values.size(-1), values.dtype)
        pooled = (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))


class PBCInstructionNet(nn.Module):
    def __init__(self, geometry_features: int = 24, content_features: int = 144, num_classes: int = 10) -> None:
        super().__init__()
        self.content_encoder = nn.Sequential(
            nn.Linear(content_features, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.GELU(),
        )
        self.geometry_encoder = nn.Sequential(nn.Linear(geometry_features, 32), nn.GELU())
        self.patch_projection = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.empty_patch = nn.Parameter(torch.zeros(128))
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, num_classes))

    def forward(
        self,
        geometry: torch.Tensor,
        content: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        if geometry.ndim != 3 or content.ndim != 3 or patch_mask.ndim != 2:
            raise ValueError("PBC inputs must have shapes [batch, patches, features]")
        if geometry.shape[:2] != content.shape[:2] or geometry.shape[:2] != patch_mask.shape:
            raise ValueError("PBC patch dimensions do not align")
        values = self.patch_projection(
            torch.cat((self.geometry_encoder(geometry.float()), self.content_encoder(content.float())), dim=-1)
        )
        mask = patch_mask.bool().clone()
        empty = ~mask.any(dim=1)
        if empty.any():
            values = values.clone()
            values[empty, 0] = self.empty_patch
            mask[empty, 0] = True
        values = self.set_encoder(values, src_key_padding_mask=~mask)
        valid = mask.unsqueeze(-1)
        mean = (values * valid.to(dtype=values.dtype)).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        maximum = values.masked_fill(~valid, torch.finfo(values.dtype).min).max(dim=1).values
        return self.classifier(torch.cat((mean, maximum), dim=-1))


class Residual2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.layers(values))


class JPEGDCTNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.y_stem = nn.Sequential(nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.GELU())
        self.y_block = Residual2d(32)
        self.y_downsample = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU()
        )
        self.y_block2 = Residual2d(64)
        self.chroma_stem = nn.Sequential(nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.GELU())
        self.chroma_block = Residual2d(32)
        self.quantization_encoder = nn.Sequential(nn.Linear(128, 128), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            Residual2d(96),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, num_classes)

    def forward(
        self,
        y_coefficients: torch.Tensor,
        cb_coefficients: torch.Tensor,
        cr_coefficients: torch.Tensor,
        quantization_tables: torch.Tensor,
    ) -> torch.Tensor:
        if y_coefficients.ndim != 4 or cb_coefficients.ndim != 4 or cr_coefficients.ndim != 4:
            raise ValueError("DCT coefficient inputs must have shapes [batch, 64, block_y, block_x]")
        if quantization_tables.shape[1:] != (2, 8, 8):
            raise ValueError("JPEG quantization tables must have shape [batch, 2, 8, 8]")
        y = self.y_block(self.y_stem(y_coefficients.float() / 1024.0))
        y = self.y_block2(self.y_downsample(y))
        cb = self.chroma_block(self.chroma_stem(cb_coefficients.float() / 1024.0))
        cr = self.chroma_block(self.chroma_stem(cr_coefficients.float() / 1024.0))
        cb = F.interpolate(cb, size=y.shape[-2:], mode="bilinear", align_corners=False)
        cr = F.interpolate(cr, size=y.shape[-2:], mode="bilinear", align_corners=False)
        quantization = self.quantization_encoder(quantization_tables.float().flatten(1) / 1024.0)
        fused = torch.cat((y, cb, cr), dim=1) + quantization[:, :, None, None]
        return self.classifier(self.fusion(fused).flatten(1))


class SmallBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.activation(self.bn1(self.conv1(values)))
        update = self.bn2(self.conv2(update))
        return self.activation(update + self.shortcut(values))


class RGBResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 32) -> None:
        super().__init__()
        self.in_channels = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(width, 2, stride=1)
        self.layer2 = self._make_layer(width * 2, 2, stride=2)
        self.layer3 = self._make_layer(width * 4, 2, stride=2)
        self.layer4 = self._make_layer(width * 8, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width * 8, num_classes)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [SmallBasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(SmallBasicBlock(channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images.float() / 255.0 - self.rgb_mean) / self.rgb_std
        values = self.stem(images)
        values = self.layer1(values)
        values = self.layer2(values)
        values = self.layer3(values)
        values = self.layer4(values)
        return self.classifier(self.pool(values).flatten(1))
