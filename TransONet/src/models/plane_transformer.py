import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SpatialTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2):
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.local_conv = nn.Conv2d(
            hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, height, width):
        normalized = self.norm1(x)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False)[0]
        x = x + attended

        local = self.fc1(self.norm2(x))
        batch_size, _, channels = local.shape
        local = local.transpose(1, 2).reshape(
            batch_size, channels, height, width)
        local = self.local_conv(local)
        local = local.flatten(2).transpose(1, 2)
        local = self.fc2(self.activation(local))
        return x + local


class HybridPlaneTransformer(nn.Module):
    """Shared U-shaped transformer for a set of dynamic feature planes."""

    def __init__(self, in_channels=32, num_planes=3, dim=128,
                 depth=4, num_heads=4):
        super().__init__()
        middle_dim = dim // 2
        low_dim = dim * 3 // 4
        self.num_planes = num_planes

        self.high_resolution = ConvBlock(in_channels, in_channels)
        self.down1 = nn.Conv2d(in_channels, middle_dim, 2, stride=2)
        self.middle_resolution = ConvBlock(middle_dim, middle_dim)
        self.down2 = nn.Conv2d(middle_dim, low_dim, 2, stride=2)
        self.low_resolution = ConvBlock(low_dim, low_dim)
        self.down3 = nn.Conv2d(low_dim, dim, 2, stride=2)

        self.position = nn.Parameter(torch.zeros(1, 64, dim))
        self.plane_embedding = nn.Parameter(
            torch.zeros(1, num_planes, 1, dim))
        self.blocks = nn.ModuleList([
            SpatialTransformerBlock(dim, num_heads)
            for _ in range(depth)
        ])
        self.plane_norm = nn.LayerNorm(dim)
        self.plane_attention = nn.MultiheadAttention(
            dim, num_heads, batch_first=True)

        self.up1 = nn.ConvTranspose2d(dim, low_dim, 2, stride=2)
        self.decode1 = ConvBlock(low_dim * 2, low_dim)
        self.up2 = nn.ConvTranspose2d(
            low_dim, middle_dim, 2, stride=2)
        self.decode2 = ConvBlock(middle_dim * 2, middle_dim)
        self.up3 = nn.ConvTranspose2d(
            middle_dim, in_channels, 2, stride=2)
        self.decode3 = ConvBlock(in_channels * 2, in_channels)
        self.output = nn.Conv2d(in_channels, in_channels, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.plane_embedding, std=0.02)

    def forward(self, planes):
        batch_size, num_planes, channels, height, width = planes.shape
        x = planes.reshape(batch_size * num_planes, channels, height, width)

        skip_high = self.high_resolution(x)
        x = self.down1(skip_high)
        skip_middle = self.middle_resolution(x)
        x = self.down2(skip_middle)
        skip_low = self.low_resolution(x)
        x = self.down3(skip_low)

        token_height, token_width = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch_size, num_planes, -1, tokens.size(-1))
        tokens = tokens + self.position[:, :tokens.size(2)].unsqueeze(1)
        tokens = tokens + self.plane_embedding[:, :num_planes]

        for block in self.blocks:
            flat_tokens = tokens.reshape(
                batch_size * num_planes, tokens.size(2), tokens.size(3))
            flat_tokens = block(flat_tokens, token_height, token_width)
            tokens = flat_tokens.reshape(
                batch_size, num_planes, tokens.size(2), tokens.size(3))

        summaries = self.plane_norm(tokens.mean(dim=2))
        context = self.plane_attention(
            summaries, summaries, summaries, need_weights=False)[0]
        tokens = tokens + context.unsqueeze(2)

        x = tokens.reshape(
            batch_size * num_planes, -1, tokens.size(-1))
        x = x.transpose(1, 2).reshape(
            batch_size * num_planes, tokens.size(-1),
            token_height, token_width)
        x = self.up1(x)
        x = self.decode1(torch.cat([x, skip_low], dim=1))
        x = self.up2(x)
        x = self.decode2(torch.cat([x, skip_middle], dim=1))
        x = self.up3(x)
        x = self.decode3(torch.cat([x, skip_high], dim=1))
        x = (planes.reshape(batch_size * num_planes, channels, height, width)
             + self.residual_scale * self.output(x))
        return x.reshape(batch_size, num_planes, channels, height, width)
