import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class EnhancedCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

        self.se = SEBlock(256)

        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 32)),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.se(x)

        x = self.reduce(x)       # B, 256, 1, 32
        x = x.squeeze(2)         # B, 256, 32
        x = x.transpose(1, 2)    # B, 32, 256

        return x


def apply_rope(x):
    b, h, t, d = x.shape
    half = d // 2

    freqs = torch.arange(half, device=x.device).float()
    freqs = 1.0 / (10000 ** (freqs / half))

    pos = torch.arange(t, device=x.device).float()
    angles = pos[:, None] * freqs[None, :]

    sin = angles.sin()[None, None, :, :]
    cos = angles.cos()[None, None, :, :]

    x1 = x[..., :half]
    x2 = x[..., half:half * 2]

    out = torch.cat(
        [
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ],
        dim=-1
    )

    if d % 2 == 1:
        out = torch.cat([out, x[..., -1:]], dim=-1)

    return out


class GQAWithRoPE(nn.Module):
    def __init__(self, dim=128, heads=4, kv_groups=2, dropout=0.2):
        super().__init__()

        self.heads = heads
        self.kv_groups = kv_groups
        self.head_dim = dim // heads
        self.heads_per_group = heads // kv_groups

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, kv_groups * self.head_dim)
        self.v_proj = nn.Linear(dim, kv_groups * self.head_dim)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, d = x.shape

        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_groups, self.head_dim).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        k = k.repeat_interleave(self.heads_per_group, dim=1)
        v = v.repeat_interleave(self.heads_per_group, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, d)

        return self.out_proj(out)


class TMSABranch(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.local3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.local5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.local7 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.global_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        xt = x.transpose(1, 2)

        local = (
            self.local3(xt)
            + self.local5(xt)
            + self.local7(xt)
        ).transpose(1, 2)

        global_out, _ = self.global_attn(x, x, x)

        x = self.norm1(x + local + global_out)
        x = self.norm2(x + self.ffn(x))

        return x


class GQABranch(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gqa = GQAWithRoPE(
            dim=dim,
            heads=4,
            kv_groups=2
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        attn_out = self.gqa(x)

        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        return x


class FeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, tmsa_out, gqa_out):
        combined = torch.cat([tmsa_out, gqa_out], dim=-1)

        gate = self.gate(combined)

        fused = gate * tmsa_out + (1 - gate) * gqa_out
        fused = fused + self.feature_fusion(combined)

        return fused


class CompactTCN(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(dim),
            nn.ELU(),

            nn.Conv1d(dim, dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(dim),
            nn.ELU(),

            nn.Conv1d(dim, dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(dim),
            nn.ELU()
        )

    def forward(self, x):
        y = self.block(x.transpose(1, 2)).transpose(1, 2)
        return x + y


class STFTHHATNet(nn.Module):
    def __init__(self, n_classes=4, dim=128):
        super().__init__()

        # Your CNN backbone
        self.cnn_backbone = EnhancedCNNBackbone()

        self.se_attention = SEBlock(256)

        self.feature_projection = nn.Linear(256, dim)

        # Full HHAT modules
        self.tmsa = TMSABranch(dim)
        self.gqa_rope = GQABranch(dim)

        self.feature_fusion = FeatureFusion(dim)
        self.compact_tcn = CompactTCN(dim)

        self.feature_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        # Input: B, 3, 120, 32

        x = self.cnn(x)          # B, 32, 256
        x = self.project(x)      # B, 32, 128

        tmsa_out = self.tmsa(x)
        gqa_out = self.gqa_rope(x)

        x = self.feature_fusion(tmsa_out, gqa_out)
        x = self.compact_tcn(x)

        x = x.mean(dim=1)

        x = self.feature_embedding(x)

        return self.classifier(x)