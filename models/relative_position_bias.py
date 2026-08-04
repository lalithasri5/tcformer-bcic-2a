import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    """
    Learnable Relative Position Bias.

    Returns:
        bias of shape (num_heads, seq_len, seq_len)
    """

    def __init__(self, num_heads: int, max_position: int = 128):
        super().__init__()

        self.num_heads = num_heads
        self.max_position = max_position

        self.relative_bias_table = nn.Parameter(
            torch.zeros(2 * max_position - 1, num_heads)
        )

        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def forward(self, seq_len: int):
        device = self.relative_bias_table.device

        coords = torch.arange(seq_len, device=device)

        relative_coords = coords[:, None] - coords[None, :]

        relative_coords = relative_coords.clamp(
            -self.max_position + 1,
            self.max_position - 1
        )

        relative_coords += self.max_position - 1

        bias = self.relative_bias_table[relative_coords]

        # (seq_len, seq_len, num_heads)
        bias = bias.permute(2, 0, 1).contiguous()

        # (num_heads, seq_len, seq_len)
        return bias