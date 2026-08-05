import torch
import torch.nn as nn
import torch.nn.functional as F

class GEGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        x, gate = self.proj(x).chunk(2, dim=-1)

        x = x * F.gelu(gate)

        x = self.out(x)

        return self.drop(x)