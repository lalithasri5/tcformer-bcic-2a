import torch
import torch.nn as nn


class SimAM(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module
    """

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: [B,C,H,W]

        n = x.shape[2] * x.shape[3] - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)

        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)
        ) + 0.5

        return x * torch.sigmoid(y)