import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveLayerNorm(nn.Module):

    def __init__(self, n_channels, c_dim):
        super().__init__()
        self.n_channels = n_channels
        self.fc = nn.Sequential(nn.SiLU(),
                                nn.Linear(c_dim, 2 * n_channels, bias=True))

    def forward(self, x, c=None):
        beta = torch.mean(x, dim=1, keepdim=True)
        alpha = torch.var(x, dim=1, keepdim=True, unbiased=False).sqrt()
        x = (x - beta) / (alpha + 1e-5)
        scale, bias = torch.chunk(self.fc(c), chunks=2, dim=1)
        scale = scale[:, :, None, None]
        bias = bias[:, :, None, None]
        return x.mul(1 + scale).add(bias)


class AdaptiveGroupNorm(nn.Module):

    def __init__(self, n_channels, c_dim, num_groups=32):
        super().__init__()
        self.num_groups = num_groups
        self.n_channels = n_channels
        self.fc = nn.Sequential(nn.SiLU(),
                                nn.Linear(c_dim, 2 * n_channels, bias=True))

    def forward(self, x, c=None):
        x = F.group_norm(x, self.num_groups, eps=1e-5)
        scale, bias = torch.chunk(self.fc(c), chunks=2, dim=1)
        scale = scale[:, :, None, None]
        bias = bias[:, :, None, None]
        return x.mul(1 + scale).add(bias)


class TransformerEncoder(nn.Module):
    pass
