import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveLayerNorm(nn.Module):

    def __init__(self, n_channels, c_dim):
        super().__init__()
        self.n_channels = n_channels
        self.fc = nn.Sequential(nn.SiLU(),
                                nn.Linear(c_dim, 2 * n_channels, bias=True))

    def forward(self, x, c):
        x = F.layer_norm(x, self.n_channels)
        scale, bias = torch.chunk(self.fc(c), chunks=2, dim=1)
        return x.mul(1 + scale).add(bias)


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 fc_dim: int,
                 c_dim: int,
                 dropout: float = 0.1,
                 norm_first: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_c = nn.Sequential(nn.SiLU(),
                                     nn.Linear(c_dim, 4 * embed_dim, bias=True))
        self.norm_first = norm_first
        self.self_attn = nn.MultiheadAttention(embed_dim,
                                               num_heads,
                                               dropout=dropout,
                                               kdim=None,
                                               vdim=None,
                                               batch_first=True)
        self.linear1 = nn.Linear(embed_dim, fc_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(fc_dim, embed_dim, bias=True)

        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, bias=False)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()  # nn.GELU()

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        gamma1, beta1, gamma2, beta2 = torch.chunk(self.embed_c(c), chunks=4, dim=1)
        if not self.norm_first:
            h = self.self_attn(x, x, x, need_weights=False)[0]
            h = self.dropout1(h)
            x = self.norm1(x + h)
            x = x.mul(1 + gamma1).add(beta1)
            h = self.linear1(x)
            h = self.activation(h)
            h = self.dropout(h)
            h = self.linear2(h)
            h = self.dropout2(h)
            x = self.norm2(x + h)
            x = x.mul(1 + gamma2).add(beta2)
        elif self.norm_first:
            h = self.norm1(x)
            try:
                h = h.mul(1 + gamma1).add(beta1)
            except:
                print(h.shape,gamma1.shape)
            h = self.self_attn(h, h, h, need_weights=False)[0]
            h = self.dropout1(h)
            x = x + h
            h = self.norm2(h)
            h = h.mul(1 + gamma2).add(beta2)
            h = self.linear1(h)
            h = self.activation(h)
            h = self.dropout(h)
            h = self.linear2(h)
            h = self.dropout2(h)
            x = x + h
        return x


class AdaLNZeroTransformerEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 fc_dim: int,
                 c_dim: int,
                 dropout: float = 0.1,
                 norm_first: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_c = nn.Sequential(nn.SiLU(),
                                     nn.Linear(c_dim, 6 * embed_dim, bias=True))
        self.norm_first = norm_first
        assert norm_first
        self.self_attn = nn.MultiheadAttention(embed_dim,
                                               num_heads,
                                               dropout=dropout,
                                               kdim=None,
                                               vdim=None,
                                               batch_first=True)
        self.linear1 = nn.Linear(embed_dim, fc_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(fc_dim, embed_dim, bias=True)

        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, bias=False)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()  # nn.GELU()

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = torch.chunk(self.embed_c(c), chunks=6, dim=1)
        h = self.norm1(x)
        h = h.mul(1 + gamma1).add(beta1)
        h = self.self_attn(h, h, h, need_weights=False)[0]
        h = self.dropout1(h)
        x = x + h.mul(alpha1)
        h = self.norm2(h)
        h = h.mul(1 + gamma2).add(beta2)
        h = self.linear1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout2(h)
        x = x + h.mul(alpha2)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self,
                 n_layers: int,
                 embed_dim: int,
                 num_heads: int,
                 fc_dim: int,
                 c_dim: int,
                 dropout: float = 0.1,
                 norm_first: bool = True,
                 **ignoredkwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim,
                                                             num_heads,
                                                             fc_dim,
                                                             c_dim,
                                                             dropout,
                                                             norm_first)
                                     for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        for layer in self.layers:
            x = layer(x, c)
        return x


