from taming import VQModel
from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self,
                 ae_config):

        super().__init__()
        ddconfig = ae_config['dd_config']
        n_embed = ae_config['n_embed']
        embed_dim = ae_config['embed_dim']
        self.model = VQModel(ddconfig,
                             n_embed,
                             embed_dim,
                             ckpt_path=ae_config['ckpt_path']).eval().requires_grad_(False)

    def forward(self, x):
        idx = self.encode(x)
        return idx, self.decode(idx)

    def encode(self, x):
        quant, _, (_, _, idx) = self.model.encode(x)  # idx[b,h,w]
        return idx

    def decode(self,idx):
        quant = self.model.quantize.embedding[idx]
        return self.model.decode(quant)

    def decode_quant(self, x):
        return self.model.decode(x)