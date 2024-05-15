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
                             ckpt_path=ae_config['ckpt_path'])

    def forward(self, x):
        return self.model(x).sample

    def encode(self, x, mode=True):
        dist = self.model.encode(x).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()

    def decode(self, x):
        return self.model.decode(x).sample