import torch
from torch import nn
import torch.nn.functional as F
from autoencoder import AutoEncoder
from modules import TransformerEncoder


class MaskGIT(nn.Module):
    # image <-> token_idx

    def __init__(self,
                 maskgit_config,
                 ae_config):
        super(MaskGIT, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.latent_size = ae_config['latent_size']
        self.n_pos = self.latent_size[0] * self.latent_size[1]
        self.maskgit = MaskGITIndex(**maskgit_config)
        self.ae = AutoEncoder(ae_config)

    def train_step(self, token_idx, class_idx):  # [batch_size, n_pos], [batch_size,]
        out = self.maskgit.train_val_step(token_idx, class_idx)
        loss = self.maskgit.calculate_loss(token_idx,out['logits'],out['mask'])
        return loss['tot_loss'], loss['log_perplexity']

    @torch.no_grad()
    def conditional_generation(self, temperature=(1, 1), n_steps=10, b=9, c=0):
        return self.maskgit.conditional_generation(temperature, n_steps, b, c)

    @torch.no_grad()
    def conditional_generation_gradually(self, temperature=(1, 1), n_steps=10, b=4, c=0):
        return self.maskgit.conditional_generation_gradually(temperature, n_steps, b, c)

    @torch.no_grad()
    def decode(self, idx):
        idx = idx.view(idx.shape[0], *self.latent_size).contiguous()
        return torch.clip(self.ae.decode(idx), -1, 1)

    @torch.no_grad()
    def encode(self, imgs: torch.Tensor):
        token_map = self.ae.encode(imgs)
        flattened_token_map = token_map.view(imgs.shape[0], self.n_pos).contiguous()
        return flattened_token_map


class MaskGITIndex(nn.Module):
    # during training: unmasked_token_idx[b, seq] + unmasked_class_idx[b,] ->
    #                  (masked_token_tensor + masked_class_tensor -> logits) -> loss[scalar]
    # during inference: class_idx[b,] -> (class_tensor -> logits) -> token_map[b, seq]
    # seq = h * w = n_pos

    def __init__(self,
                 n_tokens=512,
                 n_pos=143,
                 embed_dim=128,
                 num_heads=8,
                 fc_dim=1024,
                 n_layers=6,
                 n_steps=10,
                 c_dim=256,
                 n_classes=1000,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.n_steps = n_steps
        self.n_pos = n_pos
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.n_classes = n_classes

        self.kernel = MaskGITTensor(n_tokens=n_tokens,
                                    n_pos=n_pos,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    fc_dim=fc_dim,
                                    n_layers=n_layers,
                                    n_steps=n_steps,
                                    c_dim=c_dim,
                                    n_classes=n_classes,
                                    dropout=dropout)

        weight = torch.randn(n_tokens, embed_dim)
        self.token_embed = nn.Parameter(weight, requires_grad=True)
        nn.init.trunc_normal_(self.token_embed, 0, 0.02)
        weight = torch.randn(n_pos, embed_dim)
        self.pos_embed = nn.Parameter(weight, requires_grad=True)
        nn.init.trunc_normal_(self.pos_embed, 0, 0.02)
        weight = torch.randn(n_classes + 1, c_dim)
        self.class_embed = nn.Parameter(weight, requires_grad=True)
        nn.init.trunc_normal_(self.pos_embed, 0, 0.02)

        self.mask_embed = nn.Parameter(torch.zeros([embed_dim, ]), requires_grad=True)

        # self.embed_out = nn.ModuleList([nn.GELU(),
        #                                 nn.LayerNorm(embed_dim)])

    def forward(self, masked_embedding, c):
        # only used for training !!!
        # for layer in self.embed_out:
        #     h = layer(h)
        logits = self.kernel(masked_embedding, c, self.token_embed)
        return logits

    @torch.no_grad()
    def calculate_n_mask(self, x: torch.Tensor):
        n = torch.cos(x * 3.1415926535 / 2) * self.n_pos
        n = torch.round(n).int()
        return n

    @torch.no_grad()
    def sample_mask(self, b):
        n = self.calculate_n_mask(x=torch.rand((b,)))
        mask = torch.full((b, self.n_pos), False, dtype=torch.bool)
        r = torch.rand((b, self.n_pos,))
        for i in range(b):
            _, selected_positions = torch.topk(r[i], k=n[i], dim=-1)  # (n_masked,)
            mask[i, selected_positions] = True
        class_mask = torch.rand((b,)) < 1 / (self.n_classes + 1)
        return mask, class_mask

    def embed(self, idx, c_idx, mask=None, class_mask=None):  # ind/mask [batch_size,n_pos] or [n_pos,]
        embedding = self.token_embed[idx]  # [batch_size,n_pos,embed_dim] or [n_pos,embed_dim]
        if mask is not None:
            embedding[mask] = self.mask_embed  # [n_masked,embed_dim] <- [embed_dim,]
        embedding = embedding + self.pos_embed  # [(batch_size,)n_pos,embed_dim]+[n_pos,embed_dim]
        if class_mask is not None:
            c_idx = c_idx.detach()
            c_idx[class_mask] = self.n_classes
        class_embedding = self.class_embed[c_idx]
        return embedding, class_embedding

    def train_val_step(self, token_idx, class_idx):  # [batch_size, n_pos], [batch_size,]
        mask, class_mask = self.sample_mask(b=token_idx.shape[0])
        token_embedding, class_embedding = self.embed(token_idx, class_idx,
                                                      mask=mask, class_mask=class_mask)
        logits = self.forward(masked_embedding=token_embedding, c=class_embedding)
        return {'logits': logits, 'mask': mask}

    def calculate_loss(self, x, logits, mask):
        logits_ = logits[mask].view(-1, self.n_tokens).contiguous()
        x_ = x[mask].view(-1).contiguous().long()
        ce_loss = F.cross_entropy(logits_,
                                  target=x_,
                                  label_smoothing=0.1).mean()
        log_proba = F.log_softmax(logits_.detach(), dim=-1)
        log_perplexity = -torch.gather(log_proba, dim=1, index=x_.unsqueeze(-1)).mean()
        return {"tot_loss": ce_loss, "log_perplexity": log_perplexity}

    @torch.no_grad()
    def conditional_generation_gradually(self, temperature=(1, 1), n_steps=10, b=4, c=0):
        if n_steps is None:
            n_steps = self.n_steps
        self.eval()
        ind_ls = []
        class_idx = torch.full((b,), c, dtype=torch.long)
        current_ind = (torch.rand((b, self.n_pos)) * (self.n_tokens - 1)).long()  # [b,n_pos]
        n_masked = self.n_pos
        mask = torch.full((b, self.n_pos,), True, dtype=torch.bool)  # [b, n_pos,]
        for t in range(1, n_steps + 1):
            embedding, class_embedding = self.embed(current_ind, class_idx,
                                                    mask=mask, class_mask=None)
            logits = self.forward(masked_embedding=embedding,
                                  c=class_embedding)  # [b, n_pos, n_tokens]

            # sample confident idx start
            masked_logits = logits.clone()[mask] / temperature[0]  # [b * n_masked, n_tokens]
            token_dis = torch.distributions.categorical.Categorical(logits=masked_logits)
            token_sample = token_dis.sample()  # [b * n_masked,]
            current_ind[mask] = token_sample
            token_confidence = torch.gather(token_dis.probs,  # [b * n_masked,]
                                            dim=-1,
                                            index=token_sample.unsqueeze(-1)).squeeze(-1)
            token_confidence = token_confidence.view([b, n_masked])
            sorted_confidence, _ = torch.sort(token_confidence,  # [b, n_masked]
                                              dim=1, descending=True)
            n = self.calculate_n_mask(x=torch.tensor(t / n_steps,
                                                     dtype=torch.float32).view(1, ))
            dn = n_masked - n
            n_masked = n
            threshold_confidence = sorted_confidence[:, dn][:, None]  # [b, 1]
            confident_token_flag = (token_confidence > threshold_confidence).view(-1).cpu()  # [b * n_masked]
            # current_ind[mask] = torch.where(confident_token_flag,
            #                                 token_sample.cpu(),  # [b * n_masked,]
            #                                 current_ind[mask])  # [b * n_masked,]
            mask[mask.clone()] = ~confident_token_flag
            # sample confident idx end
            assert torch.abs(torch.sum(mask) - n_masked).cpu() <= 1
            ind_ls.append(current_ind.clone())
        return ind_ls  # list[n_step, tensor[b, n_pos]]

    @torch.no_grad()
    def conditional_generation(self, temperature=(1, 1), n_steps=10, b=9, c=0):
        if n_steps is None:
            n_steps = self.n_steps
        self.eval()
        class_idx = torch.full((b,), c, dtype=torch.long)
        current_ind = (torch.rand((b, self.n_pos)) * (self.n_tokens - 1)).long()  # [b,n_pos]
        n_masked = self.n_pos
        mask = torch.full((b, self.n_pos,), True, dtype=torch.bool)  # [b, n_pos,]
        for t in range(1, n_steps + 1):
            embedding, class_embedding = self.embed(current_ind, class_idx,
                                                    mask=mask, class_mask=None)
            logits = self.forward(masked_embedding=embedding,
                                  c=class_embedding)  # [b, n_pos, n_tokens]

            # sample confident idx start
            masked_logits = logits.clone()[mask] / temperature[0]  # [b * n_masked, n_tokens]
            token_dis = torch.distributions.categorical.Categorical(logits=masked_logits)
            token_sample = token_dis.sample()  # [b * n_masked,]
            current_ind[mask] = token_sample
            token_confidence = torch.gather(token_dis.probs,  # [b * n_masked,]
                                            dim=-1,
                                            index=token_sample.unsqueeze(-1)).squeeze(-1)
            token_confidence = token_confidence.view([b, n_masked])
            sorted_confidence, _ = torch.sort(token_confidence,  # [b, n_masked]
                                              dim=1, descending=True)
            n = self.calculate_n_mask(x=torch.tensor(t / n_steps,
                                                     dtype=torch.float32).view(1, ))
            dn = n_masked - n
            n_masked = n
            threshold_confidence = sorted_confidence[:, dn][:, None]  # [b, 1]
            confident_token_flag = (token_confidence > threshold_confidence).view(-1).cpu()  # [b * n_masked]
            # sample confident idx end
            mask[mask.clone()] = ~confident_token_flag
            assert torch.abs(torch.sum(mask) - n_masked).cpu() <= 1
        return current_ind  # [b, n_pos]


class MaskGITTensor(nn.Module):
    # masked_token_tensor[b, seq, dim] + masked_class_tensor[b, c_dim]
    # -> logits[b, seq, n_tokens]
    # seq = h * w = n_pos

    def __init__(self,
                 n_tokens=512,
                 n_pos=143,
                 embed_dim=128,
                 num_heads=8,
                 fc_dim=1024,
                 n_layers=6,
                 c_dim=256,
                 dropout=0.1,
                 **ignoredkwargs):
        super().__init__()

        self.encoder = TransformerEncoder(n_layers=n_layers,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          fc_dim=fc_dim,
                                          c_dim=c_dim,
                                          dropout=dropout)

        self.proj_out = nn.ModuleList([nn.Linear(embed_dim, embed_dim),
                                       nn.GELU(),
                                       nn.LayerNorm(embed_dim)])
        self.bias = nn.Parameter(torch.zeros([n_pos, n_tokens]), requires_grad=True)

    def forward(self, masked_embedding, c, token_embed):
        h = self.encoder(masked_embedding, c)
        for layer in self.proj_out:
            h = layer(h)
        logits = torch.matmul(h, token_embed.T) + self.bias
        return logits
