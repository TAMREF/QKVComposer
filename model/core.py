import torch
import numpy as np
from torch import nn
from omegaconf import DictConfig

class Transformer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=cfg.model.d_hidden,
            nhead=cfg.model.n_head,
            num_encoder_layers=cfg.model.num_layers,
            num_decoder_layers=cfg.model.num_layers,
            dim_feedforward=cfg.model.d_ff,
            dropout=cfg.model.dropout
        )
        self.cfg = cfg
        self.embedding = nn.Embedding(
            num_embeddings=cfg.model.num_tokens,
            embedding_dim=cfg.model.d_embed,
            padding_idx=cfg.model.padding_idx
        )
        self.temporal_encoding = TemporalEncoding(cfg)

    def forward(self, x):
        embed = self.embedding(x)
        encode = self.temporal_encoding(embed)
        output = self.transformer(encode)
        return output

class TemporalEncoding(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(TemporalEncoding, self).__init__()
        self.cfg = cfg
        self.embed = cfg.model.d_model
        position = np.arange(0, cfg.model.max_time)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed, 2) * -np.log(10000) / self.embed)
        sinusoids = np.stack([np.sin(position * div_term), np.cos(position * div_term)], axis=-1)
        self.embeddings = torch.from_numpy(sinusoids.reshape(cfg.model.max_time, -1))
        self.register_buffer(name='embeddings', tensor=self.embeddings)

    def forward(self, batch):
        x, time = batch
        return x + self.embeddings[time]
