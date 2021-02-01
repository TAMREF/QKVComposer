import torch
import numpy as np
from torch import nn
from omegaconf import DictConfig

class Transformer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Transformer, self).__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(
            num_embeddings=cfg.model.num_tokens,
            embedding_dim=cfg.model.d_embed,
            padding_idx=cfg.model.padding_idx
        )
        self.temporal_encoding = TemporalEncoding(cfg)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model.d_hidden,
            nhead=cfg.model.n_head,
            dim_feedforward=cfg.model.d_ff,
            dropout=cfg.model.dropout
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.model.num_layers,
            norm=nn.LayerNorm(cfg.model.d_hidden)
        )
        self.linear = nn.Linear(
            in_features=cfg.model.d_hidden,
            out_features=cfg.model.num_tokens + cfg.model.data_len
        )

    def forward(self, batch):
        x, time = batch
        embed = self.embedding(x)
        encode = self.temporal_encoding(embed, time)
        output = self.linear(self.transformer(encode))
        token, time = torch.split(output, [self.cfg.model.num_tokens, self.cfg.model.data_len], dim=-1)
        return token, time

class TemporalEncoding(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(TemporalEncoding, self).__init__()
        self.cfg = cfg
        self.embed = cfg.model.d_embed
        position = np.arange(0, cfg.data.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed, 2) * -np.log(10000) / self.embed)
        sinusoids = np.stack([np.sin(position * div_term), np.cos(position * div_term)], axis=-1).astype(np.float32)
        embeddings = torch.from_numpy(sinusoids.reshape(cfg.data.max_len, -1))
        self.register_buffer(name='embeddings', tensor=embeddings)

    def forward(self, embed, time):
        return embed + self.embeddings[time]
