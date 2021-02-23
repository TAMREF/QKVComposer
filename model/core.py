import torch
import numpy as np
from torch import nn
from omegaconf import DictConfig

class JBob(nn.Module):
    def __init__(self, cfg:DictConfig):
        super(JBob, self).__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(
            num_embeddings=cfg.model.num_tokens,
            embedding_dim=cfg.model.d_embed,
            padding_idx=cfg.model.padding_idx
        )
        self.l = cfg.model.data_len*cfg.model.d_embed
        self.out_len = cfg.model.num_tokens+cfg.model.num_time_token
        self.activation = torch.relu
        self.linear1 = nn.Linear(self.l, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, cfg.model.data_len*self.out_len)
    def forward(self, batch):
        batch = batch[0]
        batch = self.embedding(batch).reshape(batch.shape[0], -1)
        output = self.linear3(self.bn2(self.activation(self.linear2(self.bn1((self.activation(self.linear1(batch)))))))).reshape(batch.shape[0], -1, self.out_len)
        token, time = torch.split(output, [self.cfg.model.num_tokens, self.cfg.model.num_time_token], dim=-1)
        return token, time



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
        mask = nn.Transformer().generate_square_subsequent_mask(sz=self.cfg.model.data_len)
        self.register_buffer(name='mask', tensor=mask)
        self.linear = nn.Linear(
            in_features=cfg.model.d_hidden,
            out_features=cfg.model.num_tokens + cfg.model.num_time_token
        )

    def forward(self, batch):
        x, time = batch
        mask = self.mask[:x.shape[1], :x.shape[1]]
        embed = self.embedding(x)
        #batch first
        encode = self.temporal_encoding(embed, time) if self.cfg.model.use_temporal_encoding else embed
        #batch second to use fucking nn.Transformer implementation
        encode = encode.permute(1,0,2).contiguous()
        output = self.linear(self.transformer(encode, mask=mask))
        #batch first
        output = output.permute(1,0,2).contiguous()
        token, time = torch.split(output, [self.cfg.model.num_tokens, self.cfg.model.num_time_token], dim=-1)
        return token, time

class TemporalEncoding(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(TemporalEncoding, self).__init__()
        self.cfg = cfg
        self.embed = cfg.model.d_embed
        # position = np.arange(0, cfg.data.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed, 2) * -np.log(10000) / self.embed)
        self.register_buffer(name='div_term', tensor=torch.from_numpy(div_term.astype(np.float32)))
        # sinusoids = np.stack([np.sin(position * div_term), np.cos(position * div_term)], axis=-1).astype(np.float32)
        # embeddings = torch.from_numpy(sinusoids.reshape(cfg.data.max_len, -1))
        # self.register_buffer(name='embeddings', tensor=embeddings)

    def forward(self, embed, time):
        time = torch.unsqueeze(time, dim=-1) * self.cfg.model.sinusoid_const
        sinusoids = torch.stack([torch.sin(time * self.div_term), torch.sin(time * self.div_term)], axis=-1)
        embeddings = sinusoids.reshape(-1, self.embed)
        return embed + embeddings
