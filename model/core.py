from omegaconf import DictConfig
from model.layers import *
from model.layers import Encoder
import dataset.utils as utils
from preprocess.postprocess_util import NoteEventPalette

import sys
import torch
import torch.distributions as dist
import random
from progress.bar import Bar


class CoreModel(torch.nn.Module):
    def __init__(self, cfg:DictConfig):
        super(CoreModel, self).__init__()
        self.cfg = cfg
        self.Decoder = Encoder(
            num_layers=cfg.model.num_layer, d_model=cfg.model.embedding_dim,
            input_vocab_size=cfg.model.vocab_size, rate=cfg.model.dropout, max_len=cfg.model.max_seq)
        self.fc = torch.nn.Linear(cfg.model.embedding_dim, cfg.model.vocab_size)
    def forward(self, x, length=None, writer=None):
        """if self.cfg.train.sample:
            return self.generate(x, length, None).contiguous()
        else:"""
        seq_mask = utils.get_mask_tensor(self.cfg.model.max_seq).to(x.device)
        decoder, w = self.Decoder(x, mask=seq_mask)
        fc = self.fc(decoder)
        return fc.contiguous()
    def generate(self,
                 prior: torch.Tensor,
                 length=2048):
        decode_array = prior
        result_array = prior

        NEP = NoteEventPalette()

        for _ in Bar('generating').iter(range(length)):
            if decode_array.size(1) > self.cfg.model.max_seq: 
                decode_array = decode_array[:, 1:]
            
            result, _ = self.Decoder(decode_array, None)
            result = self.fc(result)
            result = result.softmax(-1)
            
            pdf = dist.OneHotCategorical(probs=result[:, -1])
            pdf_sampled = pdf.sample()
            result = -1
            while True:
                result = pdf_sampled.argmax(-1).unsqueeze(-1)
                if NEP.registerFromIndex(result):
                    break
                else:
                    pdf_sampled[result] = 0
            
            decode_array = torch.cat((decode_array, result), dim=-1)
            result_array = torch.cat((result_array, result), dim=-1)
        return result_array[0]

    def test(self):
        self.eval()
        self.infer = True
