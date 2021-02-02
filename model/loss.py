import torch
from torch import nn
from omegaconf import DictConfig

class TemporalLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(TemporalLoss, self).__init__()
        self.cfg = cfg
        self.token_loss = nn.CrossEntropyLoss()
        if cfg.train.time_loss_mode == 'one_hot':
            self.time_loss = nn.CrossEntropyLoss()
        elif cfg.train.time_loss_mode == 'linear':
            self.time_loss = nn.L1Loss()
        else:
            raise NotImplementedError

    def forward(self, logits, target):
        logit_token, logit_time = logits
        target_token, target_time = target
        return self.token_loss(logit_token.transpose(1, 2), target_token) + self.time_loss(logit_time.transpose(1, 2), target_time) * self.cfg.train.time_loss_mul

def get_accuracy(logits: torch.Tensor, target: torch.Tensor):
    return (logits.argmax(dim=-1) == target).type(torch.float32).mean()
