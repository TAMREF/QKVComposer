import torch
from torch import nn
from omegaconf import DictConfig

class TimeZeroLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
    def forward(self, logit_time, target_time):
        return nn.BCELoss(logit_time[:, 0], target_time == 0)\
        +nn.CrossEntropyLoss(logit_time, target_time)

class TemporalLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(TemporalLoss, self).__init__()
        self.cfg = cfg
        self.token_loss = nn.CrossEntropyLoss()
        if cfg.train.time_loss_mode == 'one_hot':
            self.time_loss = nn.CrossEntropyLoss()
        elif cfg.train.time_loss_mode == 'linear':
            self.time_loss = nn.L1Loss()
        elif cfg.train.time_loss_mode == 'time_zero':
            self.time_loss = TimeZeroLoss(cfg)
        else:
            raise NotImplementedError

    def forward(self, logits, target):
        logit_token, logit_time = logits
        target_token, target_time = target
        token_loss = self.token_loss(logit_token.transpose(1, 2), target_token)
        time_loss = self.time_loss(logit_time.transpose(1, 2), target_time) * self.cfg.train.time_loss_mul
        return  token_loss + time_loss

def get_accuracy(logits: torch.Tensor, target: torch.Tensor):
    return (logits.argmax(dim=-1) == target).type(torch.float32).mean()
