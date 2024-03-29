import torch
from torch import nn
from omegaconf import DictConfig

class TimeZeroLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(TimeZeroLoss, self).__init__()
        self.cfg = cfg
        self.zero_criterian = nn.BCEWithLogitsLoss()
        self.time_criterian = nn.CrossEntropyLoss()
    def forward(self, logit_time, target_time):
        zero_loss = self.zero_criterian(logit_time[:, 0], (target_time == 0).type(torch.float))\
        #check time_criterian plz
        logit_time = logit_time.contiguous().view(-1 ,self.cfg.model.num_time_token)
        target_time = target_time.contiguous().view(-1)
        time_loss = self.time_criterian(logit_time[target_time != 0, 1:], target_time[target_time != 0] - 1)
        return zero_loss + time_loss * self.cfg.train.time_loss_mul

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
            self.time_loss = TimeZeroLoss(cfg).forward
        else:
            raise NotImplementedError

    def forward(self, logits, target):
        logit_token, logit_time = logits
        target_token, target_time = target
        token_loss = self.token_loss(logit_token.transpose(1, 2), target_token)
        time_loss = self.time_loss(logit_time.transpose(1, 2), target_time) * self.cfg.train.time_loss_mul
        return  token_loss + time_loss


def get_accuracy(logits: torch.Tensor, target: torch.Tensor, cfg:DictConfig):
    def match_acc(logit, target):
        if logit.shape[0] == 0:
            return torch.zeros(1).type(torch.float32).to(logit.device)
        return (logit.argmax(dim=-1) == target).type(torch.float32).mean()
    
    #Devide logits and target
    token_logits = logits[0].reshape(-1, cfg.model.num_tokens)
    time_logits = logits[1].reshape(-1, cfg.model.num_time_token)
    token_target = target[0].reshape(-1)
    time_target = target[1].reshape(-1)

    #Calculate indices
    vel_indices = token_target < 128
    note_on_indices = torch.logical_and(token_target >= 128, token_target < 256) 
    note_off_indices = torch.logical_and(token_target >= 256, token_target <384)
    time_zero_indices = time_target == 0
    time_nonzero_indices = time_target != 0

    #Calculate acc
    vel_acc = match_acc(token_logits[vel_indices, :], token_target[vel_indices])
    note_on_acc = match_acc(token_logits[note_on_indices, :], token_target[note_on_indices])
    note_off_acc = match_acc(token_logits[note_off_indices, :], token_target[note_off_indices])
    if cfg.train.time_loss_mode == 'time_zero':
        time_zero_acc = ((time_logits[:, 0] > 0) == (time_target == 0)).type(torch.float32).mean()
        time_nonzero_acc = match_acc(time_logits[time_nonzero_indices, 1:], time_target[time_nonzero_indices]-1)
    else:
        time_zero_acc = match_acc(time_logits[time_zero_indices, :], time_target[time_zero_indices])
        time_nonzero_acc = match_acc(time_logits[time_nonzero_indices, :], time_target[time_nonzero_indices])
    return vel_acc, note_on_acc, note_off_acc, time_zero_acc, time_nonzero_acc