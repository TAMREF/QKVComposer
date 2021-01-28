from omegaconf import DictConfig
import torch.nn as nn

class GenericLoss:
    def __init__(self, cfg: DictConfig):
        super(GenericLoss, self).__init__()
        self.cfg = cfg
        self.loss_func = 

    def get_loss(self, logits, target):
        return self.loss_func(logits, target)

def get_accuracy(logits, target):
    pass
