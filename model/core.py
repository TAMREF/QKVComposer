from torch import nn
from omegaconf import DictConfig

class CoreModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(CoreModel, self).__init__()
        self.cfg = cfg

    def forward(self, x):
        return x
