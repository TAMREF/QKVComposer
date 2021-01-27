from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(BaseDataModule, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size

    def setup(self, stage=None):
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch)

