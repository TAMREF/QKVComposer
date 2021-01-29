from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(BaseDataModule, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

