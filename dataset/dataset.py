import os
import glob

from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

from dataset.utils import MidiParser

class MidiDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(MidiDataModule, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.file_list = glob.glob(os.path.join(cfg.data.dataset_path, '**', '*.[mM][iI][dD]'))
        self.file_list.extend(glob.glob(os.path.join(cfg.data.dataset_path, '**', '*.[mM][iI][dD][iI]')))

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.total_dataset = MidiDataset(self.cfg, self.file_list)
        train_len = round(len(self.file_list) * 0.8)
        val_len = round(len(self.file_list) * 0.1)
        test_len = len(self.file_list) - train_len - val_len
        (
            self.train_dataset, 
            self.val_dataset, 
            self.test_dataset
        ) = random_split(self.total_dataset, [train_len, val_len, test_len])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            drop_last=True
        )

class MidiDataset(Dataset):
    def __init__(self, cfg: DictConfig, pathlist):
        super(MidiDataset, self).__init__()
        self.cfg = cfg
        self.pathlist = pathlist
        self.parser = MidiParser(cfg)

    def __len__(self):
        return len(self.pathlist)

    def __getitem__(self, index):
        return self.parser.parse_midi(self.pathlist[index])
