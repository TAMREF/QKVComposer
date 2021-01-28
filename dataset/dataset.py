from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import dataset.utils as utils
import torch
import glob
import os
import hydra
import random

class MusicDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.files = list(glob.glob(os.path.join(hydra.utils.get_original_cwd(), cfg.dataset.dir_path, '*.pt')))
        # self.files = list(utils.find_files_by_extensions(cfg.dataset.dir_path, ['.pt']))
        print(os.getcwd())
        print(cfg.dataset.dir_path)
        print('len = ', len(self.files))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        data = self._get_seq(self.files[idx], self.cfg.model.max_seq+1)
        x = data[:-1]
        y = data[1:]
        return x, y
    def _get_seq(self, fname, max_length=None):
        with open(fname, 'rb') as f:
            data = torch.load(f)
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0,len(data) - max_length)
                data = data[start:start + max_length]
            else:
                raise IndexError
        return data

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(BaseDataModule, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        full_dataset = MusicDataset(self.cfg)
        len_data = len(full_dataset)
        len_train = int(len_data*0.8)
        len_val = int(len_data*0.1)
        len_test = len_data-len_train-len_val
        self.train_dataset, self.val_dataset, self.test_dataset \
        = random_split(full_dataset, [len_train, len_val, len_test], generator=torch.Generator().manual_seed(self.cfg.train.random_seed))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

