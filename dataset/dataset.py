from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import glob
import os
import hydra
import random
from preprocess.preprocess_utils import list2tensor, midi2list

class MusicDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        #Load file names of event tensors(long tensor)
        self.files = list(glob.glob(os.path.join(hydra.utils.get_original_cwd(), cfg.dataset.dir_path, '*/*.midi')))
        self.files = list(filter(lambda f : list2tensor(midi2list(ifpath = f)).size() >= cfg.model.max_seq+1, self.files))
        print('length of full dataset = ', len(self.files))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        #Get input and target tensors
        data = self._get_seq(self.files[idx], self.cfg.model.max_seq + 1)
        return data[:-1], data[1:]
    def _get_seq(self, fname, max_length=None):
        #Return event tensor(long tensor) from tensor file
        data = list2tensor(midi2list(ifpath = fname))
        #Raise error if length of tensor is less then max_length
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

        #Split dataset for train, eval, test.
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

