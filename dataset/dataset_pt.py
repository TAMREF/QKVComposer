import os
import glob

from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from tqdm import tqdm_notebook

from dataset.utils import MidiParser
from pathlib import Path
import hydra

class MidiDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(MidiDataModule, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.total_dataset = MidiDataset(self.cfg)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        total_data_len = self.total_dataset.__len__()
        train_len = round(total_data_len* 0.8)
        val_len = round(total_data_len * 0.1)
        test_len = total_data_len - train_len - val_len
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
    def __init__(self, cfg: DictConfig):
        super(MidiDataset, self).__init__()
        self.cfg = cfg
        self.parser = MidiParser(cfg)
        if cfg.data.make_note_tensor:
            self.generate_note_tensor()
        self.note_file_list = glob.glob(os.path.join(
            hydra.utils.get_original_cwd(),
            self.cfg.data.notetensor_dir,
            '*.pt'
        ))

    def generate_note_tensor(self):
        midi_file_list = glob.glob(os.path.join(
            hydra.utils.get_original_cwd(),
            self.cfg.data.datamidi_dir,
            '**', '*.[mM][iI][dD]'
        ))
        midi_file_list.extend(glob.glob(os.path.join(
            hydra.utils.get_original_cwd(),
            self.cfg.data.datamidi_dir,
            '**', '*.[mM][iI][dD][iI]'
        )))

        for f in tqdm_notebook(midi_file_list):
            parsed_note_tensor = torch.tensor(self.parser.parse_midi(f), dtype = torch.long)
            notetensor_path = os.path.join(
                hydra.utils.get_original_cwd(),
                self.cfg.data.notetensor_dir,
                str(Path(f).stem) + '.pt'
            )
            torch.save(parsed_note_tensor, notetensor_path)

    def __len__(self):
        return len(self.note_file_list)

    def __getitem__(self, index):
        parsed_data = torch.load(self.note_file_list[index])
        return (parsed_data[0][0], parsed_data[0][1]), (parsed_data[1][0], parsed_data[1][1])
