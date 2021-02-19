import os
import glob

from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from tqdm import tqdm

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
        if cfg.data.make_data_tensor:
            self.generate_note_tensor()
        self.data_file_list = glob.glob(os.path.join(
            hydra.utils.get_original_cwd(),
            self.cfg.data.datatensor_dir,
            '*.pt'
        ))
        def filter_data_tensor(f):
            time_token_tensor = torch.load(f)
            time = time_token_tensor[0]
            max_time_gap = torch.max(time[1:]-time[:-1]).item()
            if max_time_gap >= self.cfg.model.num_time_token:
                return False

            #바꿀 수 있다면 pad token 이용하는 것으로 바꿔주세요
            if self.cfg.data.datamode=='time_token':
                return time_token_tensor.size()[-1] >= cfg.model.data_len + 1
            elif self.cfg.data.datamode=='time_note_vel':
                return torch.sum(time_token_tensor[1] >= 128).cpu().item() >= cfg.model.data_len + 1
            else:
                raise Exception('datamode is invalid')
        self.data_file_list = list(filter(filter_data_tensor, self.data_file_list))
            
    def generate_note_tensor(self):
        midi_file_list = []
        for i in range(1, 10):
            sub_folder = os.path.join(*(["**"]*i))
            midi_file_list.extend(glob.glob(os.path.join(
                hydra.utils.get_original_cwd(),
                self.cfg.data.datamidi_dir,
                sub_folder, '*.[mM][iI][dD]'
            )))
            midi_file_list.extend(glob.glob(os.path.join(
                hydra.utils.get_original_cwd(),
                self.cfg.data.datamidi_dir,
                sub_folder, '*.[mM][iI][dD][iI]'
            )))

        for f in tqdm(midi_file_list):
            time_list, token_list = self.parser.parse_full_midi(f)
            parsed_time_token_tensor = torch.tensor((time_list, token_list), dtype = torch.long)
            time_token_tensor_path = os.path.join(
                hydra.utils.get_original_cwd(),
                self.cfg.data.datatensor_dir,
                str(Path(f).stem) + "time_token_tensor" + '.pt'
            )
            torch.save(parsed_time_token_tensor, time_token_tensor_path)

    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, index):
        time_token_tensor = torch.load(self.data_file_list[index])
        time_tensor = time_token_tensor[0]
        token_tensor = time_token_tensor[1]
        if self.cfg.data.datamode == 'time_token':
            return self.parser.random_choice_from_notetensor(time_token_tensor)
        elif self.cfg.data.datamode == 'time_note_vel':

            indices = time_token_tensor[1] >= 128

            time_tensor = time_tensor[indices]
            token_tensor = token_tensor[indices]
            return self.parser.random_choice_from_notetensor((time_tensor, token_tensor))
        else:
            raise Exception('datamode is invalid')