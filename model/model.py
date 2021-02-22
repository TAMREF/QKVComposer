from omegaconf.dictconfig import DictConfig
import torch
import pytorch_lightning as pl

from torch import optim

from model.loss import TemporalLoss, get_accuracy
from model.core import Transformer, JBob

class Baseline(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(Baseline, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.loss = TemporalLoss(cfg)
        self.model = Transformer(cfg)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        loss = self.loss(logits, target)
        vel_acc, note_on_acc, note_off_acc, time_zero_acc, time_nonzero_acc = get_accuracy(logits, target, self.cfg)
        self.log('train_loss', loss)
        self.log('train_vel_acc', vel_acc)
        self.log('train_note_on_acc', note_on_acc)
        self.log('train_note_off_acc', note_off_acc)
        self.log('train_time_zero_acc', time_zero_acc)
        self.log('train_time_nonzero_acc', time_nonzero_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        loss = self.loss(logits, target)
        vel_acc, note_on_acc, note_off_acc, time_zero_acc, time_nonzero_acc = get_accuracy(logits, target, self.cfg)
        return {
            'val_loss': loss,
            'val_vel_acc': vel_acc,
            'val_note_on_acc': note_on_acc,
            'val_note_off_acc': note_off_acc,
            'val_time_zero_acc': time_zero_acc,
            'val_time_nonzero_acc': time_nonzero_acc
        }
    
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        vel_acc = torch.stack([x['val_vel_acc'] for x in outputs]).mean()
        note_on_acc = torch.stack([x['val_note_on_acc'] for x in outputs]).mean()
        note_off_acc = torch.stack([x['val_note_off_acc'] for x in outputs]).mean()
        time_zero_acc = torch.stack([x['val_time_zero_acc'] for x in outputs]).mean()
        time_nonzero_acc = torch.stack([x['val_time_nonzero_acc'] for x in outputs]).mean()
        self.log('val_loss', loss)
        self.log('val_vel_acc', vel_acc)
        self.log('val_note_on_acc', note_on_acc)
        self.log('val_note_off_acc', note_off_acc)
        self.log('val_time_zero_acc', time_zero_acc)
        self.log('val_time_nonzero_acc', time_nonzero_acc)
    
    def test_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        loss = self.loss(logits, target)
        
        vel_acc, note_on_acc, note_off_acc, time_zero_acc, time_nonzero_acc = get_accuracy(logits, target, self.cfg)
        return {
            'test_loss': loss,
            'test_vel_acc': vel_acc,
            'test_note_on_acc': note_on_acc,
            'test_note_off_acc': note_off_acc,
            'test_time_zero_acc': time_zero_acc,
            'test_time_nonzero_acc': time_nonzero_acc
        }

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        vel_acc = torch.stack([x['test_vel_acc'] for x in outputs]).mean()
        note_on_acc = torch.stack([x['test_note_on_acc'] for x in outputs]).mean()
        note_off_acc = torch.stack([x['test_note_off_acc'] for x in outputs]).mean()
        time_zero_acc = torch.stack([x['test_time_zero_acc'] for x in outputs]).mean()
        time_nonzero_acc = torch.stack([x['test_time_nonzero_acc'] for x in outputs]).mean()
        self.log('test_loss', loss)
        self.log('test_vel_acc', vel_acc)
        self.log('test_note_on_acc', note_on_acc)
        self.log('test_note_off_acc', note_off_acc)
        self.log('test_time_zero_acc', time_zero_acc)
        self.log('test_time_nonzero_acc', time_nonzero_acc)
    

    def configure_optimizers(self):
        if self.cfg.train.optim == 'adam':
            return optim.Adam(
                self.parameters(),
                lr=self.cfg.train.lr,
                betas=self.cfg.train.betas,
                amsgrad=True
            )

        raise NotImplementedError
