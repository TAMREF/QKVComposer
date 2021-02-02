from omegaconf.dictconfig import DictConfig
import torch
import pytorch_lightning as pl

from torch import optim

from model.loss import TemporalLoss, get_accuracy
from model.core import Transformer

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
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        loss = self.loss(logits, target)
        note_acc = get_accuracy(logits[0], target[0])
        time_acc = get_accuracy(logits[1], target[1])
        return {
            'val_loss': loss,
            'val_note_acc': note_acc,
            'val_time_acc': time_acc
        }
    
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        note_acc = torch.stack([x['val_note_acc'] for x in outputs]).mean()
        time_acc = torch.stack([x['val_time_acc'] for x in outputs]).mean()
        self.log('val_loss', loss)
        self.log('val_note_acc', note_acc)
        self.log('val_time_acc', time_acc)
    
    def test_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        loss = self.loss(logits, target)
        note_acc = get_accuracy(logits[0], target[0])
        time_acc = get_accuracy(logits[1], target[1])
        return {
            'test_loss': loss,
            'test_note_acc': note_acc,
            'test_time_acc': time_acc
        }

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        note_acc = torch.stack([x['test_note_acc'] for x in outputs]).mean()
        time_acc = torch.stack([x['test_time_acc'] for x in outputs]).mean()
        self.log('test_loss', loss)
        self.log('test_note_acc', note_acc)
        self.log('test_time_acc', time_acc)

    def configure_optimizers(self):
        if self.cfg.train.optim == 'adam':
            return optim.Adam(
                self.parameters(),
                lr=self.cfg.train.lr,
                betas=self.cfg.train.betas,
                amsgrad=True
            )

        raise NotImplementedError
