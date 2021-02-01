from omegaconf.dictconfig import DictConfig
import torch
import pytorch_lightning as pl

from torch import optim

from model.loss import GenericLoss, get_accuracy
from model.core import Transformer

class Baseline(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(Baseline, self).__init__()
        self.cfg = cfg
        self.loss = GenericLoss(cfg)
        self.model = Transformer(cfg)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        loss = self.loss(logits, target)
        tensorboard_log = {
            'train_loss': loss
        }

        return {'loss': loss, 'log': tensorboard_log}

    def validation_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        loss = self.loss(logits, target)
        acc = get_accuracy(logits, target)

        return {
            'val_loss': loss,
            'val_acc': acc
        }
    
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_log = {
            'val_loss': loss,
            'val_acc': acc
        }

        return {'val_loss': loss, 'log': tensorboard_log}
    
    def test_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        loss = self.loss(logits, target)
        acc = get_accuracy(logits, target)

        return {
            'test_loss': loss,
            'test_acc': acc
        }

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_log = {
            'test_loss': loss,
            'test_acc': acc
        }

        return {'test_loss': loss, 'log': tensorboard_log}

    def configure_optimizers(self):
        if self.cfg.train.optim == 'adam':
            return optim.Adam(
                self.parameters(),
                lr=self.cfg.train.lr,
                betas=self.cfg.train.betas,
                amsgrad=True
            )

        raise NotImplementedError
