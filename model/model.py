from omegaconf.dictconfig import DictConfig
import torch
import pytorch_lightning as pl

from torch import optim

from model.loss import GenericLoss, CategoricalAccuracy
from model.core import CoreModel

class BaseModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(BaseModel, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.loss = GenericLoss(cfg)
        self.accuracy = CategoricalAccuracy()
        self.model = CoreModel(cfg)

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
        acc = self.accuracy(logits, target)

        return {
            'val_loss': loss,
            'val_acc': acc
        }
    
    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([x['val_loss'] for x in outputs]))
        acc = torch.mean(torch.stack([x['val_acc'] for x in outputs]))
        tensorboard_log = {
            'val_loss': loss,
            'val_acc': acc
        }

        return {'val_loss': loss, 'log': tensorboard_log}
    
    def test_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        loss = self.loss(logits, target)
        acc = self.accuracy(logits, target)

        return {
            'test_loss': loss,
            'test_acc': acc
        }

    def test_epoch_end(self, outputs):
        loss = torch.mean([x['test_loss'] for x in outputs])
        acc = torch.mean([x['test_acc'] for x in outputs])
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
