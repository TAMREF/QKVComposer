from omegaconf.dictconfig import DictConfig
import torch
import pytorch_lightning as pl

from torch import nn

class BaseModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(BaseModel, self).__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor):
        pass

    def training_step(self, batch, batch_idx):
        tensorboard_log = {
            'train_loss': 0
        }
        return {'loss': 0, 'log': tensorboard_log}

    def validation_step(self, batch, batch_idx):
        return {
            'val_loss': 0,
            'val_acc': 0
        }
    
    def validation_epoch_end(self, outputs):
        loss = torch.mean([x['val_loss'] for x in outputs])
        acc = torch.mean([x['val_acc'] for x in outputs])
        tensorboard_log = {
            'val_loss': loss,
            'val_acc': acc
        }

        return {'val_loss': loss, 'log': tensorboard_log}
    
    def test_step(self, batch, batch_idx):
        return {
            'test_loss': 0,
            'test_acc': 0
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
            return torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.train.lr,
                betas=self.cfg.train.betas,
                amsgrad=True
            )

        raise NotImplementedError
