import os
import platform
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, config
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from model.model import BaseModel
from dataset.dataset import BaseDataModule

@hydra.main(config_path=os.path.join('config', 'config.yaml'), strict=False)
def main(cfg: DictConfig):
    basemodel = BaseModel(cfg)
    basedata = BaseDataModule(cfg)
    logger = pl_loggers.TensorBoardLogger(save_dir=cfg.train.log_dir, version=cfg.train.version)
    trainer = Trainer(
        accelerator=None if platform.system() == 'Windows' else 'ddp',
        accumulate_grad_batches=cfg.train.accumulate,
        auto_scale_batch_size=True,
        default_root_dir=cfg.train.log_dir,
        fast_dev_run=cfg.train.fast_dev_run,
        gpus=cfg.train.gpus,
        logger=logger,
        terminate_on_nan=True
    )

    trainer.fit(basemodel, datamodule=basedata)

if __name__ == '__main__':
    main()
