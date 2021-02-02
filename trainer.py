import os
import platform
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning import loggers as pl_loggers

from model.model import Baseline
from dataset.dataset_pt import MidiDataModule

@hydra.main(config_name=os.path.join('config', 'config.yaml'))
def main(cfg: DictConfig):
    basemodel = Baseline(cfg)
    basedata = MidiDataModule(cfg)
    logger = pl_loggers.TensorBoardLogger(
        save_dir=cfg.log.log_dir,
        version=cfg.log.version
    )
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg.log.checkpoint_dir,
        filename='ckpt-{epoch:04d}-{val_loss:.4f}',
        save_top_k=cfg.train.save_top_k
    )
    trainer = Trainer(
        accelerator=None if platform.system() == 'Windows' else 'ddp',
        accumulate_grad_batches=cfg.train.accumulate,
        auto_scale_batch_size=True,
        callbacks=[checkpoint_callback],
        default_root_dir=cfg.log.log_dir,
        fast_dev_run=cfg.train.fast_dev_run,
        gpus=None if cfg.train.gpus == 0 else cfg.train.gpus,
        log_every_n_steps=1,
        logger=logger,
        resume_from_checkpoint=None if cfg.train.resume is '' else cfg.train.resume,
        terminate_on_nan=True,
        weights_save_path=cfg.log.checkpoint_dir
    )

    trainer.fit(basemodel, datamodule=basedata)

if __name__ == '__main__':
    main()
