# Template
Template code for future endeavors

## Libraries

See [requirements.txt](requirements.txt)

## Structure

The outer training loop is located at [trainer.py](trainer.py).

The inference functions are to be located at [inference.py](inference.py).

The model, loss function, and LightningModule is defined under `model/`.

The dataset is defined under `dataset/`.

The configuration YAML files are located under `config/`.

### Model

Note that `model/model.py` contains the LightningModule, and `model/core.py` contains the actual model.

All additional modules should be designed in files under `model/` and included in `model/model.py`.

Refer to documentation regarding [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html) for further info.

### Dataset

All data modification functions should be put under `dataset/utils.py` and included as necessary.

Refer to documentation regarding [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html) for further info.

### Training

All hyperparameters must be defined under `config/` and modified as necessary.

Checkpointing, tensorboard logging, early stopping, gradient accumulation, multi-gpu training and etc. are automatically handled within `trainer.py`.

Refer to documentation regarding [hydra](https://hydra.cc) and [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html) for further info.


## TODO

 - [x] Initialize requirements.txt
 - [x] Add loss functions
 - [x] Add datamodule
 - [ ] Lay out proper datamodule
 - [ ] Add more optimizers
 - [ ] Add inference
 - [ ] Prettify documentation

## Author

- [June Young Yi](https://github.com/Rick-McCoy)
