# Music Transformer with Template
Readable Music-Transformer using Template code

## Libraries

See [requirements.txt](requirements.txt)

## Structure

The outer training loop is located at [trainer.py](trainer.py).

The inference functions(generating music) are to be located at [inference.py](inference.py).

The model, loss function, and LightningModule is defined under `model/`.

The dataset is defined under `dataset/`.

The configuration YAML files are located under `config/`.

The preprocess functions are located under `preprocess/`.

### Config

All configurations(batch size, model size, etc...) for the dataset, model, train, inference are in this file.


### Model

Use music_transformer model with trainable relative positional encoding and sinusoidal positional encoding.

### Dataset

All data modification functions should be put under `dataset/utils.py` and included as necessary.

The event_tensor created in the generate_tensor_database should be located here.

### Preprocess

Preprocess functions to train a model. 

### Training

All hyperparameters must be defined under `config/` and modified as necessary.

Checkpointing, tensorboard logging, early stopping, gradient accumulation, multi-gpu training and etc. are automatically handled within `trainer.py`.

Refer to documentation regarding [hydra](https://hydra.cc) and [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html) for further info.

## Run trainer and inference

1. Locate midi dataset to `dataset/midi` and run python `generate_tensor_datasets.py`.

```bash
python preprocess/generate_tensor_datasets.py
```

2. Set configuration file `config/config.yaml` for train and inference.

3. Run `trainer.py`.

```bash
python trainer.py
```

4. Run `inference.py`

```bash
python inference.py
```

## TODO

 - [x] Initialize requirements.txt
 - [x] Add loss functions
 - [x] Add datamodule
 - [x] Add inference
 - [x] Add music generate code 
 - [x] Delete folder music_transformer
 - [ ] Lay out proper datamodule
 - [ ] Add more optimizers
 - [ ] Prettify documentation
