from model.model import BaseModel
import hydra
import torch
import os
from omegaconf import DictConfig
from preprocess.preprocess_utils import *


@hydra.main(config_path=os.path.join('config', 'config.yaml'), strict=True)
def main(cfg: DictConfig):
    base_path = hydra.utils.get_original_cwd()
    PATH = os.path.join(base_path, 'outputs/2021-01-29/12-44-46/checkpoints/epoch=3677-step=14711.ckpt')
    model = BaseModel.load_from_checkpoint(PATH)
    model.eval()
    if cfg.inference.condition_midi == None:
        inputs = np.array([[24, 28, 31]])
    else:
        condition_midi = torch.load(cfg.inference.condition_midi)
        inputs = condition_midi[:cfg.inference.condition_length].unsqueeze(0).numpy()
    
    inputs = torch.from_numpy(inputs)
    result = model.model.generate(inputs, cfg.inference.inference_length)

    LIST_rec = tensor2list(result)
    list2midi(LIST_rec, ofpath = os.path.join(base_path, cfg.inference.save_path, "test_rec.midi"))


if __name__ == '__main__':
    main()