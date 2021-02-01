from model.model import BaseModel
import hydra
import torch
import os
from omegaconf import DictConfig
import numpy as np
from preprocess.preprocess_utils import tensor2list, list2midi


@hydra.main(config_path=os.path.join('config', 'config_sisamcom.yaml'), strict=True)
def main(cfg: DictConfig):
    base_path = hydra.utils.get_original_cwd()
    PATH = os.path.join(base_path, cfg.inference.checkpoint_path)
    device = 'cuda' if cfg.train.gpus > 0 else 0
    model = BaseModel.load_from_checkpoint(PATH).to(device)
    model.eval()
    if cfg.inference.condition_pt == None:
        inputs = np.array([[200]])
    else:
        condition_pt = torch.load(os.path.join(hydra.utils.get_original_cwd(), cfg.inference.condition_pt))
        inputs = condition_pt[:cfg.inference.condition_length].unsqueeze(0).numpy()
    
    inputs = torch.from_numpy(inputs).to(device)
    result = model.model.generate(inputs, cfg.inference.inference_length)

    LIST_rec = tensor2list(result.to('cpu'))
    list2midi(LIST_rec, ofpath = os.path.join(base_path, cfg.inference.save_path, "test_rec.midi"))


if __name__ == '__main__':
    main()