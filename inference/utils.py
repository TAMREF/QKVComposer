import torch
import os
import hydra
def load_prior_tensors(cfg, device):
    if cfg.inference.condition_pt is None:
        prior_token = torch.tensor([[200]])
        prior_time = torch.tensor([[5]])
    else:
        condition_pt = torch.load(os.path.join(hydra.utils.get_original_cwd(), cfg.inference.condition_pt))
        if cfg.data.datamode == 'time_note_vel':
            indices = condition_pt[1] >= 128
            time_tensor = condition_pt[0][indices]
            token_tensor = condition_pt[1][indices]
        
        prior_time = time_tensor[:cfg.inference.condition_length]
        prior_token = token_tensor[:cfg.inference.condition_length]
    return prior_time, prior_token