from model.model import Baseline
import hydra
import torch
import torch.distributions as dist
import os
from omegaconf import DictConfig
import numpy as np
from dataset.utils import MidiParser
from tqdm import tqdm

def generate(cfg, model, prior_token: torch.Tensor, prior_time:torch.Tensor, length=2048):
    decode_token_array = prior_token
    result_token_array = prior_token
    decode_time_array = prior_time
    result_time_array = prior_time
    for _ in tqdm(range(length)):
        if decode_token_array.size(1) > cfg.model.data_len: 
            decode_token_array = decode_token_array[:, 1:]
        
        token, timegap = model((decode_token_array, decode_time_array))
        token = token.softmax(-1)
        timegap = timegap.softmax(-1)

        if cfg.inference.sample_mode == 'OneHotCategorical':
            pdf_token = dist.OneHotCategorical(probs=token[:, -1])
            pdf_time = dist.OneHotCategorical(probs=timegap[:, -1])
            token = pdf_token.sample().argmax(-1).unsqueeze(-1)
            timegap = pdf_time.sample().argmax(-1).unsqueeze(-1)
        elif cfg.inference.sample_mode == 'Argmax':
            token = token[:, -1].argmax(-1).unsqueeze(-1)
            timegap = timegap[:, -1].argmax(-1).unsqueeze(-1)

        decode_token_array = torch.cat((decode_token_array, token), dim=-1)
        result_token_array = torch.cat((result_token_array, token), dim=-1)
        
        decode_time_array = torch.cat((decode_time_array, decode_time_array[0][-1]+timegap), dim=-1)
        result_time_array = torch.cat((result_time_array, decode_time_array[0][-1]+timegap), dim=-1)
    result_token_array = result_token_array[0]
    result_time_array = result_time_array[0]
    
    return result_token_array, result_time_array

@hydra.main(config_path=os.path.join('config', 'config.yaml'), strict=True)
def main(cfg: DictConfig):
    base_path = hydra.utils.get_original_cwd()
    PATH = os.path.join(base_path, cfg.inference.checkpoint_path)
    device = 'cuda' if cfg.train.gpus > 0 else 0
    model = Baseline.load_from_checkpoint(PATH).to(device)
    model.eval()
    parser = MidiParser(cfg)
    if cfg.inference.condition_pt is None:
        prior_token = torch.tensor([[200]]).to(device)
        prior_time = torch.tensor([[5]]).to(device)
    else:
        pass
        #condition_pt = torch.load(os.path.join(hydra.utils.get_original_cwd(), cfg.inference.condition_pt))
        #inputs = condition_pt[:cfg.inference.condition_length].unsqueeze(0).numpy()
    
    result_token_array, result_time_array = generate(cfg, model, prior_token, prior_time, length=cfg.inference.length)
    parser.recon_midi(result_token_array.to('cpu').tolist(), result_time_array.to('cpu').tolist(), name = 'temporal_encoding.mid')

if __name__ == '__main__':
    main()