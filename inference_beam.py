from inference.beam_search import BeamSearch
from inference.utils import load_prior_tensors
from model.model import Baseline
import hydra
import torch
import os
from omegaconf import DictConfig
from dataset.utils import MidiParser

@hydra.main(config_path=os.path.join('config', 'config.yaml'), strict=True)
def main(cfg: DictConfig):
    base_path = hydra.utils.get_original_cwd()
    PATH = os.path.join(base_path, cfg.inference.checkpoint_path)
    device = 'cuda' if cfg.train.gpus > 0 else 'cpu'
    model = Baseline.load_from_checkpoint(PATH).to(device)
    model.eval()
    parser = MidiParser(cfg)
    prior_time, prior_token = load_prior_tensors(cfg, device)
    top_state = BeamSearch(cfg, prior_token, prior_time, model).generate()
    result_token_array, result_time_array = top_state.token_tensor, top_state.time_tensor
    parser.recon_midi(result_token_array.to('cpu').tolist(), result_time_array.to('cpu').tolist(), name = 'temporal_encoding.mid')

if __name__ == '__main__':
    main()