import hydra
import torch
from omegaconf import DictConfig
class next_distribution:
    def __init__(self, cfg:DictConfig, model):
        self.model = model
        self.cfg = cfg
    
    def __call__(self, state):
        decode_time_array = state.time_tensor.unsqueeze(0)
        decode_token_array = state.token_tensor.unsqueeze(0)

        if decode_token_array.size(1) > self.model.cfg.model.data_len:
            s_idx = decode_token_array.size(1) - self.model.cfg.model.data_len
            decode_token_array = decode_token_array[:, s_idx:]
            decode_time_array = decode_time_array[:, s_idx:]
        
        with torch.no_grad():
            token, timegap = self.model((decode_token_array, decode_time_array))
        token.detach()
        timegap.detach()
        
        #logit smoothization
        token = token*self.cfg.inference.one_hot_smooth
        #timegap = timegap*self.cfg.inference.one_hot_smooth

        token = token.softmax(-1)
        
        #should change if batchsize != 0

        if self.cfg.inference.sample_mode == 'OneHotCategorical':
            if self.cfg.train.time_loss_mode=='one_hot':
                pdf_time = timegap.softmax(-1)[0, -1, :].clone()
            else:
                raise Exception("time_loss_mode != one_hot")
        else:
            raise Exception("sample_mode != OneHotCategorical")
            

        if self.cfg.inference.sample_mode == 'OneHotCategorical':
            pdf_token = token[0, -1, :].clone()
        else:
            raise Exception("sample_mode != OneHotCategorical")
        
        del token, timegap, decode_time_array, decode_token_array
        torch.cuda.empty_cache()
        return pdf_time, pdf_token