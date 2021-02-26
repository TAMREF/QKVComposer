from inference.utils import load_prior_tensors
import torch
from omegaconf import DictConfig
from dataset.utils import MidiParser
from tqdm import tqdm
from inference.next_distribution import next_distribution

class State:
    #All tensors in State are flattened
    def __init__(self, token_tensor, time_tensor, logit):
        self.token_tensor = token_tensor
        self.time_tensor = time_tensor
        self.logit = logit
        self.device = token_tensor.device
    def append(self, token, time_gap, logit):
        token_tensor = torch.cat((self.token_tensor, token.unsqueeze(0).to(self.device)))
        time_tensor = torch.cat((self.time_tensor, (self.time_tensor[-1] + time_gap).unsqueeze(0).to(self.device)))
        return State(token_tensor, time_tensor, logit)

def pseudo_log(x):
    re = torch.full_like(x, -1e9)
    idx =  (x > 1e-9)
    re[idx] = torch.log(x[idx])
    return re

#For now, implemented only token(vel, note), not time_gap
class BeamSearch:
    def __init__(self, cfg:DictConfig, init_token_tensor, init_time_tensor, model):
        self.cfg = cfg
        self.k = cfg.inference.beam_search.k
        self.states = [State(init_token_tensor.to(model.device), init_time_tensor.to(model.device), 0)]
        self.next_distribution = next_distribution(cfg, model)

    def get_next_topk(self, state : State):
        pdf_time, pdf_token = self.next_distribution(state)

        log_prob_dist = torch.log(torch.max(pdf_time)) + pseudo_log(pdf_token)
        topk = log_prob_dist.topk(self.k)
        logits = topk.values + state.logit
        token_indices = topk.indices
        time_gap = torch.argmax(pdf_time).cpu().item()
        
        del pdf_time, pdf_token
        return logits, token_indices, time_gap

    def topk_indices(self, logits_list):
        viewed_topk_indices = torch.cat(logits_list).reshape(-1).topk(self.k).indices
        
        prev_state_indices = viewed_topk_indices.cpu().numpy() // self.k
        next_state_token_indices = viewed_topk_indices.cpu().numpy() % self.k

        del viewed_topk_indices
        return prev_state_indices, next_state_token_indices

    def beam_search_step(self, ):
        logits_list = []
        token_indices_list = []
        time_gap_list = []
        for state in self.states:
            logits, token_indices, time_gap = self.get_next_topk(state)
            logits_list.append(logits)
            token_indices_list.append(token_indices)
            time_gap_list.append(time_gap)
        prev_state_indices, next_state_token_indices = self.topk_indices(logits_list)

        next_states = []
        for i in range(self.k):
            next_state = self.states[prev_state_indices[i]].append(
                token_indices_list[prev_state_indices[i]][next_state_token_indices[i]],\
                time_gap_list[prev_state_indices[i]],\
                logits_list[prev_state_indices[i]][next_state_token_indices[i]]
            )
            next_states.append(next_state)
        
        del self.states, logits_list, token_indices_list, time_gap_list
        self.states = next_states
        torch.cuda.empty_cache()
    def generate(self,):
        for _ in tqdm(range(self.cfg.inference.length)):
            self.beam_search_step()
        
        top_state = State(torch.tensor([0]), torch.tensor([0]), -1e20)
        for s in self.states:
            if(s.logit > top_state.logit):
                top_state = s

        return top_state