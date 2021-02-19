import os
import numpy as np
import pretty_midi as pm

from numpy import random
from omegaconf import DictConfig
import torch
import hydra

VELOCITY_CHANGE = 0
NOTE_START = 1
NOTE_END = 2

class MidiParser:
    def __init__(self, cfg: DictConfig):
        super(MidiParser, self).__init__()
        self.cfg = cfg
        self.data_len = cfg.model.data_len + 1
    
    def parse_midi(self, path: str):
        time_list, token_list = self.parse_full_midi(path)
        if len(token_list) <= self.data_len:
            token_list += [self.cfg.model.padding_idx] * (self.data_len - len(token_list))
            time_list += [0] * (self.data_len - len(time_list))
        else:
            random_split = random.randint(0, len(token_list) - self.data_len)
            token_list = token_list[random_split:random_split + self.data_len]
            time_list = time_list[random_split:random_split + self.data_len]

        token_list = np.array(token_list, dtype=np.int64)
        time_list = np.array(time_list, dtype=np.int64)

        return (token_list[:-1], time_list[:-1]), (token_list[1:], np.maximum(time_list[1:] - time_list[:-1], 0))
    
    def get_token_list(self, event_list):
        token_list = []
        for event in event_list:
            if event[1] == VELOCITY_CHANGE:
                token_list.append(event[2])
            elif event[1] == NOTE_START:
                token_list.append(event[2] + 128)
            elif event[1] == NOTE_END:
                token_list.append(event[2] + 256)
            else:
                raise IndexError
        return token_list


    def parse_full_midi(self, path:str):
        midi = pm.PrettyMIDI(midi_file=path)
        midi.remove_invalid_notes()
        event_list = []

        for inst in midi.instruments:
            if inst.program != 0:
                continue

            for note in inst.notes:
                event_list.append((note.start, NOTE_START, note.pitch))
                event_list.append((note.end, NOTE_END, note.pitch))
                event_list.append((note.start, VELOCITY_CHANGE, note.velocity))

        event_list.sort()
        #1 idx shifts for 10ms interval
        time_list = [round(event[0]*100) for event in event_list]
        token_list = self.get_token_list(event_list)
        return time_list, token_list

    def random_choice_from_notetensor(self, data_tensor):
        time_tensor = data_tensor[0]
        token_tensor = data_tensor[1]

        random_split = random.randint(0, token_tensor.shape[0] - self.data_len)
        token_tensor = token_tensor[random_split:random_split + self.data_len]
        time_tensor = time_tensor[random_split:random_split + self.data_len]
        return (token_tensor[:-1], time_tensor[:-1]), (token_tensor[1:], torch.relu(time_tensor[1:] - time_tensor[:-1]))

    def recon_midi(self, token_list, time_list, name):
        velocity = 80
        start_times = [0] * 128

        midi = pm.PrettyMIDI(midi_file=None)
        inst = pm.Instrument(0, name='piano')

        for token, time in zip(token_list, time_list):
            if 0 <= token < 128:
                velocity = token
            elif 128 <= token < 256:
                start_times[token-128] = time / 100
            elif 256 <= token < 384:
                pitch = token - 256
                inst.notes.append(pm.Note(velocity, pitch, start_times[pitch], time / 100))
            else:
                break

        midi.instruments.append(inst)
        midi.write(filename=os.path.join(hydra.utils.get_original_cwd(), self.cfg.inference.save_path, name))
