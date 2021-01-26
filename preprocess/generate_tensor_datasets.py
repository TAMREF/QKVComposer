#Generate long tensor events for each song
import torch
from preprocess_utils import *
from tqdm import tqdm
import os

midi_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "music_transformer", "dataset", "midi", "maestro-v3.0.0")
save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "music_transformer", "dataset", "event_tensor")

midiFiles = glob(os.path.join(midi_dir, ['.mid','midi']))
pbar = tqdm(enumerate(midiFiles))
for idx, fpath in pbar:
    dataTensor = list2tensor(midi2list(ifpath = fpath))
    pbar.set_description(f"Datasize : {dataTensor.shape[0]}")
    torch.save(dataTensor, os.path.join(save_dir, os.path.splitext(os.path.basename(fpath))[0]+".pt"))