#Generate long tensor events for each song
import torch
from preprocess.preprocess_utils import list2tensor, midi2list
from tqdm import tqdm
import os
import glob

midi_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "midi")
save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "event_tensor")

midiFiles = glob.glob(os.path.join(midi_dir, '*/*.midi'))
pbar = tqdm(enumerate(midiFiles))
for idx, fpath in pbar:
    dataTensor = list2tensor(midi2list(ifpath = fpath))
    pbar.set_description(f"Datasize : {dataTensor.shape[0]}")
    torch.save(dataTensor, os.path.join(save_dir, os.path.splitext(os.path.basename(fpath))[0]+".pt"))