import torch
from preprocess_utils import *
from tqdm import tqdm
if __name__ == '__main__':

    midiFiles = glob("data/*.midi")

    fullData = torch.zeros(len(midiFiles), 6400)
    pbar = tqdm(enumerate(midiFiles))

    for idx, fpath in pbar:
        dataTensor = list2tensor(midi2list(ifpath = fpath))
        pbar.set_description(f"Datasize : {dataTensor.shape[0]}")
        dataTensor = torch.cat([dataTensor + 1, torch.zeros(6000)])[:6400]
        fullData[idx] = dataTensor
        
    
    torch.save(fullData, "train_data.dat")
        