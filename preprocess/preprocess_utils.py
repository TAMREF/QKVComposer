import note_seq
from note_seq.protobuf import music_pb2

from glob import glob
import json
 
import numpy as np
import torch
from noteseq2tensor import rawData2Indices

def list2midi(note_json, ofpath = "results/test.mid"):
    ## note_json is list, output saved as midi file at filepath
    ## note_json should have [[x0, x1, x2, x3], ...] where x0 is pitch, x1 is velocity, x2 is start_time, x3 is end_time

    seqs = music_pb2.NoteSequence()
    seqs.tempos.add(qpm = 120)
    for note in note_json:
        seqs.notes.add(pitch = note[0], velocity = note[1], start_time = note[2], end_time = note[3])
    
    note_seq.note_sequence_to_midi_file(seqs, ofpath)

def midi2list(ifpath = "data/test.mid"):
    # input is midi path, returns note_json list.
    ns = note_seq.midi_file_to_note_sequence(ifpath)

    sirials = list(map(lambda x : [x.pitch, x.velocity, x.start_time, x.end_time], list(ns.notes)))    
    return sirials

def list2tensor(note_json):

    # return one-hots from decoded json file.
    indices = rawData2Indices(note_json)
    tensor = torch.tensor(indices).long()
    return tensor

## THIS PART SHOULD BE REFACTORED


IN_EVO = 128
OUT_EVO = 128 + IN_EVO
TS_EVO = 128 + OUT_EVO

###

def tensor2list(torch_tensor):
    VELOCITY = 0
    TIME = 0.0
    note_buffer = [0 for _ in range(128)]
    cur_buffer = [0 for _ in range(128)]
    song_notes = []
    index_list = torch_tensor.tolist()

    for idx in index_list:
        if 0 <= idx < IN_EVO:
            VELOCITY = idx
        elif IN_EVO <= idx < OUT_EVO:
            in_note = idx - IN_EVO
            note_buffer[in_note] = TIME
        elif OUT_EVO <= idx < TS_EVO:
            out_note = idx - OUT_EVO
            song_notes.append([out_note, VELOCITY, note_buffer[out_note], TIME])
            note_buffer[out_note] = 0
        else:
            TIME += 0.01 * (idx - TS_EVO + 1)
    
    return song_notes
    


if __name__ == '__main__': # Testing environment

    LIST = midi2list(ifpath = "data/a.midi")
    TENSOR = list2tensor(LIST)
    LIST_rec = tensor2list(TENSOR)
    list2midi(LIST_rec, ofpath = "test_rec.midi")
    