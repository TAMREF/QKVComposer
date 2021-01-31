import note_seq
from note_seq.protobuf import music_pb2
import json
 
import numpy as np
import torch
from noteseq2tensor import rawData2Indices
from eventconst import eventConst as EC

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

def tensor2list(torch_tensor):
    VELOCITY = 0
    TIME = 0.0
    note_buffer = [0 for _ in range(128)]
    cur_buffer = [0 for _ in range(128)]
    song_notes = []
    index_list = torch_tensor.tolist()

    for idx in index_list:
        et, v, p, d = EC.getEntitiesFromIndex(idx)
        if v:
            VELOCITY = v
        if d:
            TIME += 0.01 * d
        if et == EC.ON:
            note_buffer[p] = TIME
        if et == EC.OFF:
            song_notes.append([p, VELOCITY, note_buffer[p], TIME])
            note_buffer[p] = 0

    return song_notes
    


if __name__ == '__main__': # Testing environment

    LIST = midi2list(ifpath = "data/a.midi")
    TENSOR = list2tensor(LIST)
    LIST_rec = tensor2list(TENSOR)
    list2midi(LIST_rec, ofpath = "test_rec.midi")
    
