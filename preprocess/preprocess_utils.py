import note_seq
from note_seq.protobuf import music_pb2

from glob import glob
import json
 
import numpy as np
import torch
import json
import os
from typing import List
    
EVENT_TYPE_TO_PRIORITY = {
    "SET_VELOCITY" : 1,
    "NOTE_OFF" : 2,
    "NOTE_ON" : 3,
    "TIME_SHIFT" : 4
}

ONEHOT_INDEX_OFFSET = {
    "SET_VELOCITY" : 0,
    "NOTE_ON" : 128,
    "NOTE_OFF" : 256,
    "TIME_SHIFT" : 384
}

class NoteEvent(object):
    """
    eventType : 'SET_VELOCITY' | 'NOTE_OFF' | 'NOTE_ON' | 'TIME_SHIFT'
    pitch ranges [1-128]
    velocity ranges through [1-128]
    duration ranges through [1-100], with unit 10ms and max 1000ms
    """
    def __init__(self, time, eventType, pitch = None, velocity = None, duration = None):
        self.time = time
        assert EVENT_TYPE_TO_PRIORITY.get(eventType)
        self.eventType = eventType
        self.priority = EVENT_TYPE_TO_PRIORITY[self.eventType]
        self.pitch = pitch
        self.velocity = velocity
        self.duration = duration

    def __str__(self):
        return 'Event occurred in {:.3f}, type: {}, pitch = {}, velocity = {}, duration = {}'.format(self.time, self.eventType, self.pitch, self.velocity, self.duration)
    
    def __lt__(self, other):
        if self.time == other.time:
            return self.priority < other.priority
        return self.time < other.time
    
    def getIndex(self):
        index = ONEHOT_INDEX_OFFSET[self.eventType]
        if self.eventType == 'SET_VELOCITY':
            index += self.velocity
        elif self.eventType == 'NOTE_ON' or self.eventType == 'NOTE_OFF':
            index += self.pitch
        elif self.eventType == 'TIME_SHIFT':
            index += (self.duration - 1) # duration is 1 - 100 unit, thus convert to 0-based index
        else:
            assert False
        return index

# NoteSeq object is a representation of a single note.
class NoteSeq(object):
    def __init__(self, pitch:int, velocity:int, startTime:float, endTime:float):
        self.pitch = pitch
        self.velocity = velocity
        self.startTime = startTime
        self.endTime = endTime

        if pitch < 1 or pitch > 128:
            raise Exception('pitch is out of range')
        
        if startTime > endTime:
            raise Exception('startTime goes after endTime')

    def __str__(self):
        return 'Note with pitch = {}, velocity = {}, timeline = [{}, {}]'.format(self.pitch, self.velocity, self.startTime, self.endTime)

    def makeControlEvents(self):
        return [self.makeSetVelocityEvents(), 
                self.makeNoteOffEvents(), 
                self.makeNoteOnEvents()]

    def makeSetVelocityEvents(self):
        return NoteEvent(self.startTime, 'SET_VELOCITY', velocity = self.velocity)

    def makeNoteOnEvents(self):
        return NoteEvent(self.startTime, 'NOTE_ON',      pitch = self.pitch)

    def makeNoteOffEvents(self):
        return NoteEvent(self.endTime,   'NOTE_OFF',     pitch = self.pitch)

#makeTimeShiftEvents function makes TIME_SHIFT events, for durations possibly exceed 1000ms.
def makeTimeShiftEvents(startTime, duration):
    timeShiftEvents = []
    quantDuration = round(duration * 100) #unit is 10ms

    nowTime = startTime
    while quantDuration > 100:
        timeShiftEvents.append( NoteEvent(nowTime, 'TIME_SHIFT', duration = 100) )
        quantDuration -= 100
        nowTime += 1.0
    if quantDuration > 0:
        timeShiftEvents.append( NoteEvent(nowTime, 'TIME_SHIFT', duration = quantDuration) )
    
    return timeShiftEvents

# makeEventTimeLine function takes a list of NoteSeq objects and generates the list of NoteEvent.
def makeEventTimeline(notes: List[NoteSeq]):
    eventList = []
    for note in notes:
        eventList += note.makeControlEvents()
    eventList.sort()
    
    timeShiftEvents = []
    for i, event in enumerate(eventList):
        if i + 1 == len(eventList):
            continue
        timeShiftEvents += makeTimeShiftEvents( event.time,  eventList[i+1].time - event.time )
    
    eventList = sorted(eventList + timeShiftEvents)
    return eventList

# generateIndices function converts the sorted list of NoteEvents into array of one_hot indices.
def generateIndices(events: List[NoteEvent]):
    indices = []
    for event in events:
        indices.append(event.getIndex())
    return indices

# the main access point of this file. converts a raw JSON file into the indices.
def json2Indices(path: str):
    with open(path, 'rt') as f:
        jsonArr = json.load(f)
        
        for elem in jsonArr:
            assert len(elem) == 4
        
        noteSeqArr = [ NoteSeq(elem[0], elem[1], elem[2], elem[3]) for elem in jsonArr ]
        eventArr = makeEventTimeline(noteSeqArr)
        #print(*[str(x) for x in eventArr[:50]],sep='\n') #for debug
        indices = generateIndices(eventArr)
        return indices

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

    # return one-hots.

    for elem in note_json:
            assert len(elem) == 4
    
    noteSeqArr = [ NoteSeq(elem[0], elem[1], elem[2], elem[3]) for elem in note_json ]
    eventArr = makeEventTimeline(noteSeqArr)
    #print(*[str(x) for x in eventArr[:50]],sep='\n') #for debug
    indices = generateIndices(eventArr)
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
            TIME += 0.01 * (idx - TS_EVO)
    
    return song_notes
    


if __name__ == '__main__': # Testing environment

    LIST = midi2list(ifpath = "data/test.midi")
    TENSOR = list2tensor(LIST)
    LIST_rec = tensor2list(TENSOR)
    list2midi(LIST_rec, ofpath = "test_rec.midi")