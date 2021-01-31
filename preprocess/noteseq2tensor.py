import numpy as np
import torch
import json
import os
from typing import List
from eventconst import eventConst as EC

NUM_FEATURES = 484

class NoteEvent(object):
    """
    eventType : 'SET_VELOCITY' | 'NOTE_OFF' | 'NOTE_ON' | 'TIME_SHIFT'
    pitch ranges [1-128]
    velocity ranges through [1-128]
    duration ranges through [1-100], with unit 10ms and max 1000ms
    """
    def __init__(self, time, eventType, pitch = None, velocity = None, duration = None):
        self.time = time
        self.eventType = eventType
        self.priority = EC.priority(self.eventType)
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
        index = EC.offset(self.eventType)
        if self.eventType == EC.VEL:
            index += self.velocity #velocity - 1?
        elif self.eventType == EC.ON or self.eventType == EC.OFF:
            index += self.pitch #pitch ? pitch - 1?
        elif self.eventType == EC.TS:
            index += (self.duration - 1) # duration is 1 - 100 unit, thus convert to 0-based index
        else:
            assert False
        return index
    
    def validIndexRange(self, index: int):
        offset = EC.offset(self.eventType)
        if index < offset or index >= offset + EC.length(self.eventType):
            print('type = {}, offset = {}, index = {}, offsetEnds = {}'.format(self.eventType.name, offset, index, offset + EC.length(self.eventType)))
            return False
        return True

# NoteSeq object is a representation of a single note.
class NoteSeq(object):
    def __init__(self, pitch:int, velocity:int, startTime:float, endTime:float):
        self.pitch = pitch
        self.velocity = velocity
        self.startTime = startTime
        self.endTime = endTime

        if pitch < 0 or pitch > 127:
            raise Exception('pitch is out of range')
        if velocity < 0 or velocity > 127:
            raise Exception('velocity is out of range')
        
        if startTime > endTime:
            raise Exception('startTime goes after endTime')

    def __str__(self):
        return 'Note with pitch = {}, velocity = {}, timeline = [{}, {}]'.format(self.pitch, self.velocity, self.startTime, self.endTime)

    def makeControlEvents(self):
        return [self.makeSetVelocityEvents(), 
                self.makeNoteOffEvents(), 
                self.makeNoteOnEvents()]

    def makeSetVelocityEvents(self):
        return NoteEvent(self.startTime, EC.VEL, velocity = self.velocity)

    def makeNoteOnEvents(self):
        return NoteEvent(self.startTime, EC.ON ,      pitch = self.pitch)

    def makeNoteOffEvents(self):
        return NoteEvent(self.endTime,   EC.OFF,     pitch = self.pitch)

#makeTimeShiftEvents function makes TIME_SHIFT events, for durations possibly exceed 1000ms.
def makeTimeShiftEvents(startTime, duration):
    timeShiftEvents = []
    quantDuration = round(duration * 100) #unit is 10ms

    nowTime = startTime
    while quantDuration > 100:
        timeShiftEvents.append( NoteEvent(nowTime, EC.TS, duration = 100) )
        quantDuration -= 100
        nowTime += 1.0
    if quantDuration > 0:
        timeShiftEvents.append( NoteEvent(nowTime, EC.TS, duration = quantDuration) )
    
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
        index = event.getIndex()
        assert event.validIndexRange(index)
        indices.append(index)
    return indices

# converts a list of [Pitch, Velocity, Time_start, Time_End] to indices.
def rawData2Indices(noteRawData: List[List]):
    for elem in noteRawData:
        assert len(elem) == 4
    
    noteSeqArr = [NoteSeq(elem[0], elem[1], elem[2], elem[3]) for elem in noteRawData]
    eventArr = makeEventTimeline(noteSeqArr)
    #print(*[str(x) for x in eventArr[:50]],sep='\n') #for debug
    indices = generateIndices(eventArr)
    return indices

# the main access point of this file. converts a raw JSON file into the indices.
def json2Indices(path: str):
    with open(path, 'rt') as f:
        jsonArr = json.load(f)
        return rawData2Indices(jsonArr)
        
if __name__ == '__main__': # Testing environment
    print(json2Indices(path = os.path.join(os.getcwd(), 'outputs', '0.json')))