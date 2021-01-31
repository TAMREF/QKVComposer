from preprocess.noteseq2tensor import NoteEvent
from preprocess.eventconst import eventConst as EC

class NoteEventPalette(object):
    def __init__(self):
        self.PianoRoll = [0] * EC.length(EC.ON)
        self.validNoteCount = 0
        self.lastEventType = None
    
    def registerEvent(self, e: NoteEvent):
        tmpEventType = self.lastEventType
        self.lastEventType = e.eventType
        if e.eventType == EC.ON:
            assert e.pitch != None
            if self.PianoRoll[e.pitch] == 1:
                return False
            self.PianoRoll[e.pitch] = 1
            return True
        elif e.eventType == EC.OFF:
            assert e.pitch != None
            if self.PianoRoll[e.pitch] == 0:
                return False
            self.PianoRoll[e.pitch] = 0
            self.validNoteCount += 1
            return True
        elif e.eventType == EC.TS:
            assert e.duration != None
            if tmpEventType == EC.TS:
                return False
            return True
        elif e.eventType == EC.VEL:
            assert e.velocity != None
            if tmpEventType == EC.VEL:
                return False
            return True
        else:
            return True

    def registerFromIndex(self, i: int):
        et, v, p, d = EC.getEntitiesFromIndex(i)
        #print('\nevent = {}, vel = {}, p = {}, d = {}'.format(et.name, v, p, d))
        return self.registerEvent(NoteEvent(None, eventType=et, pitch=p, velocity=v, duration=d))