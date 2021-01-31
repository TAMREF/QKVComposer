from noteseq2tensor import NoteEvent
from eventconst import eventConst as EC

class NoteEventPalette(object):
    def __init__(self):
        self.PianoRoll = [0] * EC.length(EC.ON)
    
    def registerEvent(self, e: NoteEvent):
        if e.eventType != EC.ON and e.eventType != EC.OFF:
            return True
        elif e.eventType == EC.ON:
            assert e.pitch
            if self.PianoRoll[e.pitch] == 1:
                return False
            self.PianoRoll[e.pitch] = 1
            return True
        else:
            assert e.pitch
            if self.PianoRoll[e.pitch] == 0:
                return False
            self.PianoRoll[e.pitch] = 0
            return True

    def registerFromIndex(self, i: int):
        et, v, p, d = EC.getEntitiesFromIndex(i)
        return self.registerEvent(NoteEvent(None, eventType=et, pitch=p, velocity=v, duration=d))