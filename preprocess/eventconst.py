from enum import Enum

class eventConst(Enum):
    ON = 'NOTE_ON'
    OFF = 'NOTE_OFF'
    VEL = 'SET_VELOCITY'
    TS = 'TIME_SHIFT'

    @classmethod
    def priority(cls, e):
        return PRIORITY[e]

    @classmethod
    def offset(cls, e):
        return OFFSET[e]
    
    @classmethod
    def length(cls, e):
        return LENGTH[e]

    @classmethod
    def getVelocity(cls, i):
        return i
    
    @classmethod
    def getPitchFromOn(cls, i):
        return i - OFFSET[cls.ON]
    
    @classmethod
    def getPitchFromOff(cls, i):
        return i - OFFSET[cls.OFF]
    
    @classmethod
    def getDurationFromTimeShift(cls, i):
        return i - OFFSET[cls.TS] + 1

    @classmethod #returns Vel, Pitch, Duration tuple from index to compose noteEvent
    def getEntitiesFromIndex(cls, i):
        if OFFSET[cls.VEL] <= i < OFFSET[cls.VEL] + LENGTH[cls.VEL]:
            return cls.VEL, cls.getVelocity(i), None, None
        if OFFSET[cls.ON] <= i < OFFSET[cls.ON] + LENGTH[cls.ON]:
            return cls.ON,  None, cls.getPitchFromOn(i),  None
        if OFFSET[cls.OFF] <= i < OFFSET[cls.OFF] + LENGTH[cls.OFF]:
            return cls.OFF, None, cls.getPitchFromOff(i), None
        if OFFSET[cls.TS] <= i < OFFSET[cls.TS] + LENGTH[cls.TS]:
            return cls.TS,  None, None, cls.getDurationFromTimeShift(i)

PRIORITY = {
    eventConst.VEL: 1,
    eventConst.OFF: 2,
    eventConst.ON:  3,
    eventConst.TS:  4
}

OFFSET = {
    eventConst.VEL: 0,
    eventConst.ON:  128,
    eventConst.OFF: 256,
    eventConst.TS:  384
}

LENGTH = {
    eventConst.VEL: 128,
    eventConst.ON:  128,
    eventConst.OFF: 128,
    eventConst.TS:  100
}