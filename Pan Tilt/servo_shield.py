import pyb

_tim4 = pyb.Timer(4, freq=50)
_PERIOD = _tim4.period()  # 63999

def _angle_to_ticks(angle):
    # 0.5ms to 2.5ms mapped to 0-180°
    return int(_PERIOD * (0.025 + (angle / 180.0) * 0.100))

class Servo0:
    def __init__(self):
        self._ch = _tim4.channel(1, pyb.Timer.PWM, pin=pyb.Pin("P7"))
        self._angle = 90
        self.angle(90)

    def angle(self, a=None):
        if a is None:
            return self._angle
        self._angle = a
        self._ch.pulse_width(_angle_to_ticks(a))

    def center(self):
        self.angle(90)

class Servo1:
    def __init__(self):
        self._ch = _tim4.channel(2, pyb.Timer.PWM, pin=pyb.Pin("P8"))
        self._angle = 90
        self.angle(90)

    def angle(self, a=None):
        if a is None:
            return self._angle
        self._angle = a
        self._ch.pulse_width(_angle_to_ticks(a))

    def center(self):
        self.angle(90)

class Servo2:
    def __init__(self):
        self._ch = _tim4.channel(3, pyb.Timer.PWM, pin=pyb.Pin("P9"))
        self._angle = 90
        self.angle(90)

    def angle(self, a=None):
        if a is None:
            return self._angle
        self._angle = a
        self._ch.pulse_width(_angle_to_ticks(a))

    def center(self):
        self.angle(90)
