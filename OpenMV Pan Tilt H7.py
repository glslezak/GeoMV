import pyb
import sensor
import image
import time
from pid import PID
from pyb import Servo

pan_servo=Servo(1)
tilt_servo=Servo(2)
led = pyb.LED(3)





red_threshold  = (13, 49, 18, 61, 6, 47)

pan_pid = PID(p=0.07, i=0, imax=90)
tilt_pid = PID(p=0.05, i=0, imax=90)


pan_servo.speed(100)
tilt_servo.speed(100)
tilt_servo.angle(-60)
pan_servo.angle(0)

sensor.reset()
sensor.set_contrast(3)
sensor.set_gainceiling(16)
sensor.set_pixformat(sensor.GRAYSCALE) # use RGB565. GRAYSCALE
sensor.set_framesize(sensor.QQVGA) # use QQVGA for speed.
sensor.set_vflip(False)
sensor.skip_frames(10) # Let new settings take affect.
sensor.set_auto_whitebal(False) # turn this off.
clock = time.clock() # Tracks FPS.

# Get center x, y of camera image
WIDTH = sensor.width()
HEIGHT = sensor.height()
CENTER_X = int(WIDTH / 2 + 0.5)
CENTER_Y = int(HEIGHT / 2 + 0.5)

while True:
    clock.tick()

    img = sensor.snapshot().lens_corr(1.8)
    led.off()
    for c in img.find_circles(threshold=3500,x_margin=10,y_margin=10,r_margin=10,r_min=2,r_max=100,r_step=2,):

        led.on()
        img.draw_circle(c.x(), c.y(), 1, color=(255, 0, 0))
        img.draw_line(CENTER_X, CENTER_Y, c.x(), c.y())

        pan_output = c.x() - CENTER_X
        tilt_output = c.y() - CENTER_Y

        print(c)

        pan_servo.angle(int(pan_servo.angle()-pan_output/9))
        tilt_servo.angle(int(tilt_servo.angle()-tilt_output/9))
