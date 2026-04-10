import csi, time, ml
from ml.postprocessing.mediapipe import BlazeFace
from servo_shield import Servo0, Servo1
pan_servo  = Servo0()
tilt_servo = Servo1()
pan_servo.angle(40)
tilt_servo.center()
csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(csi.RGB565)
csi0.framesize(csi.VGA)
csi0.window((400, 400))
model = ml.Model("/rom/blazeface_front_128.tflite", postprocess=BlazeFace(threshold=0.4))
WIDTH  = 400
HEIGHT = 400
CX = WIDTH  // 2
CY = HEIGHT // 2
pan_servo.angle(90)
tilt_servo.angle(25)
DEADZONE = 25
PAN_MIN  = 20
PAN_MAX  = 160
TILT_MIN = 20
TILT_MAX = 160
def clamp(val, lo, hi):
	return max(lo, min(hi, val))
def scaled_step(error):
	a = abs(error)
	if a > 100: return 4
	if a > 50:  return 3
	if a > 25:  return 2
	return 1
while True:
	img = csi0.snapshot()
	xf = 0
	yf = 0
	for r, score, keypoints in model.predict([img]):
		sub_x, sub_y, w, h = r
		x = sub_x + w/2
		y = sub_y + h/4
		img.draw_circle(int(x), int(y), 5, color=(255, 0, 0))
		img.draw_line(CX, CY, int(x), int(y))
		img.draw_line(int(x), CY, int(x), int(y))
		img.draw_line(CX, int(y), int(x), int(y))
		xf = int(x) - CX
		yf = CY - int(y)
		break
	if abs(xf) > DEADZONE:
		step = scaled_step(xf)
		pan_servo.angle(clamp(pan_servo.angle() - (step if xf > 0 else -step), PAN_MIN, PAN_MAX))
	if abs(yf) > DEADZONE:
		step = scaled_step(yf)
		tilt_servo.angle(clamp(tilt_servo.angle() + (step if yf > 0 else -step), TILT_MIN, TILT_MAX))
	time.sleep_ms(30)
