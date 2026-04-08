import csi, time, ml, struct
from ml.postprocessing.mediapipe import BlazeFace
from machine import UART

csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(csi.RGB565)
csi0.framesize(csi.VGA)
csi0.window((400, 400))
model = ml.Model(
    "/rom/blazeface_front_128.tflite",
    postprocess=BlazeFace(threshold=0.4)
)
WIDTH = 400
HEIGHT = 400
CX = WIDTH // 2
CY = HEIGHT // 2
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

	print((xf, yf))
	time.sleep_ms(30)
