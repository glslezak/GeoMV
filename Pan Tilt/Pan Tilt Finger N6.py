import csi, time, ml
from ml.preprocessing import Normalization
from ml.postprocessing.mediapipe import BlazePalm
from ml.postprocessing.mediapipe import HandLandmarks
from servo_shield import Servo0, Servo1

pan_servo  = Servo0()
tilt_servo = Servo1()
pan_servo.angle()
tilt_servo.angle(25)

csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(csi.RGB565)
csi0.framesize(csi.VGA)
csi0.window((400, 400))

palm_detection = ml.Model("/rom/palm_detection_full_192.tflite", postprocess=BlazePalm(threshold=0.4))
hand_landmarks = ml.Model("/rom/hand_landmarks_full_224.tflite", postprocess=HandLandmarks(threshold=0.4))

hand_lines = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
              (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
              (13,17),(17,18),(18,19),(19,20),(0,17))

WIDTH  = 400
HEIGHT = 400
CX = WIDTH  // 2
CY = HEIGHT // 2

DEADZONE = 45
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

n = None
wider_rect = (0, 0, 0, 0)

while True:
    img = csi0.snapshot()

    if n is None:
        for r, score, keypoints in palm_detection.predict([img]):
            wider_rect = (r[0] - r[2], r[1] - r[3], r[2] * 3, r[3] * 3)
            n = Normalization(roi=wider_rect)
            break
    else:
        hands = hand_landmarks.predict([n(img)])
        if not hands:
            n = None
        else:
            right = hands[1] if len(hands) > 1 else None
            if right:
                for r, score, keypoints in right:
                    ml.utils.draw_skeleton(img, keypoints, hand_lines, kp_color=(255,0,0), line_color=(0,255,0))

                    tip = keypoints[8]
                    x, y = int(tip[0]), int(tip[1])

                    img.draw_circle(x, y, 5, color=(255, 0, 0))
                    img.draw_line(CX, CY, x, y)
                    img.draw_line(x,  CY, x, y)
                    img.draw_line(CX, y,  x, y)

                    xf = x - CX
                    yf = CY - y

                    if abs(xf) > DEADZONE:
                        step = scaled_step(xf)
                        pan_servo.angle(clamp(pan_servo.angle() - (step if xf > 0 else -step), PAN_MIN, PAN_MAX))
                    if abs(yf) > DEADZONE:
                        step = scaled_step(yf)
                        tilt_servo.angle(clamp(tilt_servo.angle() + (step if yf > 0 else -step), TILT_MIN, TILT_MAX))

                    new_wider_rect = (r[0] + (r[2]//2) - (wider_rect[2]//2),
                                      r[1] + (r[3]//2) - (wider_rect[3]//2),
                                      wider_rect[2], wider_rect[3])
                    n = Normalization(roi=new_wider_rect)
                    break

    time.sleep_ms(30)
