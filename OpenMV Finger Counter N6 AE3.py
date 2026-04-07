import csi, time, ml, struct
from ml.preprocessing import Normalization
from ml.postprocessing.mediapipe import BlazePalm
from ml.postprocessing.mediapipe import HandLandmarks
from machine import UART



csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(csi.RGB565)
csi0.framesize(csi.VGA)
csi0.window((400, 400))

palm_detection = ml.Model("/rom/palm_detection_full_192.tflite", postprocess=BlazePalm(threshold=0.4))
hand_landmarks = ml.Model("/rom/hand_landmarks_full_224.tflite", postprocess=HandLandmarks(threshold=0.4))

hand_lines = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))

WIDTH  = 400
HEIGHT = 400
CX = WIDTH  // 2
CY = HEIGHT // 2

def tip_dist(kp, a, b):
    dx = kp[a][0] - kp[b][0]
    dy = kp[a][1] - kp[b][1]
    return (dx*dx + dy*dy) ** 0.5

n = None
wider_rect = (0, 0, 0, 0)
gesture = 1

while True:
    img = csi0.snapshot()
    xf = 0
    yf = 0

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
                    ml.utils.draw_skeleton(img, keypoints, hand_lines, kp_color=(255, 0, 0), line_color=(0, 255, 0))

                    # Index finger tip = keypoint 8
                    tip = keypoints[8]
                    x, y = int(tip[0]), int(tip[1])

                    # Pan/tilt graphics
                    img.draw_circle(x, y, 5, color=(255, 0, 0))
                    img.draw_line(CX, CY, x, y)
                    img.draw_line(x, CY, x, y)
                    img.draw_line(CX, y,  x, y)

                    xf = x - CX
                    yf = CY - y

                    # Finger extension check (tip Y must be above MCP Y)
                    idx_up  = keypoints[8][1]  < keypoints[5][1]
                    mid_up  = keypoints[12][1] < keypoints[9][1]
                    ring_up = keypoints[16][1] < keypoints[13][1]
                    pnk_up  = keypoints[20][1] < keypoints[17][1]

                    # Tip distances (only meaningful if both fingers are up)
                    d_idx_mid  = tip_dist(keypoints,  8, 12)
                    d_mid_ring = tip_dist(keypoints, 12, 16)
                    d_ring_pnk = tip_dist(keypoints, 16, 20)

                    if ring_up and pnk_up and d_ring_pnk < 35:
                        gesture = 4
                    elif mid_up and ring_up and d_mid_ring < 35:
                        gesture = 3
                    elif idx_up and mid_up and d_idx_mid < 35:
                        gesture = 2
                    else:
                        gesture = 1

                    img.draw_string(10, 10, str(gesture), color=(255, 255, 0), scale=4)

                    new_wider_rect = (r[0] + (r[2] // 2) - (wider_rect[2] // 2),
                                      r[1] + (r[3] // 2) - (wider_rect[3] // 2),
                                      wider_rect[2],
                                      wider_rect[3])
                    n = Normalization(roi=new_wider_rect)
                    break

    
    time.sleep_ms(30)
