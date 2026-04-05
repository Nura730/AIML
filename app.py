import cv2
import numpy as np
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import time

# ------------------ INIT ------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_brightness = 50
smooth_factor = 6
mode = "DISTANCE"
last_switch = 0
last_update = 0

pTime = 0

# ------------------ LOOP ------------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    h, w, _ = img.shape
    lmList = []

    # ------------------ HAND DETECTION ------------------
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        for id, lm in enumerate(hand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append((id, cx, cy))

        mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    # ------------------ MAIN LOGIC ------------------
    if lmList:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        length = hypot(x2 - x1, y2 - y1)

        # -------- Finger Detection --------
        fingers = []
        fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)

        tips = [8, 12, 16, 20]
        for tip in tips:
            fingers.append(1 if lmList[tip][2] < lmList[tip - 2][2] else 0)

        totalFingers = fingers.count(1)

        # -------- Mode Switch (3 fingers) --------
        if totalFingers == 3 and time.time() - last_switch > 1.5:
            mode = "GESTURE" if mode == "DISTANCE" else "DISTANCE"
            last_switch = time.time()

        # -------- Brightness Logic --------
        if mode == "DISTANCE":
            brightness = np.interp(length, [30, 200], [0, 100])
        else:
            if totalFingers == 0:
                brightness = 0
            elif totalFingers == 1:
                brightness = 25
            elif totalFingers == 2:
                brightness = 50
            elif totalFingers == 4:
                brightness = 75
            else:
                brightness = 100

        # -------- Smoothing --------
        smooth = prev_brightness + (brightness - prev_brightness) / smooth_factor

        # -------- Cooldown (avoid rapid changes) --------
        if time.time() - last_update > 0.2:
            sbc.set_brightness(int(smooth))
            last_update = time.time()

        prev_brightness = smooth

        # -------- UI Elements --------
        bar = np.interp(smooth, [0, 100], [400, 150])

        # Panel background
        overlay = img.copy()
        cv2.rectangle(overlay, (30, 120), (110, 420), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        # Bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 255), 2)
        cv2.rectangle(img, (50, int(bar)), (85, 400), (0, 255, 255), -1)

        # Text
        cv2.putText(img, f'{int(smooth)} %', (40, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(img, f'MODE: {mode}', (380, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.putText(img, f'Fingers: {totalFingers}', (380, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # ------------------ FPS COUNTER ------------------
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ------------------ DISPLAY ------------------
    cv2.imshow("AI Smart Control System", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()