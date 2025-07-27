
import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Helper function
def fingers_up(landmarks):
    finger_tips = [8, 12, 16, 20]
    fingers = []

    # Thumb (compare x for right hand)
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Start webcam
cap = cv2.VideoCapture(0)
prev_click_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm_list = hand_landmarks.landmark
            finger_states = fingers_up(lm_list)

            # Get index finger tip position
            x = int(lm_list[8].x * w)
            y = int(lm_list[8].y * h)

            screen_x = screen_w * lm_list[8].x
            screen_y = screen_h * lm_list[8].y

            # Action based on gesture
            if finger_states == [0,1,0,0,0]:  # Index finger only
                pyautogui.moveTo(screen_x, screen_y)

            elif finger_states == [0,1,1,0,0]:  # Index and middle up
                pyautogui.click()

            elif finger_states == [1,1,0,0,0]:  # Thumb + index (pinch)
                # Check distance
                x1, y1 = lm_list[4].x * w, lm_list[4].y * h
                x2, y2 = lm_list[8].x * w, lm_list[8].y * h
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist < 40 and time.time() - prev_click_time > 1:
                    pyautogui.doubleClick()
                    prev_click_time = time.time()

            elif finger_states == [1,0,0,0,0]:  # Thumb up
                pyautogui.scroll(20)

            elif finger_states == [0,0,0,0,1]:  # Pinky only (thumb down alternative)
                pyautogui.scroll(-20)

            elif finger_states == [0,1,1,1,0]:  # Index, middle, ring
                pyautogui.hotkey('alt', 'right')

            elif sum(finger_states) == 0:  # Fist
                pyautogui.hotkey('alt', 'left')

    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


