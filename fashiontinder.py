import cv2
import mediapipe as mp
import pygame
import numpy as np
from playsound import playsound

def generate_sine_wave(freq, duration=DURATION):
    playsound('/sounds/like.wav')

# --- MediaPipe Hand Tracker ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Webcam ---
cap = cv2.VideoCapture(0)

previous_note = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        x_pos = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

        # Map y_pos (0 top, 1 bottom) to MIDI note range
        note = int(72 - x_pos * 24)  # invert y so up = high pitch
        freq = 440.0 * 2 ** ((note - 69) / 12.0)  # MIDI to frequency

        # Play note only if changed
        if previous_note != note:
            sound = generate_sine_wave(freq)
            sound.play()
            previous_note = note

    cv2.imshow("Hand Synth", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
