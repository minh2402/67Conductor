import cv2
import mediapipe as mp
import subprocess
import sys
import time

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils



def compute_angle_from_landmarks(landmarks, idx1, idx2, idx3):
    # Extract normalized x,y (MediaPipe gives [0,1])
    p1 = (landmarks[idx1].x, landmarks[idx1].y)
    p2 = (landmarks[idx2].x, landmarks[idx2].y)
    p3 = (landmarks[idx3].x, landmarks[idx3].y)
    
    # Send to Haskell via stdin
    line = f"{p1[0]} {p1[1]} {p2[0]} {p2[1]} {p3[0]} {p3[1]}"
    haskell_proc.stdin.write(line + '\n')
    haskell_proc.stdin.flush()
    
    # Read result from stdout
    angle_str = haskell_proc.stdout.readline().strip()
    if angle_str:
        return float(angle_str)
    return None  # Error case

# Camera loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Example: Angle at elbow (indices from MediaPipe: 11=left shoulder, 13=left elbow, 15=left wrist)
        angle = compute_angle_from_landmarks(landmarks, 11, 13, 15)
        if angle is not None:
            print(f"Inner angle: {angle} radians ({angle * 180 / 3.14159:.1f} degrees)")
        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('MediaPipe Pose', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
haskell_proc.stdin.close()
haskell_proc.wait()  # Ensure Haskell exits cleanly