from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
import socket
import json
import time
import os
import threading
from collections import deque


def main():
  if len(sys.argv) < 2:
    print("Usage: python poseture.py <camera_index>")
    return 
  run_video_poselandmarker()


# --- Configuration for gesture detection and signaling ---
# Use UDP to send a single message when a pattern is detected and a RESET when gestures stop.
UDP_HOST = os.environ.get('BODYPOSE_UDP_HOST', '127.0.0.1')
UDP_PORT = int(os.environ.get('BODYPOSE_UDP_PORT', '5005'))
# Margins for considering wrists "up" or "down" relative to shoulder (normalized coords)
UP_MARGIN = 0.03
DOWN_MARGIN = 0.03
# How many stable frames (no movement / neutral) to consider the gesture stopped / reset
RESET_STABLE_FRAMES = 25
# Max sequence length to keep (if still needed for local buffering)
MAX_SEQ_LEN = 128

# UDP settings: Python will SEND landmark frames to Haskell at (HOST:PORT).
# Haskell can send back a notification to PY_NOTIFY_PORT to tell Python a pattern
# was detected (single-shot). Configure via environment variables.
UDP_HOST = os.environ.get('BODYPOSE_UDP_HOST', '127.0.0.1')
UDP_PORT = int(os.environ.get('BODYPOSE_UDP_PORT', '5005'))
PY_NOTIFY_PORT = int(os.environ.get('BODYPOSE_NOTIFY_PORT', '5006'))
# Rate-limit sending (milliseconds) and movement threshold for sending updates
SEND_INTERVAL_MS = int(os.environ.get('BODYPOSE_SEND_INTERVAL_MS', '100'))
SEND_MOVE_THRESH = float(os.environ.get('BODYPOSE_SEND_MOVE_THRESH', '0.02'))


def send_udp_message(payload: bytes, host=UDP_HOST, port=UDP_PORT, sock=None):
  try:
    if sock is not None:
      sock.sendto(payload, (host, port))
      return
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
      s.sendto(payload, (host, port))
  except Exception as e:
    logging.getLogger(__name__).exception("Failed to send UDP message: %s", e)


def start_notify_listener(state):
  """Start a background thread that listens for notifications from Haskell.

  When a UDP packet is received on PY_NOTIFY_PORT, set state['last_notify'] to
  (time, message). This allows the main loop to react (e.g., display one-shot overlay).
  """
  def listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', PY_NOTIFY_PORT))
    logger = logging.getLogger(__name__)
    logger.info("Notify listener bound on port %d", PY_NOTIFY_PORT)
    while not state.get('stop_listener'):
      try:
        sock.settimeout(0.5)
        data, addr = sock.recvfrom(4096)
        msg = data.decode('utf-8', errors='replace')
        state['last_notify'] = (time.time(), msg)
        logger.info("Received notify from %s: %s", addr, msg)
      except socket.timeout:
        continue
      except Exception:
        logger.exception("Notify listener error")
        break
    try:
      sock.close()
    except Exception:
      pass

  t = threading.Thread(target=listener, daemon=True)
  t.start()
  return t


def classify_AB_from_landmarks(landmarks):
  """Return 'A' or 'B' or None from a single pose landmarks list.

  A = left hand up, right hand down
  B = left hand down, right hand up
  """
  # Expect landmarks sequence (normalized). Use indices from Mediapipe Pose:
  # left_shoulder=11, right_shoulder=12, left_wrist=15, right_wrist=16
  try:
    l_sh = landmarks[11].y
    r_sh = landmarks[12].y
    l_wr = landmarks[15].y
    r_wr = landmarks[16].y
  except Exception:
    return None

  left_up = l_wr < (l_sh - UP_MARGIN)
  left_down = l_wr > (l_sh + DOWN_MARGIN)
  right_up = r_wr < (r_sh - UP_MARGIN)
  right_down = r_wr > (r_sh + DOWN_MARGIN)

  if left_up and right_down:
    return 'A'
  if left_down and right_up:
    return 'B'
  return None

def run_video_poselandmarker():
  # STEP 2: Create a PoseLandmarker object.
  base_options = python.BaseOptions(model_asset_path='lib/pose_landmarker_heavy.task')
  options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    num_poses=1  # Allow detection of up to 2 people (adjust as needed).
  )
  detector = vision.PoseLandmarker.create_from_options(options)

  # Open the camera stream.
  cap = cv2.VideoCapture(int(sys.argv[1]))
  if not cap.isOpened():
    logging.getLogger(__name__).error("Error: Could not open camera.")
    return

  # initialize state for gesture detection / sending
  state = {
    'stable_frames': 0,
    'last_lw_y': None,
    'last_rw_y': None,
    'last_notify': None,
    'stop_listener': False,
  }

  # start listener thread to receive single-shot notifications from Haskell
  start_notify_listener(state)

  # prepare UDP socket for sending landmark frames (reused)
  send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  send_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  last_sent = {'time': 0.0, 'landmarks': None}
  indices_to_send = [11, 12, 13, 14, 15, 16]

  while True:
    ret, frame = cap.read()
    if not ret:
      logging.getLogger(__name__).error("Error: Could not read frame.")
      break

    # Convert the frame to RGB as Mediapipe expects RGB images.
    rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect pose landmarks from the camera frame.
    detection_result = detector.detect(image)

    # Visualize the detection result.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # If Haskell sent a notification recently, show a one-shot overlay (2s)
    if state.get('last_notify') is not None:
      ts, msg = state['last_notify']
      if time.time() - ts < 2.0:
        try:
          cv2.putText(annotated_image, f"NOTIFY: {msg}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception:
          pass

    # ---------------- Send landmarks to Haskell (UDP) ----------------
    if detection_result.pose_landmarks:
      lm = detection_result.pose_landmarks[0]
      # collect the selected indices
      try:
        lm_payload = {}
        for idx in indices_to_send:
          pt = lm[idx]
          lm_payload[str(idx)] = {'x': float(pt.x), 'y': float(pt.y), 'z': float(pt.z)}
      except Exception:
        lm_payload = None

      # compute movement magnitude vs last sent
      should_send = False
      now = time.time()
      if lm_payload is not None:
        if last_sent['landmarks'] is None:
          should_send = True
        else:
          max_delta = 0.0
          for k, v in lm_payload.items():
            prev = last_sent['landmarks'].get(k)
            if prev is None:
              max_delta = max_delta or 0.0
              continue
            max_delta = max(max_delta, abs(v['x'] - prev['x']), abs(v['y'] - prev['y']), abs(v['z'] - prev['z']))
          if max_delta >= SEND_MOVE_THRESH:
            should_send = True
        # also enforce rate limit
        if (now - last_sent['time']) < (SEND_INTERVAL_MS / 1000.0):
          should_send = False

        if should_send:
          payload_obj = {'ts': now, 'landmarks': lm_payload}
          try:
            send_udp_message(json.dumps(payload_obj).encode('utf-8'), sock=send_sock)
            last_sent['time'] = now
            last_sent['landmarks'] = lm_payload
            logging.getLogger(__name__).debug("Sent landmarks to %s:%d", UDP_HOST, UDP_PORT)
          except Exception:
            logging.getLogger(__name__).exception("Failed to send landmarks")

      # update last wrist positions for simple movement-based reset (optional local use)
      try:
        lw_y = lm[15].y
        rw_y = lm[16].y
      except Exception:
        lw_y = None
        rw_y = None
      if state['last_lw_y'] is not None and lw_y is not None and state['last_rw_y'] is not None:
        if abs(lw_y - state['last_lw_y']) > UP_MARGIN or abs(rw_y - state['last_rw_y']) > UP_MARGIN:
          state['stable_frames'] = 0
      state['last_lw_y'] = lw_y
      state['last_rw_y'] = rw_y


  # # Display landmarks 0 to 18 on the camera streaming window.
  #   if detection_result.pose_landmarks:
  #     for i in range(19):  # Loop through landmarks 0 to 18.
  #       landmark = detection_result.pose_landmarks[0][i]
  #       h, w, _ = rgb_frame.shape
  #       cx, cy = int(landmark.x * w), int(landmark.y * h)
  #       cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)  # Draw a green circle for each landmark.
  #       cv2.putText(annotated_image, str(i), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
  #       cv2.putText(annotated_image, f"x:{landmark.x:.2f}, y:{landmark.y:.2f}, z:{landmark.z:.2f}", 
  #             (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imshow("Pose Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Exit the loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Release the camera and close all OpenCV windows.
  # signal listener to stop and close sending socket
  try:
    state['stop_listener'] = True
  except Exception:
    pass
  try:
    send_sock.close()
  except Exception:
    pass
  cap.release()
  cv2.destroyAllWindows()


# def tutorial_onlystillpicture():
#   # STEP 2: Create a PoseLandmarker object with multi-pose detection enabled.
#   base_options = python.BaseOptions(model_asset_path='lib/pose_landmarker.task')
#   options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     output_segmentation_masks=True,
#     num_poses=2  # Allow detection of up to 2 people (adjust as needed).
#   )
#   detector = vision.PoseLandmarker.create_from_options(options)

#   # STEP 3: Load the input image.
#   image = mp.Image.create_from_file("lib/pic/girl.jpg")

#   # STEP 4: Detect pose landmarks from the input image.
#   detection_result = detector.detect(image)

#   # STEP 5: Process the detection result. In this case, visualize it.
#   annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
#   cv2.imshow("hi", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#   cv2.waitKey(0)

#   segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
#   visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
#   cv2.imshow("hi", visualized_mask)
#   cv2.waitKey(0)


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


if __name__ == "__main__":
  main()