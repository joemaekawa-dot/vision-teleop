#!/usr/bin/env python3
"""Diagnostic: report RAW HandLandmarker detections (any handedness) per camera,
so we can tell 'camera never sees a hand' from 'hand seen but handedness-filtered'.

  python hand_probe.py [camera_index] [seconds]
"""
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

CAM = sys.argv[1] if len(sys.argv) > 1 else "0"
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")

lm = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL),
    running_mode=mp_vision.RunningMode.VIDEO, num_hands=2,
    min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))

cap = cv2.VideoCapture(int(CAM) if CAM.isdigit() else CAM)
print(f"camera[{CAM}] opened={cap.isOpened()} — hold your hand in view now")
t0 = time.perf_counter(); ts = 0; frames = 0; any_hand = 0
labels = Counter()
last = t0
while time.perf_counter() - t0 < SECS:
    ok, frame = cap.read()
    if not ok:
        continue
    frames += 1; ts += 33
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
    n = len(res.hand_landmarks) if res.hand_landmarks else 0
    if n:
        any_hand += 1
        for h in res.handedness:
            labels[h[0].category_name] += 1
    now = time.perf_counter()
    if now - last >= 1.0:
        print(f"t={now-t0:4.1f}s frames={frames} frames_with_hand={any_hand} "
              f"raw_labels={dict(labels)}")
        last = now
cap.release(); lm.close()
print(f"\nDONE cam[{CAM}]: {frames} frames, {any_hand} had a hand, "
      f"handedness_seen={dict(labels)}")
if any_hand == 0:
    print(">> No hand EVER detected on this camera → wrong camera or hand not in frame.")
else:
    print(">> Hand IS detected. If teleop showed hand=0, it's the handedness filter/mirror.")
