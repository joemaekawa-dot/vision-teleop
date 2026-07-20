#!/usr/bin/env python3
"""Send synthetic CALIBRATED EEFrames (sweeping EE delta) to the live Pi to verify
the IK->servo path moves the arm, independent of hand/gesture."""
import math
import sys
import time

from eeframe import EEFrame, FLAG_VALID, FLAG_ENABLED, FLAG_CALIBRATED
from transport_tx import ControlSender

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.3.22"
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
tx = ControlSender(HOST)
CAL = FLAG_VALID | FLAG_ENABLED | FLAG_CALIBRATED
t0 = time.perf_counter()
n = 0
while time.perf_counter() - t0 < SECS:
    t = time.perf_counter() - t0
    f = EEFrame(pos=(0.10 * math.sin(t * 1.2),   # X forward/back
                     0.10 * math.sin(t * 0.8),   # Y left/right
                     -0.06 * (0.5 + 0.5 * math.sin(t * 0.6))),  # Z down
                gripper=0.5 + 0.5 * math.sin(t * 1.5), flags=CAL)
    tx.send(f)
    n += 1
    time.sleep(1 / 60)
tx.close()
print(f"sent {n} CALIBRATED frames to {HOST}")
