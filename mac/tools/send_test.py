#!/usr/bin/env python3
"""Send a synthetic EEFrame stream to the Pi controller to validate the link
end-to-end without perception or arm motion. Sweeps pose + toggles gripper."""
import math
import sys
import time

from eeframe import EEFrame, FLAG_VALID, FLAG_ENABLED
from transport_tx import ControlSender

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.3.22"
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

tx = ControlSender(HOST)
t0 = time.perf_counter()
n = 0
while time.perf_counter() - t0 < SECS:
    t = time.perf_counter() - t0
    f = EEFrame(
        pos=(0.6 * math.sin(t * 1.5), 0.4 * math.sin(t * 0.9), 0.3 * math.cos(t)),
        quat=(1.0, 0.0, 0.0, 0.0),
        gripper=0.5 + 0.5 * math.sin(t * 2.0),
        confidence=1.0,
        flags=FLAG_VALID | FLAG_ENABLED,
    )
    tx.send(f)
    n += 1
    time.sleep(1 / 60)
tx.close()
print(f"sent {n} frames to {HOST} over {SECS}s (~{n/SECS:.0f} Hz)")
