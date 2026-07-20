#!/usr/bin/env python3
"""IK adapter dry check with URDF link lengths + the max-range workspace box.
Verifies zero-delta==home and that the horizontal/vertical range corners are
REACHABLE (ik != None), so the arm won't stall/hold at the edges of hand motion."""
import itertools
import math
from eeframe import EEFrame, FLAG_VALID, FLAG_ENABLED, FLAG_CALIBRATED
from so101_adapter import SO101Adapter
from kinematics import ik, home_ee, HOME_ANGLES, REACH

HOME = {1: 2048, 2: 2418, 3: 2113, 4: 2073, 5: 2048, 6: 3480}
CAL = FLAG_VALID | FLAG_ENABLED | FLAG_CALIBRATED
RH, VH = 0.16, 0.12   # must match retarget.REACH_HALF / VERT_HALF
a = SO101Adapter(HOME)
he, pit = home_ee()
print(f"link reach={REACH:.3f}m  home_EE=({he[0]:.3f},{he[1]:.3f},{he[2]:.3f}) "
      f"pitch={math.degrees(pit):.0f}")
z = a.retarget(EEFrame(pos=(0, 0, 0), gripper=1.0, flags=CAL))
print("zero-delta==home(1-5):", z is not None and all(abs(z[i] - HOME[i]) <= 2 for i in range(1, 6)))
print("\nmax-range box corners (reachable = arm tracks the edge, not stall):")
reach = hold = 0
for sx, sy, sz in itertools.product((-1, 0, 1), repeat=3):
    d = (sx * RH, sy * RH, sz * VH)
    g = a.retarget(EEFrame(pos=d, gripper=0.5, flags=CAL))
    tag = "HOLD(unreach)" if g is None else "reach"
    reach += g is not None
    hold += g is None
    if g is None or (sx or sy or sz) == 0:
        print(f"  delta({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f}) -> {tag}")
print(f"\nreachable {reach}/27 corners, unreachable {hold}/27")
