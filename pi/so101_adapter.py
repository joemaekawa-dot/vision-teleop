"""
SO101Adapter — maps an EEFrame onto the SO-101's 5-DOF arm + gripper.

STATUS: MVP mapping. The SO-101 is physically 5-DOF, so it cannot realize an
arbitrary 6-DOF pose (end-effector yaw is coupled to base azimuth — the
"dropped DOF" documented in the README). This adapter uses a bounded
workspace->joint teleop mapping (normalized pose deltas around home), NOT metric
IK. It is intentionally conservative for first hardware bring-up. A URDF-based
analytical IK (planar 3R in the shoulder_pan plane) is the next iteration and
must be validated in sim before going live — see README Phase 4.

Input convention (produced by the Mac retargeter):
  frame.pos  = normalized workspace target, each component ~[-1, 1]
  frame.quat = desired tool orientation (only pitch & roll are honored)
  frame.gripper = 0 (closed) .. 1 (open)
"""
from __future__ import annotations

import math

from eeframe import EEFrame
from embodiment import EmbodimentAdapter

# per-axis authority in ticks (kept inside the safety window; tune from URDF).
# Widened for a broader x,y workspace so the EE tracks a larger hand travel.
# NOTE: the controller soft-limit window MUST be >= the largest span (see
# controller.py `window`), or these are clipped back.
SPAN_PAN = 600    # id1 shoulder_pan  <- pos.x  (wide left/right; free joint per ROM)
SPAN_LIFT = 450   # id2 shoulder_lift <- pos.y (up = negative tick, tune sign live)
SPAN_ELBOW = 300  # id3 elbow_flex    <- pos.z (reach)
SPAN_WFLEX = 250  # id4 wrist_flex    <- tool pitch
SPAN_WROLL = 350  # id5 wrist_roll    <- tool roll
# Gripper: absolute open/closed ticks. FULL mechanical ROM measured by rom_scan
# (2026-07-20): closed hard-stop ~2060, open limit ~3504 (servo EEPROM max 3552).
# Old GRIP_OPEN=2400 used only ~24% of travel; now spans the real ~127deg range so
# thumb-index open/close maps to the gripper's full open/close. g=1 (open hand)
# ->OPEN, g=0 (fist/pinch)->CLOSED. Small margins off the hard stops.
GRIP_OPEN = 3480
GRIP_CLOSED = 2075


def _quat_to_pitch_roll(q) -> tuple[float, float]:
    """Return (pitch, roll) in radians from quaternion (w, x, y, z)."""
    w, x, y, z = q
    # roll (x-axis)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    # pitch (y-axis), clamped
    s = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(s)
    return pitch, roll


def _clip(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


class SO101Adapter(EmbodimentAdapter):
    name = "so101"

    def __init__(self, home: dict[int, int]):
        self._home = dict(home)

    @property
    def ids(self):
        return [1, 2, 3, 4, 5, 6]

    def home_ticks(self):
        return dict(self._home)

    def retarget(self, frame: EEFrame) -> dict[int, int]:
        h = self._home
        px, py, pz = frame.pos
        pitch, roll = _quat_to_pitch_roll(frame.quat)
        g = _clip(frame.gripper, 0.0, 1.0)
        return {
            1: int(h[1] + _clip(px) * SPAN_PAN),
            2: int(h[2] - _clip(py) * SPAN_LIFT),
            3: int(h[3] + _clip(pz) * SPAN_ELBOW),
            4: int(h[4] + _clip(pitch / (math.pi / 2)) * SPAN_WFLEX),
            5: int(h[5] + _clip(roll / math.pi) * SPAN_WROLL),
            6: int(GRIP_CLOSED + g * (GRIP_OPEN - GRIP_CLOSED)),  # g=1 open..g=0 closed
        }

    def capabilities(self):
        return {"name": self.name, "dof": 5, "gripper": True,
                "dropped_dof": "end-effector yaw (coupled to base azimuth)",
                "mapping": "mvp-bounded-workspace (not metric IK)"}
