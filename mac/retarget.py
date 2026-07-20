"""
Retarget: HandState -> EEFrame, with One-Euro smoothing for stability.

One-Euro adaptively low-passes: heavy smoothing when the hand is still (kills
jitter that would make the arm buzz), light smoothing when moving fast (keeps
responsiveness). Applied to position and gripper. Orientation is lightly
low-passed and renormalized.
"""
from __future__ import annotations

import math

from eeframe import EEFrame, FLAG_VALID, FLAG_ENABLED, FLAG_HOME, FLAG_CALIBRATED

# Screen-position -> base-frame EE mapping (RELATIVE to the calibrated origin).
# hs.pos is the INDEX-FINGERTIP normalized screen position (mirror + xy_gain
# already applied in perception), each component ~[-1,1]; pos.z is size-based
# relative depth (near=+1). Full comfortable hand travel (offset +/-1 from the
# calibrated centre) maps to +/-REACH_HALF of the arm's HORIZONTAL workspace, so
# the planar/parallel-to-floor motion uses the MAXIMUM reachable range. Depth
# drives the vertical (perpendicular-to-floor) axis.
#   screen up/down  -> EE forward/back (base +X)
#   screen left/right -> EE left/right (base +/-Y)
#   hand closer     -> EE DOWN toward the floor (base -Z)
REACH_HALF = 0.16   # m horizontal half-extent; generous so full hand travel reaches
VERT_HALF = 0.12    # the arm's limit, where the Pi clamp saturates it (max range)
SGN_X, SGN_Y, SGN_Z = +1.0, +1.0, -1.0   # live-tunable signs
DEPTH_DEAD = 0.18   # normalized depth deadband: small hand-size wobble -> no vertical


def _c1(v):
    return max(-1.0, min(1.0, v))


def _deadzone(v, dead):
    """Zero within +/-dead, then rescale so the output still spans [-1,1]."""
    if abs(v) <= dead:
        return 0.0
    return (v - (dead if v > 0 else -dead)) / (1.0 - dead)


class OneEuro:
    def __init__(self, mincutoff=1.2, beta=0.02, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self._x = None
        self._dx = 0.0
        self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t is None or t <= self._t:
            self._t, self._x = t, x
            return x
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.dcutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.mincutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


class Retargeter:
    def __init__(self):
        # mincutoff/beta raised vs. the original (1.2/0.02) to cut tracking lag
        # so the EE follows the hand firmly; still low-passes jitter when still.
        self._fx = OneEuro(mincutoff=1.7, beta=0.05)
        self._fy = OneEuro(mincutoff=1.7, beta=0.05)
        self._fz = OneEuro(mincutoff=1.7, beta=0.05)
        self._fg = OneEuro(mincutoff=2.0, beta=0.01)
        self._q = None
        self._ref = None   # calibrated index-fingertip origin (camera frame, m)

    def relative_eeframe(self, hs, t_s, tracking: bool,
                         just_calibrated: bool, gripper: float) -> EEFrame:
        """RELATIVE IK path. EE delta (m, base frame) from the INDEX-FINGERTIP
        screen offset since calibration, scaled to the arm's reach (max planar
        range). `gripper` is the operator-calibrated 0..1 opening. A depth
        deadzone keeps small size wobble from drifting the vertical axis, so
        horizontal hand motion stays PARALLEL to the floor."""
        p = hs.pos                       # (screen-x, screen-y, size-depth) ~[-1,1]
        if just_calibrated or self._ref is None:
            self._ref = p                # capture origin ("1")
        g = self._fg(gripper, t_s)
        if not tracking:
            return EEFrame(gripper=g, confidence=hs.confidence, flags=FLAG_VALID)
        dx = _c1(p[0] - self._ref[0])    # screen offset, clipped -> predictable max range
        dy = _c1(p[1] - self._ref[1])
        dz = _deadzone(_c1(p[2] - self._ref[2]), DEPTH_DEAD)  # ignore small depth wobble
        ee_x = self._fx(SGN_X * dy * REACH_HALF, t_s)   # screen up/down -> fwd/back (X)
        ee_y = self._fy(SGN_Y * dx * REACH_HALF, t_s)   # screen L/R     -> L/R (Y)
        ee_z = self._fz(SGN_Z * dz * VERT_HALF, t_s)    # closer         -> DOWN (-Z)
        return EEFrame(pos=(ee_x, ee_y, ee_z), gripper=g, confidence=hs.confidence,
                       flags=FLAG_VALID | FLAG_ENABLED | FLAG_CALIBRATED)

    def notify_hand_lost(self):
        """Invalidate the calibrated origin so a re-acquired hand re-seats it
        instead of producing a violent jump from a stale origin (review CRIT-1)."""
        self._ref = None

    def to_eeframe(self, hs, t_s, home=False, enabled=True) -> EEFrame:
        x = self._fx(hs.pos[0], t_s)
        y = self._fy(hs.pos[1], t_s)
        z = self._fz(hs.pos[2], t_s)
        g = self._fg(hs.gripper, t_s)
        q = self._smooth_quat(hs.quat)
        flags = FLAG_VALID
        if enabled:
            flags |= FLAG_ENABLED
        if home:
            flags |= FLAG_HOME
        return EEFrame(pos=(x, y, z), quat=q, gripper=g,
                       confidence=hs.confidence, flags=flags)

    def _smooth_quat(self, q, a=0.5):
        if self._q is None:
            self._q = q
            return q
        # sign-align to avoid flips, lerp, renormalize
        d = sum(p * c for p, c in zip(self._q, q))
        s = -1.0 if d < 0 else 1.0
        blended = tuple(self._q[i] * (1 - a) + s * q[i] * a for i in range(4))
        n = math.sqrt(sum(c * c for c in blended)) or 1.0
        self._q = tuple(c / n for c in blended)
        return self._q
