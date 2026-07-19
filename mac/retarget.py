"""
Retarget: HandState -> EEFrame, with One-Euro smoothing for stability.

One-Euro adaptively low-passes: heavy smoothing when the hand is still (kills
jitter that would make the arm buzz), light smoothing when moving fast (keeps
responsiveness). Applied to position and gripper. Orientation is lightly
low-passed and renormalized.
"""
from __future__ import annotations

import math

from eeframe import EEFrame, FLAG_VALID, FLAG_ENABLED, FLAG_HOME


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
