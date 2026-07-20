"""
Two-step calibration + gripper-range clutch.

Spec (replaces the old fist-5s gesture): with the RIGHT hand,
  STEP 1 (3 s): index + thumb straight/open, the other 3 fingers folded
               -> captures the gripper's FULLY-OPEN pinch distance.
  STEP 2 (3 s): thumb and index touching (pinched)
               -> captures the gripper's FULLY-CLOSED pinch distance, then
                  tracking starts and the index-fingertip position becomes the
                  relative-control origin.
The captured [closed, open] pinch range makes the gripper reach full open/close
per operator. Redoing STEP 1 while tracking re-clutches. Hand-loss -> reset.

Only cheap scalar/boolean comparisons per frame (no per-frame allocation).
"""
from __future__ import annotations


class CalibrationFSM:
    def __init__(self, step_secs=3.0, pinch_touch=0.35, pinch_spread=0.70,
                 pinch_release=0.55):
        self.step_secs = step_secs
        self.pinch_touch = pinch_touch          # raw pinch <= this == "touching"
        self.pinch_spread = pinch_spread        # raw pinch >= this == "spread open"
        self.pinch_release = pinch_release      # pinch rises past this == released
        self.state = "WAIT"                      # WAIT->STEP2->SETTLE->TRACKING
        self._since = None
        self.pinch_open = None
        self.pinch_closed = None
        self._lo, self._hi = 0.30, 1.50          # default pinch range (fallback)

    def _is_open_pose(self, hs):
        # STEP1: index extended, middle/ring/pinky folded, thumb SPREAD from index.
        # (Use pinch-spread instead of the unreliable thumb radial-extension test,
        # and require this ONLY from WAIT so it can't collide with tracking poses.)
        e = hs.fingers_ext
        return (e[1] and not e[2] and not e[3] and not e[4]
                and hs.pinch >= self.pinch_spread)

    def _is_pinch(self, hs):
        return hs.pinch <= self.pinch_touch

    def update(self, hs, now_s):
        """Return (state_label, progress[0..1], just_calibrated)."""
        just = False
        prog = 0.0
        if self.state == "WAIT":                 # re-clutch ONLY here (never in TRACKING)
            if self._is_open_pose(hs):
                if self._since is None:
                    self._since = now_s
                prog = min(1.0, (now_s - self._since) / self.step_secs)
                if prog >= 1.0:
                    self.pinch_open = hs.pinch
                    self.state = "STEP2"
                    self._since = None
            else:
                self._since = None
        elif self.state == "STEP2":
            if self._is_pinch(hs):
                if self._since is None:
                    self._since = now_s
                prog = min(1.0, (now_s - self._since) / self.step_secs)
                if prog >= 1.0:
                    self.pinch_closed = hs.pinch
                    self._finalize()
                    self.state = "SETTLE"        # wait for release before setting origin
                    self._since = None
            else:
                self._since = None
        elif self.state == "SETTLE":
            # C2 fix: capture the relative origin only AFTER the pinch releases, so
            # the origin is a neutral hand position, not the pinched fingertip.
            prog = 1.0
            if hs.pinch >= self.pinch_release:
                self.state = "TRACKING"
                just = True
        return self.state, prog, just

    def _finalize(self):
        if (self.pinch_open is not None and self.pinch_closed is not None
                and self.pinch_open > self.pinch_closed + 0.05):
            self._lo, self._hi = self.pinch_closed, self.pinch_open

    def gripper_value(self, pinch: float) -> float:
        """Calibrated gripper opening 0..1 (0=closed .. 1=open)."""
        return max(0.0, min(1.0, (pinch - self._lo) / (self._hi - self._lo + 1e-6)))

    def reset(self):
        self.state = "WAIT"
        self._since = None

    @property
    def tracking(self) -> bool:
        return self.state == "TRACKING"
