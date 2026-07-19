"""
Gesture logic. MVP spec: right hand, hold a fist for 5 s -> command HOME.
Deadman: motion is authorized only while a valid hand is tracked.
"""
from __future__ import annotations


class FistHold:
    """Latches HOME once a fist is held >= hold_secs; releases when the hand
    opens again (edge-triggered so HOME isn't re-fired every frame forever)."""

    def __init__(self, hold_secs=5.0, fist_openness=0.35):
        self.hold_secs = hold_secs
        self.fist_openness = fist_openness
        self._since = None
        self._latched = False

    def update(self, openness: float, now_s: float):
        """Return (home_active: bool, progress: float in [0,1])."""
        is_fist = openness <= self.fist_openness
        if not is_fist:
            self._since = None
            self._latched = False
            return False, 0.0
        if self._since is None:
            self._since = now_s
        held = now_s - self._since
        if held >= self.hold_secs:
            self._latched = True
        return self._latched, min(1.0, held / self.hold_secs)
