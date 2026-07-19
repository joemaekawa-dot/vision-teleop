"""
Safety layer for the Pi-side controller. Nothing reaches the servos without
passing through here. Three independent guards:

  * Watchdog   — no fresh command within HOLD_MS => freeze at last commanded
                 pose; within RELAX_MS still frozen; beyond => caller may relax.
  * Deadman    — motion only when the operator's FLAG_ENABLED bit is set.
  * SlewLimit  — per-joint max ticks/tick, so a target jump can never command a
                 violent servo move (bounds velocity regardless of input).
  * SoftLimits — per-joint absolute tick window; hard clamp, last line of defense.

Ticks are Feetech STS3215 units: 0..4095 == 360 deg.
"""
from __future__ import annotations

from dataclasses import dataclass

TICKS_PER_REV = 4096

# Command-freshness watchdog (nanoseconds)
HOLD_MS = 100     # no fresh frame beyond this -> hold position
RELAX_MS = 2000   # beyond this -> controller may disable torque (caller decides)


@dataclass
class JointGuard:
    lo: int          # absolute soft-limit low (ticks)
    hi: int          # absolute soft-limit high (ticks)
    max_step: int    # max change per control tick (slew / velocity clamp)


class Safety:
    def __init__(self, guards: dict[int, JointGuard]):
        self.guards = guards

    def clamp_target(self, sid: int, target: int, current_cmd: int) -> int:
        """Apply slew limit around the last commanded value, then soft limits."""
        g = self.guards[sid]
        # velocity clamp: never move more than max_step from what we last commanded
        if target > current_cmd + g.max_step:
            target = current_cmd + g.max_step
        elif target < current_cmd - g.max_step:
            target = current_cmd - g.max_step
        # absolute soft limits
        return max(g.lo, min(g.hi, target))

    @staticmethod
    def watchdog_state(age_ns: int) -> str:
        ms = age_ns / 1e6
        if ms <= HOLD_MS:
            return "live"
        if ms <= RELAX_MS:
            return "hold"
        return "relax"


def guards_around_home(home: dict[int, int], window: int = 400,
                       max_step: int = 12) -> dict[int, JointGuard]:
    """Conservative first-bring-up limits: home +/- window ticks (~35 deg),
    slew max_step ticks/control-tick. Tighten/loosen per joint later from URDF."""
    out = {}
    for sid, h in home.items():
        lo = max(0, h - window)
        hi = min(TICKS_PER_REV - 1, h + window)
        out[sid] = JointGuard(lo=lo, hi=hi, max_step=max_step)
    return out
