"""
Pi-side control loop. Decoupled from perception: it runs at a fixed local rate
and interpolates toward the latest network target, so WiFi jitter/dropouts can
never stall the servo loop.

Pipeline each tick:
  latest EEFrame (RX) --watchdog--> adapter.retarget --safety(slew+limits)--> bus

Modes:
  dry   -- full pipeline, logs intended goals, NEVER energizes/moves the arm.
  live  -- enables torque and writes goals. Guarded by watchdog + deadman +
           slew + soft-limits. Use only with the arm in view and power at hand.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

from transport_rx import ControlReceiver
from safety import Safety, JointGuard, guards_around_home, HOLD_MS
from feetech_bus import FeetechBus
from so101_adapter import SO101Adapter, GRIP_OPEN, GRIP_CLOSED

# Home = the hardware-VERIFIED unfolded "ready" pose (far-center, tip extended
# forward above the table), NOT the folded rest pose. Reaching it from the folded
# pose is a coordinated unfold (proven on hardware 2026-07-20). The old folded
# home {2010,1603,2724,...} sat at the table/servo corner and could not teleop.
HOME = {1: 2048, 2: 2418, 3: 2113, 4: 2073, 5: 2048, 6: GRIP_OPEN}

# Per-joint absolute soft limits = measured servo-EEPROM ROM (ticks). These ARE
# the position safety; velocity safety is the slew limit (max_step). Using the
# full servo range (not a tight window around home) lets the arm slew-safely
# unfold from ANY start pose without the soft-limit clamp overriding the slew
# (a tight window far from the start pose causes a one-tick jump — unsafe).
SERVO_LIMITS = {1: (731, 3461), 2: (1576, 3878), 3: (490, 2745),
                4: (176, 2508), 5: (60, 4035),
                6: (GRIP_CLOSED - 15, GRIP_OPEN + 15)}


class Controller:
    def __init__(self, mode="dry", port="/dev/ttyACM0", hz=100, home=None,
                 window=0, max_step=24):
        self.mode = mode
        self.hz = hz
        self.home = dict(home or HOME)
        self.adapter = SO101Adapter(self.home)
        # Soft limits = full measured servo ROM (position safety); slew=velocity
        # safety. window>0 optionally tightens to home+/-window INTERSECTED with
        # the servo ROM (still safe to home into, since it can only shrink the
        # range, and the start pose is inside the servo ROM anyway -> no clamp jump
        # only if window is large enough; keep window=0 for full-range homing).
        guards = {}
        for sid, (lo, hi) in SERVO_LIMITS.items():
            if window > 0:
                lo = max(lo, self.home[sid] - window)
                hi = min(hi, self.home[sid] + window)
            guards[sid] = JointGuard(lo=lo, hi=hi, max_step=max_step)
        self.safety = Safety(guards)
        self.rx = ControlReceiver(port=47800)
        self.bus = FeetechBus(port=port, ids=self.adapter.ids)
        pos = self.bus.read_positions_complete()   # F2: validated, complete or None
        self.cmd = pos if pos else dict(self.home)
        self._run = False

    def start_live(self):
        # F2: never energize without a confirmed reading of every joint.
        pos = self.bus.read_positions_complete()
        if pos is None:
            raise RuntimeError("refusing LIVE: could not read all joint positions")
        self.cmd = pos
        self.bus.set_profile()
        self.bus.enable_torque_hold(self.cmd)   # F1: preload goal=present, then energize
        print("!! LIVE: torque ENABLED (held at present pose)")

    def _targets(self):
        """Raw (pre-clamp) targets for this tick + a status tag."""
        f = self.rx.latest()
        # RX-thread death is a fault, not silence: force the stalest state.
        state = "relax" if not self.rx.rx_alive() else \
            self.safety.watchdog_state(self.rx.age_ns())
        if f is None or state != "live":
            return dict(self.cmd), state          # hold at last commanded
        if f.home:
            return dict(self.home), "home"
        if not f.enabled:                          # deadman released
            return dict(self.cmd), "deadman"
        raw = self.adapter.retarget(f)
        if raw is None:                            # IK unreachable -> HOLD (no home yank)
            return dict(self.cmd), "unreach"
        return raw, "track"

    def _home_arm(self, period, secs=5.0):
        """Actively slew to HOME (incl. opening the gripper) before tracking.
        In live this physically moves the arm and waits until PRESENT actually
        reaches home (so the gripper fully opens); in dry it only updates cmd."""
        print(f"[{self.mode}] homing to {self.home} ...")
        deadline = time.monotonic() + secs
        while time.monotonic() < deadline:
            goals = {i: self.safety.clamp_target(i, self.home[i], self.cmd[i])
                     for i in self.adapter.ids}
            self.bus.write_goals(goals)
            self.cmd = goals
            cmd_at_home = all(abs(goals[i] - self.home[i]) <= 2 for i in self.adapter.ids)
            if self.mode == "live" and cmd_at_home:
                pres = self.bus.read_positions()
                if pres and all(abs(pres.get(i, 10**9) - self.home[i]) <= 12
                                for i in self.adapter.ids):
                    break
            elif cmd_at_home:
                break
            time.sleep(period)
        print(f"[{self.mode}] homed: cmd={self.cmd}")

    def run(self):
        self._run = True
        period = 1.0 / self.hz
        self._home_arm(period)          # start at HOME + gripper OPEN
        next_t = time.monotonic()
        ticks = 0
        last_log = 0.0
        relaxed = False
        while self._run:
            raw, tag = self._targets()
            goals = {i: self.safety.clamp_target(i, raw[i], self.cmd[i])
                     for i in self.adapter.ids}
            if self.mode == "live":
                if tag == "relax" and not relaxed:
                    self.bus.enable_torque(False)   # link dead too long -> limp
                    relaxed = True
                    print("!! watchdog RELAX -> torque disabled")
                elif tag != "relax" and relaxed:
                    # F3: only re-enable if we can confirm ALL joint positions;
                    # otherwise stay limp rather than energize toward a guess.
                    pos = self.bus.read_positions_complete()
                    if pos is not None:
                        self.cmd = pos
                        self.bus.enable_torque_hold(self.cmd)  # F1: preload + enable
                        relaxed = False
                self.bus.write_goals(goals)
            self.cmd = goals
            ticks += 1
            now = time.monotonic()
            if now - last_log >= 1.0:
                st = self.rx.stats()
                pres = self.bus.read_positions() if self.mode == "live" else {}
                presdump = (" pres={" + ", ".join(f"{i}:{pres[i]}" for i in self.adapter.ids
                            if i in pres) + "}") if pres else ""
                print(f"[{self.mode}] {tag:7s} age={self.rx.age_ns()/1e6:6.1f}ms "
                      f"pkts={st['pkts']} recov={st['recovered']} "
                      f"goals={{{', '.join(f'{i}:{goals[i]}' for i in self.adapter.ids)}}}"
                      f"{presdump}")
                last_log = now
            next_t += period
            dt = next_t - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                next_t = time.monotonic()          # fell behind; resync

    def stop(self):
        self._run = False
        self.rx.stop()
        self.bus.close()   # disables torque
        print("stopped; torque disabled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "live"], default="dry")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--hz", type=int, default=100)
    ap.add_argument("--window", type=int, default=0, help="0 = soft limits are the full servo ROM (safe homing from any pose); >0 tightens to home+/-window")
    ap.add_argument("--max-step", type=int, default=24, help="slew: max ticks/control-tick = velocity safety (lower = gentler)")
    a = ap.parse_args()

    c = Controller(mode=a.mode, port=a.port, hz=a.hz,
                   window=a.window, max_step=a.max_step)
    c.rx.start()
    if a.mode == "live":
        c.start_live()

    def _sig(*_):
        c.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    print(f"controller up: mode={a.mode} hz={a.hz} home={c.home}")
    try:
        c.run()
    finally:
        c.stop()


if __name__ == "__main__":
    main()
