"""
Feetech STS3215 bus driver for SO-101 (IDs 1..6).

Encapsulates ALL servo writes behind an explicit torque gate. Construction and
reads never energize the arm; only enable_torque()+write_goals() can move it.
Uses sync read/write so a full 6-servo cycle is one bus transaction.
"""
from __future__ import annotations

from scservo_sdk import (PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite,
                         COMM_SUCCESS, SCS_LOBYTE, SCS_HIBYTE, SCS_LOWORD, SCS_HIWORD)

STS_PROTOCOL_END = 0
ADDR_TORQUE_ENABLE = 40
ADDR_ACCELERATION = 41
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_PRESENT_POSITION = 56
LEN_GOAL_POSITION = 2
LEN_PRESENT_POSITION = 2

SO101_IDS = [1, 2, 3, 4, 5, 6]


class FeetechBus:
    def __init__(self, port="/dev/ttyACM0", baud=1_000_000, ids=None):
        self.ids = ids or list(SO101_IDS)
        self.port = PortHandler(port)
        self.ph = PacketHandler(STS_PROTOCOL_END)
        if not self.port.openPort():
            raise RuntimeError(f"cannot open {port}")
        if not self.port.setBaudRate(baud):
            raise RuntimeError(f"cannot set baud {baud}")
        self._reader = GroupSyncRead(self.port, self.ph, ADDR_PRESENT_POSITION,
                                     LEN_PRESENT_POSITION)
        for i in self.ids:
            self._reader.addParam(i)
        self._torque_on = False

    @staticmethod
    def _plausible(v) -> bool:
        # F2: a real STS3215 present position is 1..4094; 0/4095/None are the
        # tell-tales of an unpopulated sync-read slot or a comms glitch. Reject
        # them so a bad read can never seed a full-range commanded jump.
        return v is not None and 0 < v < 4095

    def read_positions(self) -> dict[int, int]:
        """Return ONLY plausible readings (bad/zero slots omitted, never 0-filled)."""
        out = {}
        if self._reader.txRxPacket() == COMM_SUCCESS:
            for i in self.ids:
                v = self._reader.getData(i, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                if self._plausible(v):
                    out[i] = v
        # fill any missing/rejected id via individual read
        for i in self.ids:
            if i not in out:
                v, comm, _ = self.ph.read2ByteTxRx(self.port, i, ADDR_PRESENT_POSITION)
                if comm == COMM_SUCCESS and self._plausible(v):
                    out[i] = v
        return out

    def read_positions_complete(self) -> dict[int, int] | None:
        """All ids present & plausible, or None. Use before energizing."""
        pos = self.read_positions()
        return pos if all(i in pos for i in self.ids) else None

    def set_profile(self, accel=20, speed=800):
        """Conservative internal accel/speed. Our slew-limiter is the real governor."""
        for i in self.ids:
            self.ph.write1ByteTxRx(self.port, i, ADDR_ACCELERATION, accel & 0xFF)
            self.ph.write2ByteTxRx(self.port, i, ADDR_GOAL_SPEED, speed & 0xFFFF)

    def _write_goal_regs(self, goals: dict[int, int]):
        """Write goal-position registers regardless of torque state (used to
        PRELOAD before enabling torque). Writing the reg while limp is harmless;
        the servo only acts on it once torque is on."""
        w = GroupSyncWrite(self.port, self.ph, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        for i, tick in goals.items():
            tick = max(0, min(4095, int(tick)))
            w.addParam(i, [SCS_LOBYTE(tick), SCS_HIBYTE(tick)])
        w.txPacket()

    def enable_torque(self, on: bool, retries: int = 4):
        """Set torque on every servo, RETRYING transient serial glitches per servo
        (Feetech writes occasionally NAK). Only a servo that fails all retries is a
        real fault. Enabling that still fails on any servo aborts to all-off and
        raises; disabling that fails leaves torque_on True (never claim limp falsely)."""
        failed = []
        for i in self.ids:
            ok = False
            for _ in range(retries):
                comm, _ = self.ph.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, 1 if on else 0)
                if comm == COMM_SUCCESS:
                    ok = True
                    break
            if not ok:
                failed.append(i)
        if on:
            if failed:
                for i in self.ids:  # partial enable -> force everything back off
                    self.ph.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, 0)
                self._torque_on = False
                raise RuntimeError(f"torque enable failed on {failed}; aborted to limp")
            self._torque_on = True
        else:
            self._torque_on = bool(failed)  # only truly off when all confirmed
            if failed:
                raise RuntimeError(f"torque DISABLE failed on {failed}; arm may be live")

    def enable_torque_hold(self, positions: dict[int, int]):
        """F1: preload goal=present, THEN energize, so enabling holds the current
        pose instead of snapping to a stale Goal_Position register."""
        self._write_goal_regs(positions)
        self.enable_torque(True)

    @property
    def torque_on(self) -> bool:
        return self._torque_on

    def write_goals(self, goals: dict[int, int]):
        """Sync-write goal positions. No-op unless torque is enabled (safety gate)."""
        if not self._torque_on:
            return
        self._write_goal_regs(goals)

    def close(self):
        try:
            if self._torque_on:
                self.enable_torque(False)
        finally:
            self.port.closePort()
