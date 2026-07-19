"""
Control-stream RX (Pi side).

A background thread drains the socket and keeps only the FRESHEST frame
(latest-wins by seq, with wraparound handling). The control loop reads
`latest()` at its own rate; `age_ns()` feeds the watchdog. Redundant frames in
each packet let us recover the newest even when the carrying datagram is lost.
"""
from __future__ import annotations

import socket
import threading
import time

from eeframe import EEFrame, unpack_packet


def _seq_newer(a: int, b: int) -> bool:
    """True if seq `a` is STRICTLY newer than `b` under 32-bit wraparound.
    Equal seq is NOT newer (a duplicate must not refresh the watchdog clock)."""
    d = (a - b) & 0xFFFFFFFF
    return 0 < d < 0x80000000


class ControlReceiver:
    def __init__(self, port: int = 47800, bind: str = "0.0.0.0", pin_peer: bool = True):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 18)
        self.sock.bind((bind, port))
        self.sock.settimeout(0.2)
        self._pin_peer = pin_peer
        self._peer = None            # locked to the first sender seen
        self._lock = threading.Lock()
        self._frame: EEFrame | None = None
        self._last_seq = -1
        self._recv_mono_ns = 0
        self._pkts = 0
        self._recovered = 0
        self._rejected = 0           # bad magic/crc/schema/drift
        self._foreign = 0            # dropped: wrong source address
        self._healthy = True         # RX thread alive & socket ok
        self._run = False
        self._thr: threading.Thread | None = None

    def start(self):
        self._run = True
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        backoff = 0.0
        while self._run:
            try:
                data, addr = self.sock.recvfrom(2048)
                backoff = 0.0
            except socket.timeout:
                continue
            except OSError:
                if not self._run:      # expected: socket closed by stop()
                    break
                # transient (WiFi roam, ICMP-driven error): stay alive, back off
                with self._lock:
                    self._rejected += 1
                backoff = min(0.1, backoff + 0.01)
                time.sleep(backoff)
                continue
            # source pinning: lock to the first sender; drop foreign packets
            if self._pin_peer:
                if self._peer is None:
                    self._peer = addr[0]
                elif addr[0] != self._peer:
                    with self._lock:
                        self._foreign += 1
                    continue
            try:
                frames = unpack_packet(data)
            except ValueError:
                with self._lock:
                    self._rejected += 1
                continue  # garbage / corrupt / drifted / foreign-schema packet
            now = time.monotonic_ns()
            with self._lock:
                self._pkts += 1
                for i, f in enumerate(frames):  # newest first
                    if self._last_seq < 0 or _seq_newer(f.seq, self._last_seq):
                        self._frame = f
                        self._last_seq = f.seq
                        self._recv_mono_ns = now
                        if i > 0:
                            self._recovered += 1
                        break

    def latest(self) -> EEFrame | None:
        with self._lock:
            return self._frame

    def age_ns(self) -> int:
        """Nanoseconds since the freshest frame was received (watchdog input)."""
        with self._lock:
            if self._recv_mono_ns == 0:
                return 1 << 62
            return time.monotonic_ns() - self._recv_mono_ns

    def stats(self):
        alive = bool(self._thr and self._thr.is_alive())
        with self._lock:
            return {"pkts": self._pkts, "recovered": self._recovered,
                    "rejected": self._rejected, "foreign": self._foreign,
                    "last_seq": self._last_seq, "peer": self._peer,
                    "rx_alive": alive}

    def rx_alive(self) -> bool:
        return bool(self._thr and self._thr.is_alive())

    def stop(self):
        self._run = False
        if self._thr:
            self._thr.join(timeout=1)
        self.sock.close()
