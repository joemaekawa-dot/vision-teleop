"""
Control-stream TX (Mac side).

Fastest reasonable path to the Pi:
  * Raw UDP, unicast, to the Pi's LAN IP (NOT the Tailscale 100.x address) —
    the overlay adds jitter and a DERP-relay tail we cannot tolerate on the
    control loop. Tailscale stays for SSH/telemetry only.
  * Best-effort, latest-wins: never retransmit (a newer target supersedes a
    lost one). Robustness comes from REDUNDANT STATE PACKING — every datagram
    carries the last 3 frames, so one dropped packet is recovered by the next.
  * DSCP EF marking so a QoS/WMM-enabled AP puts control in the voice queue.
"""
from __future__ import annotations

import socket
import time
from collections import deque

from eeframe import EEFrame, pack_packet, snapshot, MAX_FRAMES_PER_PACKET

# macOS QoS: IP_TOS/DSCP is unreliable on Darwin; the WMM voice class is reached
# via SO_NET_SERVICE_TYPE = NET_SERVICE_TYPE_VO. Set both, best-effort.
_SO_NET_SERVICE_TYPE = 0x1116
_NET_SERVICE_TYPE_VO = 8


class ControlSender:
    def __init__(self, host: str, port: int = 47800):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        # small send buffer: we only ever want the freshest datagram in flight
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 16)
        for level, opt, val in (
            (socket.IPPROTO_IP, socket.IP_TOS, 0xB8),            # DSCP EF
            (socket.SOL_SOCKET, _SO_NET_SERVICE_TYPE, _NET_SERVICE_TYPE_VO),  # WMM VO
        ):
            try:
                self.sock.setsockopt(level, opt, val)
            except OSError:
                pass
        self._ring: deque[EEFrame] = deque(maxlen=MAX_FRAMES_PER_PACKET)
        self._seq = 0

    def send(self, frame: EEFrame) -> None:
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        frame.seq = self._seq
        frame.send_ts_ns = time.time_ns()
        self._ring.appendleft(snapshot(frame))  # copy: caller may reuse the object
        pkt = pack_packet(list(self._ring))  # newest first + up to 2 redundant
        try:
            self.sock.sendto(pkt, self.addr)
        except (BlockingIOError, InterruptedError):
            pass  # drop rather than block the perception loop

    def close(self):
        self.sock.close()
