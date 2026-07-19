#!/usr/bin/env python3
"""Measure real control-path RTT Mac->Pi->Mac over UDP. Reports loss + tail latency."""
import socket
import struct
import sys
import time

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.3.22"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
RATE = 60.0
PORT = 47801

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0xB8)  # DSCP EF
except OSError:
    pass
sock.settimeout(0.5)

payload = b"\x00" * 164  # match real control packet size
rtts = []
lost = 0
period = 1.0 / RATE
for i in range(N):
    t0 = time.perf_counter_ns()
    sock.sendto(struct.pack("<I", i) + payload[:160], (HOST, PORT))
    try:
        data, _ = sock.recvfrom(2048)
        t1 = time.perf_counter_ns()
        rtts.append((t1 - t0) / 1e6)
    except socket.timeout:
        lost += 1
    dt = period - (time.perf_counter_ns() - t0) / 1e9
    if dt > 0:
        time.sleep(dt)

rtts.sort()
if rtts:
    def pct(p):
        return rtts[min(len(rtts) - 1, int(len(rtts) * p))]
    print(f"host={HOST} sent={N} recv={len(rtts)} loss={lost}/{N} ({100*lost/N:.1f}%)")
    print(f"RTT ms: min={rtts[0]:.2f} p50={pct(.5):.2f} p95={pct(.95):.2f} "
          f"p99={pct(.99):.2f} max={rtts[-1]:.2f}")
else:
    print(f"host={HOST} NO replies (all {N} lost)")
