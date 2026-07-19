#!/usr/bin/env python3
"""UDP reflector for RTT measurement. Bounces every datagram back to sender."""
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 18)
s.bind(("0.0.0.0", 47801))
print("echo on :47801")
while True:
    data, addr = s.recvfrom(2048)
    s.sendto(data, addr)
