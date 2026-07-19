#!/usr/bin/env python3
"""
SO-101 Feetech STS3215 bus scanner — READ-ONLY, SAFE.

Enumerates servos on the bus and reads present position/temp/voltage/torque
state. It NEVER enables torque and NEVER writes a goal position, so running it
CANNOT move the arm. Use it as the first bring-up step to confirm the bus,
discover servo IDs, and capture the raw home pose.

  python3 servo_scan.py                 # scan ids 1..20 @ 1 Mbps on /dev/ttyACM0
  python3 servo_scan.py --relax         # additionally DISABLE torque (arm goes limp)
"""
import argparse
import sys

from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS

# STS/SMS control table
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMP = 63
STS_PROTOCOL_END = 0  # STS/SMS series is little-endian

# SO-101 follower canonical joint order (Feetech IDs 1..6)
JOINT_NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
               4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}


def scan(port_name="/dev/ttyACM0", baud=1_000_000, id_range=range(1, 21), relax=False):
    port = PortHandler(port_name)
    ph = PacketHandler(STS_PROTOCOL_END)
    if not port.openPort():
        print(f"FAIL: cannot open {port_name}", file=sys.stderr)
        return None
    if not port.setBaudRate(baud):
        print(f"FAIL: cannot set baud {baud}", file=sys.stderr)
        return None
    print(f"port {port_name} @ {baud} baud")
    found = {}
    for sid in id_range:
        model, comm, err = ph.ping(port, sid)
        if comm != COMM_SUCCESS:
            continue
        pos, c1, _ = ph.read2ByteTxRx(port, sid, ADDR_PRESENT_POSITION)
        volt, _, _ = ph.read1ByteTxRx(port, sid, ADDR_PRESENT_VOLTAGE)
        temp, _, _ = ph.read1ByteTxRx(port, sid, ADDR_PRESENT_TEMP)
        torq, _, _ = ph.read1ByteTxRx(port, sid, ADDR_TORQUE_ENABLE)
        name = JOINT_NAMES.get(sid, "?")
        deg = pos * 360.0 / 4096.0
        print(f"  id={sid:2d} model={model:5d} {name:13s} pos={pos:4d} ({deg:6.1f}deg) "
              f"torque={'ON' if torq else 'off'} {volt/10:.1f}V {temp}C")
        found[sid] = {"model": model, "pos": pos, "torque": bool(torq),
                      "volt": volt / 10, "temp": temp, "name": name}
        if relax and torq:
            ph.write1ByteTxRx(port, sid, ADDR_TORQUE_ENABLE, 0)
            print(f"     -> torque disabled (id {sid})")
    port.closePort()
    if not found:
        print("NO servos responded. Check power, baud, and cabling.", file=sys.stderr)
    else:
        print(f"\n{len(found)} servos found. Home-pose snapshot (ticks): "
              + ", ".join(f"{i}:{found[i]['pos']}" for i in sorted(found)))
    return found


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--relax", action="store_true", help="disable torque (arm limp)")
    a = ap.parse_args()
    r = scan(a.port, a.baud, relax=a.relax)
    sys.exit(0 if r else 1)
