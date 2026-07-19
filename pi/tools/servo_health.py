#!/usr/bin/env python3
"""
SO-101 pre-flight health check — home pose + per-servo torque/thermal/error.

No arm motion: the torque-integrity test PRELOADS goal=present before enabling,
so each servo just holds its current pose while we confirm torque toggles on/off
cleanly. Reads hardware error flags, temperature, voltage, and present load.

  python3 servo_health.py
"""
import sys

from scservo_sdk import (PortHandler, PacketHandler, COMM_SUCCESS,
                         ERRBIT_VOLTAGE, ERRBIT_ANGLE, ERRBIT_OVERHEAT,
                         ERRBIT_OVERELE, ERRBIT_OVERLOAD)

ADDR_TORQUE, ADDR_GOAL, ADDR_LOAD, ADDR_VOLT, ADDR_TEMP, ADDR_MOVING, ADDR_PRES = \
    40, 42, 60, 62, 63, 66, 56
END = 0
HOME = {1: 2010, 2: 1603, 3: 2724, 4: 2245, 5: 1872, 6: 2044}
NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
         4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
TEMP_WARN, VOLT_MIN, VOLT_MAX, HOME_TOL = 55, 4.2, 13.0, 60


def err_flags(e):
    out = []
    for bit, name in ((ERRBIT_OVERHEAT, "OVERHEAT"), (ERRBIT_OVERLOAD, "OVERLOAD"),
                      (ERRBIT_VOLTAGE, "VOLTAGE"), (ERRBIT_ANGLE, "ANGLE"),
                      (ERRBIT_OVERELE, "OVERCURRENT")):
        if e & bit:
            out.append(name)
    return out


def decode_load(raw):
    # STS load: bit10 = direction, bits0..9 = magnitude (~0..1000 = 0..100%)
    mag = (raw & 0x3FF) / 10.0
    return mag, ("cw" if raw & 0x400 else "ccw")


def main():
    port = PortHandler("/dev/ttyACM0"); ph = PacketHandler(END)
    if not port.openPort() or not port.setBaudRate(1_000_000):
        print("FAIL open/baud", file=sys.stderr); return 1
    ids = sorted(NAMES)
    ok_all = True
    print(f"{'id':>2} {'joint':13} {'pos':>5} {'home':>5} {'dHome':>6} "
          f"{'volt':>5} {'temp':>4} {'load':>6} {'torq':>4} {'errors'}")
    try:
        for i in ids:
            model, comm, err = ph.ping(port, i)
            if comm != COMM_SUCCESS:
                print(f"{i:>2} {NAMES[i]:13} PING FAIL — servo not responding"); ok_all = False; continue
            pos, _, _ = ph.read2ByteTxRx(port, i, ADDR_PRES)
            volt, _, _ = ph.read1ByteTxRx(port, i, ADDR_VOLT)
            temp, _, _ = ph.read1ByteTxRx(port, i, ADDR_TEMP)
            load, _, _ = ph.read2ByteTxRx(port, i, ADDR_LOAD)
            torq, _, _ = ph.read1ByteTxRx(port, i, ADDR_TORQUE)
            dhome = pos - HOME[i]
            lm, ld = decode_load(load)
            flags = err_flags(err)
            warn = []
            if abs(dhome) > HOME_TOL: warn.append(f"off-home {dhome:+d}t")
            if temp >= TEMP_WARN: warn.append("HOT")
            if not (VOLT_MIN <= volt / 10 <= VOLT_MAX): warn.append("volt?")
            if flags: warn.append("+".join(flags))
            status = "OK" if not warn else "WARN: " + ", ".join(warn)
            if warn: ok_all = False
            print(f"{i:>2} {NAMES[i]:13} {pos:>5} {HOME[i]:>5} {dhome:>+6} "
                  f"{volt/10:>4.1f}V {temp:>3}C {lm:>4.0f}%{ld:>2} "
                  f"{'ON' if torq else 'off':>4} {status}")

        # torque-integrity toggle (no motion: preload goal=present first)
        print("\n-- torque toggle test (preloaded to present pose; no motion) --")
        for i in ids:
            pos, _, _ = ph.read2ByteTxRx(port, i, ADDR_PRES)
            ph.write2ByteTxRx(port, i, ADDR_GOAL, pos)      # F1 preload
            ph.write1ByteTxRx(port, i, ADDR_TORQUE, 1)
            on, _, _ = ph.read1ByteTxRx(port, i, ADDR_TORQUE)
            ph.write1ByteTxRx(port, i, ADDR_TORQUE, 0)
            off, _, _ = ph.read1ByteTxRx(port, i, ADDR_TORQUE)
            good = (on == 1 and off == 0)
            ok_all &= good
            print(f"  id{i} {NAMES[i]:13} enable->{on} disable->{off}  "
                  f"{'OK' if good else 'FAIL torque control'}")
    finally:
        for i in ids:
            ph.write1ByteTxRx(port, i, ADDR_TORQUE, 0)      # ensure all limp
        port.closePort()
    print(f"\nVERDICT: {'ALL HEALTHY' if ok_all else 'ATTENTION NEEDED (see WARN/FAIL above)'}")
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
