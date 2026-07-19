#!/usr/bin/env python3
"""
SO-101 range-of-motion characterization — measures REALITY vs the adapter's
assumptions. Moves ONE joint at a time in small steps, reading present position
back after each step to learn:
  * is the motor bus actually powered?  (does present track goal at all)
  * real mechanical travel each way before it stops following (hard limit)
  * home position within that travel (mounting offset)
  * backlash (return-to-start error)

Safety design:
  * only the joint under test is energized; all others stay limp.
  * start position is VALIDATED (plausible, nonzero) before any command — a bad
    read can otherwise command a full-range crash toward 0.
  * every goal is clamped 0..4095; steps are small; each step settles + reads
    back; 2 consecutive non-follows => stop that direction (limit/obstruction).
  * always ramps back to start in steps (never a jump) and disables torque in
    finally.

  python3 rom_scan.py --ids 6 --max-delta 60          # SAFE first probe (gripper)
  python3 rom_scan.py --ids 1,2,3,4,5,6 --max-delta 250
"""
import argparse
import sys
import time

from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS

ADDR_TORQUE, ADDR_ACC, ADDR_GOAL, ADDR_SPEED, ADDR_PRES = 40, 41, 42, 46, 56
END = 0
NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
         4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}


def rd(ph, port, jid):
    v, comm, err = ph.read2ByteTxRx(port, jid, ADDR_PRES)
    return v if comm == COMM_SUCCESS else None


def sweep(ph, port, jid, start, sign, step, max_delta, settle, tol):
    """Step from start in one direction; return (reached_pos, moved_ticks, hit)."""
    reached = start
    nofollow = 0
    hit = None
    n = max(1, max_delta // step)
    for k in range(1, n + 1):
        goal = max(0, min(4095, start + sign * k * step))
        ph.write2ByteTxRx(port, jid, ADDR_GOAL, goal)
        time.sleep(settle)
        pres = rd(ph, port, jid)
        if pres is None:
            hit = "read-fail"; break
        if abs(pres - goal) <= tol:
            reached = pres; nofollow = 0
        else:
            nofollow += 1
            if nofollow >= 2:
                hit = "limit/no-follow"; break
    # ramp back to start
    cur = rd(ph, port, jid) or reached
    while abs(cur - start) > step:
        cur = max(0, min(4095, cur - sign * step))
        ph.write2ByteTxRx(port, jid, ADDR_GOAL, cur)
        time.sleep(settle)
    ph.write2ByteTxRx(port, jid, ADDR_GOAL, start)
    time.sleep(settle * 2)
    return reached, abs(reached - start), hit


def characterize(port_name, ids, step, max_delta, settle, tol, speed, accel):
    port = PortHandler(port_name); ph = PacketHandler(END)
    if not port.openPort() or not port.setBaudRate(1_000_000):
        print("FAIL open/baud", file=sys.stderr); return 1
    print(f"ROM scan ids={ids} step={step} max_delta={max_delta} settle={settle}s")
    results = {}
    try:
        for jid in ids:
            start = rd(ph, port, jid)
            volt, _, _ = ph.read1ByteTxRx(port, jid, 62)
            # validate start read before energizing/commanding
            if start is None or not (0 <= start <= 4095) or start == 0:
                print(f"id{jid}: bad start read ({start}) -> SKIP (no crash-commanding)")
                continue
            name = NAMES.get(jid, "?")
            print(f"\n== id{jid} {name}  start={start} ({start*360/4096:.1f}deg) {volt/10:.1f}V ==")
            ph.write1ByteTxRx(port, jid, ADDR_ACC, accel)
            ph.write2ByteTxRx(port, jid, ADDR_SPEED, speed)
            # F1 fix: preload goal=present BEFORE torque-on so enabling holds the
            # current pose instead of snapping to a stale Goal_Position register.
            ph.write2ByteTxRx(port, jid, ADDR_GOAL, start)
            ph.write1ByteTxRx(port, jid, ADDR_TORQUE, 1)
            time.sleep(0.1)
            pos_r, mv_p, hit_p = sweep(ph, port, jid, start, +1, step, max_delta, settle, tol)
            neg_r, mv_n, hit_n = sweep(ph, port, jid, start, -1, step, max_delta, settle, tol)
            ph.write1ByteTxRx(port, jid, ADDR_TORQUE, 0)
            moved = max(mv_p, mv_n)
            powered = moved >= 8
            results[jid] = dict(name=name, start=start, hi=pos_r, lo=neg_r,
                                up=mv_p, down=mv_n, powered=powered,
                                hit_hi=hit_p, hit_lo=hit_n)
            print(f"   +dir: reached {pos_r} (+{mv_p} ticks) [{hit_p or 'full'}]")
            print(f"   -dir: reached {neg_r} (-{mv_n} ticks) [{hit_n or 'full'}]")
            print(f"   => {'MOVED (powered)' if powered else 'NO MOTION (motor power off / stuck)'}"
                  f"  range=[{neg_r}..{pos_r}] home@{start}")
    finally:
        for jid in ids:  # belt-and-suspenders: everything limp
            ph.write1ByteTxRx(port, jid, ADDR_TORQUE, 0)
        port.closePort()

    print("\n=== SUMMARY (measured reality) ===")
    for jid, r in results.items():
        print(f" id{jid} {r['name']:13s} range[{r['lo']}..{r['hi']}] "
              f"span={r['hi']-r['lo']}t (~{(r['hi']-r['lo'])*360/4096:.0f}deg) "
              f"home@{r['start']} {'POWERED' if r['powered'] else 'NO-MOTION'}")
    if results and not any(r["powered"] for r in results.values()):
        print(" >> No joint moved -> motor power almost certainly NOT connected.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--ids", default="6")
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--max-delta", type=int, default=60)
    ap.add_argument("--settle", type=float, default=0.25)
    ap.add_argument("--tol", type=int, default=20)
    ap.add_argument("--speed", type=int, default=300)
    ap.add_argument("--accel", type=int, default=10)
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",") if x.strip()]
    sys.exit(characterize(a.port, ids, a.step, a.max_delta, a.settle, a.tol,
                          a.speed, a.accel))
