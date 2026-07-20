"""
SO-101 kinematics — 5-DOF: shoulder_pan(yaw) + shoulder_lift/elbow/wrist_flex
(pitches in the yawed vertical plane) + wrist_roll.

Base frame: origin at the base yaw axis, Z UP (perpendicular to the floor),
X forward, Y left. So a target Z increase raises the EE vertically off the floor
and a Z decrease drives it straight down toward the floor.

The EE point is the tip of the gripper (which the operator drives with their
INDEX FINGERTIP). We command a Cartesian EE target and solve joint angles.

⚠️ LINK LENGTHS / HOME ANGLES here are APPROXIMATE (no URDF was available). Since
control is RELATIVE (deltas from a calibrated reference) with an operator gain,
self-consistent FK/IK + live sign/gain tuning is sufficient; refine L1..L3 from a
real URDF for metric accuracy. Per-joint sign/tick conversion is verified live.
"""
from __future__ import annotations

import math

TICKS_PER_REV = 4096
TICKS_PER_RAD = TICKS_PER_REV / (2 * math.pi)   # ~651.9

# link lengths (m) — from the official URDF joint origins
# (TheRobotStudio/SO-ARM100 Simulation/SO101/so101_new_calib.urdf):
#   shoulder_lift->elbow_flex |(-0.11257,-0.028)| = 0.116
#   elbow_flex->wrist_flex    |(-0.1349, 0.0052)| = 0.135
#   wrist_flex->wrist_roll 0.064 + wrist_roll->gripper 0.036 (+jaw) ≈ 0.12 to EE tip
L1 = 0.116   # shoulder -> elbow
L2 = 0.135   # elbow -> wrist
L3 = 0.120   # wrist -> EE (gripper tip)
BASE_H = 0.10  # base -> shoulder pitch axis height (base_pan 0.0624 + offset)
REACH = L1 + L2 + L3   # ~0.371 m max straight-line reach

# Nominal joint angles (rad) at the HOME servo ticks. a2/a3/a4 are pitch of each
# link from horizontal (cumulative handled in fk). APPROX; used only as the
# reference configuration for relative control.
HOME_ANGLES = {
    "pan": 0.0,        # yaw, facing +X
    "lift": math.radians(35),    # shoulder pitch up
    "elbow": math.radians(-70),  # elbow bend (relative to previous link)
    "wrist": math.radians(0),    # wrist pitch
    "roll": 0.0,
}


def fk(pan, a2, a3, a4):
    """Forward kinematics -> (X, Y, Z) EE position and ee_pitch (approach angle
    from horizontal). a2,a3,a4 are the pitch angles of links 1,2,3 RELATIVE to the
    previous link (a2 from horizontal)."""
    p2 = a2
    p3 = a2 + a3
    p4 = a2 + a3 + a4
    r = L1 * math.cos(p2) + L2 * math.cos(p3) + L3 * math.cos(p4)
    z = BASE_H + L1 * math.sin(p2) + L2 * math.sin(p3) + L3 * math.sin(p4)
    return (r * math.cos(pan), r * math.sin(pan), z), p4


def clamp_target(x, y, z, ee_pitch, margin=0.99):
    """Clamp an EE target so IK ALWAYS has a solution: project the wrist point onto
    the reachable 2-link annulus [|L1-L2|, L1+L2] in the yawed plane, then rebuild
    the EE. Beyond-reach hand motion then SATURATES at the arm's limit (using the
    full range) instead of stalling on an unreachable target. Returns (x,y,z)."""
    pan = math.atan2(y, x)
    r = math.hypot(x, y)
    zc = z - BASE_H
    rw = r - L3 * math.cos(ee_pitch)
    zw = zc - L3 * math.sin(ee_pitch)
    d = math.hypot(rw, zw)
    dmax = (L1 + L2) * margin
    dmin = abs(L1 - L2) + 0.01
    if d > 1e-6:
        d2 = min(dmax, max(dmin, d))
        if d2 != d:
            rw *= d2 / d
            zw *= d2 / d
    r2 = rw + L3 * math.cos(ee_pitch)
    z2 = zw + L3 * math.sin(ee_pitch) + BASE_H
    return (r2 * math.cos(pan), r2 * math.sin(pan), z2)


def ik(x, y, z, ee_pitch):
    """Inverse kinematics. Given EE target (x,y,z) in base frame and a desired EE
    pitch (approach angle from horizontal), return (pan, a2, a3, a4) or None if
    unreachable. a2/a3/a4 are per-link relative pitches (matching fk)."""
    pan = math.atan2(y, x)
    r = math.hypot(x, y)
    zc = z - BASE_H
    # wrist position = EE minus the last link along the approach direction
    rw = r - L3 * math.cos(ee_pitch)
    zw = zc - L3 * math.sin(ee_pitch)
    d2 = rw * rw + zw * zw
    d = math.sqrt(d2)
    if d > (L1 + L2) - 1e-6 or d < abs(L1 - L2) + 1e-6:
        return None  # out of reach
    # law of cosines for the 2-link (L1,L2) reaching (rw,zw)
    cos_e = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_e = max(-1.0, min(1.0, cos_e))
    a3 = math.acos(cos_e)                 # elbow interior (choose elbow-up)
    a3 = -a3                              # elbow-down convention (sign tuned live)
    beta = math.atan2(zw, rw)
    psi = math.atan2(L2 * math.sin(a3), L1 + L2 * math.cos(a3))
    a2 = beta - psi
    a4 = ee_pitch - (a2 + a3)             # wrist pitch closes the approach angle
    return pan, a2, a3, a4


# --- angle <-> tick conversion (per joint: home tick + sign) ---
# sign = +1 if increasing tick increases the joint's positive angle (verify LIVE).
JOINT_SIGN = {"pan": 1, "lift": 1, "elbow": 1, "wrist": 1, "roll": 1}


def angles_to_ticks(pan, a2, a3, a4, roll, home_ticks):
    """Map absolute joint angles to servo ticks using the home tick as the zero of
    HOME_ANGLES (relative, so exact home angles need not be perfect)."""
    dp = {
        1: pan - HOME_ANGLES["pan"],
        2: a2 - HOME_ANGLES["lift"],
        3: a3 - HOME_ANGLES["elbow"],
        4: a4 - HOME_ANGLES["wrist"],
        5: roll - HOME_ANGLES["roll"],
    }
    keys = {1: "pan", 2: "lift", 3: "elbow", 4: "wrist", 5: "roll"}
    return {i: int(round(home_ticks[i] + JOINT_SIGN[keys[i]] * dp[i] * TICKS_PER_RAD))
            for i in (1, 2, 3, 4, 5)}


def home_ee():
    """EE position at HOME_ANGLES (the relative-control origin)."""
    (x, y, z), pitch = fk(HOME_ANGLES["pan"], HOME_ANGLES["lift"],
                          HOME_ANGLES["elbow"], HOME_ANGLES["wrist"])
    return (x, y, z), pitch


if __name__ == "__main__":
    # correctness: fk(ik(target)) round-trips
    (hx, hy, hz), hp = home_ee()
    print(f"home_EE=({hx:.3f},{hy:.3f},{hz:.3f}) pitch={math.degrees(hp):.1f}deg")
    ok = 0
    for dx, dy, dz in [(0, 0, 0), (0.03, 0, 0), (0, 0, -0.04), (0.02, 0.02, 0.02)]:
        tgt = (hx + dx, hy + dy, hz + dz)
        sol = ik(*tgt, hp)
        if sol is None:
            print(f"  delta({dx},{dy},{dz}) -> UNREACHABLE"); continue
        (fx, fy, fz), _ = fk(*sol)
        err = math.dist(tgt, (fx, fy, fz))
        print(f"  delta({dx:+.2f},{dy:+.2f},{dz:+.2f}) roundtrip_err={err*1000:.2f}mm")
        ok += err < 1e-4
    print(f"round-trip exact: {ok}/3 nonzero deltas")
