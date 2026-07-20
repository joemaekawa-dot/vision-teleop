#!/usr/bin/env bash
# vision-teleop — one-command operation from the Mac.
#
#   ./teleop.sh cams         # list cameras by name (built-in marked); nothing hard-coded
#   ./teleop.sh view [cam]   # camera viewfinder only (no arm, no torque)
#   ./teleop.sh dry  [cam]   # full pipeline to the arm but torque NEVER on (rehearsal)
#   ./teleop.sh live [cam]   # FULL TELEOP: energize arm + track your right hand
#   ./teleop.sh home         # drive the arm to HOME, then relax
#   ./teleop.sh stop         # stop controller + relax (torque off)
#
# [cam] = builtin (default) | a name substring e.g. Webcam | an index e.g. 1
# Camera is opened NON-EXCLUSIVELY (device format not reconfigured) so other
# apps can record the same camera at once; name resolution is one-time at
# startup => zero per-frame latency.
#
# During live/dry:  move your RIGHT hand = move arm | pinch = gripper |
#                   make a FIST and hold 5 s = return HOME | ESC = quit
set -euo pipefail
PI=omakase-pi
PI_IP=192.168.3.22
H="$HOME/vision-teleop"
PY="$H/mac/.venv/bin/python"
export PYTHONPATH="$H/mac:$H/proto"
CMD=${1:-help}
CAM=${2:-builtin}   # resolve the Mac built-in camera by identity, not a fragile index
# zsh (interactive) does NOT treat a trailing '#' as a comment, so `live #` passes
# '#' as the camera arg. Guard against that / empty -> default to builtin.
case "$CAM" in ""|\#*) CAM=builtin;; esac

pi_ctrl() { ssh "$PI" "bash ~/vision-teleop/pi/run_controller.sh $1"; }

case "$CMD" in
  view)
    echo "Viewfinder cam[$CAM] — green dots = hand tracked. keys 0/1/2 switch cam, ESC quit."
    exec "$PY" "$H/mac/run_mac.py" --selftest --camera "$CAM" ;;

  dry|live)
    pi_ctrl "$CMD"
    echo ">>> $CMD up. Move your RIGHT hand in front of cam[$CAM]. Fist 5s=HOME. ESC=quit."
    "$PY" "$H/mac/run_mac.py" --host "$PI_IP" --camera "$CAM" || true
    pi_ctrl stop
    echo ">>> session ended, arm relaxed (torque off)." ;;

  home)
    pi_ctrl live
    "$PY" - <<PY
import time
from eeframe import EEFrame, FLAG_VALID, FLAG_ENABLED, FLAG_HOME
from transport_tx import ControlSender
tx = ControlSender("$PI_IP"); t = time.time()
while time.time() - t < 3.0:
    tx.send(EEFrame(flags=FLAG_VALID | FLAG_ENABLED | FLAG_HOME)); time.sleep(1/60)
tx.close(); print("homing frames sent")
PY
    pi_ctrl stop
    echo ">>> homed and relaxed." ;;

  cams)
    "$PY" - <<'PYEOF'
from camera import list_cameras
cams = list_cameras()
if not cams:
    print("no cameras found")
for n, uid, t, b in cams:
    print(f"{n}  [{t}]  uid={uid}" + ("   <-- BUILTIN (Mac internal, default)" if b else ""))
print("\nSelected by hardware IDENTITY (uniqueID/type), not index:")
print("  ./teleop.sh live builtin   |   ./teleop.sh live Webcam   |   ./teleop.sh live <uniqueID>")
PYEOF
    ;;
  stop) pi_ctrl stop ;;
  *) grep '^#' "$0" | sed 's/^# \{0,1\}//' ;;
esac
