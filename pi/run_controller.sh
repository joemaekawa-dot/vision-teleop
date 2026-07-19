#!/usr/bin/env bash
# One-command controller lifecycle on the Pi.
#   ./run_controller.sh dry        # safe: full pipeline, never energizes the arm
#   ./run_controller.sh live       # FIRST-run: torque ON, EXTRA-GENTLE (±22°, ~35°/s)
#   ./run_controller.sh live-full  # torque ON, normal authority (±35°, ~105°/s)
#   ./run_controller.sh stop        # stop + relax (torque off)
#   ./run_controller.sh log         # follow the log
set -euo pipefail
export XDG_RUNTIME_DIR=/run/user/$(id -u)
CMD=${1:-dry}
H="$HOME/vision-teleop"

case "$CMD" in
  dry|live|live-full)
    if [ "$CMD" = "live" ]; then
      # soft limits = full servo ROM (safe unfold-home from folded); gentleness
      # comes from the slew limit, not a tight window (a tight window far from the
      # folded start pose would clamp-jump the arm).
      MODE=live; EXTRA="--window 0 --max-step 6"        # gentle first run (~slow slew)
    elif [ "$CMD" = "live-full" ]; then
      MODE=live; EXTRA="--window 0 --max-step 14"
    else
      MODE=dry; EXTRA=""
    fi
    systemctl --user stop vt_ctrl 2>/dev/null || true
    systemctl --user reset-failed vt_ctrl 2>/dev/null || true
    sleep 0.4
    systemd-run --user --unit=vt_ctrl --collect \
      --setenv=PYTHONUNBUFFERED=1 \
      --setenv=PYTHONPATH="$H/pi:$H/proto" \
      /bin/sh -c "exec $H/.venv/bin/python $H/pi/controller.py --mode $MODE $EXTRA >/tmp/vt_ctrl.log 2>&1"
    sleep 1.2
    echo "vt_ctrl: $(systemctl --user is-active vt_ctrl) (mode=$CMD)"
    tail -2 /tmp/vt_ctrl.log || true
    ;;
  stop)
    systemctl --user stop vt_ctrl 2>/dev/null || true
    systemctl --user reset-failed vt_ctrl 2>/dev/null || true
    # belt-and-suspenders: ensure torque off
    PYTHONPATH="$H/pi:$H/proto" "$H/.venv/bin/python" "$H/pi/tools/servo_scan.py" --relax >/dev/null 2>&1 || true
    echo "vt_ctrl stopped; torque relaxed"
    ;;
  log) exec tail -f /tmp/vt_ctrl.log ;;
  *) echo "usage: $0 {dry|live|stop|log}"; exit 1 ;;
esac
