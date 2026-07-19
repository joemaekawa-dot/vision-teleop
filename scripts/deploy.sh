#!/usr/bin/env bash
# Sync Pi-side code + shared schema to the robot. Never touches .venv.
set -euo pipefail
PI=${1:-omakase-pi}
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='mac' \
  "$HOME/vision-teleop/pi"    "$PI:~/vision-teleop/"
rsync -az --exclude='__pycache__' \
  "$HOME/vision-teleop/proto" "$PI:~/vision-teleop/"
echo "deployed to $PI"
