#!/usr/bin/env bash
# Fetch model weights not stored in git.
#  - Depth Anything V2 small ONNX (~97MB) for the viewer depth pane.
#  (hand_landmarker.task is committed in mac/models/.)
set -euo pipefail
MODELS="$(cd "$(dirname "$0")/.." && pwd)/mac/models"
mkdir -p "$MODELS"
if [ ! -f "$MODELS/depth_anything_v2_vits.onnx" ]; then
  echo "downloading Depth Anything V2 small ONNX ..."
  curl -fSL -o "$MODELS/depth_anything_v2_vits.onnx" \
    "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
fi
echo "models ready in $MODELS"
