"""
Monocular depth (Depth Anything V2 small, ONNX) — VIEWER ONLY.

Runs on its own thread at low resolution/rate to produce a colourised relative
depth map for the 2nd GUI pane. It does NOT feed the control loop, so it adds
ZERO latency to the arm (the arm's depth signal remains the fist-calibrated
relative hand-size). This pane just lets you SEE the depth the camera sees.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import onnxruntime as ort

_MODEL = os.path.join(os.path.dirname(__file__), "models", "depth_anything_v2_vits.onnx")
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class DepthAnything:
    def __init__(self, size=252, threads=2):
        self.size = size - (size % 14)      # must be a multiple of 14
        # CPU-ONLY on purpose: MediaPipe (control thread) owns the GPU/Metal via its
        # GL delegate; running depth on CoreML/Metal too starves the control loop.
        # CPU depth in its own thread (~5-10 Hz) keeps the arm path at full rate.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads      # don't hog all cores from control
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(_MODEL, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name

    def infer(self, frame_bgr, out_w, out_h):
        """frame_bgr -> colourised depth (BGR) at (out_w,out_h). Near=bright."""
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        depth = self.sess.run(None, {self.iname: x})[0][0]     # (size,size), larger=nearer
        d = depth - depth.min()
        d = d / (d.max() + 1e-6)
        u8 = (d * 255).astype(np.uint8)
        color = cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)
        return cv2.resize(color, (out_w, out_h))
