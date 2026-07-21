"""
Monocular depth (Depth Anything V2 small, ONNX).

Two uses, both off the control thread (own thread, CPU) so the arm loop stays at
full rate:
  * viewer: colourised depth map for the 2nd GUI pane.
  * control Z: the RAW depth sampled at the index-MCP pixel. Because this is
    per-point SCENE depth (physical distance of that knuckle), it is invariant to
    palm ORIENTATION and to lateral SHIFT — unlike the old apparent-size cue — so
    horizontal hand motion / hand tilt no longer leak into the arm's vertical axis.

CPU-ONLY on purpose: MediaPipe (control thread) owns the GPU/Metal via its GL
delegate; running depth on CoreML/Metal too starves the control loop.
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
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads      # don't hog all cores from control
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(_MODEL, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name

    def infer_raw(self, frame_bgr):
        """frame_bgr -> raw relative-depth map (size x size, float32; larger=nearer)."""
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        return self.sess.run(None, {self.iname: x})[0][0]

    def colorize(self, raw, out_w, out_h):
        """Raw depth -> MAGMA BGR image (near=bright) for the viewer pane."""
        d = raw - raw.min()
        d = d / (d.max() + 1e-6)
        color = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        return cv2.resize(color, (out_w, out_h))

    def sample(self, raw, u_frame, v_frame, fw, fh, k=2):
        """Median RAW depth in a (2k+1) patch at frame pixel (u,v). Point scene
        depth -> invariant to hand orientation / lateral position."""
        mu = int(round(u_frame * self.size / max(1, fw)))
        mv = int(round(v_frame * self.size / max(1, fh)))
        mu = min(self.size - k - 1, max(k, mu))
        mv = min(self.size - k - 1, max(k, mv))
        return float(np.median(raw[mv - k:mv + k + 1, mu - k:mu + k + 1]))

    # kept for the standalone smoke test
    def infer(self, frame_bgr, out_w, out_h):
        return self.colorize(self.infer_raw(frame_bgr), out_w, out_h)
