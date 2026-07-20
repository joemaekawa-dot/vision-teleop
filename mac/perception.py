"""
Perception: RGB frame -> HandState (6-DOF-ish hand pose + gripper + fist).

Two pluggable stages so the pipeline is honest about what's metric:
  * HandTracker (MediaPipe Hands) — 21 landmarks (image + metric "world"),
    handedness. Orientation comes from the palm frame (wrist, index-MCP,
    pinky-MCP); gripper from thumb-index pinch; openness (for the fist gesture)
    from finger extension.
  * DepthSource — supplies the forward axis. Default HeuristicDepth uses the
    hand's apparent size (camera-agnostic, no calibration, NORMALIZED not
    metric). OnnxDepth (Depth Anything V2) is the drop-in that makes Z metric;
    it loads only if a model path is given, so it never blocks bring-up.

pos is emitted NORMALIZED to ~[-1,1] per axis (visual-servoing space); the Pi
adapter maps that to joint spans. Swapping in metric depth changes only the Z
axis source, not the interface.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

# landmark indices
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP = 0, 4, 5, 8
MIDDLE_MCP, PINKY_MCP = 9, 17
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]


@dataclass
class HandState:
    pos: tuple             # normalized x,y,z ~[-1,1] (legacy planar path)
    quat: tuple            # palm orientation w,x,y,z
    gripper: float         # 0=closed .. 1=open (pinch)
    openness: float        # 0=fist .. 1=open hand (for gesture)
    handedness: str
    confidence: float
    landmarks_px: list = field(default_factory=list)
    pinch: float = 0.0                     # raw thumb-tip↔index-tip / hand_scale
    fingers_ext: tuple = (False,) * 5      # extended? [thumb,index,middle,ring,pinky]


def _mat_to_quat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return (w / n, x / n, y / n, z / n)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _clip_unit(v):
    return max(-1.0, min(1.0, v))


class HeuristicDepth:
    """Normalized forward axis from hand apparent size. Bigger hand => nearer.
    Not metric; a stand-in until OnnxDepth is enabled."""
    approximate = True

    def __init__(self, near_px=260.0, far_px=90.0):
        self.near_px, self.far_px = near_px, far_px

    def z_norm(self, size_px, frame=None, uv=None):
        # map size range -> [-1,1]; near(large)->+1, far(small)->-1
        t = (size_px - self.far_px) / max(1.0, (self.near_px - self.far_px))
        return float(max(-1.0, min(1.0, 2 * t - 1)))


class HandTracker:
    def __init__(self, target="Right", mirror=True, depth=None,
                 min_det=0.6, min_track=0.5, model_path=None,
                 swap_handedness=False, xy_gain=1.6, hfov_deg=60.0):
        self.target = target
        self.mirror = mirror
        self._hfov = hfov_deg   # for the solvePnP camera matrix (approx)
        # Handedness is DECOUPLED from the position mirror. MediaPipe reports
        # handedness assuming a selfie-mirrored image; the AVFoundation raw
        # buffer is NOT mirrored, but empirically this pipeline already labels
        # the physical hand correctly WITHOUT a swap here (swapping made a
        # right hand register as "Left"). Flip swap_handedness only if your
        # camera path inverts this. Position mirroring stays on self.mirror so
        # the arm moves the same direction as the hand on screen.
        self.swap_handedness = swap_handedness
        # xy_gain amplifies hand XY so a comfortable hand motion (not reaching
        # the frame edges) can span the full [-1,1] workspace -> wider reach.
        self.xy_gain = xy_gain
        self.depth = depth or HeuristicDepth()
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path or _DEFAULT_MODEL),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_det,
            min_tracking_confidence=min_track)
        self.hands = mp_vision.HandLandmarker.create_from_options(opts)
        self._ts_ms = 0

    def _pick(self, res):
        if not res.hand_landmarks:
            return None
        worlds = res.hand_world_landmarks or res.hand_landmarks
        for lm, world, handed in zip(res.hand_landmarks, worlds, res.handedness):
            label = handed[0].category_name  # 'Left'/'Right' (image space)
            if self.swap_handedness:         # only if the camera path inverts it
                label = "Right" if label == "Left" else "Left"
            if label == self.target:
                return lm, world, handed[0].score
        return None

    def process(self, frame_bgr, intrinsics=None) -> HandState | None:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.monotonic() * 1000)        # real cadence (not a fixed 33ms)
        if ts <= self._ts_ms:                     # keep strictly increasing
            ts = self._ts_ms + 1
        self._ts_ms = ts
        res = self.hands.detect_for_video(mp_img, ts)
        picked = self._pick(res)
        if picked is None:
            return None
        lm, world, score = picked

        px = [(int(p.x * w), int(p.y * h)) for p in lm]
        W = np.array([[p.x, p.y, p.z] for p in world])  # meters

        # --- orientation: palm frame from wrist, index-MCP, pinky-MCP ---
        v1 = _unit(W[INDEX_MCP] - W[WRIST])
        v2 = W[PINKY_MCP] - W[WRIST]
        z_axis = _unit(np.cross(v1, v2))         # palm normal
        x_axis = v1
        y_axis = _unit(np.cross(z_axis, x_axis))
        x_axis = _unit(np.cross(y_axis, z_axis))
        quat = _mat_to_quat(np.column_stack([x_axis, y_axis, z_axis]))

        # --- position: INDEX FINGERTIP (EE point) image XY + size depth axis ---
        cx = float(px[INDEX_TIP][0])
        cy = float(px[INDEX_TIP][1])
        # gain centers on the frame middle, amplifies, then clips to [-1,1] so a
        # comfortable hand travel reaches the full workspace (wider effective x,y)
        xn = _clip_unit(2 * (cx / w - 0.5) * self.xy_gain)
        if self.mirror:
            xn = -xn
        yn = _clip_unit(-2 * (cy / h - 0.5) * self.xy_gain)   # up = +y
        size_px = math.hypot(px[MIDDLE_MCP][0] - px[WRIST][0],
                             px[MIDDLE_MCP][1] - px[WRIST][1])
        zn = self.depth.z_norm(size_px, frame=frame_bgr, uv=(cx, cy))

        # --- pinch, per-finger extension, openness from world coords ---
        hand_scale = np.linalg.norm(W[MIDDLE_MCP] - W[WRIST]) or 1e-3
        pinch = float(np.linalg.norm(W[THUMB_TIP] - W[INDEX_TIP]) / hand_scale)
        gripper = float(max(0.0, min(1.0, (pinch - 0.3) / 1.2)))   # legacy fallback
        # extended? [thumb, index, middle, ring, pinky]
        _tips, _pips = (4, 8, 12, 16, 20), (3, 6, 10, 14, 18)
        fingers_ext = tuple(
            np.linalg.norm(W[t] - W[WRIST]) > np.linalg.norm(W[p] - W[WRIST])
            for t, p in zip(_tips, _pips))
        openness = sum(fingers_ext[1:]) / 4.0     # 4 non-thumb fingers

        conf = score * (0.7 if getattr(self.depth, "approximate", True) else 1.0)
        return HandState(pos=(xn, yn, zn), quat=quat, gripper=gripper,
                         openness=openness, handedness=self.target,
                         confidence=float(conf), landmarks_px=px,
                         pinch=pinch, fingers_ext=fingers_ext)

    def _fingertip_cam(self, W, px, w, h):
        """Metric-ish index-fingertip position in the camera frame (meters) via
        solvePnP over the 21 landmarks (3D metric model = MediaPipe world
        landmarks, 2D = image pixels). Approx camera matrix from HFOV — fine for
        RELATIVE control (deltas), where absolute scale is absorbed by the gain."""
        f = (w / 2) / math.tan(math.radians(self._hfov) / 2)
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
        obj = np.ascontiguousarray(W, dtype=np.float64)          # 21x3 (hand frame, m)
        img = np.ascontiguousarray(px, dtype=np.float64)         # 21x2 (pixels)
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
        except Exception:
            ok = False
        if not ok:
            return (0.0, 0.0, 0.0)
        R, _ = cv2.Rodrigues(rvec)
        p = (R @ W[INDEX_TIP].reshape(3, 1) + tvec).ravel()      # fingertip in cam frame
        return (float(p[0]), float(p[1]), float(p[2]))

    def close(self):
        self.hands.close()
