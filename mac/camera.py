"""
CameraSource — camera input, selected by DEVICE IDENTITY (not a fragile index).

Why not OpenCV for selection: cv2.VideoCapture takes only an integer index and
exposes NO device name / uniqueID / serial, and its index order does NOT match
AVFoundation's (empirically verified) and shifts between launches. So OpenCV
alone cannot deterministically pick "the built-in camera".

Solution: select the AVCaptureDevice by its stable uniqueID / type via
AVFoundation and capture from it directly (AVFCamera). No index guessing, no
image-content heuristics — pure hardware identity. AVCaptureSession is
non-exclusive (other apps can capture the same device), and we don't reconfigure
the device format. OpenCVCamera remains for explicit indices, files and RTSP URLs.
"""
from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

# ---- AVFoundation (identity-based capture) ----
try:
    import objc  # noqa: F401
    import AVFoundation as AVF
    from Foundation import NSObject
    from libdispatch import dispatch_queue_create
    from CoreMedia import CMSampleBufferGetImageBuffer
    from Quartz import (CVPixelBufferLockBaseAddress, CVPixelBufferUnlockBaseAddress,
                        CVPixelBufferGetBaseAddress, CVPixelBufferGetWidth,
                        CVPixelBufferGetHeight, CVPixelBufferGetBytesPerRow,
                        kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_32BGRA)
    _AVF_OK = True
except Exception as _e:   # pragma: no cover
    _AVF_OK = False
    _AVF_ERR = _e


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    approximate: bool = True

    @classmethod
    def from_fov(cls, width, height, hfov_deg=60.0):
        fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        return cls(fx=fx, fy=fx, cx=width / 2, cy=height / 2,
                   width=width, height=height, approximate=True)

    def backproject(self, u, v, z):
        return ((u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z)


class CameraSource(ABC):
    intrinsics: Intrinsics

    @abstractmethod
    def read(self):
        """Return (ts_ns, frame_bgr) or (None, None)."""

    @abstractmethod
    def release(self):
        ...


# ---------------------------------------------------------------------------
# AVFoundation identity-based camera
# ---------------------------------------------------------------------------
def _sbuf_to_np(sbuf):
    pb = CMSampleBufferGetImageBuffer(sbuf)
    if pb is None:
        return None
    CVPixelBufferLockBaseAddress(pb, 0)
    try:
        w = CVPixelBufferGetWidth(pb)
        h = CVPixelBufferGetHeight(pb)
        bpr = CVPixelBufferGetBytesPerRow(pb)
        base = CVPixelBufferGetBaseAddress(pb)
        buf = base.as_buffer(bpr * h)
        # 32BGRA -> take BGR
        return np.frombuffer(buf, np.uint8).reshape(h, bpr // 4, 4)[:, :w, :3].copy()
    finally:
        CVPixelBufferUnlockBaseAddress(pb, 0)


if _AVF_OK:
    class _FrameDelegate(NSObject):
        def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sbuf, conn):
            cam = getattr(self, "_camera", None)
            if cam is None:
                return
            try:
                arr = _sbuf_to_np(sbuf)
            except Exception:
                arr = None
            if arr is not None:
                cam._push(arr)


class AVFCamera(CameraSource):
    """Streams from a specific AVCaptureDevice (chosen by identity). Latest-frame
    with drop-late, so read() is always the freshest frame (low latency)."""

    def __init__(self, device, hfov_deg=60.0):
        self._lock = threading.Lock()
        self._latest = None
        self._ts = 0
        self._session = AVF.AVCaptureSession.alloc().init()
        inp = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)[0]
        if inp is None or not self._session.canAddInput_(inp):
            raise RuntimeError(f"cannot add input for {device.localizedName()}")
        self._session.addInput_(inp)
        out = AVF.AVCaptureVideoDataOutput.alloc().init()
        out.setAlwaysDiscardsLateVideoFrames_(True)
        out.setVideoSettings_({kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA})
        self._deleg = _FrameDelegate.alloc().init()
        self._deleg._camera = self
        out.setSampleBufferDelegate_queue_(self._deleg, dispatch_queue_create(b"vt.avfcam", None))
        if not self._session.canAddOutput_(out):
            raise RuntimeError("cannot add video output")
        self._session.addOutput_(out)
        self._session.startRunning()
        t0 = time.time()
        while self._latest is None and time.time() - t0 < 5.0:
            time.sleep(0.03)
        if self._latest is None:
            self._session.stopRunning()
            raise RuntimeError("AVFCamera got no frames (camera permission or device busy?)")
        h, w = self._latest.shape[:2]
        self.intrinsics = Intrinsics.from_fov(w, h, hfov_deg)

    def _push(self, arr):
        with self._lock:
            self._latest = arr
            self._ts = time.monotonic_ns()

    def read(self):
        with self._lock:
            if self._latest is None:
                return None, None
            return self._ts, self._latest   # delegate replaces (never mutates) -> safe

    def release(self):
        try:
            self._session.stopRunning()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OpenCV camera (explicit index / file / RTSP only)
# ---------------------------------------------------------------------------
class OpenCVCamera(CameraSource):
    def __init__(self, spec, width=1280, height=720, fps=60, hfov_deg=60.0,
                 intrinsics: Intrinsics | None = None, force_res=False):
        self.cap = None
        for _ in range(4):
            cap = cv2.VideoCapture(spec)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
            time.sleep(0.3)
        if self.cap is None:
            raise RuntimeError(f"cannot open camera {spec!r} after retries")
        if isinstance(spec, int) or (isinstance(spec, str) and str(spec).isdigit()):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if force_res:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.intrinsics = intrinsics or Intrinsics.from_fov(w, h, hfov_deg)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        return time.monotonic_ns(), frame

    def release(self):
        self.cap.release()


# ---------------------------------------------------------------------------
# Device enumeration & identity selection (AVFoundation)
# ---------------------------------------------------------------------------
def list_cameras():
    """[(name, uniqueID, type, is_builtin)] — identity, no index (index is
    meaningless because OpenCV/AVFoundation orders differ)."""
    if not _AVF_OK:
        return []
    out = []
    for d in AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo):
        t = d.deviceType()
        out.append((str(d.localizedName()), str(d.uniqueID()), t.split("Type")[-1],
                    t == AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera))
    return out


def _avf_select(spec):
    """Return an AVCaptureDevice by identity: 'builtin' (type), a uniqueID, or a
    name substring. Raises if not found (never silently picks the wrong one)."""
    devs = list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo))
    if spec == "builtin":
        for d in devs:
            if d.deviceType() == AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera:
                return d
        raise RuntimeError("no built-in camera (BuiltInWideAngleCamera) found")
    d = AVF.AVCaptureDevice.deviceWithUniqueID_(spec)      # exact serial/uniqueID
    if d is not None:
        return d
    for d in devs:                                          # name substring
        if str(spec).lower() in str(d.localizedName()).lower():
            return d
    raise RuntimeError(f"no camera matches {spec!r}; have: "
                       + ", ".join(f"{n} [{u}]" for n, u, *_ in list_cameras()))


def open_camera(spec="builtin", hfov_deg=60.0, **kw) -> CameraSource:
    """spec: 'builtin' | a uniqueID | a name substring  -> AVFoundation identity
    capture (deterministic). An int/digit index, a file path or rtsp:// URL ->
    OpenCV. Selection is one-time at startup (zero per-frame latency)."""
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return OpenCVCamera(int(spec), hfov_deg=hfov_deg, **kw)
    if isinstance(spec, str) and ("://" in spec or "/" in spec
                                  or spec.lower().endswith((".mp4", ".mov", ".avi"))):
        return OpenCVCamera(spec, hfov_deg=hfov_deg, **kw)
    if not _AVF_OK:
        raise RuntimeError(f"AVFoundation unavailable ({_AVF_ERR}); pass a numeric index")
    dev = _avf_select(spec)
    print(f"[camera] identity-select: '{dev.localizedName()}' "
          f"uid={dev.uniqueID()} type={dev.deviceType().split('Type')[-1]}")
    return AVFCamera(dev, hfov_deg=hfov_deg)
