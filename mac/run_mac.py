"""
Mac operator loop: camera -> hand tracking -> gesture -> retarget -> UDP TX,
with a live GUI viewfinder.

THREADING (so the GUI adds ZERO latency to the arm):
  * CONTROL thread runs capture -> track -> retarget -> tx.send() at the camera's
    full rate, never blocked by the display. This is the path to the robot.
  * MAIN thread only draws the latest frame + landmarks + HAND status and calls
    cv2.imshow/waitKey (Cocoa GUI must be on the main thread). It reads a shared
    snapshot; it never gates the control loop.

  python run_mac.py --host 192.168.3.22 --camera builtin            # GUI
  python run_mac.py --host 192.168.3.22 --camera builtin --headless # no GUI
  python run_mac.py --selftest --camera builtin                     # no network
"""
from __future__ import annotations

import argparse
import threading
import time

import cv2

from camera import open_camera
from perception import HandTracker
from gestures import CalibrationFSM
from retarget import Retargeter
from eeframe import EEFrame, FLAG_VALID


class Shared:
    """Latest snapshot handed from the control thread to the GUI thread."""
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.hs = None
        self.prog = 0.0
        self.status = "no hand"
        self.rate = 0.0
        self.frames = 0
        self.hands = 0
        self.depth = None      # colourised depth map (BGR), from the depth thread
        self.stop = False

    def set_depth(self, d):
        with self.lock:
            self.depth = d

    def get_depth(self):
        with self.lock:
            return self.depth

    def publish(self, frame, hs, prog, status, rate, frames, hands):
        with self.lock:
            self.frame = frame
            self.hs = hs
            self.prog = prog
            self.status = status
            self.rate = rate
            self.frames = frames
            self.hands = hands

    def snapshot(self):
        with self.lock:
            return (self.frame, self.hs, self.prog, self.status, self.rate,
                    self.frames, self.hands)


def control_loop(cam, static_img, tracker, fist, retarget, tx, shared, duration):
    t0 = time.perf_counter()
    frames = hands = sent = 0
    last_log = t0
    rate = 0.0
    lost = 0
    LOST_RESET = 20      # ~0.25s of no hand -> require a fresh calibration clutch
    while not shared.stop:
        if static_img is not None:
            ts, frame = time.monotonic_ns(), static_img
        else:
            ts, frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue
        now = time.perf_counter()
        hs = tracker.process(frame)
        frames += 1
        prog = 0.0
        status = "no hand"
        if hs:
            lost = 0
            hands += 1
            status, prog, just = fist.update(hs, now)
            g = fist.gripper_value(hs.pinch)     # operator-calibrated 0..1
            if tx:
                tx.send(retarget.relative_eeframe(hs, now, tracking=fist.tracking,
                                                  just_calibrated=just, gripper=g))
                sent += 1
        else:
            lost += 1
            if lost == LOST_RESET:        # hand gone -> drop tracking + clear origin
                fist.reset()
                retarget.notify_hand_lost()
            if tx:
                tx.send(EEFrame(flags=FLAG_VALID))   # hold (deadman off)
                sent += 1
        if now - last_log >= 1.0:
            rate = frames / (now - t0)
            last_log = now
        shared.publish(frame, hs, prog, status, rate, frames, hands)
        if duration and (now - t0) >= duration:
            break
    shared.stop = True
    print(f"control loop done: frames={frames} hands={hands} sent={sent}")


def depth_loop(depther, shared, hz=7.0):
    """VIEWER-ONLY monocular depth on its own thread (never touches control).
    Rate-capped so it can't pin CPU cores / cause thermal throttling that would
    slow the control loop over a long session (review S1)."""
    period = 1.0 / hz
    while not shared.stop:
        t0 = time.perf_counter()
        frame, *_ = shared.snapshot()
        if frame is None:
            time.sleep(0.05)
            continue
        try:
            shared.set_depth(depther.infer(frame, 480, 480))
        except Exception as e:
            print("depth thread stopped:", e)
            break
        dt = period - (time.perf_counter() - t0)
        if dt > 0:
            time.sleep(dt)


def draw(frame, hs, prog, status, rate, cam_name):
    col = (0, 255, 0) if hs else (0, 0, 255)
    cv2.putText(frame, f"{cam_name}   HAND: {'YES' if hs else 'NO'}   {rate:.0f}Hz",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    if hs:
        pts = hs.landmarks_px
        for i, (x, y) in enumerate(pts):
            # highlight the index fingertip (EE reference point) in magenta
            c = (255, 0, 255) if i == 8 else (0, 255, 0)
            cv2.circle(frame, (x, y), 6 if i == 8 else 4, c, -1)
        for a, b in [(0, 5), (5, 8), (0, 9), (9, 12), (0, 17), (0, 4), (4, 8)]:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
        cv2.putText(frame, f"grip={hs.gripper:.2f}  open={hs.openness:.2f}  "
                    f"tip=({hs.pos[0]:+.2f},{hs.pos[1]:+.2f},{hs.pos[2]:+.2f})",
                    (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "show your RIGHT hand to the camera", (12, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # calibration state banner
    if status == "TRACKING":
        cv2.putText(frame, "TRACKING (index fingertip -> arm)", (12, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elif status == "STEP2":
        cv2.putText(frame, f"STEP2: PINCH thumb+index  {prog*100:.0f}%", (12, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    elif prog > 0:
        cv2.putText(frame, f"STEP1: index+thumb OPEN, fold 3 fingers  {prog*100:.0f}%",
                    (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    elif hs:
        cv2.putText(frame, "STEP1: index+thumb OPEN, fold other 3 fingers (3s)",
                    (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.3.22")
    ap.add_argument("--camera", default="builtin",
                    help="'builtin' | a camera name | a uniqueID | an index")
    ap.add_argument("--target", default="Right")
    ap.add_argument("--mirror", type=int, default=1)
    ap.add_argument("--swap-handedness", type=int, default=0,
                    help="1 = invert MediaPipe L/R labels (flip if right hand reads as left)")
    ap.add_argument("--xy-gain", type=float, default=1.6,
                    help="amplify hand XY so comfortable motion reaches full workspace")
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--depth", type=int, default=1,
                    help="1 = show the monocular depth pane (viewer-only, own thread)")
    ap.add_argument("--selftest", action="store_true", help="no network")
    ap.add_argument("--image", default="", help="feed a static image instead of a camera")
    a = ap.parse_args()

    static_img = None
    cam = None
    cam_name = str(a.camera)
    if a.image:
        static_img = cv2.imread(a.image)
        if static_img is None:
            print(f"cannot read image {a.image}"); return
        cam_name = f"image:{a.image}"
    else:
        cam = open_camera(a.camera, hfov_deg=a.hfov)

    tracker = HandTracker(target=a.target, mirror=bool(a.mirror),
                          swap_handedness=bool(a.swap_handedness),
                          xy_gain=a.xy_gain)
    fist = CalibrationFSM()
    retarget = Retargeter()
    tx = None
    if not a.selftest:
        from transport_tx import ControlSender
        tx = ControlSender(a.host)

    shared = Shared()
    args = (cam, static_img, tracker, fist, retarget, tx, shared, a.duration)
    print(f"source={cam_name} host={a.host} "
          f"mode={'selftest' if a.selftest else 'live-tx'} "
          f"gui={'off' if a.headless else 'on'}")

    try:
        if a.headless:
            control_loop(*args)                 # no GUI: run control in this thread
        else:
            th = threading.Thread(target=control_loop, args=args, daemon=True)
            th.start()
            # optional depth pane on its OWN thread (viewer-only, zero control cost)
            if a.depth:
                try:
                    from depth_model import DepthAnything
                    dth = threading.Thread(target=depth_loop,
                                           args=(DepthAnything(), shared), daemon=True)
                    dth.start()
                except Exception as e:
                    print("depth pane disabled:", e)
            # MAIN thread = GUI only (never gates control -> zero control latency)
            DH = 540
            while not shared.stop:
                frame, hs, prog, status, rate, _f, _h = shared.snapshot()
                if frame is not None:
                    disp = frame.copy()          # copy so control thread is untouched
                    draw(disp, hs, prog, status, rate, cam_name)
                    h0, w0 = disp.shape[:2]
                    dw = int(w0 * DH / h0)
                    disp = cv2.resize(disp, (dw, DH))
                    d = shared.get_depth()
                    if d is not None:
                        dd = cv2.resize(d, (dw, DH))
                        cv2.putText(dd, "DEPTH (monocular)", (12, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        disp = cv2.hconcat([disp, dd])
                    cv2.imshow("vision-teleop", disp)
                if (cv2.waitKey(15) & 0xFF) == 27:   # ~66 Hz GUI cap, ESC to quit
                    shared.stop = True
            th.join(timeout=2)
    finally:
        shared.stop = True
        if cam is not None:
            cam.release()
        tracker.close()
        if tx:
            tx.close()
        if not a.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
