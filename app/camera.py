"""Camera reader: RTSP / IP CCTV streams, video files or webcams.

Runs in its own thread, always keeps only the newest frame (so a slow analysis never builds up
a delay), and reconnects automatically if the stream drops.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time

# RTSP options are dynamically set in FrameSource._open
import cv2  # noqa: E402

log = logging.getLogger("camera")


def parse_source(source):
    """'0' -> 0 (webcam index); everything else is kept as text (RTSP url or file path)."""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    return int(s) if s.isdigit() else s


class FrameSource:
    def __init__(self, name: str, source, loop_video: bool = True, reconnect_s: float = 3.0):
        self.name = name
        self.source = parse_source(source)
        self.loop_video = loop_video
        self.reconnect_s = reconnect_s
        self.status = "connecting"          # connecting | online | offline
        self.last_frame_time = 0.0          # time.time() of the last good frame
        self._frame = None
        self._frame_id = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{name}", daemon=True)

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str) and not self.source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def read(self, last_id: int = 0, timeout: float = 1.0):
        """Wait for a frame newer than last_id. Returns (frame, frame_id) or (None, last_id)."""
        with self._cond:
            if self._frame_id == last_id:
                self._cond.wait(timeout)
            if self._frame is None or self._frame_id == last_id:
                return None, last_id
            return self._frame, self._frame_id

    def seconds_since_frame(self) -> float:
        return time.time() - self.last_frame_time if self.last_frame_time else float("inf")

    # ------------------------------------------------------------------ thread
    def _open(self):
        if isinstance(self.source, str) and self.source.lower().startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        else:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        if isinstance(self.source, int) and sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass
        return cap

    def _run(self) -> None:
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.status = "offline"
                log.warning("[%s] cannot open %s - retrying in %.0fs", self.name, self._safe_source(), self.reconnect_s)
                cap.release()
                self._stop.wait(self.reconnect_s)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_delay = 1.0 / fps if (self.is_file and 1 < fps < 120) else (1 / 25 if self.is_file else 0)
            log.info("[%s] connected to %s", self.name, self._safe_source())
            failures = 0
            while not self._stop.is_set():
                t0 = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.is_file and self.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        failures += 1
                        if failures > 3:
                            break
                        continue
                    if self.is_file:
                        self.status = "offline"
                        log.info("[%s] video finished", self.name)
                        self._stop.wait(1.0)
                        break
                    failures += 1
                    if failures > 5:
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                self.status = "online"
                self.last_frame_time = time.time()
                with self._cond:
                    self._frame = frame
                    self._frame_id += 1
                    self._cond.notify_all()
                if frame_delay:
                    left = frame_delay - (time.time() - t0)
                    if left > 0:
                        time.sleep(left)
            cap.release()
            if self.is_file and not self.loop_video:
                # keep the thread alive but idle so the last frame stays visible
                self._stop.wait(3600)
                return
            self.status = "offline"
            log.warning("[%s] stream lost - reconnecting", self.name)
            self._stop.wait(self.reconnect_s)

    def _safe_source(self) -> str:
        """Never print camera passwords in logs."""
        s = str(self.source)
        if "@" in s and "://" in s:
            scheme, rest = s.split("://", 1)
            return f"{scheme}://***@{rest.split('@', 1)[1]}"
        return s
