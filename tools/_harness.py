"""Shared helpers for the tools: run the REAL recognition pipeline offline (no cameras, no MongoDB server, no printer).

The pipeline code is exactly the one used by run.bat; only the camera is replaced by a list of frames and the clock
by a virtual clock, so a recording is replayed fast and gives the same result every time.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402,F401  (makes third_party importable)
from app.config import load_config  # noqa: E402
from app.main import build_plate_format  # noqa: E402
from app.pipeline import CameraWorker  # noqa: E402
from app.recognizer import build_recognizers  # noqa: E402
from app.slips import SlipEngine  # noqa: E402
from app.store import Store  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}


class _NullSource:
    status = "online"
    last_frame_time = 0.0

    def seconds_since_frame(self) -> float:
        return 0.0


@dataclass
class ClipResult:
    name: str
    slips: list[dict] = field(default_factory=list)       # entries created (kind == slip)
    whitelisted: list[dict] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    frames: int = 0
    analysed_ms: float = 0.0                               # average milliseconds per analysed frame
    data_dir: str = ""


class Harness:
    def __init__(self, config_path: str | None = None, camera: str = "camera1", device: str | None = None):
        self.cfg = load_config(config_path)
        if device:
            self.cfg["models"]["device"] = device
        self.camera = camera
        self.fmt = build_plate_format(self.cfg)
        self.rec = build_recognizers(self.cfg["models"], self.cfg["vehicles"].get("confidence", 0.45))

    def run_clip(self, name: str, frames, fps: float, must_move: bool | None = None, tail_s: float = 12.0,
                 keep_dir: Path | None = None) -> ClipResult:
        """frames: iterable of BGR images. Returns what the system would have done (slips, alerts)."""
        import copy
        import mongomock
        cfg = copy.deepcopy(self.cfg)
        tmp = Path(tempfile.mkdtemp(prefix="eval_"))
        (tmp / "snapshots").mkdir()
        (tmp / "slips").mkdir()
        cfg["storage"]["data_dir"] = str(tmp)
        cfg["mongodb"]["database"] = "eval"
        if must_move is not None:
            cfg["vehicles"]["must_be_moving"] = must_move
        store = Store(cfg, client=mongomock.MongoClient(tz_aware=True))
        store.start()
        engine = SlipEngine(store, None, cfg["matching"])
        worker = CameraWorker(self.camera, _NullSource(), self.rec, engine, store, self.fmt, cfg)
        n, t, last = 0, 0.0, None
        import time
        t_work, n_heavy = 0.0, 0
        for fr in frames:
            t = n / fps
            before = worker._last_heavy_run
            t0 = time.perf_counter()
            worker.process_frame(fr, t)
            if worker._last_heavy_run != before:
                t_work += time.perf_counter() - t0
                n_heavy += 1
            last, n = fr, n + 1
        worker._finish_lost(t + tail_s)
        worker.stop()
        store.stop()
        res = ClipResult(name=name, frames=n, data_dir=str(tmp))
        res.analysed_ms = 1000.0 * t_work / max(1, n_heavy)
        for e in store.db["entries"].find():
            (res.slips if e["kind"] == "slip" else res.whitelisted).append(e)
        res.alerts = list(store.alerts)
        if keep_dir is not None:
            keep_dir.mkdir(parents=True, exist_ok=True)
            for src in (tmp / "snapshots").rglob("*.jpg"):
                shutil.copy(src, keep_dir / src.name)
        shutil.rmtree(tmp, ignore_errors=True)
        return res

    def run_image(self, path: Path, repeat: int = 6, **kw) -> ClipResult:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        # a still picture is replayed as a short clip so the multi-frame logic is exercised too
        return self.run_clip(path.name, (img for _ in range(repeat)), fps=6.0, must_move=False, **kw)

    def run_video(self, path: Path, stride: int = 1, max_frames: int = 0, **kw) -> ClipResult:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        def gen():
            i = 0
            while True:
                ok, f = cap.read()
                if not ok or (max_frames and i >= max_frames):
                    break
                if i % stride == 0:
                    yield f
                i += 1
            cap.release()
        return self.run_clip(path.name, gen(), fps=fps / stride, **kw)
