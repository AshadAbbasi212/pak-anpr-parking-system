"""One CameraWorker per camera.

loop:  newest frame -> motion check (cheap) -> [only if something moves] vehicle + plate detectors ->
       tracker (one track per vehicle) -> read plate text on each new vehicle -> vote over several frames ->
       SlipEngine decides (slip / already slipped / check / unreadable / whitelisted) -> coloured boxes.
"""
from __future__ import annotations

import logging
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .camera import FrameSource
from .motion import MotionDetector
from .pk_formats import decode_free, decode_templates
from .plate_reader import fuse, read_plate
from .plates import PlateFormat, display, vote, weighted_distance
from .recognizer import Recognizers
from .slips import SlipEngine
from .store import Store
from .tracker import Box, Obj, Track, VehicleTracker, center, contains, expand, iou

log = logging.getLogger("pipeline")

COLORS = {            # BGR
    "green": (0, 170, 0), "blue": (220, 120, 0), "gray": (150, 150, 150),
    "red": (0, 0, 230), "yellow": (0, 200, 255),
}
WHITE = (255, 255, 255)
STREAM_MAX_WIDTH = 960
LOOP_HZ = 12.0
LOW_LIGHT_THRESHOLD = 70       # avg luminance below this = night / very dim scene
PLATE_UPSCALE_MIN_W = 100      # plates narrower than this are upscaled before OCR


# ------------------------------------------------------------------ preprocessing helpers

def _clahe_enhance(crop: np.ndarray, clip: float = 2.5) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalise lighting."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _sharpen(crop: np.ndarray) -> np.ndarray:
    """Unsharp-mask sharpening to bring out character edges."""
    blur = cv2.GaussianBlur(crop, (0, 0), 2.0)
    return cv2.addWeighted(crop, 1.5, blur, -0.5, 0)


def _denoise(crop: np.ndarray) -> np.ndarray:
    """Light bilateral filter that reduces noise while keeping character edges."""
    return cv2.bilateralFilter(crop, d=5, sigmaColor=50, sigmaSpace=50)


def _upscale_if_small(crop: np.ndarray, min_width: int = PLATE_UPSCALE_MIN_W) -> np.ndarray:
    """Upscale small plate crops so the OCR model has more pixels to work with."""
    h, w = crop.shape[:2]
    if 0 < w < min_width and h > 0:
        scale = min_width / w
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def _scene_brightness(frame: np.ndarray) -> float:
    """Average luminance (0-255) of a downscaled frame."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def _find_split_row(crop: np.ndarray) -> int | None:
    """Find the horizontal dividing line in a two-row plate using Sobel edge detection.
    Returns the y-coordinate of the split, or None if no clear line is found."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ph = gray.shape[0]
    y_min, y_max = int(ph * 0.30), int(ph * 0.70)
    if y_max <= y_min:
        return None
    sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    row_energy = sobel[y_min:y_max, :].sum(axis=1)
    if row_energy.max() < row_energy.mean() * 1.5:
        return None            # no clear horizontal dividing line
    return y_min + int(np.argmax(row_energy))



def parse_roi(points) -> list[tuple[float, float]]:
    out = []
    for p in points or []:
        try:
            out.append((float(p[0]), float(p[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out if len(out) >= 3 else []


class CameraWorker(threading.Thread):
    def __init__(self, name: str, source: FrameSource, rec: Recognizers, engine: SlipEngine, store: Store,
                 fmt: PlateFormat, cfg: dict, roi=None):
        super().__init__(name=f"worker-{name}", daemon=True)
        self.cam = name
        self.source = source
        self.rec = rec
        self.engine = engine
        self.store = store
        self.fmt = fmt
        r, v, mo = cfg["recognition"], cfg.get("vehicles", {}), cfg.get("motion", {})
        self.snap_dir = Path(cfg["storage"]["data_dir"]) / "snapshots"
        self.jpeg_quality = int(cfg["storage"].get("jpeg_quality", 85))
        self.offline_after = float(cfg["alerts"].get("camera_offline_seconds", 20))

        self.min_reads = int(r.get("min_reads", 2))
        self.min_reads_on_loss = int(r.get("min_reads_on_loss", 1))
        self.max_reads = int(r.get("max_reads", 10))
        self.min_conf = float(r.get("min_char_confidence", 0.5))
        self.min_agreement = float(r.get("min_agreement", 0.6))
        self.min_plate_w = int(r.get("min_plate_width_px", 36))
        oc = cfg.get("ocr", {})
        self.ocr_temp = float(oc.get("temperature", 2.0))          # >1 = trust the OCR model a bit less
        self.min_quality = float(oc.get("min_quality", 0.30))      # minimum quality (0-1) of a plate reading
        self.min_margin = float(oc.get("min_margin", 0.5))         # best reading must beat the runner-up by this
        self.max_evidence = int(oc.get("max_evidence", 40))
        self.fast_accept = float(oc.get("fast_accept_quality", 0.8))   # clear plates are read once (0 = always all variants)
        self.merged_weight = float(oc.get("merged_weight", 1.0))   # weight of the "two rows written as one line" reading
        self.strip_two_row_year = bool(oc.get("strip_two_row_year", True))
        self.unusual_min_conf = float(oc.get("unusual_min_confidence", 0.55))
        self.allow_unusual = bool(cfg.get("validation", {}).get("allow_unusual", True)) and fmt.allow_unusual
        self.min_len = fmt.min_length                               # validation.min_length
        self.unreadable_after = float(r.get("unreadable_after_s", 6))
        self.heavy_interval = 1.0 / max(0.5, float(r.get("process_fps", 6)))
        self.base_missing_s = float(r.get("track_max_missing_s", 1.2))
        self.tracker = VehicleTracker(max_missing_s=self.base_missing_s)
        self._heavy_dt = 0.0          # measured time between analysed frames (grows on slow computers)
        self._last_heavy_t = 0.0
        self._last_heavy_run = -1e9   # clock value of the last analysed frame

        self.vehicle_classes = set(v.get("classes", ["car", "motorcycle", "bus", "truck"]))
        self.min_vehicle_w = int(v.get("min_vehicle_width_px", 50))
        self.vehicle_every = max(1, int(v.get("every_n_frames", 2)))
        self.must_move = bool(v.get("must_be_moving", True))
        self.bike_plate_conf = float(v.get("bike_plate_confidence", 0.25))     # 2nd-stage plate search on motorcycles
        self.second_stage_conf = float(v.get("second_stage_confidence", 0.30))  # ... on other vehicles
        self.second_stage_max = int(v.get("second_stage_max_vehicles", 4))      # vehicles searched per analysed frame

        self.motion = MotionDetector()
        self.motion_enabled = bool(mo.get("enabled", True))
        self.gate_fraction = float(mo.get("gate_fraction", 0.004))
        self.hold_s = float(mo.get("hold_seconds", 2.0))
        self.moving_fraction = float(mo.get("moving_fraction", 0.03))

        self.roi = parse_roi(roi)
        self._roi_mask: np.ndarray | None = None
        self._roi_size: tuple[int, int] | None = None

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"veh-{name}")
        self._stop_evt = threading.Event()
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0
        self._jpeg_cond = threading.Condition()
        self.fps = 0.0
        self.last_plate = ""
        self.analysing = False
        self._offline_alerted = False
        self._t_created = time.time()
        self._last_motion = 0.0
        self._heavy_n = 0
        self._ghosts: list[tuple[Track, float]] = []     # finished vehicles still drawn for a moment
        self._done_boxes: list[tuple[Box, float, str]] = []   # where vehicles with a decision were, with plate text
        self._last_frame: np.ndarray | None = None
        self._brightness: float = 128.0                  # running average scene brightness (0-255)

    # ------------------------------------------------------------------ public
    def stop(self) -> None:
        self._stop_evt.set()
        with self._jpeg_cond:
            self._jpeg_cond.notify_all()
        self._pool.shutdown(wait=False)

    def status(self) -> dict:
        return {"camera": self.cam, "status": self.source.status, "fps": round(self.fps, 1),
                "last_plate": self.last_plate, "analysing": self.analysing}

    def get_jpeg(self, last_seq: int, timeout: float = 2.0):
        with self._jpeg_cond:
            if self._jpeg_seq == last_seq:
                self._jpeg_cond.wait(timeout)
            return self._jpeg, self._jpeg_seq

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        last_id = 0
        last_placeholder = 0.0
        last_publish = 0.0
        period = 1.0 / LOOP_HZ
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            frame, last_id = self.source.read(last_id, timeout=1.0)
            if frame is None:
                self._handle_no_signal()
                if time.time() - last_placeholder > 1.0:
                    last_placeholder = time.time()
                    self._publish(self._placeholder())
                continue
            self._offline_alerted = False
            try:
                now = time.monotonic()
                self.process_frame(frame, now)
                if time.monotonic() - last_publish >= 0.1:
                    last_publish = time.monotonic()
                    self._publish(self._annotate(frame, now))
            except Exception:  # noqa: BLE001 - one bad frame must never stop the camera
                log.exception("[%s] error while processing a frame", self.cam)
            left = period - (time.monotonic() - t0)
            if left > 0:
                time.sleep(left)

    def process_frame(self, frame: np.ndarray, now: float) -> None:
        """Everything that happens for one camera frame (motion gate -> detection -> reading -> decisions).

        `now` is a clock in seconds. The live program passes time.monotonic(); tools/evaluate.py passes a
        virtual clock (frame number / video fps) so recordings can be replayed fast and repeatably."""
        self._last_frame = frame
        frac = self.motion.update(frame, self._roi_mask_for(frame))
        if frac >= self.gate_fraction:
            self._last_motion = now
        active = (not self.motion_enabled or bool(self.tracker.tracks)
                  or now - self._last_motion <= self.hold_s)
        self.analysing = active
        if active and now - self._last_heavy_run >= self.heavy_interval:
            self._last_heavy_run = now
            t0 = time.monotonic()
            self._heavy(frame, now)
            dt = time.monotonic() - t0
            self.fps = 0.8 * self.fps + 0.2 * (1.0 / max(dt, 1e-3)) if self.fps else 1.0 / max(dt, 1e-3)
        elif not active:
            self.fps = 0.0
            self._finish_lost(now)       # nothing moves: close any vehicle that has been gone for a while

    # ------------------------------------------------------------------ region of interest
    def _roi_mask_for(self, frame: np.ndarray) -> np.ndarray | None:
        if not self.roi:
            return None
        h, w = frame.shape[:2]
        if self._roi_size != (w, h):
            m = np.zeros((h, w), np.uint8)
            pts = np.array([[int(x * w), int(y * h)] for x, y in self.roi], np.int32)
            cv2.fillPoly(m, [pts], 1)
            self._roi_mask, self._roi_size = m, (w, h)
        return self._roi_mask

    def _in_roi(self, frame: np.ndarray, point: tuple[float, float]) -> bool:
        mask = self._roi_mask_for(frame)
        if mask is None:
            return True
        h, w = mask.shape
        x, y = int(point[0]), int(point[1])
        return 0 <= x < w and 0 <= y < h and bool(mask[y, x])

    # ------------------------------------------------------------------ analysis
    @staticmethod
    def _conf(ocr_result) -> float:
        c = ocr_result.confidence if ocr_result else 0.0
        if isinstance(c, list):
            return float(statistics.mean(c)) if c else 0.0
        return float(c or 0.0)

    def _detect(self, frame: np.ndarray):
        """Vehicle detector runs in a helper thread on CPU; on GPU it runs sequentially to protect DirectML."""
        run_vehicle = (self._heavy_n % self.vehicle_every == 0
                       or any(not t.has_vehicle and not t.finalized for t in self.tracker.tracks))
        self._heavy_n += 1
        if getattr(self.rec, "is_gpu", False):
            vehicles = self.rec.vehicles.predict(frame) if run_vehicle else []
            plates = self.rec.plates.predict(frame)
        else:
            fut = self._pool.submit(self.rec.vehicles.predict, frame) if run_vehicle else None
            plates = self.rec.plates.predict(frame)
            vehicles = fut.result() if fut is not None else []
        return vehicles, plates, run_vehicle

    def _heavy(self, frame: np.ndarray, now: float) -> None:
        h, w = frame.shape[:2]
        # On a slow computer analysed frames are far apart; be patient so one vehicle is not split into many.
        if self._last_heavy_t and now - self._last_heavy_t < 10:
            gap = now - self._last_heavy_t
            self._heavy_dt = gap if not self._heavy_dt else 0.7 * self._heavy_dt + 0.3 * gap
            self.tracker.max_missing_s = max(self.base_missing_s, 2.5 * self._heavy_dt)
        self._last_heavy_t = now
        # Update running scene brightness for night-time adaptation
        br = _scene_brightness(frame)
        self._brightness = 0.9 * self._brightness + 0.1 * br
        vehicles_raw, plates_raw, _ = self._detect(frame)

        vehicles: list[Obj] = []
        for d in vehicles_raw:
            if d.label not in self.vehicle_classes:
                continue
            b = d.bounding_box
            box = (max(0, int(b.x1)), max(0, int(b.y1)), min(w, int(b.x2)), min(h, int(b.y2)))
            if box[2] - box[0] < self.min_vehicle_w or box[3] <= box[1]:
                continue
            if not self._in_roi(frame, ((box[0] + box[2]) / 2, box[3] - 1)):    # where the tyres touch the road
                continue
            vehicles.append(Obj(box=box, cls=d.label, conf=float(d.confidence)))
        vehicles = self._drop_nested(vehicles)

        plate_only: list[Obj] = []
        for d in plates_raw:
            b = d.bounding_box
            pb = (max(0, int(b.x1)), max(0, int(b.y1)), min(w, int(b.x2)), min(h, int(b.y2)))
            if pb[2] - pb[0] < self.min_plate_w or pb[3] <= pb[1]:
                continue
            margin = max(4, int(0.004 * w))
            if pb[0] <= margin or pb[1] <= margin or pb[2] >= w - margin or pb[3] >= h - margin:
                continue                           # plate cut off by the picture edge: wait until it is fully visible
            if not self._in_roi(frame, center(pb)):
                continue
            pc = center(pb)
            # Plates are mounted within vehicle width. Require strict containment (pad=0) first,
            # then allow only tiny horizontal slack (0.02) and vertical slack (0.15 for bikes, 0.05 for cars/trucks).
            strict_owners = [v for v in vehicles if contains(v.box, pc, pad_x=0.0, pad_y=0.0)]
            owners = strict_owners if strict_owners else [
                v for v in vehicles if contains(v.box, pc, pad_x=0.02, pad_y=0.15 if v.cls == "motorcycle" else 0.05)
            ]
            if owners:
                owner = min(owners, key=lambda v: (v.box[2] - v.box[0]) * (v.box[3] - v.box[1]))
                if owner.plate_box is None or float(d.confidence) > owner.plate_conf:
                    owner.plate_box, owner.plate_conf = pb, float(d.confidence)
            else:
                # Do not spawn rogue plate-only objects if an existing vehicle track already encloses this plate
                if not any(contains(t.box, pc, pad=0.08) for t in self.tracker.tracks):
                    plate_only.append(Obj(box=expand(pb, 1.0, 2.0, w, h), cls=None, conf=float(d.confidence),
                                          plate_box=pb, plate_conf=float(d.confidence)))

        # Second-stage zoomed plate search INSIDE each vehicle that has no plate yet.
        # Motorcycle plates are small, low on the bike and often missed by the full-frame detector, so for bikes
        # we search a slightly enlarged box (rider / mudguard) with a lower confidence limit, and reject
        # candidates whose shape or position cannot be a plate (headlight, mirror, number on the rider's shirt).
        # Cost control: all motorcycles (up to 4) plus the largest MOVING vehicles, at most `second_stage_max` in total.
        todo = [v for v in vehicles if v.plate_box is None
                and (not self.must_move or self.motion.box_fraction(v.box) >= self.moving_fraction)]
        todo.sort(key=lambda v: (v.cls != "motorcycle", -(v.box[2] - v.box[0]) * (v.box[3] - v.box[1])))
        for v in todo[:max(self.second_stage_max, 1)]:
            self._second_stage_plate(frame, v)

        objs = vehicles + plate_only
        aligned, lost = self.tracker.update(objs, now)

        for track, obj in zip(aligned, objs):
            if obj.cls is None:
                track.full_view = True                          # a fully visible plate means a fully visible vehicle
            else:
                m = max(4, int(0.004 * w))
                cut = obj.box[0] <= m or obj.box[2] >= w - m
                if not cut or (obj.box[2] - obj.box[0]) >= 0.25 * w:
                    track.full_view = True                      # (a huge vehicle filling the picture also counts)
            probe = obj.box if obj.cls is not None else obj.plate_box or obj.box
            if self.motion.box_fraction(probe) >= self.moving_fraction:
                track.moving_frames += 1
            if track.finalized or obj.plate_box is None:
                self._maybe_unreadable_live(track, now)
                continue
            if self.must_move and track.moving_frames == 0 and track.frames > 3:
                continue                       # standing vehicle: don't waste CPU reading it
            self._read_plate(track, obj, frame)
            self._try_finalize(track, now, force=False)
            self._maybe_unreadable_live(track, now)

        for t in lost:
            self._lost(t, now)
        self._prune_ghosts(now)

    def _second_stage_plate(self, frame: np.ndarray, v: Obj) -> None:
        h, w = frame.shape[:2]
        bike = v.cls == "motorcycle"
        bx1, by1, bx2, by2 = v.box
        if bike:
            sx1, sy1, sx2, sy2 = expand(v.box, 0.08, 0.05, w, h)
        else:
            sx1, sy1, sx2, sy2 = v.box
        sw, sh = sx2 - sx1, sy2 - sy1
        if sw < 40 or sh < 40:
            return
        found = self.rec.plates.predict(frame[sy1:sy2, sx1:sx2])
        conf_min = self.bike_plate_conf if bike else self.second_stage_conf
        vw, vh = bx2 - bx1, by2 - by1
        best = None
        for mp in found:
            conf = float(mp.confidence)
            if conf < conf_min:
                continue
            mb = mp.bounding_box
            pb = (sx1 + max(0, int(mb.x1)), sy1 + max(0, int(mb.y1)),
                  sx1 + min(sw, int(mb.x2)), sy1 + min(sh, int(mb.y2)))
            pw_, ph_ = pb[2] - pb[0], pb[3] - pb[1]
            min_w = max(14, self.min_plate_w // 2) if bike else self.min_plate_w
            if pw_ < min_w or ph_ < 8:
                continue
            if bike:
                aspect = pw_ / max(1, ph_)
                if not (0.8 <= aspect <= 5.5):
                    continue                                       # not plate-shaped
                if pw_ * ph_ > 0.25 * vw * vh:
                    continue                                       # far too big for a plate
                if (pb[1] + pb[3]) / 2 < by1 + 0.30 * vh:
                    continue                                       # upper part of a bike = handlebar / headlight
            if best is None or conf > best[1]:
                best = (pb, conf)
        if best is not None:
            v.plate_box, v.plate_conf = best

    @staticmethod
    def _drop_nested(vehicles: list[Obj]) -> list[Obj]:
        """A box that lies almost entirely inside a bigger box of the same kind (car/truck/bus) is a duplicate
        detection of the same vehicle. Motorbikes are never dropped (a bike in front of a bus is real)."""
        four = {"car", "truck", "bus"}
        kept: list[Obj] = []
        for v in sorted(vehicles, key=lambda o: (o.box[2] - o.box[0]) * (o.box[3] - o.box[1]), reverse=True):
            area = max(1, (v.box[2] - v.box[0]) * (v.box[3] - v.box[1]))
            dup = False
            for k in kept:
                if v.cls == "motorcycle" or not (v.cls == k.cls or (v.cls in four and k.cls in four)):
                    continue
                ix = max(0, min(v.box[2], k.box[2]) - max(v.box[0], k.box[0]))
                iy = max(0, min(v.box[3], k.box[3]) - max(v.box[1], k.box[1]))
                if ix * iy / area >= 0.85:
                    dup = True
                    break
            if not dup:
                kept.append(v)
        return kept

    def _read_plate(self, track: Track, obj: Obj, frame: np.ndarray) -> None:
        """Read one plate in one frame and add it to the vehicle's evidence (see plate_reader.py)."""
        x1, y1, x2, y2 = obj.plate_box
        h_frame, w_frame = frame.shape[:2]
        pw, ph = x2 - x1, y2 - y1
        pad_x, pad_y = int(pw * 0.10), int(ph * 0.12)           # margin so edge characters are not cut off
        crop = frame[max(0, y1 - pad_y):min(h_frame, y2 + pad_y), max(0, x1 - pad_x):min(w_frame, x2 + pad_x)]
        if crop.shape[0] < 6 or crop.shape[1] < 6:
            track.invalid_reads += 1
            return
        low_light = self._brightness < LOW_LIGHT_THRESHOLD
        fr = read_plate(self.rec.reader, crop, self.fmt.templates, low_light, plate_w=pw, temperature=self.ocr_temp,
                        merged_weight=self.merged_weight, fast_accept=self.fast_accept,
                        strip_two_row_year=self.strip_two_row_year)
        if fr is None:
            track.invalid_reads += 1
            return
        track.plate_w_max = max(track.plate_w_max, pw)
        track.evidence.extend(fr.evidence)
        if len(track.evidence) > self.max_evidence:             # keep the strongest evidence
            track.evidence.sort(key=lambda e: e.weight, reverse=True)
            del track.evidence[self.max_evidence:]

        best = fr.best
        quality = best.quality if best else 0.0
        valid = best is not None and quality >= self.min_quality and len(best.text) >= self.min_len
        text = best.text if valid else ""
        if (not valid and self.fmt.enabled and fr.free_conf >= self.min_quality and len(fr.free_text) >= self.min_len
                and self.fmt.matches_custom(fr.free_text)):
            valid, text, quality = True, fr.free_text, fr.free_conf           # a custom regex format from config.yaml
        score = quality * obj.plate_conf
        if track.snap_frame is None or (valid and score > track.snap_score):
            track.snap_frame, track.snap_box, track.snap_plate_box, track.snap_score = \
                frame, track.box, obj.plate_box, score
        if valid:
            track.reads.append((text, quality))
            track.best_guess = text
            return
        track.invalid_reads += 1
        ft = fr.free_text
        if (self.allow_unusual and fr.free_conf >= self.unusual_min_conf and self.min_len <= len(ft) <= 10
                and any(c.isdigit() for c in ft) and any(c.isalpha() for c in ft)):
            track.unusual_reads.append((ft, fr.free_conf))        # steady reading that fits no known layout

    def _fused_decision(self, track: Track):
        """Decode ALL evidence of this vehicle at once. Returns (plate, quality, margin) or None."""
        fused = fuse(track.evidence)
        if fused is None:
            return None
        cands = [c for c in decode_templates(fused, self.fmt.templates, top=4) if len(c.text) >= self.min_len]
        if cands and cands[0].quality >= self.min_quality:
            return cands[0].text, cands[0].quality, cands[0].margin
        free_text, free_conf = decode_free(fused)
        if (free_text and len(free_text) >= self.min_len and self.fmt.enabled and self.fmt.matches_custom(free_text)
                and free_conf >= self.min_quality):
            return free_text, free_conf, 1.0
        return None

    def _eligible(self, track: Track) -> bool:
        return (not self.must_move) or track.moving_frames > 0

    def _try_finalize(self, track: Track, now: float, force: bool) -> bool:
        if track.finalized or not track.evidence or not self._eligible(track):
            return False
        decision = self._fused_decision(track)
        if decision is None:
            return self._try_finalize_unusual(track, now, force)
        final, quality, margin = decision
        track.best_guess = final
        n_valid = len(track.reads)
        n_read = n_valid + track.invalid_reads
        track.stable = track.stable + 1 if final == track.last_fused else 0
        track.last_fused = final
        confident = margin >= self.min_margin or n_valid >= self.min_reads + 3
        ready = ((n_valid >= self.min_reads and track.stable >= 1 and confident)
                 or n_valid >= self.max_reads
                 or (force and n_read >= self.min_reads_on_loss))
        if not ready:
            return False

        # Suppress duplicate slips for stationary / parked vehicles sitting in the same spot
        rep = getattr(getattr(self, "engine", None), "repeat_s", 60.0)
        for item in self._done_boxes:
            done_box, done_t = item[0], item[1]
            done_plate = item[2] if len(item) > 2 else ""
            if now - done_t <= rep and iou(track.box, done_box) >= 0.55:
                d = 0.0 if final == done_plate else weighted_distance(final, done_plate)
                if d <= 1.5 or (done_plate and d <= 2.2 and final[:3] == done_plate[:3]):
                    track.finalized = True
                    return False

        return self._issue_plate(track, final, quality, now, unusual=False)

    def _try_finalize_unusual(self, track: Track, now: float, force: bool) -> bool:
        """Vehicle with a steady plate reading that fits no known Pakistani layout: still gets a slip."""
        need = 1 if force else max(3, self.min_reads + 1)
        if not self.allow_unusual or not track.has_vehicle or len(track.unusual_reads) < need:
            return False
        voted = vote(track.unusual_reads)
        if not voted or voted[1] < 0.7:
            return False
        text, _agreement, _used = voted
        conf = float(statistics.mean(c for _t, c in track.unusual_reads))
        return self._issue_plate(track, text, conf, now, unusual=True)

    def _issue_plate(self, track: Track, final: str, conf: float, now: float, unusual: bool) -> bool:
        vtype = track.vehicle_type
        snap_fn = self._snap_fn(track, final)
        try:
            dec = self.engine.on_plate(self.cam, final, conf, vtype, datetime.now(timezone.utc), now, snap_fn,
                                       unusual=unusual)
        except Exception:  # noqa: BLE001
            log.exception("[%s] slip decision failed for %s", self.cam, final)
            return False
        track.decision, track.plate, track.finalized = dec, dec.plate or final, True
        self._done_boxes.append((track.box, now, final))
        self.last_plate = display(track.plate)
        log.info("[%s] %s (%s)%s -> %s", self.cam, display(final), vtype, " [unusual format]" if unusual else "",
                 dec.reason)
        return True

    def _maybe_unreadable_live(self, track: Track, now: float) -> None:
        """A vehicle that stays in view for a long time without a readable plate."""
        if (not track.finalized and not track.reads and not track.unusual_reads and self._eligible(track)
                and track.full_view
                and now - track.first_seen >= self.unreadable_after and track.moving_frames > 0):
            self._decide_unreadable(track, now)

    def _is_fragment(self, track: Track, now: float) -> bool:
        """True if this 'vehicle' is just a piece of a vehicle that was handled a moment ago
        (e.g. a partial detection while it leaves the picture), so it must not raise a second alert."""
        cutoff = 6.0 + 2 * getattr(self, "_heavy_dt", 0.0)
        rep = getattr(getattr(self, "engine", None), "repeat_s", 60.0)
        self._done_boxes = [item for item in self._done_boxes if now - item[1] <= max(rep, cutoff)]
        c = center(track.box)
        for item in self._done_boxes:
            if contains(item[0], c, pad=0.1):
                return True
        for other in self.tracker.tracks:
            if other is not track and other.finalized and other.decision is not None and other.decision.plate \
                    and contains(other.box, c, pad=0.1):
                return True
        return False

    def _decide_unreadable(self, track: Track, now: float) -> None:
        if self._is_fragment(track, now):
            track.finalized = True              # piece of a vehicle that already has a decision: stay quiet
            return
        vtype = track.vehicle_type
        try:
            dec = self.engine.on_unreadable(self.cam, vtype, datetime.now(timezone.utc), now,
                                            self._snap_fn(track, "NOPLATE"))
        except Exception:  # noqa: BLE001
            log.exception("[%s] unreadable-vehicle decision failed", self.cam)
            return
        track.decision, track.finalized = dec, True
        self._done_boxes.append((track.box, now))
        log.info("[%s] %s without readable plate -> %s", self.cam, vtype, dec.reason)

    def _lost(self, t: Track, now: float) -> None:
        if not t.finalized:
            done = bool(t.evidence) and self._try_finalize(t, now, force=True)
            if not done and t.has_vehicle and self._eligible(t) and t.frames >= 2 and t.full_view:
                self._decide_unreadable(t, now)
        if t.decision is not None:
            self._ghosts.append((t, now + 2.5))

    def _finish_lost(self, now: float) -> None:
        """Called while the road is quiet, so the last vehicle of a burst is closed without waiting for more frames."""
        _aligned, lost = self.tracker.update([], now)
        for t in lost:
            self._lost(t, now)
        self._prune_ghosts(now)

    def _prune_ghosts(self, now: float) -> None:
        self._ghosts = [g for g in self._ghosts if g[1] > now]

    # ------------------------------------------------------------------ snapshots
    def _snap_fn(self, track: Track, label: str):
        cache: list = []

        def snap():
            if not cache:
                cache.append(self._save_snapshot(track, label))
            return cache[0]
        return snap

    def _save_snapshot(self, track: Track, label: str) -> tuple[str | None, str | None]:
        frame = track.snap_frame if track.snap_frame is not None else self._last_frame
        if frame is None:
            return None, None
        h, w = frame.shape[:2]
        vis = frame.copy()
        th = max(2, w // 500)
        vb = track.snap_box or track.box
        cv2.rectangle(vis, (vb[0], vb[1]), (vb[2], vb[3]), COLORS["yellow"], th)
        pb = track.snap_plate_box
        crop = None
        if pb is not None:
            cv2.rectangle(vis, (pb[0], pb[1]), (pb[2], pb[3]), COLORS["green"], th)
            padx, pady = int((pb[2] - pb[0]) * 0.12), int((pb[3] - pb[1]) * 0.2)
            crop = frame[max(0, pb[1] - pady):min(h, pb[3] + pady), max(0, pb[0] - padx):min(w, pb[2] + padx)]
        cv2.putText(vis, display(label) if label != "NOPLATE" else "NO PLATE", (vb[0], max(25, vb[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.7, w / 1400), COLORS["yellow"], 2, cv2.LINE_AA)

        local = datetime.now().astimezone()
        day = local.strftime("%Y-%m-%d")
        folder = self.snap_dir / day
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.cam}_{label}_{local.strftime('%H%M%S')}_{int(local.microsecond / 1000):03d}"
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        full_rel, crop_rel = f"{day}/{stem}.jpg", f"{day}/{stem}_plate.jpg"
        ok1 = cv2.imwrite(str(self.snap_dir / full_rel), vis, params)
        ok2 = crop is not None and crop.size > 0 and cv2.imwrite(str(self.snap_dir / crop_rel), crop, params)
        return (full_rel if ok1 else None), (crop_rel if ok2 else None)

    # ------------------------------------------------------------------ drawing / streaming
    def _annotate(self, frame: np.ndarray, now: float) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]
        fs = max(0.55, w / 1500)
        th = max(2, w // 600)
        if self.roi:
            pts = np.array([[int(x * w), int(y * h)] for x, y in self.roi], np.int32)
            cv2.polylines(vis, [pts], True, (255, 200, 0), 1, cv2.LINE_AA)

        shown = [(t, False) for t in self.tracker.tracks] + [(t, True) for t, _ in self._ghosts]
        for t, ghost in shown:
            dec = t.decision
            if dec is not None:
                color, label = COLORS[dec.color], (f"{display(dec.plate)}  " if dec.plate else "") + dec.label
            elif not self._eligible(t) and t.has_vehicle:
                continue                                    # standing vehicle: not counted, not drawn
            else:
                color = COLORS["yellow"]
                label = f"{display(t.best_guess)}  reading..." if t.best_guess else "reading..."
            box = t.box if ghost else t.predicted_box(now)
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, th + 1)
            if t.plate_box is not None and not ghost:
                px1, py1, px2, py2 = t.plate_box
                cv2.rectangle(vis, (px1, py1), (px2, py2), color, max(1, th - 1))
            (tw, tth), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            ty = y1 - 8 if y1 - tth - 14 > 0 else y2 + tth + 12
            cv2.rectangle(vis, (x1, ty - tth - 6), (x1 + tw + 8, ty + base), color, -1)
            cv2.putText(vis, label, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, WHITE, th, cv2.LINE_AA)

        s = self.store.stats()
        title = "CAMERA 1 - MAIN GATE (front)" if self.cam == "camera1" else "CAMERA 2 - ROAD (rear)"
        header = f"{title}   slips today: {s['slips_today']}"
        (tw, tth), base = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (tw + 20, tth + base + 16), (0, 0, 0), -1)
        vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)
        cv2.putText(vis, header, (10, tth + 8), cv2.FONT_HERSHEY_SIMPLEX, fs, WHITE, th, cv2.LINE_AA)
        return vis

    def _placeholder(self) -> np.ndarray:
        img = np.zeros((360, 640, 3), np.uint8)
        msg = "NO SIGNAL" if self.source.status != "connecting" else "CONNECTING..."
        cv2.putText(img, self.cam.upper(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
        cv2.putText(img, msg, (200, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 230), 2, cv2.LINE_AA)
        return img

    def _publish(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        if w > STREAM_MAX_WIDTH:
            img = cv2.resize(img, (STREAM_MAX_WIDTH, int(h * STREAM_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self._jpeg_cond:
            self._jpeg = buf.tobytes()
            self._jpeg_seq += 1
            self._jpeg_cond.notify_all()

    def _handle_no_signal(self) -> None:
        never = self.source.last_frame_time == 0 and time.time() - self._t_created > self.offline_after
        stalled = self.source.last_frame_time and self.source.seconds_since_frame() > self.offline_after
        if not self._offline_alerted and (never or stalled):
            self._offline_alerted = True
            name = "Camera 1 (main gate)" if self.cam == "camera1" else "Camera 2 (road)"
            self.store.add_alert("camera_offline", f"{name} has no picture!", camera=self.cam, severity="danger")
