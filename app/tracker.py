"""Multi-object tracker for vehicles (with their plates).

* A track is one vehicle passing a camera. Several vehicles at once = several tracks.
* Tracks are seeded either by a vehicle box (car / motorbike / bus / truck) or - if the vehicle detector
  missed it - by a plate box alone, so a vehicle with a readable plate is never lost.
* Plate reads are collected on the track so the final plate is a vote over several frames.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

Box = tuple[int, int, int, int]

VEHICLE_TYPE = {"car": "CAR", "motorcycle": "BIKE", "bus": "BUS", "truck": "TRUCK"}


def vehicle_group(vtype: str) -> str:
    """car / bike / other  (used for the counters)."""
    return {"CAR": "car", "BIKE": "bike"}.get(vtype, "other")


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def contains(b: Box, p: tuple[float, float], pad: float = 0.0, pad_x: float | None = None, pad_y: float | None = None) -> bool:
    w, h = b[2] - b[0], b[3] - b[1]
    px = pad if pad_x is None else pad_x
    py = pad if pad_y is None else pad_y
    return b[0] - px * w <= p[0] <= b[2] + px * w and b[1] - py * h <= p[1] <= b[3] + py * h


def expand(b: Box, fx: float, fy: float, w: int, h: int) -> Box:
    bw, bh = b[2] - b[0], b[3] - b[1]
    return (max(0, int(b[0] - bw * fx)), max(0, int(b[1] - bh * fy)),
            min(w, int(b[2] + bw * fx)), min(h, int(b[3] + bh * fy)))


@dataclass
class Obj:
    """One detection in one frame: a vehicle box (cls set) or a plate on its own (cls None)."""
    box: Box
    cls: str | None = None
    conf: float = 0.0
    plate_box: Box | None = None
    plate_conf: float = 0.0


@dataclass
class Track:
    id: int
    box: Box
    first_seen: float
    last_seen: float
    has_vehicle: bool = False
    frames: int = 1
    moving_frames: int = 0
    cls_votes: Counter = field(default_factory=Counter)
    plate_box: Box | None = None
    vel: tuple[float, float] = (0.0, 0.0)          # pixels per second
    reads: list[tuple[str, float]] = field(default_factory=list)   # (valid plate text, confidence)
    invalid_reads: int = 0
    decision: Any = None
    plate: str | None = None
    best_guess: str = ""
    finalized: bool = False
    snap_frame: Any = None
    snap_box: Box | None = None
    snap_plate_box: Box | None = None
    snap_score: float = 0.0
    full_view: bool = False                        # seen at least once without being cut off by the picture edge
    evidence: list = field(default_factory=list)   # plate_reader.Evidence from every frame (fused when deciding)
    unusual_reads: list = field(default_factory=list)   # (text, confidence): stable reads that fit no known layout
    last_fused: str = ""                           # fused plate text after the previous read (stability check)
    stable: int = 0                                # how many reads in a row left the fused text unchanged
    plate_w_max: int = 0                           # widest plate seen (pixels)

    @property
    def vehicle_type(self) -> str:
        if not self.cls_votes:
            return "VEHICLE"
        return VEHICLE_TYPE.get(self.cls_votes.most_common(1)[0][0], "VEHICLE")

    def predicted_box(self, now: float) -> Box:
        dt = min(1.0, max(0.0, now - self.last_seen))
        dx, dy = self.vel[0] * dt, self.vel[1] * dt
        return (int(self.box[0] + dx), int(self.box[1] + dy), int(self.box[2] + dx), int(self.box[3] + dy))


class VehicleTracker:
    def __init__(self, max_missing_s: float = 1.2):
        self.max_missing_s = max_missing_s
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    # ------------------------------------------------------------------ matching
    @staticmethod
    def _score(t: Track, o: Obj, now: float) -> float | None:
        pb = t.predicted_box(now)
        oc, tc = center(o.box), center(pb)
        scale = max(o.box[2] - o.box[0], o.box[3] - o.box[1], pb[2] - pb[0], pb[3] - pb[1], 1)
        dist = ((oc[0] - tc[0]) ** 2 + (oc[1] - tc[1]) ** 2) ** 0.5 / scale
        ov = iou(pb, o.box)
        if o.cls is None:                                   # plate-only observation
            if t.has_vehicle:
                if not contains(pb, oc, pad_x=0.03, pad_y=0.20):
                    return None
                return 1.0 + ov - 0.3 * dist
            if ov > 0.05 or dist < 1.5:
                return ov - 0.2 * dist
            return None
        if not t.has_vehicle:                               # vehicle box meets a plate-only track
            if contains(o.box, tc, pad=0.1):
                return 1.0 + ov - 0.3 * dist
            return None
        # vehicle meets vehicle. Fast traffic moves more than a vehicle-width between analysed frames, so
        # accept a generous distance (the best-scoring pairs are matched first, so neighbours do not swap).
        if ov > 0.1 or dist < 2.0:
            return 1.0 + ov - 0.3 * dist
        return None

    def update(self, objs: list[Obj], now: float) -> tuple[list[Track], list[Track]]:
        """Returns (track for every obj, in the same order ; tracks that were lost in this update)."""
        pairs = []
        for ti, t in self._tracks.items():
            for oi, o in enumerate(objs):
                s = self._score(t, o, now)
                if s is not None:
                    pairs.append((s, ti, oi))
        pairs.sort(reverse=True)
        used_t: set[int] = set()
        assigned: dict[int, Track] = {}
        for _s, ti, oi in pairs:
            if ti in used_t or oi in assigned:
                continue
            used_t.add(ti)
            assigned[oi] = self._tracks[ti]
            self._apply(self._tracks[ti], objs[oi], now)

        aligned: list[Track] = []
        for oi, o in enumerate(objs):
            if oi in assigned:
                aligned.append(assigned[oi])
                continue
            t = Track(id=self._next_id, box=o.box, first_seen=now, last_seen=now)
            self._next_id += 1
            if o.cls is not None:
                t.has_vehicle = True
                t.cls_votes[o.cls] += max(o.conf, 0.01)
            t.plate_box = o.plate_box
            self._tracks[t.id] = t
            aligned.append(t)

        lost: list[Track] = []
        for ti in list(self._tracks):
            t = self._tracks[ti]
            if t.last_seen != now and now - t.last_seen > self.max_missing_s:
                lost.append(t)
                del self._tracks[ti]
        return aligned, lost

    @staticmethod
    def _apply(t: Track, o: Obj, now: float) -> None:
        dt = max(1e-3, now - t.last_seen)
        old_c = center(t.box)
        if o.cls is not None:
            t.box = o.box
            t.has_vehicle = True
            t.cls_votes[o.cls] += max(o.conf, 0.01)
        elif t.has_vehicle:
            # plate only: move the (stale) vehicle box along with the plate
            if t.plate_box is not None and o.plate_box is not None:
                pc, nc = center(t.plate_box), center(o.plate_box)
                dx, dy = int(nc[0] - pc[0]), int(nc[1] - pc[1])
                t.box = (t.box[0] + dx, t.box[1] + dy, t.box[2] + dx, t.box[3] + dy)
        else:
            t.box = o.box
        new_c = center(t.box)
        vx, vy = (new_c[0] - old_c[0]) / dt, (new_c[1] - old_c[1]) / dt
        t.vel = (0.6 * t.vel[0] + 0.4 * vx, 0.6 * t.vel[1] + 0.4 * vy)
        if o.plate_box is not None:
            t.plate_box = o.plate_box
        t.last_seen = now
        t.frames += 1
