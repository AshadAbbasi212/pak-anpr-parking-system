"""Slip decisions for the two entrance cameras.

camera1  main gate, sees the FRONT plate  -> first chance to issue a slip
camera2  ~20 m further down, sees the REAR plate -> backup:
           * a plate camera 1 already slipped      -> no new slip (gray)
           * a plate very similar to one of those  -> no new slip, quiet "please check" note (gray)
           * a plate camera 1 did not slip         -> slip issued by camera 2 (green)
           * no readable plate, but camera 1 slipped a vehicle a moment ago that nobody has claimed yet
                                                   -> assumed to be that vehicle, quiet "please check" note (gray)
           * no readable plate and nothing to explain it -> LOUD alert, operator gives a manual slip (red)

Whitelisted plates never get a slip; they are logged quietly (blue).
The same plate seen again by the same camera within `repeat_protection_s` is the same vehicle (no second slip).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .plates import display, weighted_distance
from .printing import PrintQueue
from .store import Store, StoreError

log = logging.getLogger("slips")

CAM1 = "camera1"
CAM2 = "camera2"
SnapFn = Callable[[], "tuple[str | None, str | None]"]


@dataclass
class Decision:
    kind: str           # slip | whitelisted | already | check | unreadable | waiting
    color: str          # green | blue | gray | red | yellow
    label: str          # short text on the box
    reason: str         # longer text for the live feed
    plate: str | None = None
    entry_id: str | None = None
    slip_code: str | None = None


@dataclass
class Cam1Record:
    plate: str
    t: float
    entry_id: str
    label: str          # slip code, or "whitelisted"
    claimed: bool = False


class SlipEngine:
    def __init__(self, store: Store, printq: PrintQueue | None, matching: dict | None = None):
        m = matching or {}
        self.store = store
        self.printq = printq
        self.travel_min = float(m.get("travel_time_min_s", 2))
        self.travel_max = float(m.get("travel_time_max_s", 30))
        self.repeat_s = float(m.get("repeat_protection_s", 60))
        self.fuzzy_repeat_s = float(m.get("fuzzy_repeat_s", 6))
        self.unsure_cost = float(m.get("unsure_max_cost", 1.5))
        self._lock = threading.RLock()
        self._seen: dict[str, list[tuple[str, float, Decision]]] = {}
        self._cam1: list[Cam1Record] = []
        self._sources: dict = {}

    def set_sources(self, sources: dict) -> None:
        """Attach camera FrameSources to monitor real-time online/offline status."""
        self._sources = sources

    def is_online(self, cam: str) -> bool:
        """Returns True if the camera is currently online and receiving frames."""
        if not self._sources:
            return True   # default to collaborative mode when sources aren't attached (e.g. tests)
        src = self._sources.get(cam)
        if src is None:
            return False
        return src.status == "online" and src.seconds_since_frame() <= 10.0

    # ------------------------------------------------------------------ helpers
    def _repeat(self, cam: str, plate: str, now: float) -> Decision | None:
        items = [x for x in self._seen.get(cam, []) if now - x[1] <= self.repeat_s]
        self._seen[cam] = items
        for p, t, dec in reversed(items):
            d = 0.0 if p == plate else weighted_distance(plate, p)
            if d <= 0.5 or (d <= 1.0 and now - t <= self.fuzzy_repeat_s):
                return dec
        return None

    def _purge(self, now: float) -> None:
        keep = self.travel_max + 15
        self._cam1 = [r for r in self._cam1 if now - r.t <= keep]

    def _cam1_candidates(self, now: float, min_age: float = 0.0) -> list[Cam1Record]:
        # +5 s slack: camera 2 decides a moment after the vehicle was first seen
        return [r for r in self._cam1 if not r.claimed and min_age <= now - r.t <= self.travel_max + 5]

    def _log(self, cam: str, dec: Decision) -> Decision:
        self.store.push_event(camera=cam, plate=display(dec.plate) if dec.plate else "", kind=dec.kind,
                              color=dec.color, label=dec.label, reason=dec.reason)
        return dec

    # ------------------------------------------------------------------ plate read
    def on_plate(self, cam: str, plate: str, conf: float, vtype: str, ts: datetime, now: float,
                 snap_fn: SnapFn, unusual: bool = False) -> Decision:
        with self._lock:
            self._purge(now)
            rep = self._repeat(cam, plate, now)
            if rep is not None:
                return rep
            # If camera 2 already slipped this vehicle recently, don't issue a second slip at camera 1
            if cam == CAM1:
                cam2_rep = self._repeat(CAM2, plate, now)
                if cam2_rep is not None and cam2_rep.kind in ("slip", "whitelisted", "already"):
                    if cam2_rep.kind == "whitelisted":
                        return Decision("whitelisted", "blue", "WHITELISTED",
                                        "Whitelisted vehicle (already logged at camera 2)", plate, cam2_rep.entry_id)
                    code = cam2_rep.slip_code or cam2_rep.label
                    return Decision("already", "gray", f"ALREADY SLIPPED {code}",
                                    f"Camera 2 already issued slip {code}", plate, cam2_rep.entry_id, code)

            if cam == CAM1:
                dec = self._camera1(plate, conf, vtype, ts, now, snap_fn)
            else:
                dec = self._camera2(plate, conf, vtype, ts, now, snap_fn)
            self._seen.setdefault(cam, []).append((dec.plate or plate, now, dec))
            if unusual and dec.kind == "slip" and dec.entry_id:
                self._flag_unusual(dec, plate, cam)
            return self._log(cam, dec)

    def _flag_unusual(self, dec: Decision, plate: str, cam: str) -> None:
        """The plate was read steadily but fits none of the known Pakistani layouts: slip is issued anyway
        (never miss a vehicle) and the operator gets a quiet note to check it."""
        try:
            e = self.store.get_entry(dec.entry_id)
            note = ((e.get("notes") or "") + " | " if e and e.get("notes") else "") + \
                "unusual plate format - please check the plate"
            self.store._update(dec.entry_id, {"notes": note})
            self.store.add_alert(
                "check", f"Slip {dec.slip_code}: the plate was read as {display(plate)}, which does not match a "
                f"known Pakistani layout. Slip issued anyway - please check the plate and use Fix plate if needed.",
                plate=plate, camera=cam, snapshot=(e or {}).get("snapshot"), severity="info", entry_id=dec.entry_id)
        except Exception:  # noqa: BLE001 - a note must never break slip issuing
            log.exception("could not flag unusual plate")

    def _issue(self, kind: str, cam: str, plate: str, conf: float, vtype: str, ts: datetime,
               snap_fn: SnapFn, notes: str = "") -> dict:
        snap, crop = snap_fn()
        entry = self.store.create_entry(kind, plate, vtype, cam, conf, ts, snap, crop, notes)
        if kind == "slip" and self.printq is not None:
            self.printq.enqueue(entry["_id"])
        return entry

    def _camera1(self, plate, conf, vtype, ts, now, snap_fn) -> Decision:
        if self.store.is_whitelisted(plate):
            e = self._issue("whitelisted", CAM1, plate, conf, vtype, ts, snap_fn)
            self._cam1.append(Cam1Record(plate, now, e["_id"], "whitelisted"))
            return Decision("whitelisted", "blue", "WHITELISTED", "Whitelisted vehicle - no slip",
                            plate, e["_id"])
        e = self._issue("slip", CAM1, plate, conf, vtype, ts, snap_fn)
        self._cam1.append(Cam1Record(plate, now, e["_id"], e["slip_code"]))
        return Decision("slip", "green", f"SLIP {e['slip_code']}", f"Slip {e['slip_code']} issued by camera 1",
                        plate, e["_id"], e["slip_code"])

    def _camera2(self, plate, conf, vtype, ts, now, snap_fn) -> Decision:
        cam1_alive = self.is_online(CAM1)
        if not cam1_alive:
            # Standalone failover mode: camera 1 is down, camera 2 handles all gate duties
            if self.store.is_whitelisted(plate):
                e = self._issue("whitelisted", CAM2, plate, conf, vtype, ts, snap_fn, "logged by camera 2 (camera 1 offline)")
                return Decision("whitelisted", "blue", "WHITELISTED", "Whitelisted vehicle - no slip", plate, e["_id"])
            e = self._issue("slip", CAM2, plate, conf, vtype, ts, snap_fn,
                            "issued by camera 2 (camera 1 is offline)")
            return Decision("slip", "green", f"SLIP {e['slip_code']}",
                            f"Slip {e['slip_code']} issued by camera 2 (camera 1 offline)",
                            plate, e["_id"], e["slip_code"])

        # Dual-camera cooperative mode (camera 1 is alive)
        best, best_d = None, 99.0
        for r in self._cam1_candidates(now):
            d = 0.0 if r.plate == plate else weighted_distance(plate, r.plate)
            if d < best_d:
                best, best_d = r, d

        if best is not None and best_d <= 0.5:
            best.claimed = True
            if best.label == "whitelisted":
                return Decision("whitelisted", "blue", "WHITELISTED", "Whitelisted vehicle (already logged at camera 1)",
                                plate, best.entry_id)
            return Decision("already", "gray", f"ALREADY SLIPPED {best.label}",
                            f"Camera 1 already issued slip {best.label}", plate, best.entry_id, best.label)

        if best is not None and best_d <= self.unsure_cost:
            best.claimed = True
            snap, _crop = snap_fn()
            self.store.add_alert(
                "check",
                f"Camera 2 read {display(plate)}, very similar to {display(best.plate)} "
                f"(slip {best.label}) from camera 1. Treated as the same vehicle - no new slip. "
                f"Please check if unsure.",
                plate=plate, camera=CAM2, snapshot=snap, severity="info", entry_id=best.entry_id)
            return Decision("check", "gray", "CHECK - SAME AS " + best.label,
                            f"Similar to slip {best.label}; skipped", plate, best.entry_id, best.label)

        if self.store.is_whitelisted(plate):
            e = self._issue("whitelisted", CAM2, plate, conf, vtype, ts, snap_fn, "logged by camera 2")
            return Decision("whitelisted", "blue", "WHITELISTED", "Whitelisted vehicle - no slip", plate, e["_id"])

        e = self._issue("slip", CAM2, plate, conf, vtype, ts, snap_fn,
                        "issued by camera 2 (camera 1 did not read this vehicle)")
        return Decision("slip", "green", f"SLIP {e['slip_code']}",
                        f"Slip {e['slip_code']} issued by camera 2 (camera 1 missed it)",
                        plate, e["_id"], e["slip_code"])

    # ------------------------------------------------------------------ vehicle without a readable plate
    def on_unreadable(self, cam: str, vtype: str, ts: datetime, now: float, snap_fn: SnapFn) -> Decision:
        with self._lock:
            self._purge(now)
            other_cam = CAM2 if cam == CAM1 else CAM1
            other_alive = self.is_online(other_cam)

            if cam == CAM1:
                if other_alive:
                    # normal for bikes (rear plate only): camera 2 is the backup
                    return self._log(cam, Decision("waiting", "yellow", "NO PLATE - CAM 2 WILL CHECK",
                                                   f"{vtype} passed camera 1 without a readable plate"))
                # Standalone mode: Camera 2 is offline, so Camera 1 must raise manual slip alert!
                snap, _crop = snap_fn()
                self.store.add_alert(
                    "unreadable_vehicle",
                    f"A {vtype} passed camera 1 with NO READABLE PLATE (camera 2 is offline). Give a manual slip.",
                    camera=CAM1, snapshot=snap, severity="danger")
                return self._log(cam, Decision("unreadable", "red", "NO PLATE - MANUAL SLIP",
                                               f"{vtype} with unreadable plate (camera 2 offline) - manual slip needed"))

            # cam == CAM2:
            if not other_alive:
                # Standalone mode: Camera 1 is offline, so Camera 2 must raise manual slip alert directly
                snap, _crop = snap_fn()
                self.store.add_alert(
                    "unreadable_vehicle",
                    f"A {vtype} passed camera 2 with NO READABLE PLATE (camera 1 is offline). Give a manual slip.",
                    camera=CAM2, snapshot=snap, severity="danger")
                return self._log(cam, Decision("unreadable", "red", "NO PLATE - MANUAL SLIP",
                                               f"{vtype} with unreadable plate (camera 1 offline) - manual slip needed"))

            cands = sorted(self._cam1_candidates(now, self.travel_min), key=lambda r: r.t)
            snap, _crop = snap_fn()
            if cands:
                r = cands[0]
                r.claimed = True
                self.store.add_alert(
                    "check",
                    f"A {vtype} passed camera 2 and its plate could not be read. Assumed to be the vehicle "
                    f"camera 1 slipped a moment ago ({display(r.plate)}, slip {r.label}). Please check if unsure.",
                    camera=CAM2, snapshot=snap, severity="info", entry_id=r.entry_id)
                return self._log(cam, Decision("check", "gray", "CHECK - PROBABLY " + r.label,
                                               f"Unreadable; probably slip {r.label}", display(r.plate),
                                               r.entry_id, r.label))
            self.store.add_alert(
                "unreadable_vehicle",
                f"A {vtype} passed with NO READABLE PLATE and no matching slip. Give a manual slip.",
                camera=CAM2, snapshot=snap, severity="danger")
            return self._log(cam, Decision("unreadable", "red", "NO PLATE - MANUAL SLIP",
                                           f"{vtype} with unreadable plate - manual slip needed"))
