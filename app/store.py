"""Data layer: MongoDB + an in-memory copy + an offline buffer.

* Everything the gates need to decide (whitelist, recent slips, counters, slip numbers) is kept in memory,
  so the system keeps working even if MongoDB stops.
* Every change is written to MongoDB. If MongoDB is not reachable the change is appended to
  data/pending.jsonl and replayed (in order) as soon as MongoDB comes back.
* A copy of the current state is kept in data/cache.json, and the last slip number in data/slip_counter.txt,
  so a restart while MongoDB is down still continues the numbering correctly.

Collections: entries (slips + whitelisted passes), whitelist, alerts, meta.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.errors import DuplicateKeyError, PyMongoError
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pymongo is not installed. Run setup.bat first.") from exc

from .tracker import vehicle_group

log = logging.getLogger("store")

GROUPS = ("car", "bike", "other")


class StoreError(Exception):
    """Raised for user-facing problems (unknown id, invalid plate ...)."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _ser(obj):
    if isinstance(obj, datetime):
        return {"$dt": as_aware(obj).astimezone(timezone.utc).isoformat()}
    if isinstance(obj, dict):
        return {k: _ser(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ser(v) for v in obj]
    return obj


def _deser(obj):
    if isinstance(obj, dict):
        if set(obj) == {"$dt"}:
            return datetime.fromisoformat(obj["$dt"])
        return {k: _deser(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deser(v) for v in obj]
    return obj


def local_day(dt: datetime | None = None) -> str:
    return (dt or utcnow()).astimezone().strftime("%Y-%m-%d")


def _empty_counters() -> dict:
    return {"day": local_day(), "today": {"car": 0, "bike": 0, "other": 0, "whitelisted": 0},
            "total_slips": 0, "slip_counter": 0}


class Store:
    def __init__(self, cfg: dict, client=None):
        self.cfg = cfg
        self.data_dir = Path(cfg["storage"]["data_dir"])
        self.cache_path = self.data_dir / "cache.json"
        self.pending_path = self.data_dir / "pending.jsonl"
        self.counter_path = self.data_dir / "slip_counter.txt"
        slip = cfg.get("slip", {})
        self.number_start = int(slip.get("number_start", 1))
        self.number_digits = int(slip.get("number_digits", 6))
        self.number_prefix = str(slip.get("number_prefix", ""))

        self._lock = threading.RLock()      # protects the in-memory state
        self._io_lock = threading.RLock()   # serialises MongoDB / pending-file access
        self._client = client
        self.db = None
        self.online = False
        self._pending_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.recent: OrderedDict[str, dict] = OrderedDict()   # newest last; the last ~600 entries
        self.whitelist: dict[str, dict] = {}
        self.alerts: deque = deque(maxlen=300)
        self.events: deque = deque(maxlen=60)
        self.counters = _empty_counters()

    # ------------------------------------------------------------------ start / stop
    def start(self) -> None:
        self._count_pending()
        self._connect()
        if self.online:
            self.flush_pending()
            self._load_from_db()
        else:
            log.warning("MongoDB is not reachable - running from the local cache. "
                        "Changes are buffered and will be saved when MongoDB is back.")
            self._load_from_cache()
        self._restore_slip_counter()
        self._save_cache()
        self._thread = threading.Thread(target=self._maintenance_loop, name="store-maint", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._save_cache()

    # ------------------------------------------------------------------ connection
    def _connect(self) -> bool:
        with self._io_lock:
            try:
                if self._client is None:
                    m = self.cfg["mongodb"]
                    self._client = MongoClient(m["uri"], tz_aware=True,
                                               serverSelectionTimeoutMS=int(m["server_selection_timeout_ms"]))
                self._client.admin.command("ping")
                self.db = self._client[self.cfg["mongodb"]["database"]]
                self._ensure_indexes()
                if not self.online:
                    log.info("MongoDB connected.")
                self.online = True
            except PyMongoError as exc:
                self.online = False
                log.debug("MongoDB connect failed: %s", exc)
            return self.online

    def _ensure_indexes(self) -> None:
        try:
            e = self.db["entries"]
            e.create_index([("created_at", DESCENDING)])
            e.create_index([("plate", ASCENDING), ("created_at", DESCENDING)])
            e.create_index([("kind", ASCENDING), ("handed_over", ASCENDING), ("voided", ASCENDING)])
            e.create_index([("slip_no", ASCENDING)], unique=True,
                           partialFilterExpression={"kind": "slip"}, name="unique_slip_no")
            self.db["alerts"].create_index([("time", DESCENDING)])
        except Exception as exc:  # noqa: BLE001 - index problems must never stop the system
            log.warning("Could not create indexes: %s", exc)

    def _maintenance_loop(self) -> None:
        while not self._stop.wait(5.0):
            try:
                if not self.online:
                    if self._connect():
                        self.flush_pending()
                        self._save_cache()
                elif self._pending_count:
                    self.flush_pending()
                self._roll_day()
            except Exception:  # noqa: BLE001
                log.exception("maintenance loop error")

    # ------------------------------------------------------------------ pending buffer
    def _count_pending(self) -> None:
        try:
            with open(self.pending_path, "r", encoding="utf-8") as f:
                self._pending_count = sum(1 for line in f if line.strip())
        except FileNotFoundError:
            self._pending_count = 0

    def _apply(self, coll: str, op: str, doc_id, doc: dict | None) -> None:
        c = self.db[coll]
        if op == "replace":
            c.replace_one({"_id": doc_id}, {**(doc or {}), "_id": doc_id}, upsert=True)
        elif op == "set":
            c.update_one({"_id": doc_id}, {"$set": doc or {}}, upsert=True)
        elif op == "delete":
            c.delete_one({"_id": doc_id})

    def _write(self, coll: str, op: str, doc_id, doc: dict | None = None) -> None:
        with self._io_lock:
            if self.online and self._pending_count == 0:
                try:
                    self._apply(coll, op, doc_id, doc)
                    return
                except DuplicateKeyError as exc:
                    log.error("Database refused %s/%s (duplicate): %s", coll, doc_id, exc)
                    return
                except PyMongoError as exc:
                    log.warning("MongoDB write failed, buffering locally: %s", exc)
                    self.online = False
            line = json.dumps({"c": coll, "op": op, "id": doc_id, "doc": _ser(doc)})
            with open(self.pending_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._pending_count += 1

    def flush_pending(self) -> None:
        with self._io_lock:
            if not self.online or not self.pending_path.exists():
                return
            with open(self.pending_path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            done = 0
            for ln in lines:
                try:
                    item = json.loads(ln)
                    self._apply(item["c"], item["op"], item["id"], _deser(item.get("doc")))
                except DuplicateKeyError as exc:
                    log.error("Skipping buffered write (duplicate): %s", exc)
                except PyMongoError as exc:
                    log.warning("MongoDB went away while replaying buffer: %s", exc)
                    self.online = False
                    break
                except (ValueError, KeyError):
                    log.error("Skipping unreadable buffered line")
                done += 1
            remaining = lines[done:]
            if remaining:
                with open(self.pending_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(remaining) + "\n")
            else:
                try:
                    os.remove(self.pending_path)
                except FileNotFoundError:
                    pass
            self._pending_count = len(remaining)
            if done:
                log.info("Replayed %d buffered changes to MongoDB.", done)

    @property
    def pending_count(self) -> int:
        return self._pending_count

    # ------------------------------------------------------------------ cache + loading
    def _save_cache(self) -> None:
        with self._lock:
            data = {
                "recent": list(self.recent.values())[-300:],
                "whitelist": list(self.whitelist.values()),
                "alerts": list(self.alerts)[-100:],
                "counters": self.counters,
            }
            payload = json.dumps(_ser(data))
        tmp = self.cache_path.with_suffix(".tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self.cache_path)
        except OSError as exc:
            log.warning("Could not write cache: %s", exc)

    def _remember(self, entry: dict) -> None:
        self.recent[entry["_id"]] = entry
        self.recent.move_to_end(entry["_id"])
        while len(self.recent) > 600:
            self.recent.popitem(last=False)

    def _load_from_cache(self) -> None:
        try:
            data = _deser(json.loads(self.cache_path.read_text(encoding="utf-8")))
        except (FileNotFoundError, ValueError):
            return
        with self._lock:
            for e in data.get("recent", []):
                self._remember(e)
            self.whitelist = {w["plate"]: w for w in data.get("whitelist", [])}
            self.alerts.extend(data.get("alerts", []))
            self.counters.update(data.get("counters", {}))
        self._roll_day()

    def _load_from_db(self) -> None:
        try:
            with self._lock:
                coll = self.db["entries"]
                recent = list(coll.find().sort("created_at", DESCENDING).limit(300))
                pending = list(coll.find({"kind": "slip", "handed_over": False, "voided": False})
                               .sort("created_at", DESCENDING).limit(300))
                merged = {e["_id"]: e for e in recent + pending}
                for e in sorted(merged.values(), key=lambda x: x["created_at"]):
                    self._remember(e)
                self.whitelist = {w["plate"]: w for w in self.db["whitelist"].find()}
                self.alerts.clear()
                self.alerts.extend(reversed(list(self.db["alerts"].find().sort("time", DESCENDING).limit(200))))
                meta = self.db["meta"].find_one({"_id": "counters"})
                if meta:
                    meta.pop("_id", None)
                    self.counters.update(meta)
                else:
                    self._recount_from_db()
            self._roll_day()
        except PyMongoError as exc:
            log.warning("Could not load state from MongoDB (%s) - using cache.", exc)
            self.online = False
            self._load_from_cache()

    def _recount_from_db(self) -> None:
        coll = self.db["entries"]
        self.counters["total_slips"] = coll.count_documents({"kind": "slip", "voided": False})
        self._recount_today()

    def _recount_today(self) -> None:
        start = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        today = {g: 0 for g in GROUPS}
        today["whitelisted"] = 0
        coll = self.db["entries"]
        for e in coll.find({"created_at": {"$gte": start}, "voided": False}, {"kind": 1, "vehicle_group": 1}):
            if e.get("kind") == "whitelisted":
                today["whitelisted"] += 1
            else:
                g = e.get("vehicle_group", "other")
                today[g] = today.get(g, 0) + 1
        self.counters["day"] = local_day()
        self.counters["today"] = today

    def _roll_day(self) -> None:
        with self._lock:
            if self.counters.get("day") != local_day():
                self.counters["day"] = local_day()
                self.counters["today"] = {"car": 0, "bike": 0, "other": 0, "whitelisted": 0}
                self._write("meta", "set", "counters", dict(self.counters))
                self._save_cache()

    # ------------------------------------------------------------------ slip numbers
    def _restore_slip_counter(self) -> None:
        """Continue the numbering from the highest number ever used (cache, counter file, MongoDB)."""
        best = int(self.counters.get("slip_counter", 0))
        try:
            best = max(best, int(self.counter_path.read_text().strip() or 0))
        except (FileNotFoundError, ValueError):
            pass
        if self.online:
            try:
                top = self.db["entries"].find_one({"kind": "slip"}, sort=[("slip_no", DESCENDING)])
                if top:
                    best = max(best, int(top["slip_no"]))
            except PyMongoError:
                pass
        with self._lock:
            self.counters["slip_counter"] = best

    def _persist_slip_counter(self, n: int) -> None:
        tmp = self.counter_path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                f.write(str(n))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.counter_path)
        except OSError as exc:
            log.warning("Could not write slip counter file: %s", exc)

    def slip_code(self, n: int) -> str:
        return f"{self.number_prefix}{n:0{self.number_digits}d}"

    # ------------------------------------------------------------------ whitelist
    def is_whitelisted(self, plate: str) -> bool:
        return plate in self.whitelist

    def whitelist_add(self, plate: str, note: str = "") -> dict:
        doc = {"plate": plate, "note": note, "added_at": utcnow()}
        with self._lock:
            self.whitelist[plate] = doc
        self._write("whitelist", "replace", plate, doc)
        self._save_cache()
        return doc

    def whitelist_add_many(self, items: list[tuple[str, str]]) -> int:
        n = 0
        with self._lock:
            for plate, note in items:
                doc = {"plate": plate, "note": note, "added_at": utcnow()}
                self.whitelist[plate] = doc
                self._write("whitelist", "replace", plate, doc)
                n += 1
        self._save_cache()
        return n

    def whitelist_remove(self, plate: str) -> None:
        with self._lock:
            if plate not in self.whitelist:
                raise StoreError("That plate is not on the whitelist")
            del self.whitelist[plate]
        self._write("whitelist", "delete", plate)
        self._save_cache()

    # ------------------------------------------------------------------ entries (slips / whitelisted passes)
    def create_entry(self, kind: str, plate: str, vtype: str, camera: str, conf: float, ts: datetime,
                     snapshot: str | None = None, plate_img: str | None = None, notes: str = "") -> dict:
        if kind not in ("slip", "whitelisted"):
            raise StoreError("unknown entry kind")
        with self._lock:
            self._roll_day()
            group = vehicle_group(vtype)
            entry = {
                "_id": uuid.uuid4().hex, "kind": kind, "plate": plate, "vehicle_type": vtype,
                "vehicle_group": group, "created_at": ts, "camera": camera, "conf": round(float(conf), 3),
                "snapshot": snapshot, "plate_img": plate_img, "slip_no": None, "slip_code": None,
                "print_status": "none" if kind != "slip" else "queued", "print_count": 0, "print_error": "",
                "handed_over": False, "handed_at": None, "voided": False, "voided_at": None, "notes": notes,
            }
            if kind == "slip":
                n = max(int(self.counters["slip_counter"]) + 1, self.number_start)
                self.counters["slip_counter"] = n
                self._persist_slip_counter(n)
                entry["slip_no"], entry["slip_code"] = n, self.slip_code(n)
                self.counters["today"][group] = self.counters["today"].get(group, 0) + 1
                self.counters["total_slips"] += 1
            else:
                self.counters["today"]["whitelisted"] = self.counters["today"].get("whitelisted", 0) + 1
            self._remember(entry)
            doc, counters = dict(entry), dict(self.counters)
        self._write("entries", "replace", entry["_id"], doc)
        self._write("meta", "set", "counters", counters)
        self._save_cache()
        return entry

    def get_entry(self, entry_id: str) -> dict | None:
        with self._lock:
            e = self.recent.get(entry_id)
            if e is not None:
                return e
        if self.online:
            try:
                with self._io_lock:
                    return self.db["entries"].find_one({"_id": entry_id})
            except PyMongoError:
                self.online = False
        return None

    def _update(self, entry_id: str, fields: dict, save_cache: bool = True) -> dict:
        entry = self.get_entry(entry_id)
        if entry is None:
            raise StoreError("Slip not found")
        with self._lock:
            entry.update(fields)
            if entry_id in self.recent:
                self.recent[entry_id] = entry
            else:
                self._remember(entry)
        self._write("entries", "set", entry_id, dict(fields))
        if save_cache:
            self._save_cache()
        return entry

    def set_print_result(self, entry_id: str, ok: bool, status: str, error: str = "") -> dict:
        entry = self.get_entry(entry_id)
        count = (int(entry.get("print_count", 0)) + (1 if ok else 0)) if entry else 0
        return self._update(entry_id, {"print_status": status, "print_error": error, "print_count": count})

    def set_handed_over(self, entry_id: str, value: bool) -> dict:
        return self._update(entry_id, {"handed_over": bool(value), "handed_at": utcnow() if value else None})

    def void_entry(self, entry_id: str) -> dict:
        entry = self.get_entry(entry_id)
        if entry is None:
            raise StoreError("Slip not found")
        if entry.get("voided"):
            raise StoreError("This slip is already voided")
        with self._lock:
            if entry["kind"] == "slip":
                self.counters["total_slips"] = max(0, self.counters["total_slips"] - 1)
            if local_day(as_aware(entry["created_at"])) == self.counters.get("day"):
                key = "whitelisted" if entry["kind"] == "whitelisted" else entry.get("vehicle_group", "other")
                self.counters["today"][key] = max(0, self.counters["today"].get(key, 0) - 1)
            counters = dict(self.counters)
        self._write("meta", "set", "counters", counters)
        return self._update(entry_id, {"voided": True, "voided_at": utcnow()})

    def edit_plate(self, entry_id: str, plate: str) -> dict:
        """Operator corrects a plate. The first value is kept in `original_plate` and `plate_corrected` is set:
        corrected slips are the most valuable examples for improving the reader (tools/export_training_set.py)."""
        entry = self.get_entry(entry_id)
        if entry is None:
            raise StoreError("Slip not found")
        note = ((entry.get("notes") or "") + " | " if entry.get("notes") else "") + \
            f"plate corrected manually (was {entry['plate']})"
        fields = {"plate": plate, "notes": note, "plate_corrected": True}
        if not entry.get("original_plate"):
            fields["original_plate"] = entry["plate"]
        return self._update(entry_id, fields)

    # ------------------------------------------------------------------ alerts + live events
    def add_alert(self, kind: str, message: str, plate: str | None = None, camera: str | None = None,
                  snapshot: str | None = None, severity: str = "warning", entry_id: str | None = None) -> dict:
        """severity: danger = loud (sound) | warning | info = quiet 'please check' note"""
        doc = {"_id": uuid.uuid4().hex, "time": utcnow(), "type": kind, "severity": severity,
               "message": message, "plate": plate, "camera": camera, "snapshot": snapshot,
               "entry_id": entry_id, "ack": False}
        with self._lock:
            self.alerts.append(doc)
        self._write("alerts", "replace", doc["_id"], dict(doc))
        self._save_cache()
        log.warning("ALERT [%s/%s] %s", severity, kind, message)
        return doc

    def ack_alert(self, alert_id: str) -> None:
        with self._lock:
            for a in self.alerts:
                if a["_id"] == alert_id:
                    a["ack"] = True
                    break
            else:
                raise StoreError("Alert not found")
        self._write("alerts", "set", alert_id, {"ack": True})
        self._save_cache()

    def push_event(self, **event) -> None:
        event.setdefault("time", utcnow())
        with self._lock:
            self.events.appendleft(event)

    # ------------------------------------------------------------------ counters
    def stats(self) -> dict:
        with self._lock:
            t = self.counters["today"]
            pending = sum(1 for e in self.recent.values()
                          if e["kind"] == "slip" and not e["voided"] and not e["handed_over"])
            return {
                "slips_today": t.get("car", 0) + t.get("bike", 0) + t.get("other", 0),
                "cars_today": t.get("car", 0), "bikes_today": t.get("bike", 0), "other_today": t.get("other", 0),
                "whitelisted_today": t.get("whitelisted", 0),
                "total_slips": self.counters["total_slips"],
                "pending_handover": pending,
            }

    def queue(self, include_handed: bool = False, limit: int = 60) -> list[dict]:
        with self._lock:
            rows = [e for e in self.recent.values()
                    if e["kind"] == "slip" and not e["voided"] and (include_handed or not e["handed_over"])]
        rows.sort(key=lambda e: e["created_at"], reverse=True)
        return rows[:limit]

    # ------------------------------------------------------------------ history queries
    @staticmethod
    def _day_range(date_from: str, date_to: str) -> dict:
        rng: dict = {}
        try:
            if date_from:
                d = datetime.strptime(date_from, "%Y-%m-%d").astimezone()
                rng["$gte"] = d.astimezone(timezone.utc)
            if date_to:
                d = datetime.strptime(date_to, "%Y-%m-%d").astimezone() + timedelta(days=1)
                rng["$lt"] = d.astimezone(timezone.utc)
        except ValueError:
            pass
        return rng

    def _history_filter(self, q: str = "", kind: str = "", date_from: str = "", date_to: str = "",
                        include_voided: bool = False) -> dict:
        flt: dict = {}
        q = re.sub(r"[^A-Za-z0-9]", "", q or "").upper()
        if q:
            flt["$or"] = [{"plate": {"$regex": re.escape(q)}}, {"slip_code": {"$regex": re.escape(q)}}]
        if kind in ("slip", "whitelisted"):
            flt["kind"] = kind
        elif kind in GROUPS:
            flt["kind"] = "slip"
            flt["vehicle_group"] = kind
        if not include_voided:
            flt["voided"] = False
        rng = self._day_range(date_from, date_to)
        if rng:
            flt["created_at"] = rng
        return flt

    def _offline_filter(self, q, kind, date_from, date_to, include_voided) -> list[dict]:
        qn = re.sub(r"[^A-Za-z0-9]", "", q or "").upper()
        rng = self._day_range(date_from, date_to)
        out = []
        with self._lock:
            for e in self.recent.values():
                if qn and qn not in e["plate"] and qn not in (e.get("slip_code") or ""):
                    continue
                if kind in ("slip", "whitelisted") and e["kind"] != kind:
                    continue
                if kind in GROUPS and (e["kind"] != "slip" or e["vehicle_group"] != kind):
                    continue
                if not include_voided and e["voided"]:
                    continue
                ts = as_aware(e["created_at"])
                if "$gte" in rng and ts < rng["$gte"]:
                    continue
                if "$lt" in rng and ts >= rng["$lt"]:
                    continue
                out.append(e)
        out.sort(key=lambda e: e["created_at"], reverse=True)
        return out

    def query_entries(self, q: str = "", kind: str = "", date_from: str = "", date_to: str = "",
                      include_voided: bool = False, skip: int = 0, limit: int = 50) -> tuple[list[dict], int, bool]:
        """Returns (rows, total, complete). complete=False means MongoDB is offline and only the most
        recent entries held in memory are shown."""
        if self.online:
            try:
                with self._io_lock:
                    flt = self._history_filter(q, kind, date_from, date_to, include_voided)
                    total = self.db["entries"].count_documents(flt)
                    rows = list(self.db["entries"].find(flt).sort("created_at", DESCENDING).skip(skip).limit(limit))
                return rows, total, True
            except PyMongoError as exc:
                log.warning("History query failed: %s", exc)
                self.online = False
        rows = self._offline_filter(q, kind, date_from, date_to, include_voided)
        return rows[skip:skip + limit], len(rows), False

    def export_entries(self, q="", kind="", date_from="", date_to="", include_voided=False,
                       cap: int = 200000) -> list[dict]:
        rows, _total, complete = self.query_entries(q, kind, date_from, date_to, include_voided, 0, cap)
        if not complete:
            raise StoreError("MongoDB is offline - history export is not available right now")
        return rows

    def hourly_summary(self, date_str: str) -> dict:
        """Entries per hour (local time) for one day, plus the totals per vehicle group."""
        rng = self._day_range(date_str, date_str)
        if not rng:
            raise StoreError("Please give a date like 2026-10-03")
        rows: list[dict] = []
        complete = True
        if self.online:
            try:
                with self._io_lock:
                    rows = list(self.db["entries"].find(
                        {"created_at": rng, "voided": False}, {"created_at": 1, "kind": 1, "vehicle_group": 1}))
            except PyMongoError:
                self.online = False
        if not self.online:
            complete = False
            rows = self._offline_filter("", "", date_str, date_str, False)
        hours = [{"hour": h, "car": 0, "bike": 0, "other": 0, "whitelisted": 0} for h in range(24)]
        totals = {"car": 0, "bike": 0, "other": 0, "whitelisted": 0}
        for e in rows:
            h = as_aware(e["created_at"]).astimezone().hour
            key = "whitelisted" if e.get("kind") == "whitelisted" else e.get("vehicle_group", "other")
            hours[h][key] += 1
            totals[key] += 1
        return {"date": date_str, "hours": hours, "totals": totals, "complete": complete}
