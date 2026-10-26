"""The pipeline's decision logic (fusion -> slip), driven by a fake OCR so every case is exact and fast."""
from types import SimpleNamespace

import numpy as np
import pytest
from test_pk_formats import probs_for

from app.config import load_config
from app.main import build_plate_format
from app.pipeline import CameraWorker
from app.slips import CAM1, CAM2, SlipEngine
from app.tracker import Obj


class FakeReader:
    """Pretends to be the OCR model: always returns the same position probabilities."""

    def __init__(self, probs):
        self.probs = probs

    def infer(self, images):
        return np.stack([self.probs] * len(images)).astype(np.float32)


@pytest.fixture
def make_worker(store, cfg, tmp_path):
    def _make(probs, cam=CAM1):
        full = load_config()
        full["storage"]["data_dir"] = str(tmp_path)
        full["vehicles"]["must_be_moving"] = False
        engine = SlipEngine(store, None, full["matching"])
        rec = SimpleNamespace(reader=FakeReader(probs), plates=None, vehicles=None, ocr=None, is_gpu=False)
        w = CameraWorker(cam, SimpleNamespace(status="online", last_frame_time=0.0), rec, engine, store,
                         build_plate_format(full), full)
        return w
    return _make


FRAME = np.full((720, 1280, 3), 120, np.uint8)
BOX, PLATE = (300, 200, 800, 650), (480, 520, 640, 580)


def feed(w, n, now0=0.0, cls="motorcycle"):
    track = None
    for i in range(n):
        obj = Obj(box=BOX, cls=cls, conf=0.9, plate_box=PLATE, plate_conf=0.9)
        (track,), _ = w.tracker.update([obj], now0 + i * 0.2)
        track.full_view = True
        w._read_plate(track, obj, FRAME)
        w._try_finalize(track, now0 + i * 0.2, force=False)
    return track


def slips(store):
    return list(store.db["entries"].find({"kind": "slip"}))


def test_confident_look_alike_is_fixed_by_the_grammar(make_worker, store):
    w = make_worker(probs_for("JML2472", overrides={3: {"Z": 0.90, "2": 0.05}}))
    t = feed(w, 4)
    assert t.finalized and [s["plate"] for s in slips(store)] == ["JML2472"]
    assert slips(store)[0]["vehicle_type"] == "BIKE"


def test_slip_is_issued_after_two_agreeing_reads_not_one(make_worker, store):
    w = make_worker(probs_for("LEA1234"))
    feed(w, 1)
    assert slips(store) == []
    feed(w, 3, now0=1.0)
    assert [s["plate"] for s in slips(store)] == ["LEA1234"]


def test_vehicle_leaving_after_one_read_still_gets_its_slip(make_worker, store):
    w = make_worker(probs_for("LEA1234"))
    t = feed(w, 1)
    assert slips(store) == []
    w._lost(t, 5.0)
    assert [s["plate"] for s in slips(store)] == ["LEA1234"]


def test_two_row_plate_year_format_is_issued_as_one_plate(make_worker, store):
    w = make_worker(probs_for("LES181573"))
    feed(w, 3)
    assert [s["plate"] for s in slips(store)] == ["LES181573"]


def test_unusual_plate_still_gets_a_slip_with_a_check_note(make_worker, store):
    w = make_worker(probs_for("AB1C23D", conf=0.995))          # fits no Pakistani layout
    t = feed(w, 5)
    s = slips(store)
    assert t.finalized and len(s) == 1 and s[0]["plate"] == "AB1C23D"
    assert "unusual" in s[0]["notes"]
    assert any(a["type"] == "check" and a["severity"] == "info" for a in store.alerts)


def test_unusual_plate_needs_a_vehicle_not_just_a_sign(make_worker, store):
    w = make_worker(probs_for("AB1C23D", conf=0.995))
    track = None
    for i in range(6):
        obj = Obj(box=(480, 520, 640, 580), cls=None, plate_box=PLATE, plate_conf=0.9)      # plate only, no vehicle box
        (track,), _ = w.tracker.update([obj], i * 0.2)
        w._read_plate(track, obj, FRAME)
        w._try_finalize(track, i * 0.2, force=False)
    assert slips(store) == []


def test_unusual_plates_can_be_switched_off(make_worker, store):
    w = make_worker(probs_for("AB1C23D", conf=0.995))
    w.allow_unusual = False
    feed(w, 6)
    assert slips(store) == []


def test_unreadable_reading_gives_no_slip_and_a_red_alert_at_camera2(make_worker, store):
    flat = np.full((10, 37), 1 / 37, np.float32)
    w = make_worker(flat, cam=CAM2)
    t = feed(w, 5)
    assert slips(store) == [] and not t.finalized
    w._lost(t, 9.0)
    assert any(a["type"] == "unreadable_vehicle" and a["severity"] == "danger" for a in store.alerts)


def test_ambiguous_reading_waits_for_more_frames(make_worker, store):
    w = make_worker(probs_for("LEA1234", overrides={6: {"4": 0.46, "9": 0.44}}))     # last digit is a coin flip
    t = feed(w, 3)
    assert not t.finalized and slips(store) == []
    feed_more = feed(w, 4, now0=2.0)
    assert feed_more.finalized                                   # after min_reads + 3 reads it decides anyway


def test_evidence_is_capped(make_worker):
    w = make_worker(probs_for("LEA1234"))
    t = feed(w, 30)
    assert len(t.evidence) <= w.max_evidence
