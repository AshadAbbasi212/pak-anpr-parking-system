import sys
from datetime import datetime, timezone
from pathlib import Path

import mongomock
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import ROOT as _R  # noqa: E402,F401  (adds third_party to sys.path)
from app.printing import PrinterSettings, PrintQueue  # noqa: E402
from app.slips import SlipEngine  # noqa: E402
from app.store import Store  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "slips").mkdir()
    return {
        "storage": {"data_dir": str(tmp_path), "jpeg_quality": 80},
        "mongodb": {"uri": "mongodb://x", "database": "t", "server_selection_timeout_ms": 100},
        "alerts": {"camera_offline_seconds": 20},
        "slip": {"title": "PARKING SLIP", "footer_lines": ["Thank you"], "number_start": 1,
                 "number_digits": 6, "number_prefix": ""},
        "printer": {"mode": "file", "paper_width_mm": 80},
        "matching": {"travel_time_min_s": 2, "travel_time_max_s": 30, "repeat_protection_s": 60,
                     "fuzzy_repeat_s": 6, "unsure_max_cost": 1.5},
    }


@pytest.fixture
def store(cfg):
    s = Store(cfg, client=mongomock.MongoClient(tz_aware=True))
    s.start()
    yield s
    s.stop()


@pytest.fixture
def printq(store, cfg):
    q = PrintQueue(store, PrinterSettings(cfg), cfg["slip"], cfg["storage"]["data_dir"])
    q.retry_wait_s = 0
    q.start()
    yield q
    q.stop()


@pytest.fixture
def engine(store, printq, cfg):
    return SlipEngine(store, printq, cfg["matching"])


def snap():
    return ("snap.jpg", "crop.jpg")


def ts():
    return datetime.now(timezone.utc)
