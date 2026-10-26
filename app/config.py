"""Loads config.yaml and fills in defaults."""
from __future__ import annotations

from pathlib import Path

import yaml

from . import ROOT


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("cameras", {})
    cfg.setdefault("mongodb", {})
    cfg["mongodb"].setdefault("uri", "mongodb://localhost:27017")
    cfg["mongodb"].setdefault("database", "parking_system")
    cfg["mongodb"].setdefault("server_selection_timeout_ms", 2000)
    cfg.setdefault("web", {})
    cfg["web"].setdefault("host", "127.0.0.1")
    cfg["web"].setdefault("port", 5000)
    cfg.setdefault("models", {})
    cfg.setdefault("recognition", {})
    cfg.setdefault("vehicles", {})
    cfg.setdefault("motion", {})
    cfg.setdefault("matching", {})
    cfg.setdefault("printer", {})
    cfg.setdefault("slip", {})
    cfg["slip"].setdefault("title", "PARKING SLIP")
    cfg["slip"].setdefault("footer_lines", [])
    cfg["slip"].setdefault("number_start", 1)
    cfg["slip"].setdefault("number_digits", 6)
    cfg["slip"].setdefault("number_prefix", "")
    cfg.setdefault("validation", {"enabled": True, "patterns": []})
    cfg.setdefault("alerts", {})
    cfg["alerts"].setdefault("camera_offline_seconds", 20)
    cfg.setdefault("storage", {})
    cfg["storage"].setdefault("data_dir", "data")
    cfg["storage"].setdefault("jpeg_quality", 85)

    data_dir = Path(cfg["storage"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    cfg["storage"]["data_dir"] = str(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "snapshots").mkdir(exist_ok=True)
    (data_dir / "slips").mkdir(exist_ok=True)

    m = cfg["models"]
    for key in ("plate_detector", "ocr_model", "ocr_config", "vehicle_detector"):
        p = Path(m.get(key, ""))
        if key in m and not p.is_absolute():
            m[key] = str(ROOT / p)
    return cfg
