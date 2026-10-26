import socket
import threading
import time

import numpy as np
from conftest import ts
from PIL import Image

from app.printing import (PAPER_PIXELS, PrinterSettings, PrintQueue, escpos_bytes, render_slip, send_to_printer)
from app.slips import CAM1
from app.store import StoreError

ENTRY = {"slip_code": "000123", "plate": "LEA1234", "vehicle_type": "BIKE", "created_at": ts()}
SLIP_CFG = {"title": "PARKING SLIP", "footer_lines": ["Thank you"]}


def test_slip_is_rendered_for_both_paper_widths():
    for mm, px in PAPER_PIXELS.items():
        img = render_slip(ENTRY, SLIP_CFG, mm)
        assert img.width == px and 500 < img.height < 1400
        assert np.array(img).min() < 50                       # something dark was drawn


def test_reprint_marker_changes_the_picture():
    a = np.array(render_slip(ENTRY, SLIP_CFG, 80))
    b = np.array(render_slip(ENTRY, SLIP_CFG, 80, reprint=True))
    assert b.shape[0] > a.shape[0]


def test_escpos_bytes_structure():
    img = Image.new("L", (576, 300), 255)
    data = escpos_bytes(img, cut=True)
    assert data.startswith(b"\x1b\x40") and b"\x1d\x76\x30\x00" in data and data.endswith(b"\x1d\x56\x42\x00")
    assert b"\x1d\x56" not in escpos_bytes(img, cut=False)
    # 576 px wide = 72 bytes per row; 300 rows in bands of 128 -> 3 raster commands
    assert data.count(b"\x1d\x76\x30\x00") == 3


def test_network_mode_sends_escpos_to_a_printer():
    received = bytearray()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        with conn:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                received.extend(chunk)

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    img = render_slip(ENTRY, SLIP_CFG, 80)
    status = send_to_printer(img, {"mode": "network", "network_host": "127.0.0.1", "network_port": port,
                                   "auto_cut": True})
    th.join(3)
    srv.close()
    assert status == "printed" and received.startswith(b"\x1b\x40") and len(received) > 5000


def test_windows_modes_fail_cleanly_on_other_systems():
    import pytest
    img = Image.new("L", (576, 100), 255)
    with pytest.raises(Exception):
        send_to_printer(img, {"mode": "windows_gdi", "name": "x"})


def test_printer_settings_validation_and_persistence(cfg):
    s = PrinterSettings(cfg)
    assert s.get()["mode"] == "file"
    s.update({"mode": "network", "network_host": "10.0.0.5", "paper_width_mm": "58"})
    assert PrinterSettings(cfg).get()["paper_width_mm"] == 58          # survives a restart
    for bad in ({"mode": "laser"}, {"paper_width_mm": 70}, {"network_port": "abc"}):
        try:
            s.update(bad)
            raise AssertionError("should have been rejected")
        except StoreError:
            pass


def test_failed_print_raises_a_loud_alert_and_can_be_retried(store, cfg):
    cfg["printer"] = {"mode": "network", "network_host": "127.0.0.1", "network_port": 1, "paper_width_mm": 80}
    q = PrintQueue(store, PrinterSettings(cfg), cfg["slip"], cfg["storage"]["data_dir"])
    q.retries, q.retry_wait_s = 1, 0
    q.start()
    e = store.create_entry("slip", "LEA1234", "CAR", CAM1, 0.9, ts())
    q.enqueue(e["_id"])
    for _ in range(100):
        if store.get_entry(e["_id"])["print_status"] == "failed":
            break
        time.sleep(0.05)
    assert store.get_entry(e["_id"])["print_status"] == "failed"
    a = [x for x in store.alerts if x["type"] == "printer_error"]
    assert a and a[0]["severity"] == "danger" and a[0]["entry_id"] == e["_id"]
    assert q.last_error
    q.stop()


def test_test_print_in_file_mode(store, cfg):
    q = PrintQueue(store, PrinterSettings(cfg), cfg["slip"], cfg["storage"]["data_dir"])
    ok, msg = q.test_print()
    assert ok and "file" in msg and (store.data_dir / "slips" / "TEST.png").exists()
