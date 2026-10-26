"""Slip picture + printing.

The slip is drawn as a picture (so any text, the QR code and any paper width work the same everywhere)
and then sent to the printer in one of four ways:

  file         nothing is printed; the picture is saved in data/slips (default until a printer is set up)
  windows_gdi  any printer installed in Windows, through its normal driver
  windows_raw  ESC/POS thermal printer installed in Windows (faster, cuts the paper)
  network      ESC/POS thermal printer on the network (host + port 9100)

Printing runs in its own thread so a slow or jammed printer never delays the cameras.
"""
from __future__ import annotations

import json
import logging
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .plates import display
from .store import Store, StoreError, as_aware

log = logging.getLogger("printing")

MODES = ("file", "windows_gdi", "windows_raw", "network")
PAPER_PIXELS = {80: 576, 58: 384}      # printable dots at 203 dpi


# ---------------------------------------------------------------------------------------- drawing
_FONT_CANDIDATES = {
    True: ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf"],
    False: ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "FreeSans.ttf"],
}
_font_cache: dict = {}


def _font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    font = None
    for name in _FONT_CANDIDATES[bold]:
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size)
        except TypeError:   # very old Pillow
            font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def _qr_image(text: str, size: int) -> Image.Image:
    import qrcode
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=1)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img = img.get_image() if hasattr(img, "get_image") else img
    return img.convert("L").resize((size, size), Image.NEAREST)


def render_slip(entry: dict, slip_cfg: dict, paper_mm: int = 80, reprint: bool = False) -> Image.Image:
    """Draws the slip. `entry` needs: slip_code, plate, vehicle_type, created_at."""
    width = PAPER_PIXELS.get(int(paper_mm), 576)
    s = width / 576.0
    pad = int(22 * s)
    img = Image.new("L", (width, int(1500 * s)), 255)
    d = ImageDraw.Draw(img)
    y = int(18 * s)

    def centered(text: str, size: int, bold: bool = True, gap: int = 8, max_w: int | None = None) -> None:
        nonlocal y
        max_w = max_w or (width - 2 * pad)
        sz = int(size * s)
        font = _font(sz, bold)
        while d.textlength(text, font=font) > max_w and sz > 12:
            sz -= 2
            font = _font(sz, bold)
        w = d.textlength(text, font=font)
        d.text(((width - w) / 2, y), text, font=font, fill=0)
        y += int(sz * 1.18) + int(gap * s)

    def rule(dashed: bool = True) -> None:
        nonlocal y
        if dashed:
            for x in range(pad, width - pad, int(14 * s)):
                d.line([(x, y), (x + int(8 * s), y)], fill=0, width=max(2, int(2 * s)))
        else:
            d.line([(pad, y), (width - pad, y)], fill=0, width=max(2, int(3 * s)))
        y += int(14 * s)

    title = (slip_cfg.get("title") or "").strip()
    if title:
        centered(title, 40, True, 6)
        rule(False)
    centered("SLIP NO.", 24, False, 0)
    centered(str(entry["slip_code"]), 84, True, 6)
    rule()
    centered("PLATE", 24, False, 0)
    centered(display(entry["plate"]) or "-", 66, True, 6)
    centered(f"VEHICLE: {entry.get('vehicle_type', 'VEHICLE')}", 34, True, 10)
    ts = as_aware(entry["created_at"]).astimezone()
    centered(ts.strftime("%d-%b-%Y   %H:%M:%S"), 32, False, 10)
    rule()
    qr_size = int(min(width - 2 * pad, 300 * s))
    img.paste(_qr_image(str(entry["slip_code"]), qr_size), ((width - qr_size) // 2, y))
    y += qr_size + int(6 * s)
    centered(str(entry["slip_code"]), 26, False, 6)
    for line in slip_cfg.get("footer_lines") or []:
        centered(str(line), 26, False, 2)
    if reprint:
        y += int(6 * s)
        centered("*** REPRINT ***", 28, True, 4)
    y += int(34 * s)
    return img.crop((0, 0, width, y))


# ---------------------------------------------------------------------------------------- ESC/POS
def escpos_bytes(img: Image.Image, cut: bool = True, feed_lines: int = 4) -> bytes:
    """Picture -> ESC/POS raster commands (works on practically every thermal receipt printer)."""
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    if w % 8:
        gray = np.pad(gray, ((0, 0), (0, 8 - w % 8)), constant_values=255)
        w = gray.shape[1]
    black = (gray < 160).astype(np.uint8)
    packed = np.packbits(black, axis=1)            # 1 = black dot
    wb = w // 8
    out = bytearray(b"\x1b\x40")                   # initialise
    out += b"\x1b\x61\x01"                         # centre
    band = 128
    for top in range(0, h, band):
        chunk = packed[top:top + band]
        rows = chunk.shape[0]
        out += b"\x1d\x76\x30\x00" + bytes([wb & 255, wb >> 8, rows & 255, rows >> 8]) + chunk.tobytes()
    out += b"\x1b\x64" + bytes([feed_lines])       # feed
    if cut:
        out += b"\x1d\x56\x42\x00"                 # feed to cut position + partial cut
    return bytes(out)


def list_windows_printers() -> tuple[list[str], str]:
    """Names of the printers installed in Windows, and the default one. ([], '') on other systems."""
    if not sys.platform.startswith("win"):
        return [], ""
    try:
        import win32print
        flags = win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS
        names = sorted({p[2] for p in win32print.EnumPrinters(flags)})
        try:
            default = win32print.GetDefaultPrinter()
        except Exception:  # noqa: BLE001
            default = ""
        return names, default
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not list Windows printers: %s", exc)
        return [], ""


def _send_windows_raw(name: str, data: bytes) -> None:
    try:
        import win32print
    except ImportError as exc:
        raise RuntimeError("pywin32 is not installed (run setup.bat again)") from exc
    name = name or win32print.GetDefaultPrinter()
    handle = win32print.OpenPrinter(name)
    try:
        win32print.StartDocPrinter(handle, 1, ("Parking slip", None, "RAW"))
        try:
            win32print.StartPagePrinter(handle)
            win32print.WritePrinter(handle, data)
            win32print.EndPagePrinter(handle)
        finally:
            win32print.EndDocPrinter(handle)
    finally:
        win32print.ClosePrinter(handle)


def _send_windows_gdi(name: str, img: Image.Image) -> None:
    try:
        import win32print
        import win32ui
        from PIL import ImageWin
    except ImportError as exc:
        raise RuntimeError("pywin32 is not installed (run setup.bat again)") from exc
    name = name or win32print.GetDefaultPrinter()
    hdc = win32ui.CreateDC()
    hdc.CreatePrinterDC(name)
    try:
        printable_w = hdc.GetDeviceCaps(8)       # HORZRES
        scale = printable_w / img.width
        height = int(img.height * scale)
        hdc.StartDoc("Parking slip")
        hdc.StartPage()
        ImageWin.Dib(img.convert("RGB")).draw(hdc.GetHandleOutput(), (0, 0, printable_w, height))
        hdc.EndPage()
        hdc.EndDoc()
    finally:
        hdc.DeleteDC()


def _send_network(host: str, port: int, data: bytes) -> None:
    if not host:
        raise RuntimeError("No printer address set (Settings > Printer)")
    with socket.create_connection((host, int(port)), timeout=6) as sock:
        sock.sendall(data)
        time.sleep(0.2)


def send_to_printer(img: Image.Image, p: dict) -> str:
    """Sends the picture using the configured mode. Returns the status word ('printed' or 'saved')."""
    mode = p.get("mode", "file")
    if mode == "file":
        return "saved"
    cut = bool(p.get("auto_cut", True))
    if mode == "windows_gdi":
        _send_windows_gdi(p.get("name", ""), img)
    elif mode == "windows_raw":
        _send_windows_raw(p.get("name", ""), escpos_bytes(img, cut))
    elif mode == "network":
        _send_network(p.get("network_host", ""), p.get("network_port", 9100), escpos_bytes(img, cut))
    else:
        raise RuntimeError(f"Unknown printer mode: {mode}")
    return "printed"


# ---------------------------------------------------------------------------------------- settings
class PrinterSettings:
    """config.yaml values, overridden by whatever was saved from the Settings tab (data/settings.json)."""

    def __init__(self, cfg: dict):
        self._lock = threading.Lock()
        self._path = Path(cfg["storage"]["data_dir"]) / "settings.json"
        base = {"mode": "file", "name": "", "network_host": "", "network_port": 9100,
                "paper_width_mm": 80, "auto_cut": True}
        base.update(cfg.get("printer") or {})
        self._data = base
        try:
            saved = json.loads(self._path.read_text(encoding="utf-8")).get("printer", {})
            self._data.update(saved)
        except (FileNotFoundError, ValueError):
            pass

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    def update(self, new: dict) -> dict:
        with self._lock:
            d = dict(self._data)
            if "mode" in new:
                if new["mode"] not in MODES:
                    raise StoreError("Unknown printer mode")
                d["mode"] = new["mode"]
            if "name" in new:
                d["name"] = str(new["name"] or "")[:200]
            if "network_host" in new:
                d["network_host"] = str(new["network_host"] or "").strip()[:200]
            if "network_port" in new:
                try:
                    d["network_port"] = int(new["network_port"] or 9100)
                except (TypeError, ValueError) as exc:
                    raise StoreError("Port must be a number") from exc
            if "paper_width_mm" in new:
                try:
                    w = int(new["paper_width_mm"])
                except (TypeError, ValueError) as exc:
                    raise StoreError("Paper width must be 80 or 58") from exc
                if w not in PAPER_PIXELS:
                    raise StoreError("Paper width must be 80 or 58")
                d["paper_width_mm"] = w
            if "auto_cut" in new:
                d["auto_cut"] = bool(new["auto_cut"])
            self._data = d
            try:
                self._path.write_text(json.dumps({"printer": d}, indent=2), encoding="utf-8")
            except OSError as exc:
                log.warning("Could not save printer settings: %s", exc)
            return dict(d)


# ---------------------------------------------------------------------------------------- queue
class PrintQueue(threading.Thread):
    def __init__(self, store: Store, settings: PrinterSettings, slip_cfg: dict, data_dir: str):
        super().__init__(name="print-queue", daemon=True)
        self.store = store
        self.settings = settings
        self.slip_cfg = slip_cfg
        self.slips_dir = Path(data_dir) / "slips"
        self.slips_dir.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue = queue.Queue()
        self._stop_evt = threading.Event()
        self.last_error = ""
        self.retries = 2
        self.retry_wait_s = 2.0

    def enqueue(self, entry_id: str, reprint: bool = False) -> None:
        self._q.put((entry_id, reprint))

    def status(self) -> dict:
        p = self.settings.get()
        return {"queued": self._q.qsize(), "mode": p["mode"], "last_error": self.last_error}

    def stop(self) -> None:
        self._stop_evt.set()
        self._q.put(None)

    def slip_path(self, entry: dict) -> Path:
        return self.slips_dir / f"{entry['slip_code']}.png"

    def run(self) -> None:
        while not self._stop_evt.is_set():
            job = self._q.get()
            if job is None:
                break
            try:
                self._handle(*job)
            except Exception:  # noqa: BLE001 - the print thread must never die
                log.exception("print job crashed")

    def _handle(self, entry_id: str, reprint: bool) -> None:
        entry = self.store.get_entry(entry_id)
        if entry is None or entry.get("voided"):
            return
        p = self.settings.get()
        img = render_slip(entry, self.slip_cfg, p["paper_width_mm"], reprint=reprint)
        try:
            img.save(self.slip_path(entry))
        except OSError as exc:
            log.warning("Could not save slip picture: %s", exc)
        last = ""
        for attempt in range(self.retries + 1):
            try:
                status = send_to_printer(img, p)
                self.last_error = ""
                self.store.set_print_result(entry_id, True, status)
                log.info("Slip %s %s", entry["slip_code"], "printed" if status == "printed" else "saved (file mode)")
                return
            except Exception as exc:  # noqa: BLE001
                last = str(exc) or exc.__class__.__name__
                log.warning("Printing slip %s failed (attempt %d): %s", entry["slip_code"], attempt + 1, last)
                if attempt < self.retries:
                    time.sleep(self.retry_wait_s)
        self.last_error = last
        self.store.add_alert("printer_error",
                             f"Slip {entry['slip_code']} ({display(entry['plate'])}) did NOT print: {last}. "
                             f"Press Reprint once the printer is ready.",
                             plate=entry["plate"], camera=entry.get("camera"), snapshot=entry.get("snapshot"),
                             severity="danger", entry_id=entry_id)
        self.store.set_print_result(entry_id, False, "failed", last)

    def test_print(self) -> tuple[bool, str]:
        """Synchronous test print from the Settings tab."""
        from datetime import datetime, timezone
        p = self.settings.get()
        fake = {"slip_code": "TEST", "plate": "TEST123", "vehicle_type": "CAR",
                "created_at": datetime.now(timezone.utc)}
        img = render_slip(fake, self.slip_cfg, p["paper_width_mm"])
        try:
            img.save(self.slips_dir / "TEST.png")
            status = send_to_printer(img, p)
        except Exception as exc:  # noqa: BLE001
            return False, f"Test print failed: {exc}"
        if status == "saved":
            return True, "Printer mode is 'file': the test slip was saved in data/slips/TEST.png (nothing was printed)."
        return True, "Test slip sent to the printer."
