"""Starts the whole system:  python -m app.main   (or run.bat)"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
import threading
import webbrowser
from pathlib import Path


def setup_logging(data_dir: str) -> None:
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    fileh = logging.handlers.RotatingFileHandler(Path(data_dir) / "parking.log", maxBytes=2_000_000,
                                                 backupCount=3, encoding="utf-8")
    fileh.setFormatter(fmt)
    root.addHandler(fileh)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def build_plate_format(cfg: dict):
    """PlateFormat from config.yaml > validation (Pakistani layouts + optional extras)."""
    from .pk_formats import templates_from_config
    from .plates import PlateFormat
    v = cfg.get("validation", {})
    return PlateFormat(v.get("enabled", True), v.get("patterns"), min_length=int(v.get("min_length", 4)),
                       templates=templates_from_config(v.get("extra_templates"), v.get("disabled_templates")),
                       allow_unusual=bool(v.get("allow_unusual", True)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Parking entrance system")
    ap.add_argument("--config", help="path to a config file (default: config.yaml)")
    ap.add_argument("--check", action="store_true", help="test MongoDB, models, printer and cameras, then exit")
    args = ap.parse_args(argv)

    from .config import load_config
    cfg = load_config(args.config)
    setup_logging(cfg["storage"]["data_dir"])
    log = logging.getLogger("main")

    from .camera import FrameSource
    from .plates import PlateFormat
    from .printing import PrinterSettings, PrintQueue
    from .recognizer import build_recognizers
    from .slips import CAM1, CAM2, SlipEngine
    from .store import Store

    cams_cfg = cfg["cameras"]
    for need in (CAM1, CAM2):
        if need not in cams_cfg or "source" not in cams_cfg[need]:
            log.error("config.yaml must define cameras: %s: source", need)
            return 1

    store = Store(cfg)
    settings = PrinterSettings(cfg)

    if args.check:
        return run_check(cfg, store, settings, build_recognizers, FrameSource)

    store.start()
    printq = PrintQueue(store, settings, cfg["slip"], cfg["storage"]["data_dir"])
    printq.start()
    fmt = build_plate_format(cfg)
    engine = SlipEngine(store, printq, cfg["matching"])

    log.info("Loading plate and vehicle recognition models (this takes a few seconds) ...")
    try:
        recs = {name: build_recognizers(cfg["models"], cfg["vehicles"].get("confidence", 0.45))
                for name in (CAM1, CAM2)}
    except Exception as exc:  # noqa: BLE001
        log.error("Could not load the models: %s", exc)
        return 1

    from .pipeline import CameraWorker
    sources, workers = {}, {}
    for name in (CAM1, CAM2):
        c = cams_cfg[name]
        sources[name] = FrameSource(name, c["source"], bool(c.get("loop_video", True)))
        workers[name] = CameraWorker(name, sources[name], recs[name], engine, store, fmt, cfg, roi=c.get("roi"))
    engine.set_sources(sources)
    for name in (CAM1, CAM2):
        sources[name].start()
        workers[name].start()

    from .web import create_app
    app = create_app(cfg, store, printq, settings, workers, fmt)
    host, port = cfg["web"]["host"], int(cfg["web"]["port"])
    url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}"
    log.info("Station screen: %s   (press Ctrl+C to stop)", url)
    if cfg["web"].get("open_browser", True):
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down ...")
        for w in workers.values():
            w.stop()
        for s in sources.values():
            s.stop()
        printq.stop()
        store.stop()
    return 0


def run_check(cfg, store, settings, build_recognizers, FrameSource) -> int:
    """Quick self-test: prints OK / PROBLEM for MongoDB, models, printer and both cameras."""
    from .printing import PrintQueue
    ok = True
    print("\n--- Parking entrance system: self-check ---")
    store._connect()
    print(f"MongoDB   : {'OK - connected' if store.online else 'PROBLEM - cannot connect (is the MongoDB service running?)'}")
    ok &= store.online
    try:
        build_recognizers(cfg["models"], cfg["vehicles"].get("confidence", 0.45))
        print("Models    : OK - vehicle detector, plate detector and plate reader loaded")
    except Exception as exc:  # noqa: BLE001
        print(f"Models    : PROBLEM - {exc}")
        ok = False
    p = settings.get()
    if p["mode"] == "file":
        print("Printer   : not set up yet (slips are saved as pictures in data/slips) - set it up in the Settings tab")
    else:
        good, msg = PrintQueue(store, settings, cfg["slip"], cfg["storage"]["data_dir"]).test_print()
        print(f"Printer   : {'OK - ' if good else 'PROBLEM - '}{msg}")
        ok &= good
    for name in ("camera1", "camera2"):
        src = FrameSource(name, cfg["cameras"][name]["source"], True)
        src.start()
        frame, _ = src.read(0, timeout=12.0)
        src.stop()
        if frame is not None:
            print(f"{name:<9} : OK - picture received ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"{name:<9} : PROBLEM - no picture from {src._safe_source()}")
            ok = False
    print("\nAll good." if ok else "\nSome checks failed - see the lines marked PROBLEM above.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
