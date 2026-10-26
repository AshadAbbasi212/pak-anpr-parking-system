"""End-to-end smoke test with the REAL models and simulated camera videos.
Needs:  pip install mongomock      Run from the project folder:  python tests/e2e_smoke.py [scenario]
Scenarios: both (default) | cam2only | noplate"""
import sys
import time
from pathlib import Path

import cv2
import mongomock
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.camera import FrameSource                       # noqa: E402
from app.config import load_config                       # noqa: E402
from app.pipeline import CameraWorker                    # noqa: E402
from app.plates import PlateFormat                       # noqa: E402
from app.pk_formats import Template                      # noqa: E402
from app.printing import PrinterSettings, PrintQueue     # noqa: E402
from app.recognizer import build_recognizers             # noqa: E402
from app.slips import CAM1, CAM2, SlipEngine             # noqa: E402
from app.store import Store                              # noqa: E402

scenario = sys.argv[1] if len(sys.argv) > 1 else "both"
OUT = Path("/tmp/e2e"); OUT.mkdir(exist_ok=True)
FPS, W, H = 25, 1280, 720

src_img = cv2.imread(str(ROOT / "tests" / "sample_plate.png"))
car = src_img[20:300, 20:470].copy()                      # just the grey car with the plate 5AU5341
px1, py1, px2, py2 = 306 - 20, 187 - 20, 387 - 20, 209 - 20
if scenario == "noplate":                                  # hide the plate (paint it body-coloured)
    pad = 4
    patch = car[py1 - 12:py1 - 6, px1:px2].mean(axis=(0, 1))
    car[py1 - pad:py2 + pad, px1 - pad:px2 + pad] = patch
car = cv2.resize(car, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
ch, cw = car.shape[:2]
bg = np.full((H, W, 3), 70, np.uint8)
cv2.rectangle(bg, (0, H - 160), (W, H), (90, 90, 90), -1)   # "road"


def make_video(path, delay_s, present=True, drive_s=10.0):   # slow drive-by so slow test computers still get enough frames
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for _ in range(int(delay_s * FPS)):
        vw.write(bg)
    if present:
        n = int(drive_s * FPS)
        for i in range(n):
            f = bg.copy()
            x = int(W - (W + cw) * i / n)                  # right -> left
            y = H - ch - 20
            x1, x2 = max(0, x), min(W, x + cw)
            if x2 > x1:
                f[y:y + ch, x1:x2] = car[:, x1 - x:x2 - x]
            vw.write(f)
    for _ in range(int(3 * FPS)):
        vw.write(bg)
    vw.release()


v1, v2 = OUT / "cam1.avi", OUT / "cam2.avi"
make_video(v1, 1.0, present=(scenario == "both"))
make_video(v2, 1.0 + (8.0 if scenario == "both" else 0.0), present=True)

cfg = load_config()
data = OUT / f"data_{scenario}"
import shutil; shutil.rmtree(data, ignore_errors=True)
(data / "snapshots").mkdir(parents=True); (data / "slips").mkdir()
cfg["storage"]["data_dir"] = str(data)
cfg["mongodb"]["database"] = "e2e"
cfg["models"]["cpu_threads"] = 1
cfg["models"]["device"] = "cpu"
store = Store(cfg, client=mongomock.MongoClient(tz_aware=True)); store.start()
settings = PrinterSettings(cfg)
printq = PrintQueue(store, settings, cfg["slip"], str(data)); printq.start()
engine = SlipEngine(store, printq, cfg["matching"])
fmt = PlateFormat(True, [], templates=[Template('NLLNNNN', (3, 4), 0.0, 'Czech sample photo')])   # sample photo is a Czech plate
recs = {n: build_recognizers(cfg["models"], 0.45) for n in (CAM1, CAM2)}
srcs = {CAM1: FrameSource(CAM1, str(v1), loop_video=False), CAM2: FrameSource(CAM2, str(v2), loop_video=False)}
workers = {n: CameraWorker(n, srcs[n], recs[n], engine, store, fmt, cfg) for n in (CAM1, CAM2)}
t0 = time.time()
for n in (CAM1, CAM2):
    srcs[n].start(); workers[n].start()

shots = {}
while time.time() - t0 < 40:
    time.sleep(0.4)
    for n, w in workers.items():
        jpeg, _ = w.get_jpeg(0, 0.05)
        if jpeg and n not in shots and any(t.decision for t in w.tracker.tracks + [g[0] for g in w._ghosts]):
            shots[n] = jpeg                       # keep the first picture that shows a decision
for n, j in shots.items():
    (OUT / f"{scenario}_{n}.jpg").write_bytes(j)
for w in workers.values(): w.stop()
for s in srcs.values(): s.stop()
time.sleep(0.5)

print(f"\n=== scenario: {scenario} ===")
print("live events:", [(e["camera"], e["label"]) for e in reversed(store.events)])
print("alerts     :", [(a["type"], a["severity"]) for a in store.alerts])
print("slips      :", [(e["slip_code"], e["plate"], e["camera"], e["vehicle_type"], e["print_status"])
                      for e in store.db["entries"].find({"kind": "slip"})])
print("stats      :", store.stats())

slips = list(store.db["entries"].find({"kind": "slip"}))
if scenario == "both":
    assert len(slips) == 1 and slips[0]["camera"] == CAM1 and slips[0]["plate"] == "5AU5341", slips
    assert any(e["label"].startswith("ALREADY SLIPPED") or e["label"].startswith("CHECK") for e in store.events)
elif scenario == "cam2only":
    assert len(slips) == 1 and slips[0]["camera"] == CAM2 and slips[0]["plate"] == "5AU5341", slips
else:
    assert len(slips) == 0
    assert any(a["type"] == "unreadable_vehicle" and a["severity"] == "danger" for a in store.alerts)
print("E2E OK:", scenario)
