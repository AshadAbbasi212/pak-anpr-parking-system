"""Process a video through the parking entrance pipeline and write an annotated output video.

Usage:
    python tools/process_video.py footage/VID_20261007_172952.mp4
    python tools/process_video.py footage/VID_20261007_172952.mp4 --out footage/output.mp4 --camera camera1
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import mongomock
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: F401
from app.config import load_config
from app.main import build_plate_format
from app.pipeline import CameraWorker
from app.plates import display
from app.recognizer import build_recognizers
from app.slips import SlipEngine
from app.store import Store
from tools._harness import _NullSource


def process_video(
    video_path: Path,
    out_path: Path,
    camera: str = "camera1",
    config_path: str | None = None,
    device: str | None = None,
    stride: int = 1,
    max_frames: int = 0,
    keep_snapshots_dir: Path | None = None,
) -> dict:
    cfg = load_config(config_path)
    if device:
        cfg["models"]["device"] = device

    tmp = Path(tempfile.mkdtemp(prefix="proc_vid_"))
    (tmp / "snapshots").mkdir()
    (tmp / "slips").mkdir()
    cfg["storage"]["data_dir"] = str(tmp)
    cfg["mongodb"]["database"] = "eval"

    store = Store(cfg, client=mongomock.MongoClient(tz_aware=True))
    store.start()
    engine = SlipEngine(store, None, cfg["matching"])
    fmt = build_plate_format(cfg)
    rec = build_recognizers(cfg["models"], cfg["vehicles"].get("confidence", 0.45))
    worker = CameraWorker(camera, _NullSource(), rec, engine, store, fmt, cfg)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = fps / stride
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    print(f"Processing '{video_path.name}' ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)...")
    print(f"Output will be saved to '{out_path}'")

    frame_idx = 0
    written_frames = 0
    t_start = time.perf_counter()

    target_total = min(total_frames, max_frames) if max_frames else total_frames
    pbar = tqdm(total=target_total, unit="frame")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if frame_idx % stride == 0:
            t = frame_idx / fps
            worker.process_frame(frame, t)
            annotated = worker._annotate(frame, t)
            writer.write(annotated)
            written_frames += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()

    # Finish any pending vehicle decisions
    t_last = frame_idx / fps
    worker._finish_lost(t_last + 12.0)
    worker.stop()
    store.stop()
    cap.release()
    writer.release()

    elapsed = time.perf_counter() - t_start
    print(f"\nFinished in {elapsed:.1f}s ({written_frames} frames written).")

    # Collect entries
    slips = [e for e in store.db["entries"].find() if e["kind"] == "slip"]
    print(f"\nSlips issued ({len(slips)}):")
    for s in slips:
        print(f"  - {display(s['plate'])} ({s.get('vehicle_type', 'unknown')}) -> Slip #{s.get('slip_code')}")

    # Save vehicle snapshots and plate crops
    if keep_snapshots_dir is not None:
        keep_snapshots_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for src in (tmp / "snapshots").rglob("*.jpg"):
            shutil.copy(src, keep_snapshots_dir / src.name)
            count += 1
        print(f"\nSaved {count} snapshot images to '{keep_snapshots_dir}'")

    # Generate printable slip receipts
    from app.printing import render_slip
    slips_dir = video_path.parent / f"{video_path.stem}_slips"
    slips_dir.mkdir(parents=True, exist_ok=True)
    html_cards = []

    for s in slips:
        try:
            img = render_slip(s, cfg["slip"])
            fn = f"Slip_{s.get('slip_code', '000000')}_{display(s['plate'])}.png"
            img.save(slips_dir / fn)
            snap_fn = Path(s.get("snapshot") or "").name if s.get("snapshot") else ""
            crop_fn = Path(s.get("plate_img") or "").name if s.get("plate_img") else ""
            html_cards.append(f"""
            <div style="background: #1e293b; border-radius: 12px; padding: 18px; margin-bottom: 24px; border: 1px solid #334155;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 16px;">
                    <div>
                        <span style="background: #22c55e; color: #022c22; font-weight: bold; padding: 4px 12px; border-radius: 9999px; font-size: 14px;">SLIP #{s.get('slip_code')}</span>
                        <span style="font-size: 20px; font-weight: bold; color: #f8fafc; margin-left: 12px;">{display(s['plate'])}</span>
                    </div>
                    <span style="background: #334155; color: #94a3b8; padding: 4px 10px; border-radius: 6px; font-size: 13px; text-transform: uppercase;">{s.get('vehicle_type')}</span>
                </div>
                <div style="display: grid; grid-template-columns: 220px 240px 1fr; gap: 20px; align-items: start;">
                    <div>
                        <div style="color: #94a3b8; font-size: 12px; margin-bottom: 6px; font-weight: 600;">PRINTED SLIP</div>
                        <img src="{video_path.stem}_slips/{fn}" style="width: 100%; border-radius: 8px; border: 1px solid #475569; background: #fff;" />
                    </div>
                    <div>
                        <div style="color: #94a3b8; font-size: 12px; margin-bottom: 6px; font-weight: 600;">PLATE CROP (ZOOMED)</div>
                        <img src="{video_path.stem}_snapshots/{crop_fn}" style="width: 100%; border-radius: 8px; border: 1px solid #475569; background: #000;" />
                    </div>
                    <div>
                        <div style="color: #94a3b8; font-size: 12px; margin-bottom: 6px; font-weight: 600;">VEHICLE SCENE SNAPSHOT</div>
                        <img src="{video_path.stem}_snapshots/{snap_fn}" style="width: 100%; border-radius: 8px; border: 1px solid #475569;" />
                    </div>
                </div>
            </div>
            """)
        except Exception as e:
            pass

    if html_cards:
        html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Slips & Plate Verification - {video_path.name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 32px; }}
        .header {{ margin-bottom: 32px; padding-bottom: 16px; border-bottom: 1px solid #334155; }}
        h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        p {{ margin: 0; color: #94a3b8; font-size: 15px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Generated Slips & Plate Verification</h1>
        <p>Source Footage: <strong>{video_path.name}</strong> &bull; Total Slips: <strong>{len(slips)}</strong></p>
    </div>
    {''.join(html_cards)}
</body>
</html>
"""
        review_path = video_path.parent / f"{video_path.stem}_slips_review.html"
        review_path.write_text(html_doc, encoding="utf-8")
        print(f"Generated slip receipts in '{slips_dir}'")
        print(f"Generated visual review page: '{review_path}'")

    shutil.rmtree(tmp, ignore_errors=True)
    return {"frames": written_frames, "slips": slips, "out_path": str(out_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="Path to input video file")
    ap.add_argument("--out", help="Path for output annotated video (default: <name>_output.mp4)")
    ap.add_argument("--camera", default="camera1", choices=["camera1", "camera2"], help="Camera profile to use")
    ap.add_argument("--config", help="Config file (default: config.yaml)")
    ap.add_argument("--device", choices=["cpu", "gpu"], help="Override models.device")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride (default 1 = every frame)")
    ap.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    ap.add_argument("--snapshots", help="Folder to save snapshot pictures of issued slips")
    args = ap.parse_args()

    vpath = Path(args.video)
    if not vpath.exists():
        print(f"Error: Video file not found: {vpath}")
        return 1

    out_path = Path(args.out) if args.out else vpath.with_name(f"{vpath.stem}_output{vpath.suffix}")
    snap_dir = Path(args.snapshots) if args.snapshots else vpath.parent / f"{vpath.stem}_snapshots"

    process_video(
        video_path=vpath,
        out_path=out_path,
        camera=args.camera,
        config_path=args.config,
        device=args.device,
        stride=args.stride,
        max_frames=args.max_frames,
        keep_snapshots_dir=snap_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
