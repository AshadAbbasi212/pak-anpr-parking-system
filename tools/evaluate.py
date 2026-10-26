"""Measure how well the system reads plates on YOUR footage - the heart of the improvement loop.

    python tools/evaluate.py footage/ --labels footage/labels.csv
    python tools/evaluate.py clip1.mp4 clip2.mp4 photo.jpg --camera camera2
    python tools/evaluate.py footage/ --labels footage/labels.csv --out eval_out --compare eval_out/last.json

footage/ may hold videos (.mp4 .mov ...) and photos (.jpg .png ...). Photos are replayed as short clips.

labels.csv  (one line per vehicle that SHOULD get a slip; the same file can appear several times):
    file,plate,type
    IMG_4397.MOV,JML-2472,bike
    IMG_4397.MOV,LES-18-1573,car
    gate_photo1.jpg,LEA-1234,car
  `type` (car / bike / other) is optional. Dashes and spaces in the plate are ignored.

PLATE CROPS instead of scenes: add --crops when every picture is already just ONE plate (for example the crops saved
in data/snapshots/*_plate.jpg, or tools/make_synthetic_plates.py output). The vehicle/plate detectors are skipped and
only the reading is measured - the quickest way to test changes to the reader (app/plate_reader.py, app/pk_formats.py).

Output: a table on screen, <out>/report.json (+ report.csv), debug photos of every slip / alert, and a comparison with a
previous report when --compare is given (so every change is judged by numbers, not by feeling).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import IMAGE_EXT, VIDEO_EXT, Harness  # noqa: E402

from app.plates import clean, display, weighted_distance  # noqa: E402


def load_labels(path: str | None) -> dict[str, list[dict]]:
    labels: dict[str, list[dict]] = {}
    if not path:
        return labels
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = Path((row.get("file") or "").strip()).name
            plate = clean(row.get("plate"))
            if name and plate:
                labels.setdefault(name, []).append({"plate": plate, "type": (row.get("type") or "").strip().lower()})
    return labels


def collect_inputs(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(x for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXT | VIDEO_EXT)
        elif p.exists():
            out.append(p)
        else:
            print(f"(skipping, not found: {it})")
    return out


def score(expected: list[dict], slips: list[dict]) -> dict:
    """Greedy matching of expected plates to slips. exact / near (<= 1.5 edits) / missed / extra."""
    got = [{"plate": s["plate"], "type": s.get("vehicle_type", ""), "unusual": "unusual" in (s.get("notes") or "")}
           for s in slips]
    used: set[int] = set()
    pairs, missed = [], []
    for exp in expected:
        best, best_d = None, 99.0
        for i, g in enumerate(got):
            if i in used:
                continue
            d = weighted_distance(exp["plate"], g["plate"])
            if d < best_d:
                best, best_d = i, d
        if best is not None and best_d <= 1.5:
            used.add(best)
            pairs.append({"expected": exp["plate"], "got": got[best]["plate"], "dist": best_d,
                          "type": exp["type"] or got[best]["type"].lower(), "exact": best_d == 0})
        else:
            missed.append(exp)
    extra = [g for i, g in enumerate(got) if i not in used]
    return {"pairs": pairs, "missed": missed, "extra": extra}


def evaluate_crops(files: list[Path], labels: dict, args) -> int:
    import cv2
    import numpy as np

    from app.pk_formats import decode_free, to_logp
    from app.plate_reader import read_plate

    h = Harness(args.config, args.camera, args.device)
    templates, temp = h.fmt.templates, float(h.cfg.get("ocr", {}).get("temperature", 2.0))
    ok = base_ok = n = 0
    rows = []
    print(f"Reading {len(files)} plate crops ...")
    for f in files:
        exp = labels.get(f.name)
        img = cv2.imread(str(f))
        if img is None or not exp:
            continue
        strip_year = bool(h.cfg.get("ocr", {}).get("strip_two_row_year", True))
        fr = read_plate(h.rec.reader, img, templates, plate_w=img.shape[1], temperature=temp, strip_two_row_year=strip_year)
        got = fr.best.text if fr and fr.best else ""
        bare = decode_free(to_logp(h.rec.reader.infer([img])[0], 1.0))[0]       # the model without our rules
        n += 1
        ok += got == truth
        base_ok += bare == truth
        rows.append((f.name, truth, got, bare, round(fr.best.quality, 2) if fr and fr.best else 0.0))
        if got != truth:
            print(f"   WRONG {f.name}: expected {truth}  got {got or '-'}   (bare model said {bare or '-'})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "crops_report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "expected", "got", "bare_model", "quality"])
        w.writerows(rows)
    print("\n================ SUMMARY (plate crops) ================")
    if n:
        print(f"crops with a label        : {n}")
        print(f"read exactly (our reader) : {ok}  ({ok / n:.0%})")
        print(f"read exactly (bare model) : {base_ok}  ({base_ok / n:.0%})   <- what the plain OCR model gives")
        print(f"details                   : {out / 'crops_report.csv'}")
    else:
        print("No crop had a label. Put a labels.csv next to the pictures (file,plate).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="videos, photos or folders")
    ap.add_argument("--labels", help="labels.csv with the expected plates")
    ap.add_argument("--camera", default="camera1", choices=["camera1", "camera2"],
                    help="which camera's settings to use (default camera1: every plate gives a slip)")
    ap.add_argument("--config", help="config file (default config.yaml)")
    ap.add_argument("--device", choices=["cpu", "gpu"], help="override models.device")
    ap.add_argument("--out", default="eval_out", help="folder for the report and debug photos")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth video frame (faster, less accurate)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop each video after N frames (0 = whole video)")
    ap.add_argument("--compare", help="report.json of an earlier run to compare with")
    ap.add_argument("--crops", action="store_true", help="every picture is a single plate crop (skip detection)")
    args = ap.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        print("No videos or photos found.")
        return 1
    labels = load_labels(args.labels)
    if args.crops:
        return evaluate_crops(files, labels, args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading models ... ({len(files)} file(s) to analyse)")
    h = Harness(args.config, args.camera, args.device)

    report = {"files": [], "summary": {}}
    tot = {"expected": 0, "exact": 0, "near": 0, "missed": 0, "extra": 0, "slips": 0, "unreadable": 0, "ms": []}
    by_type: dict[str, dict] = {}
    for f in files:
        keep = out / f.stem
        if f.suffix.lower() in IMAGE_EXT:
            res = h.run_image(f, keep_dir=keep)
        else:
            res = h.run_video(f, stride=args.stride, max_frames=args.max_frames, keep_dir=keep)
        exp = labels.get(f.name, [])
        sc = score(exp, res.slips)
        unread = sum(1 for a in res.alerts if a["type"] == "unreadable_vehicle")
        entry = {"file": f.name, "frames": res.frames, "ms_per_analysed_frame": round(res.analysed_ms, 1),
                 "slips": [{"plate": s["plate"], "type": s.get("vehicle_type"), "quality": s.get("conf"),
                            "note": s.get("notes", "")} for s in res.slips],
                 "unreadable_alerts": unread, "scored": bool(exp), **sc}
        report["files"].append(entry)
        tot["slips"] += len(res.slips)
        tot["unreadable"] += unread
        tot["ms"].append(res.analysed_ms)
        print(f"\n{f.name}: {res.frames} frames, {res.analysed_ms:.0f} ms per analysed frame")
        print("   slips :", ", ".join(f"{display(s['plate'])} ({s.get('vehicle_type')})" for s in res.slips) or "none")
        if unread:
            print(f"   red alerts (vehicle without readable plate): {unread}")
        if exp:
            tot["expected"] += len(exp)
            for p in sc["pairs"]:
                t = by_type.setdefault(p["type"] or "?", {"expected": 0, "exact": 0, "near": 0})
                t["expected"] += 1
                if p["exact"]:
                    tot["exact"] += 1
                    t["exact"] += 1
                else:
                    tot["near"] += 1
                    t["near"] += 1
                    print(f"   NEAR   expected {display(p['expected'])}  got {display(p['got'])}")
            for m in sc["missed"]:
                tot["missed"] += 1
                t = by_type.setdefault(m["type"] or "?", {"expected": 0, "exact": 0, "near": 0})
                t["expected"] += 1
                print(f"   MISSED expected {display(m['plate'])}")
            for e in sc["extra"]:
                tot["extra"] += 1
                print(f"   EXTRA  slip for {display(e['plate'])} (not in labels)")

    n = tot["expected"]
    summ = {"files": len(files), "expected": n, "exact": tot["exact"], "near": tot["near"], "missed": tot["missed"],
            "extra_slips": tot["extra"], "total_slips": tot["slips"], "unreadable_alerts": tot["unreadable"],
            "exact_rate": round(tot["exact"] / n, 3) if n else None,
            "found_rate": round((tot["exact"] + tot["near"]) / n, 3) if n else None,
            "avg_ms_per_analysed_frame": round(sum(tot["ms"]) / max(1, len(tot["ms"])), 1),
            "by_type": {k: {**v, "exact_rate": round(v["exact"] / v["expected"], 3) if v["expected"] else None}
                        for k, v in by_type.items()}}
    report["summary"] = summ
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(out / "report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "result", "expected", "got"])
        for e in report["files"]:
            for p in e["pairs"]:
                w.writerow([e["file"], "exact" if p["exact"] else "near", p["expected"], p["got"]])
            for m in e["missed"]:
                w.writerow([e["file"], "missed", m["plate"], ""])
            for x in e["extra"]:
                w.writerow([e["file"], "extra", "", x["plate"]])

    print("\n================ SUMMARY ================")
    if n:
        print(f"expected plates : {n}")
        print(f"read exactly    : {tot['exact']}  ({summ['exact_rate']:.0%})")
        print(f"read nearly     : {tot['near']}  (1-2 characters wrong)")
        print(f"missed          : {tot['missed']}")
        print(f"extra slips     : {tot['extra']}   (slips for vehicles that are not in labels.csv)")
        for k, v in summ["by_type"].items():
            print(f"   {k:6s}: {v['exact']}/{v['expected']} exact")
    else:
        print(f"{tot['slips']} slips, {tot['unreadable']} red alerts (no labels given, so no accuracy)")
    print(f"speed           : {summ['avg_ms_per_analysed_frame']} ms per analysed frame")
    print(f"report saved in : {out / 'report.json'}")

    if args.compare:
        try:
            old = json.loads(Path(args.compare).read_text(encoding="utf-8"))["summary"]
            print("\n---- compared with", args.compare)
            for k in ("exact_rate", "found_rate", "missed", "extra_slips", "unreadable_alerts", "avg_ms_per_analysed_frame"):
                print(f"   {k:28s} {old.get(k)}  ->  {summ.get(k)}")
        except (OSError, ValueError, KeyError) as exc:
            print("could not read the comparison file:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
