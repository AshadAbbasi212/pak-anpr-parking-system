"""Turn the system's own history into a labelled set of plate pictures - for measuring accuracy and, later, for
fine-tuning the OCR model on Pakistani plates.

    python tools/export_training_set.py                  # everything saved in MongoDB
    python tools/export_training_set.py --only-corrected # only plates a human fixed with "Fix plate" (gold labels)

Writes training_export/images/*.jpg and training_export/labels.csv  (file,plate,source,vehicle_type,original_read)
  source = corrected   an operator typed the right plate  -> TRUSTED
  source = auto        the system's own reading           -> may contain mistakes, review before training

Why this matters: every day the system collects real Pakistani plates from YOUR cameras. Corrected slips are exactly the
cases the reader got wrong - the most valuable examples. See docs/IMPROVEMENT_GUIDE.md ("Fine-tuning the OCR").
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--out", default="training_export")
    ap.add_argument("--only-corrected", action="store_true")
    ap.add_argument("--since", help="only entries from this date on (YYYY-MM-DD)")
    args = ap.parse_args()

    from datetime import datetime

    from pymongo import MongoClient

    cfg = load_config(args.config)
    client = MongoClient(cfg["mongodb"]["uri"], serverSelectionTimeoutMS=3000, tz_aware=True)
    try:
        client.admin.command("ping")
    except Exception as exc:  # noqa: BLE001
        print("Cannot reach MongoDB:", exc)
        return 1
    flt: dict = {"kind": "slip", "plate_img": {"$ne": None}}
    if args.only_corrected:
        flt["plate_corrected"] = True
    if args.since:
        flt["created_at"] = {"$gte": datetime.strptime(args.since, "%Y-%m-%d").astimezone()}
    snaps = Path(cfg["storage"]["data_dir"]) / "snapshots"
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    n = corrected = 0
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "plate", "source", "vehicle_type", "original_read"])
        for e in client[cfg["mongodb"]["database"]].entries.find(flt):
            src = snaps / e["plate_img"]
            if not src.exists() or e.get("voided"):
                continue
            name = f"{e['slip_code']}_{e['plate']}.jpg"
            shutil.copy(src, out / "images" / name)
            is_fixed = bool(e.get("plate_corrected"))
            corrected += is_fixed
            n += 1
            w.writerow([name, e["plate"], "corrected" if is_fixed else "auto", e.get("vehicle_type", ""),
                        e.get("original_plate", "") if is_fixed else ""])
    print(f"{n} plate pictures exported to {out}  ({corrected} corrected by a person, {n - corrected} automatic)")
    print("Measure the reader on them:  python tools/evaluate.py training_export/images --crops "
          "--labels training_export/labels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
