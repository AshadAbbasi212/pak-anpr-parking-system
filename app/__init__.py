"""Parking management system - makes the bundled third_party libraries importable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for _repo in ("fast-alpr-master", "open-image-models-main", "fast-plate-ocr-master"):
    _p = str(ROOT / "third_party" / _repo)
    if _p not in sys.path:
        sys.path.insert(0, _p)
