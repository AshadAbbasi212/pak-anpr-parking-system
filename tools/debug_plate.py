"""Show EXACTLY what the reader does with one plate picture - the first thing to run when a plate is read wrong.

    python tools/debug_plate.py data/snapshots/camera2_LEA1234_101500_plate.jpg
    python tools/debug_plate.py crop.jpg --truth LEA1234

Prints: plate shape, polarity decision, the variants sent to the OCR, what each variant reads, the top candidate
readings after the Pakistani grammar, and for every position the 3 most likely symbols.
Writes debug_<name>.png (the variants as the OCR model sees them, 128x64) next to the picture (or into --out).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import ROOT  # noqa: E402,F401

from app.main import build_plate_format  # noqa: E402
from app.config import load_config  # noqa: E402
from app.pk_formats import ALPHABET, decode_free, decode_templates, to_logp  # noqa: E402
from app.plate_reader import MODEL_H, MODEL_W, PlateReader, build_variants, fuse, read_plate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--truth", help="the real plate, to see where the reading differs")
    ap.add_argument("--config")
    ap.add_argument("--out", help="folder for the debug picture (default: next to the image)")
    args = ap.parse_args()

    import onnxruntime as ort
    cfg = load_config(args.config)
    fmt = build_plate_format(cfg)
    temp = float(cfg.get("ocr", {}).get("temperature", 2.0))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    reader = PlateReader(cfg["models"]["ocr_model"], ["CPUExecutionProvider"], so)

    img = cv2.imread(args.image)
    if img is None:
        print("cannot read", args.image)
        return 1
    h, w = img.shape[:2]
    print(f"plate picture: {w}x{h} px, aspect {w / h:.2f}")
    variants, info = build_variants(img, merged_weight=float(cfg.get("ocr", {}).get("merged_weight", 1.0)))
    print(f"polarity: {'LIGHT text on dark -> inverted' if info['inverted'] else 'dark text on light'}"
          f"{' (unsure, extra flipped variant added)' if info['ambiguous'] else ''}   two-row: {info['two_row']}")
    probs = reader.infer([v[1] for v in variants])
    tiles = []
    print("\nvariant        reads (grammar)       quality  | bare model")
    for (name, vimg, mult), p in zip(variants, probs):
        lp = to_logp(p, temp)
        c = decode_templates(lp, fmt.templates, top=1)[0]
        bare, _ = decode_free(to_logp(p, 1.0))
        print(f"{name:12s}   {c.display:18s}   {c.quality:5.2f}    | {bare}")
        t = cv2.resize(vimg if vimg.ndim == 3 else cv2.cvtColor(vimg, cv2.COLOR_GRAY2BGR), (MODEL_W * 2, MODEL_H * 2))
        cv2.putText(t, name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        tiles.append(t)

    strip_year = bool(cfg.get("ocr", {}).get("strip_two_row_year", True))
    fr = read_plate(reader, img, fmt.templates, temperature=temp, strip_two_row_year=strip_year)
    fused = fuse(fr.evidence)
    print("\nBest readings after fusing the variants:")
    for c in decode_templates(fused, fmt.templates, top=5):
        print(f"   {c.display:16s} layout {c.template.slots:11s} quality {c.quality:.2f}  margin {c.margin:.2f}")
    print("\nMost likely symbols per position (fused):")
    for pos in range(10):
        order = np.argsort(-fused[pos])[:3]
        print(f"   pos {pos}: " + "   ".join(f"{ALPHABET[i]}={np.exp(fused[pos, i]):.2f}" for i in order))
    if args.truth:
        truth = args.truth.upper().replace("-", "").replace(" ", "")
        best = decode_templates(fused, fmt.templates, top=1)[0].text
        print(f"\nTRUTH {truth}   READ {best}   -> {'OK' if truth == best else 'WRONG'}")

    out = Path(args.out) if args.out else Path(args.image).parent
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"debug_{Path(args.image).stem}.png"
    cv2.imwrite(str(dst), np.vstack([np.hstack(tiles[i:i + 2] + [np.zeros_like(tiles[0])] * (2 - len(tiles[i:i + 2])))
                                     for i in range(0, len(tiles), 2)]))
    print("\ndebug picture:", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
