"""Draw fake Pakistani-style plates (white / black / green / blue-ICT, one-row and two-row motorcycle plates) with
realistic damage (low resolution, blur, noise, tilt, glare, JPEG) and write them with a labels.csv.

    python tools/make_synthetic_plates.py --count 60 --out synthetic_plates

IMPORTANT - what this is for: checking that the LOGIC works (dark plates get inverted, two-row plates get merged, the
Pakistani grammar fixes look-alike characters, fusion beats single frames) and catching regressions after a change.
It can NOT tell you the real-world accuracy: fonts, dirt, embossing and lighting of real plates differ.
Real accuracy is measured only with real footage and tools/evaluate.py.
"""
from __future__ import annotations

import argparse
import csv
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf"]

STYLES = {
    # name: (background BGR-ish RGB, text RGB, aspect w/h, rows, vehicle type)
    "car_white":     ((245, 245, 240), (15, 15, 15), 3.2, 1, "car"),
    "bike_white":    ((240, 240, 235), (20, 20, 20), 1.9, 1, "bike"),
    "bike_two_row":  ((242, 242, 236), (20, 20, 20), 1.35, 2, "bike"),
    "commercial":    ((20, 20, 20), (240, 240, 240), 3.2, 1, "car"),
    "government":    ((20, 110, 50), (245, 245, 245), 3.2, 1, "car"),
    "ict_blue":      ((246, 246, 250), (15, 20, 60), 3.0, 1, "car"),
}


def font(size: int):
    for n in FONTS:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def random_plate(rng: random.Random, style: str) -> tuple[str, list[str]]:
    """Returns (plate without dashes, text rows as printed)."""
    L = lambda n: "".join(rng.choice(string.ascii_uppercase) for _ in range(n))     # noqa: E731
    D = lambda n: "".join(rng.choice(string.digits) for _ in range(n))              # noqa: E731
    if style == "bike_two_row":
        letters, year, serial = L(3), D(2), D(4)
        return letters + year + serial, [f"{letters}-{year}", serial]
    kind = rng.choice(["3+3", "3+4", "2+4", "2+3", "3+2+4"] if style != "bike_white" else ["3+4", "3+4", "2+4"])
    if kind == "3+2+4":
        a, y, s = L(3), D(2), D(4)
        return a + y + s, [f"{a}-{y}-{s}"]
    a, b = (int(x) for x in kind.split("+"))
    l, d = L(a), D(b)
    return l + d, [f"{l}-{d}"]


def draw_plate(rng: random.Random, style: str, rows: list[str], height: int = 220) -> np.ndarray:
    bg, fg, aspect, nrows, _ = STYLES[style]
    W, H = int(height * aspect), height
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([6, 6, W - 7, H - 7], radius=12, outline=fg, width=4)
    if style in ("bike_white", "bike_two_row"):
        d.rectangle([10, 10, 34, H - 11], fill=(30, 150, 60))                           # province stripe
    if style == "ict_blue":
        d.rectangle([10, 10, W - 11, int(H * 0.2)], fill=(30, 80, 170))
    x0 = 44 if style in ("bike_white", "bike_two_row") else 24
    top = int(H * 0.22) if style == "ict_blue" else 14
    avail_h = (H - top - 14) / len(rows)
    for i, text in enumerate(rows):
        size = int(avail_h * 0.95)
        f = font(size)
        tw = d.textlength(text, font=f)
        max_w = W - x0 - 20
        if tw > max_w:
            size = int(size * max_w / tw)
            f = font(size)
            tw = d.textlength(text, font=f)
        x = x0 + (max_w - tw) / 2
        y = top + i * avail_h + (avail_h - size) / 2 - size * 0.12
        d.text((x, y), text, font=f, fill=fg)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def degrade(rng: random.Random, plate: np.ndarray, target_w: int) -> np.ndarray:
    h, w = plate.shape[:2]
    # tilt + perspective
    ang = rng.uniform(-4, 4)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    img = cv2.warpAffine(plate, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    j = lambda: rng.uniform(-0.05, 0.05)                                                # noqa: E731
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * max(0, j()), h * max(0, j())], [w * (1 + min(0, j())), h * max(0, j())],
                      [w * (1 + min(0, j())), h * (1 + min(0, j()))], [w * max(0, j()), h * (1 + min(0, j()))]])
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    # surroundings (what a detector crop contains) - dirty background strips
    pad_x, pad_y = int(w * rng.uniform(0.03, 0.12)), int(h * rng.uniform(0.05, 0.2))
    ctx = np.full((h + 2 * pad_y, w + 2 * pad_x, 3), rng.randint(40, 120), np.uint8)
    ctx[pad_y:pad_y + h, pad_x:pad_x + w] = img
    img = ctx
    # shrink to the camera resolution, blur, noise
    s = target_w / img.shape[1]
    img = cv2.resize(img, (max(8, int(img.shape[1] * s)), max(6, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    sigma = rng.uniform(0.0, 1.4) if target_w > 70 else rng.uniform(0.0, 0.9)
    if sigma > 0.2:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = img.astype(np.float32)
    img = img * rng.uniform(0.55, 1.1) + rng.uniform(-25, 25)                           # exposure
    img += np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(1, 9), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = rng.randint(35, 90)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def make_set(count: int, seed: int = 1, widths=(60, 90, 130, 180)):
    rng = random.Random(seed)
    names = list(STYLES)
    for i in range(count):
        style = names[i % len(names)]
        plate, rows = random_plate(rng, style)
        img = degrade(rng, draw_plate(rng, style, rows), rng.choice(widths))
        yield f"{i:03d}_{style}_{plate}.png", plate, STYLES[style][4], style, img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--out", default="synthetic_plates")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "plate", "type", "style"])
        for name, plate, vtype, style, img in make_set(args.count, args.seed):
            cv2.imwrite(str(out / name), img)
            w.writerow([name, plate, vtype, style])
    print(f"{args.count} synthetic plates written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
