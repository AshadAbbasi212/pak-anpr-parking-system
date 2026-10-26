"""Plate reader: image of one plate -> per-position probabilities -> Pakistani-grammar decode -> multi-frame fusion.

Pipeline for ONE plate crop (see read_plate):
  1. polarity fix        light-on-dark plates (black commercial, green government ...) are inverted
  2. variants            raw, contrast-enhanced and - for square two-row plates - "merged" (top row + bottom row
                         written as one line, which is what the OCR model understands)
  3. one batched OCR call for all variants -> (10 positions x 37 symbols) probabilities each
  4. every variant becomes Evidence with a weight (clear + large + well-formed reads count more)

Evidence from SEVERAL frames is combined with fuse() and decoded once with pk_formats.decode_templates().

Tuning knobs live in config.yaml > ocr. Debug a single crop with:  python tools/debug_plate.py crop.jpg
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .pk_formats import (ALPHABET, PAD_IDX, Candidate, Template, decode_free, decode_templates, to_logp)

MODEL_W, MODEL_H = 128, 64          # input size of the bundled OCR model (fast-plate-ocr cct_s_v2_global)


# ---------------------------------------------------------------------------------------- model wrapper
class PlateReader:
    """Thin wrapper around the OCR ONNX model that returns the raw (10 x 37) probabilities."""

    def __init__(self, model_path: str, providers: list[str], sess_options=None, fixed_batch: int = 0):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.out_names = [o.name for o in self.session.get_outputs()]
        self.plate_out = self.out_names.index("plate") if "plate" in self.out_names else 0
        self.fixed_batch = int(fixed_batch)      # >0: always send exactly this many images (GPU/DirectML likes fixed shapes)

    @staticmethod
    def _prep(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        interp = cv2.INTER_AREA if (w > MODEL_W * 1.5 or h > MODEL_H * 1.5) else cv2.INTER_LINEAR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(cv2.resize(img, (MODEL_W, MODEL_H), interpolation=interp), cv2.COLOR_BGR2RGB)

    def infer(self, images: list[np.ndarray]) -> np.ndarray:
        """images (BGR) -> probabilities (N, 10, 37)."""
        if not images:
            return np.zeros((0, 10, len(ALPHABET)), np.float32)
        batch = [self._prep(i) for i in images]
        n = len(batch)
        if self.fixed_batch and n < self.fixed_batch:
            batch = batch + [batch[-1]] * (self.fixed_batch - n)
        out = self.session.run(None, {self.input_name: np.stack(batch).astype(np.uint8)})[self.plate_out]
        return np.asarray(out, dtype=np.float32)[:n]


# ---------------------------------------------------------------------------------------- image helpers
def clahe_enhance(img: np.ndarray, clip: float = 2.5) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4)).apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)


def upscale_to(img: np.ndarray, min_w: int = 160) -> np.ndarray:
    h, w = img.shape[:2]
    if 0 < w < min_w and h > 0:
        s = min_w / w
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return img


def text_is_light(img: np.ndarray) -> tuple[bool, bool]:
    """Detects light-on-dark plates. Returns (invert_needed, ambiguous).

    The characters are always the MINORITY of the pixels. After an Otsu split, if the bright class is the small
    one, the text is bright (black commercial plate, green government plate ...) and the picture must be inverted
    so the OCR model (trained mostly on dark-on-light plates) sees what it expects."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    inner = gray[int(h * 0.12):max(int(h * 0.88), int(h * 0.12) + 2), int(w * 0.08):max(int(w * 0.92), int(w * 0.08) + 2)]
    if inner.size < 16 or inner.std() < 6:
        return False, True
    t, _ = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = float((inner > t).mean())
    if bright < 0.36:
        return True, False
    if bright > 0.64:
        return False, False
    return False, True


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 else n + 1


def _text_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of dark text pixels (dark-on-light picture)."""
    h = gray.shape[0]
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, _odd(h // 3), 7)


def find_row_split(img: np.ndarray) -> int | None:
    """y position of the gap between the two rows of a square (two-row) plate, or None if the picture looks
    like ONE row of characters. Needs two clearly separated bands of text of similar height."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    my, mx = int(h * 0.05), int(w * 0.06)
    g = gray[my:h - my, mx:w - mx]
    if g.shape[0] < 12 or g.shape[1] < 12:
        return None
    mask = _text_mask(g)
    prof = mask.sum(axis=1).astype(np.float32) / 255.0
    k = max(1, g.shape[0] // 14)
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    H = len(prof)
    lo, hi = int(H * 0.30), int(H * 0.70)
    if hi - lo < 3:
        return None
    split = lo + int(np.argmin(prof[lo:hi]))
    top_peak, bot_peak = float(prof[:split].max(initial=0)), float(prof[split:].max(initial=0))
    if min(top_peak, bot_peak) < 1e-3:
        return None
    # both bands must be real bands: tall enough, and the valley between them clearly lower than both peaks
    top_h = int((prof[:split] > 0.35 * top_peak).sum())
    bot_h = int((prof[split:] > 0.35 * bot_peak).sum())
    if top_h < 0.14 * H or bot_h < 0.14 * H:
        return None
    if prof[split] <= 0.5 * min(top_peak, bot_peak):
        return split + my
    return None


def _trim_columns(row: np.ndarray) -> np.ndarray:
    """Cut empty left/right margins (and the province stripe / screws) off one row of a two-row plate."""
    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 6 or gray.shape[1] < 12:
        return row
    mask = _text_mask(gray)
    cols = mask.sum(axis=0).astype(np.float32) / 255.0
    cols = np.convolve(cols, np.ones(3) / 3, mode="same")
    if cols.max() < 1:
        return row
    on = np.where(cols > 0.18 * cols.max())[0]
    if len(on) < 4:
        return row
    pad = max(2, int(row.shape[0] * 0.25))
    x0, x1 = max(0, on[0] - pad), min(row.shape[1], on[-1] + 1 + pad)
    return row[:, x0:x1] if (x1 - x0) >= 0.25 * row.shape[1] else row


def merge_two_rows(img: np.ndarray, split: int | None = None, row_h: int = 48) -> np.ndarray | None:
    """Square two-row plate -> ONE line: top row, then bottom row, side by side at the same height.

    Example: top "LES-18" / bottom "1573"  ->  one picture that reads LES181573."""
    h, w = img.shape[:2]
    if split is None:
        split = find_row_split(img)
    if split is None:
        split = h // 2                                          # no clear gap: assume the plate is cut in half
    if split < 6 or h - split < 6:
        return None
    top, bot = _trim_columns(img[:split]), _trim_columns(img[split:])
    rows = []
    for r in (top, bot):
        s = row_h / max(1, r.shape[0])
        rows.append(cv2.resize(r, (max(8, int(r.shape[1] * s)), row_h), interpolation=cv2.INTER_CUBIC))
    bg = tuple(int(v) for v in np.median(img.reshape(-1, img.shape[-1]), axis=0)) if img.ndim == 3 else 255
    gap = np.full((row_h, max(6, row_h // 5), 3), bg, np.uint8)
    return np.hstack([rows[0], gap, rows[1]])


# ---------------------------------------------------------------------------------------- evidence
@dataclass
class Evidence:
    logp: np.ndarray            # (10, 37) log-probabilities from one variant of one frame
    weight: float
    variant: str
    plate_w: int = 0


@dataclass
class FrameRead:
    evidence: list[Evidence]
    logp: np.ndarray                      # fused over this frame's variants
    best: Candidate | None                # template decode of `logp`
    free_text: str
    free_conf: float
    aspect: float
    inverted: bool = False
    two_row: bool = False
    variants: list[str] = field(default_factory=list)


def fuse(evidence: list[Evidence]) -> np.ndarray | None:
    """Weighted average of log-probabilities, position by position."""
    if not evidence:
        return None
    wsum = sum(e.weight for e in evidence)
    if wsum <= 0:
        return None
    acc = np.zeros_like(evidence[0].logp)
    for e in evidence:
        acc += e.weight * e.logp
    return acc / wsum


def _splice_two_rows(top_lp: np.ndarray, bot_lp: np.ndarray) -> np.ndarray:
    """Combines top row (series letters) and bottom row (registration digits)
    into a unified (10, 37) log-probabilities matrix, stripping the registration year from the top row."""
    top_chars = [ALPHABET[np.argmax(top_lp[i])] for i in range(10)]
    num_letters = 0
    for ch in top_chars:
        if ch.isalpha():
            num_letters += 1
        elif num_letters < 2 and ch in ("0", "1", "2", "4", "5", "6", "7", "8"):
            num_letters += 1
        else:
            break
    num_letters = max(2, min(4, num_letters))

    bot_chars = [ALPHABET[np.argmax(bot_lp[i])] for i in range(10)]
    num_digits = max(1, min(4, sum(1 for ch in bot_chars if ch != "_")))

    spliced = np.full((10, len(ALPHABET)), -15.0, np.float32)
    spliced[:, PAD_IDX] = 0.0
    for i in range(num_letters):
        spliced[i] = top_lp[i]
    for j in range(num_digits):
        if num_letters + j < 10:
            spliced[num_letters + j] = bot_lp[j]
    for k in range(num_letters + num_digits, 10):
        spliced[k] = -15.0
        spliced[k, PAD_IDX] = 0.0
    return spliced


def build_variants(crop: np.ndarray, low_light: bool = False, two_row_max_aspect: float = 2.3,
                   merged_weight: float = 1.0) -> tuple[list[tuple[str, np.ndarray, float]], dict]:
    """crop (BGR) -> [(name, image, weight multiplier)], info."""
    crop = upscale_to(crop, 160)
    inv, ambiguous = text_is_light(crop)
    base = 255 - crop if inv else crop
    h, w = base.shape[:2]
    aspect = w / max(1, h)
    clip = 4.0 if low_light else 2.5
    enhanced = clahe_enhance(denoise(base), clip)
    split = find_row_split(base) if aspect < two_row_max_aspect else None
    if split is None and aspect < 1.35:
        split = h // 2                                         # almost square and no visible gap: cut in the middle
    info = {"inverted": inv, "ambiguous": ambiguous, "aspect": aspect, "two_row": False,
            "split": split, "base": base, "enhanced": enhanced}
    variants: list[tuple[str, np.ndarray, float]] = []

    merged = merge_two_rows(base, split) if split is not None else None
    if merged is not None:
        info["two_row"] = True
        # Square plates are first read as they are (the OCR model knows two-row layouts) AND as one merged line.
        # Evidence is weighted by how well-formed each reading is, so the better of the two wins.
        variants += [("raw", base, 1.0), ("enh", enhanced, 1.0),
                     ("merged_enh", merge_two_rows(enhanced, split), merged_weight), ("merged", merged, merged_weight)]
    else:
        variants += [("raw", base, 1.0), ("enh", enhanced, 1.0)]

    if ambiguous:                                              # could not tell the polarity: also try the opposite
        variants.append(("flip", 255 - base, 0.6))
    return variants[:4], info


def read_plate(reader: PlateReader, crop: np.ndarray, templates: list[Template], low_light: bool = False,
               plate_w: int | None = None, temperature: float = 2.0, merged_weight: float = 1.0,
               fast_accept: float = 0.8, strip_two_row_year: bool = False) -> FrameRead | None:
    """Read one plate picture. Returns None if the crop is unusable."""
    if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 8:
        return None
    pw = int(plate_w if plate_w else crop.shape[1])
    variants, info = build_variants(crop, low_light, merged_weight=merged_weight)
    if not variants:
        return None
    size_factor = float(np.clip(pw / 110.0, 0.25, 1.0))
    evid: list[Evidence] = []

    if info["two_row"] and strip_two_row_year and info["split"] is not None:
        base = info["base"]
        enhanced = info["enhanced"]
        # Fast path: check single row first. If it's a clear standard plate without year, accept immediately
        YEAR_LAYOUTS = {"LLLNNNNN", "LLNNNNNN", "LLNNNNN", "LLLNNNNNN", "LLNNNNNNL", "LLLNNNNNNL"}
        p_base = reader.infer([base])[0]
        cand_base = decode_templates(to_logp(p_base, temperature), templates, top=1)
        if cand_base and cand_base[0].quality >= 0.88 and cand_base[0].template and cand_base[0].template.slots not in YEAR_LAYOUTS:
            lp = to_logp(p_base, temperature)
            q = cand_base[0].quality
            evid.append(Evidence(lp, float(max(1e-4, (q ** 2) * size_factor)), "raw", pw))
            info["two_row"] = False
        else:
            split = info["split"]
            top_crop = _trim_columns(base[:split])
            bot_crop = _trim_columns(base[split:])
            top_enh = _trim_columns(enhanced[:split])
            bot_enh = _trim_columns(enhanced[split:])
            row_probs = reader.infer([top_crop, bot_crop, top_enh, bot_enh, enhanced])

            two_row_evid: list[Evidence] = []
            for (top_p, bot_p, name) in [
                (row_probs[0], row_probs[1], "two_row_spliced"),
                (row_probs[2], row_probs[3], "two_row_spliced_enh")
            ]:
                lp = _splice_two_rows(to_logp(top_p, temperature), to_logp(bot_p, temperature))
                cand = decode_templates(lp, templates, top=1)
                q = cand[0].quality if cand else 0.05
                two_row_evid.append(Evidence(lp, float(max(1e-4, (q ** 2) * size_factor)), name, pw))

            single_row_evid: list[Evidence] = [
                Evidence(to_logp(p_base, temperature), float(max(1e-4, ((cand_base[0].quality if cand_base else 0.05) ** 2) * size_factor)), "raw", pw)
            ]
            lp_enh = to_logp(row_probs[4], temperature)
            cand_enh = decode_templates(lp_enh, templates, top=1)
            q_enh = cand_enh[0].quality if cand_enh else 0.05
            single_row_evid.append(Evidence(lp_enh, float(max(1e-4, (q_enh ** 2) * size_factor)), "enh", pw))

            f_two = fuse(two_row_evid)
            cand_t = decode_templates(f_two, templates, top=1)[0] if f_two is not None else None
            f_single = fuse(single_row_evid)
            cand_s = decode_templates(f_single, templates, top=1)[0] if f_single is not None else None

            choose_two_row = False
            if cand_t and cand_t.quality >= 0.40:
                if cand_s and cand_s.template and cand_s.template.slots in YEAR_LAYOUTS:
                    choose_two_row = True
                elif cand_s and cand_t.quality >= cand_s.quality - 0.05:
                    choose_two_row = True

            if choose_two_row:
                evid = two_row_evid
            else:
                evid = single_row_evid
                info["two_row"] = False
    else:
        # Saves CPU: a clear single-row plate is read once. Only unclear ones get the extra variants.
        probs = None
        if fast_accept > 0 and not info["two_row"] and len(variants) > 1:
            p0 = reader.infer([variants[0][1]])
            c0 = decode_templates(to_logp(p0[0], temperature), templates, top=1)
            if c0 and c0[0].quality >= fast_accept:
                variants, probs = variants[:1], p0
        if probs is None:
            probs = reader.infer([v[1] for v in variants])
        for (name, _img, mult), p in zip(variants, probs):
            lp = to_logp(p, temperature)
            cand = decode_templates(lp, templates, top=1)
            q = cand[0].quality if cand else 0.05
            evid.append(Evidence(lp, float(max(1e-4, (q ** 2) * mult * size_factor)), name, pw))

    fused = fuse(evid)
    if fused is None:
        return None
    best = decode_templates(fused, templates, top=2)
    free_text, free_conf = decode_free(fused)
    return FrameRead(evid, fused, best[0] if best else None, free_text, free_conf, info["aspect"],
                     info["inverted"], info["two_row"], [e.variant for e in evid])
