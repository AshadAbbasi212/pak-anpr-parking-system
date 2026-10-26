"""Pakistani number-plate formats and grammar-constrained decoding.

WHY THIS FILE EXISTS
--------------------
The bundled plate reader (fast-plate-ocr "cct_s_v2_global") is a *general* model. For every one of its
10 output positions it gives a probability for each of 37 symbols (0-9, A-Z and "_" = empty). Left alone it
happily returns things like "JMLZ472" with 99 % confidence although every Pakistani plate has its letters FIRST
and its digits AFTER them (so the "Z" can only be a "2").

Here we use that knowledge: for every known Pakistani layout ("template", e.g. LLLNNNN = 3 letters + 4 digits)
we pick, position by position, the most probable LETTER or DIGIT as the template demands, add up the
log-probabilities and keep the best template. The same maths is applied to evidence collected over many frames
(see plate_reader.fuse), which is far more reliable than voting on whole strings.

HOW TO ADD / CHANGE A FORMAT
----------------------------
Edit TEMPLATES below (or add `validation.extra_templates` in config.yaml). One line per format:
    Template(slots, groups, prior, name)
      slots  : one character per position - L = letter, N = digit.  The reader has at most 10 positions.
      groups : how the plate is printed, e.g. (3, 2, 4) -> LEB-21-1234
      prior  : log-prior. 0 = very common, more negative = rarer. It only breaks ties between layouts.
Then run:  python -m pytest tests/test_pk_formats.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PAD_IDX = ALPHABET.index("_")
MAX_SLOTS = 10
EPS = 1e-6

LETTER_IDX = np.array([i for i, c in enumerate(ALPHABET) if c.isalpha()])
DIGIT_IDX = np.array([i for i, c in enumerate(ALPHABET) if c.isdigit()])


@dataclass(frozen=True)
class Template:
    slots: str
    groups: tuple[int, ...]
    prior: float
    name: str

    @property
    def length(self) -> int:
        return len(self.slots)

    def display(self, text: str) -> str:
        out, i = [], 0
        for g in self.groups:
            out.append(text[i:i + g])
            i += g
        if i < len(text):
            out.append(text[i:])
        return "-".join(p for p in out if p)


# Layouts seen on Pakistani plates (cars, motorcycles, rickshaws, commercial, government, all provinces / ICT).
# Two-row (square) motorcycle plates are read as ONE line: top row followed by bottom row (see plate_reader.py),
# so e.g. top "LES-18" + bottom "1573" becomes LES181573 = the (3,2,4) template.
TEMPLATES: list[Template] = [
    Template("LLLNNN",     (3, 3),       0.0,  "3 letters + 3 digits   (LEA-123, ABC-123)"),
    Template("LLLNNNN",    (3, 4),       0.0,  "3 letters + 4 digits   (LEA-1234, JML-2472, motorcycles)"),
    Template("LLNNNN",     (2, 4),      -0.4,  "2 letters + 4 digits   (LE-1234, AB-1234)"),
    Template("LLNNN",      (2, 3),      -0.6,  "2 letters + 3 digits   (AB-123)"),
    Template("LLLNNNNNN",  (3, 2, 4),   -0.8,  "3 letters + year + 4 digits   (LEB-21-1234, two-row plates)"),
    Template("LLNNNNNN",   (2, 2, 4),   -0.8,  "2 letters + year + 4 digits   (LE-16-5471)"),
    Template("LLLNNNNN",   (3, 2, 3),   -1.0,  "3 letters + year + 3 digits   (MNE-17-412)"),
    Template("LLNNNNN",    (2, 2, 3),   -1.0,  "2 letters + year + 3 digits   (MN-16-314)"),
    Template("NNLLLNNNN",  (2, 3, 4),   -1.4,  "year + 3 letters + 4 digits   (09-MNA-3082)"),
    Template("NNLLNNNN",   (2, 2, 4),   -1.5,  "year + 2 letters + 4 digits   (09-MN-3082)"),
    Template("NNLLLNNN",   (2, 3, 3),   -1.6,  "year + 3 letters + 3 digits   (09-MNA-308)"),
    Template("LLLNN",      (3, 2),      -1.4,  "3 letters + 2 digits   (older Balochistan / KP)"),
    Template("LLNNNNNNL",  (2, 2, 5),   -1.6,  "2 letters + year + 4 digits + suffix letter   (LE-16-5471A)"),
    Template("LLLNNNNNNL", (3, 2, 5),   -1.8,  "3 letters + year + 4 digits + suffix letter  (LEA-16-5471A)"),
    Template("LLLN",       (3, 1),      -3.0,  "3 letters + 1 digit    (LED-9, short/VIP)"),
    Template("LNNNN",      (1, 4),      -3.5,  "1 letter + 4 digits   (old plates, A-1234)"),
    Template("LLNN",       (2, 2),      -3.5,  "2 letters + 2 digits  (government / special)"),
    Template("LNNN",       (1, 3),      -4.0,  "1 letter + 3 digits   (old plates)"),
    Template("LLLLNNN",    (4, 3),      -4.5,  "4 letters + 3 digits  (some government / old series)"),
    Template("LLLLNN",     (4, 2),      -4.5,  "4 letters + 2 digits"),
    Template("LLN",        (2, 1),      -4.5,  "2 letters + 1 digit    (short/special)"),
]


def templates_from_config(extra: list | None = None, disabled: list[str] | None = None,
                          min_length: int = 0) -> list[Template]:
    """TEMPLATES plus optional extras from config.yaml:  extra_templates: [{slots: LLLNN, groups: [3,2], prior: -1}]"""
    out = [t for t in TEMPLATES if (not disabled or t.slots not in disabled) and (min_length <= 0 or t.length >= min_length)]
    for e in extra or []:
        try:
            slots = str(e["slots"]).upper()
            if not re.fullmatch(r"[LN]{3,10}", slots):
                continue
            if min_length > 0 and len(slots) < min_length:
                continue
            groups = tuple(int(g) for g in e.get("groups", (len(slots),)))
            out.append(Template(slots, groups, float(e.get("prior", -2.0)), str(e.get("name", "custom"))))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------------------- results
@dataclass
class Candidate:
    text: str
    template: Template | None
    score: float           # sum of log-probabilities (+ prior)
    quality: float         # geometric-mean probability over all 10 positions (0..1) - easy to threshold
    margin: float = 0.0    # how clear-cut the reading is: the smaller of (gap to the next best layout) and
                           # (gap between the chosen and the runner-up character at the least certain position)
    slot_gap: float = 9.9

    @property
    def display(self) -> str:
        return self.template.display(self.text) if self.template else self.text


def _second_best(lp: np.ndarray, idx: np.ndarray) -> np.ndarray:
    sub = lp[:, idx]
    return np.partition(sub, -2, axis=1)[:, -2]


def _prep(logp: np.ndarray):
    """Best letter / best digit / pad log-prob for every position (+ the runner-up of each, for the margin)."""
    lp = np.asarray(logp, dtype=np.float32)
    if lp.shape[0] < MAX_SLOTS:                                  # defensive: pad missing positions as 'empty'
        extra = np.full((MAX_SLOTS - lp.shape[0], lp.shape[1]), np.log(EPS), np.float32)
        extra[:, PAD_IDX] = 0.0
        lp = np.vstack([lp, extra])
    lp = lp[:MAX_SLOTS]
    rows = np.arange(MAX_SLOTS)
    li = LETTER_IDX[np.argmax(lp[:, LETTER_IDX], axis=1)]
    di = DIGIT_IDX[np.argmax(lp[:, DIGIT_IDX], axis=1)]
    return (lp, li, di, lp[rows, li], lp[rows, di], lp[:, PAD_IDX],
            lp[rows, li] - _second_best(lp, LETTER_IDX), lp[rows, di] - _second_best(lp, DIGIT_IDX))


def to_logp(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Probabilities -> log-probabilities. The OCR model is over-confident (it says 99 % for a wrong character),
    so a temperature > 1 flattens each position's distribution; the Pakistani grammar and the evidence from other
    frames then get a fair say. temperature=1 leaves the model untouched."""
    lp = np.log(np.clip(np.asarray(probs, dtype=np.float32), EPS, 1.0))
    if temperature and abs(temperature - 1.0) > 1e-3:
        lp = lp / float(temperature)
        m = lp.max(axis=-1, keepdims=True)
        lp = lp - (m + np.log(np.exp(lp - m).sum(axis=-1, keepdims=True)))      # re-normalise (log-softmax)
    return lp


def decode_templates(logp: np.ndarray, templates: list[Template] | None = None, top: int = 3) -> list[Candidate]:
    """Best reading of one set of position probabilities for each template, best first.

    `logp` is (10, 37) log-probabilities (from one read, or the fused evidence of many frames)."""
    templates = templates or TEMPLATES
    lp, li, di, lbest, dbest, pad, lgap, dgap = _prep(logp)
    pad_tail = np.concatenate([np.cumsum(pad[::-1])[::-1], [0.0]])    # pad_tail[n] = sum of pad log-probs from n on
    cands: dict[str, Candidate] = {}
    for t in templates:
        n = t.length
        if n > MAX_SLOTS:
            continue
        total, chars, slot_gap = 0.0, [], 9.9
        for pos, kind in enumerate(t.slots):
            if kind == "L":
                total += lbest[pos]
                slot_gap = min(slot_gap, float(lgap[pos]))
                chars.append(ALPHABET[li[pos]])
            else:
                total += dbest[pos]
                slot_gap = min(slot_gap, float(dgap[pos]))
                chars.append(ALPHABET[di[pos]])
        total += pad_tail[n]
        text = "".join(chars)
        quality = float(np.exp(total / MAX_SLOTS))
        score = float(total + t.prior)
        old = cands.get(text)
        if old is None or score > old.score:
            c = Candidate(text, t, score, quality)
            c.slot_gap = slot_gap
            cands[text] = c
    ranked = sorted(cands.values(), key=lambda c: -c.score)
    for i, c in enumerate(ranked):
        layout_gap = c.score - ranked[i + 1].score if i + 1 < len(ranked) else 9.9
        c.margin = min(layout_gap, c.slot_gap)       # ambiguous layout OR a character that is a coin-flip
    return ranked[:top]


def decode_free(logp: np.ndarray) -> tuple[str, float]:
    """Plain best-symbol-per-position reading (what the model says without our rules), plus its confidence
    (geometric-mean probability of the non-empty positions)."""
    lp = np.asarray(logp, dtype=np.float32)[:MAX_SLOTS]
    idx = np.argmax(lp, axis=1)
    keep = [i for i in range(len(idx)) if idx[i] != PAD_IDX]
    if not keep:
        return "", 0.0
    text = "".join(ALPHABET[idx[i]] for i in keep)
    conf = float(np.exp(np.mean([lp[i, idx[i]] for i in keep])))
    return text, conf


def best_template_match(text: str, templates: list[Template] | None = None) -> Template | None:
    """Template whose letter/digit pattern fits `text` exactly (None if no layout matches)."""
    templates = templates or TEMPLATES
    best = None
    for t in templates:
        if t.length == len(text) and all((c.isalpha() if k == "L" else c.isdigit()) for c, k in zip(text, t.slots)):
            if best is None or t.prior > best.prior:
                best = t
    return best


# Look-alike characters, used to repair a TEXT (not probabilities) when only a string is available.
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "T": "7", "A": "4"}


def repair_text(text: str, templates: list[Template] | None = None, max_changes: int = 2) -> tuple[str, Template] | None:
    """Fix a string by forcing it into a template (letters <-> look-alike digits). Fewest changes wins."""
    if not any(c.isalpha() for c in text):
        return None  # an all-digit string is never a plate
    templates = templates or TEMPLATES
    best: tuple[float, str, Template] | None = None
    for t in templates:
        if t.length != len(text):
            continue
        out, changes, ok = [], 0, True
        for c, k in zip(text, t.slots):
            if k == "L":
                if c.isalpha():
                    out.append(c)
                elif c in _TO_LETTER:
                    out.append(_TO_LETTER[c]); changes += 1
                else:
                    ok = False; break
            else:
                if c.isdigit():
                    out.append(c)
                elif c in _TO_DIGIT:
                    out.append(_TO_DIGIT[c]); changes += 1
                else:
                    ok = False; break
        if ok and changes <= max_changes:
            key = changes * 10 - t.prior
            if best is None or key < best[0]:
                best = (key, "".join(out), t)
    return (best[1], best[2]) if best else None
