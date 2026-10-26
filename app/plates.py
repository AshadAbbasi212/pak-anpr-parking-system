"""Plate text helpers: cleaning, Pakistani-format validation, OCR-error correction,
multi-frame voting and tolerant ("fuzzy") comparison."""
from __future__ import annotations

import re
from collections import defaultdict

from .pk_formats import TEMPLATES, Template, best_template_match, repair_text

# Letters that OCR often confuses with digits and the other way round.
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}

_CONFUSABLE = {
    frozenset(p)
    for p in [("0", "O"), ("0", "D"), ("0", "Q"), ("1", "I"), ("1", "L"), ("2", "Z"),
              ("5", "S"), ("6", "G"), ("8", "B"), ("O", "Q"), ("O", "D"),
              ("6", "B"), ("4", "A")]  # extra pairs common on low-res two-row plates
}

# Extra (regex) formats on top of the Pakistani layouts in pk_formats.py. Empty by default: pk_formats.TEMPLATES
# already covers every layout; add your own in config.yaml > validation > patterns if you meet a new one.
DEFAULT_PATTERNS: list[str] = []


def clean(text: str | None) -> str:
    """Upper-case and keep only letters and digits."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


class PlateFormat:
    """Validates and repairs plate text.

    A plate is valid if it fits one of the Pakistani layouts (pk_formats.TEMPLATES, plus any
    `validation.extra_templates` from config.yaml) or one of the optional regex `patterns`."""

    def __init__(self, enabled: bool = True, patterns: list[str] | None = None, min_length: int = 4,
                 templates: list[Template] | None = None, allow_unusual: bool = True):
        self.enabled = enabled
        self.min_length = max(3, min_length)
        self.patterns = [re.compile(p) for p in (patterns or DEFAULT_PATTERNS)]
        self.templates = templates or TEMPLATES
        self.allow_unusual = allow_unusual

    def _matches(self, s: str) -> bool:
        return any(p.match(s) for p in self.patterns)

    def matches_custom(self, s: str) -> bool:
        """True if `s` fits one of the optional regex patterns from config.yaml."""
        return self._matches(clean(s))

    def match_template(self, s: str) -> Template | None:
        return best_template_match(clean(s), self.templates)

    def normalize(self, text: str | None) -> str | None:
        """Return the cleaned plate, repaired if possible, or None if it is not a valid plate."""
        s = clean(text)
        if len(s) < self.min_length:
            return None
        if not self.enabled:
            return s
        if self.match_template(s) is not None or self._matches(s):
            return s
        # Repair look-alike characters (O/0, I/1, S/5 ...) so that the text fits a Pakistani layout.
        fixed = repair_text(s, self.templates, max_changes=2)
        if fixed is not None:
            return fixed[0]
        # Legacy repair for the custom regex patterns: first k characters are letters, the rest digits.
        best: tuple[int, int, str] | None = None
        for k in (3, 2, 4):
            if k >= len(s):
                continue
            prefix, rest = s[:k], s[k:]
            fixed_prefix = "".join(DIGIT_TO_LETTER.get(c, c) if c.isdigit() else c for c in prefix)
            fixed_rest = "".join(LETTER_TO_DIGIT.get(c, c) if c.isalpha() else c for c in rest)
            cand = fixed_prefix + fixed_rest
            changes = sum(1 for a, b in zip(s, cand) if a != b)
            prefix_changes = sum(1 for a, b in zip(prefix, fixed_prefix) if a != b)
            # at least one prefix character must really be a letter: a read made only of digits is not a plate
            if (changes <= 3 and prefix_changes <= 2 and prefix_changes < k and fixed_prefix.isalpha()
                    and fixed_rest.isdigit() and self._matches(cand)):
                key = (changes, abs(k - 3), cand)
                if best is None or key < best:
                    best = key
        return best[2] if best else None

    def normalize_manual(self, text: str | None) -> str | None:
        """Plate typed by the operator (Fix plate, whitelist, ...). Known layouts are accepted as usual; if
        `allow_unusual` is on, any 4-10 character text with at least one letter and one digit is accepted too
        (special plates exist that no list can cover)."""
        ok = self.normalize(text)
        if ok:
            return ok
        s = clean(text)
        if self.allow_unusual and 4 <= len(s) <= 10 and any(c.isdigit() for c in s) and any(c.isalpha() for c in s):
            return s
        return None


def display(plate: str | None) -> str:
    """Human friendly form: LEA1234 -> LEA-1234, LEB211234 -> LEB-21-1234, LE165471A -> LE-16-5471A."""
    if not plate:
        return ""
    t = best_template_match(plate)
    if t is not None:
        return t.display(plate)
    m = re.match(r"^([A-Z]+)([0-9]+)$", plate)
    if not m:
        return plate
    letters, digits = m.groups()
    if len(digits) >= 5:
        return f"{letters}-{digits[:2]}-{digits[2:]}"
    return f"{letters}-{digits}"


def weighted_distance(a: str, b: str) -> float:
    """Edit distance where OCR-confusable characters (0/O, 1/I, 8/B ...) cost only 0.5."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    prev = [float(j) for j in range(lb + 1)]
    for i in range(1, la + 1):
        cur = [float(i)] + [0.0] * lb
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                sub = 0.0
            elif frozenset((a[i - 1], b[j - 1])) in _CONFUSABLE:
                sub = 0.5
            else:
                sub = 1.0
            cur[j] = min(prev[j] + 1.0, cur[j - 1] + 1.0, prev[j - 1] + sub)
        prev = cur
    return prev[lb]


def vote(reads: list[tuple[str, float]]) -> tuple[str, float, int] | None:
    """Combine several reads of the same plate.

    reads: list of (plate_text, weight). Returns (plate_text, agreement 0-1, reads_used).
    Later reads (plate closer, larger, clearer) receive a recency bonus."""
    if not reads:
        return None
    # Apply recency bias: later reads come from a closer/larger plate and are usually better.
    n = len(reads)
    weighted_reads = []
    for i, (s, w) in enumerate(reads):
        recency = 0.7 + 0.3 * (i / max(1, n - 1))   # 0.7 for the oldest read, 1.0 for the newest
        weighted_reads.append((s, w * recency))
    weight_by_len: dict[int, float] = defaultdict(float)
    for s, w in weighted_reads:
        weight_by_len[len(s)] += w
    best_len = max(weight_by_len, key=weight_by_len.get)
    chosen = [(s, w) for s, w in weighted_reads if len(s) == best_len]
    out, agreement = [], []
    for i in range(best_len):
        tally: dict[str, float] = defaultdict(float)
        for s, w in chosen:
            tally[s[i]] += w
        ch = max(tally, key=tally.get)
        out.append(ch)
        agreement.append(tally[ch] / (sum(tally.values()) or 1.0))
    return "".join(out), sum(agreement) / max(1, best_len), len(chosen)
