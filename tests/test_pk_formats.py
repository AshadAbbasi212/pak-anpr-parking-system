"""Pakistani plate layouts + grammar-constrained decoding (app/pk_formats.py)."""
import numpy as np

from app.pk_formats import (ALPHABET, PAD_IDX, TEMPLATES, Template, best_template_match, decode_free,
                            decode_templates, repair_text, templates_from_config, to_logp)


def probs_for(text, conf=0.995, overrides=None):
    """(10, 37) probabilities that spell `text`; overrides = {slot: {char: prob}} to simulate a confused reader."""
    p = np.full((10, len(ALPHABET)), (1 - conf) / (len(ALPHABET) - 1), np.float32)
    for i in range(10):
        ch = text[i] if i < len(text) else "_"
        p[i, ALPHABET.index(ch)] = conf
    for slot, d in (overrides or {}).items():
        p[slot] = (1 - sum(d.values())) / (len(ALPHABET) - len(d))
        for ch, v in d.items():
            p[slot, ALPHABET.index(ch)] = v
    return p / p.sum(axis=1, keepdims=True)


def test_clean_reading_of_every_common_layout():
    for text in ["LEA1234", "LEA123", "AB1234", "AB123", "LEB211234", "LE165471", "LE165471A", "ABC12", "A1234"]:
        best = decode_templates(to_logp(probs_for(text)), TEMPLATES, top=1)[0]
        assert best.text == text, (text, best.text)


def test_look_alike_letter_in_the_digit_part_is_fixed():
    # the real-world bug: a bike plate JML2472 read as "JMLZ472" with 98 % confidence
    p = probs_for("JML2472", overrides={3: {"Z": 0.90, "2": 0.05}})
    assert decode_free(to_logp(p))[0] == "JMLZ472"                       # what the bare model says
    assert decode_templates(to_logp(p), TEMPLATES, top=1)[0].text == "JML2472"   # what we use


def test_overwhelming_evidence_can_still_win_over_the_grammar():
    p = probs_for("JMLZ472", overrides={3: {"Z": 0.999, "2": 0.0005}})
    assert decode_templates(to_logp(p), TEMPLATES, top=1)[0].text == "JMLZ472"   # a real 4-letter plate stays possible


def test_digit_in_the_letter_part_is_fixed():
    p = probs_for("LEA1234", overrides={0: {"1": 0.6, "L": 0.3}})
    assert decode_templates(to_logp(p), TEMPLATES, top=1)[0].text == "LEA1234"


def test_two_row_plate_written_as_one_line():
    p = probs_for("LES181573")
    best = decode_templates(to_logp(p), TEMPLATES, top=1)[0]
    assert best.text == "LES181573" and best.template.groups == (3, 2, 4) and best.display == "LES-18-1573"


def test_suffix_letter_layout():
    best = decode_templates(to_logp(probs_for("LE165471A")), TEMPLATES, top=1)[0]
    assert best.display == "LE-16-5471A"


def test_margin_is_small_when_two_readings_are_close():
    p = probs_for("LEA1234", overrides={6: {"4": 0.5, "9": 0.45}})
    c = decode_templates(to_logp(p), TEMPLATES, top=2)
    assert c[0].margin < 1.0 or len(c) == 1


def test_free_decode_ignores_empty_positions():
    text, conf = decode_free(to_logp(probs_for("AB123")))
    assert text == "AB123" and conf > 0.8
    assert decode_free(np.log(np.clip(probs_for(""), 1e-6, 1)))[0] == ""


def test_temperature_flattens_overconfidence():
    p = probs_for("LEA1234", conf=0.98)
    assert np.exp(to_logp(p, 2.0)).max() < np.exp(to_logp(p, 1.0)).max()
    assert np.allclose(np.exp(to_logp(p, 2.0)).sum(axis=1), 1.0, atol=1e-4)


def test_repair_text_only_swaps_look_alikes():
    assert repair_text("LEA12S4")[0] == "LEA1254"
    assert repair_text("1EA1234")[0] == "IEA1234"
    assert repair_text("1234567") is None          # an all-digit string is never a plate
    assert repair_text("LEA1234")[0] == "LEA1234"


def test_best_template_match_and_grouping():
    assert best_template_match("LEA1234").groups == (3, 4)
    assert best_template_match("LE1234").slots == "LLNNNN"
    assert best_template_match("12345") is None
    assert best_template_match("LEB211234").display("LEB211234") == "LEB-21-1234"


def test_extra_and_disabled_templates_from_config():
    t = templates_from_config([{"slots": "LLNNL", "groups": [2, 2, 1], "prior": -1}, {"slots": "bad!"}], ["LNNN"])
    assert any(x.slots == "LLNNL" for x in t) and not any(x.slots == "LNNN" for x in t)
    assert not any(x.slots == "bad!" for x in t)
    assert best_template_match("AB12C", t).slots == "LLNNL"


def test_every_template_fits_the_reader():
    assert all(len(t.slots) <= 10 and sum(t.groups) <= len(t.slots) for t in TEMPLATES)
    assert len({t.slots for t in TEMPLATES}) == len(TEMPLATES)


def test_custom_template_decodes():
    czech = [Template("NLLNNNN", (3, 4), 0.0, "czech")]
    assert decode_templates(to_logp(probs_for("5AU5341")), czech, top=1)[0].text == "5AU5341"
