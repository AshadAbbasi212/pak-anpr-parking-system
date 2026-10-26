"""Image-level behaviour of the plate reader (polarity, two-row plates, fusion). Uses synthetic plates, so it checks
the LOGIC and catches regressions - it does not measure real-world accuracy (use tools/evaluate.py for that)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from make_synthetic_plates import STYLES, degrade, draw_plate, random_plate  # noqa: E402
from app.pk_formats import TEMPLATES, decode_templates  # noqa: E402
from app.plate_reader import (PlateReader, build_variants, find_row_split, fuse, merge_two_rows, read_plate,  # noqa: E402
                              text_is_light)


@pytest.fixture(scope="module")
def reader():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return PlateReader(str(ROOT / "models" / "cct_s_v2_global.onnx"), ["CPUExecutionProvider"], so)


def make(style, seed, width=180):
    import random
    rng = random.Random(seed)
    plate, rows = random_plate(rng, style)
    big = draw_plate(rng, style, rows)
    return plate, big, degrade(random.Random(seed + 1), big, width)


def test_dark_plates_are_inverted_and_white_plates_are_not():
    _p, black, _ = make("commercial", 1)
    _p, green, _ = make("government", 2)
    _p, white, _ = make("car_white", 3)
    assert text_is_light(black)[0] is True
    assert text_is_light(green)[0] is True
    assert text_is_light(white)[0] is False


def test_two_row_plate_is_split_and_merged_into_one_line():
    _p, big, _ = make("bike_two_row", 5)
    split = find_row_split(big)
    assert split is not None and 0.35 * big.shape[0] < split < 0.65 * big.shape[0]
    merged = merge_two_rows(big, split)
    assert merged is not None and merged.shape[1] / merged.shape[0] > 3.0        # one long line


def test_single_row_plates_are_not_split():
    for style, seed in (("car_white", 1), ("bike_white", 2), ("commercial", 3)):
        _p, big, _ = make(style, seed)
        assert find_row_split(big) is None, style


def test_variants_for_each_plate_type():
    _p, two, _ = make("bike_two_row", 7)
    names = [v[0] for v in build_variants(two)[0]]
    assert "merged" in names and "raw" in names
    _p, one, _ = make("car_white", 7)
    assert [v[0] for v in build_variants(one)[0]][:2] == ["raw", "enh"]


@pytest.mark.parametrize("style", list(STYLES))
def test_clear_plates_of_every_style_are_read_exactly(reader, style):
    ok = 0
    for seed in range(100, 108):
        plate, big, _ = make(style, seed)
        fr = read_plate(reader, big, TEMPLATES, plate_w=big.shape[1])
        ok += bool(fr and fr.best and fr.best.text == plate)
    assert ok >= 6, f"{style}: only {ok}/8 clear plates read exactly"      # measured 8/8 when written


def test_fusing_several_frames_is_at_least_as_good_as_one_frame(reader):
    import random
    single = fused = 0
    for seed in range(200, 214):
        rng = random.Random(seed)
        plate, rows = random_plate(rng, "car_white")
        big = draw_plate(rng, "car_white", rows)
        reads = [read_plate(reader, degrade(random.Random(seed * 10 + k), big, 70), TEMPLATES, plate_w=70)
                 for k in range(4)]
        single += bool(reads[0].best and reads[0].best.text == plate)
        f = fuse([e for r in reads for e in r.evidence])
        fused += decode_templates(f, TEMPLATES, top=1)[0].text == plate
    assert fused >= single


def test_unusable_crops_return_none(reader):
    assert read_plate(reader, np.zeros((2, 3, 3), np.uint8), TEMPLATES) is None
    assert read_plate(reader, None, TEMPLATES) is None


def test_fixed_batch_pads_but_returns_requested_count(reader):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    rd = PlateReader(str(ROOT / "models" / "cct_s_v2_global.onnx"), ["CPUExecutionProvider"], so, fixed_batch=4)
    img = np.full((60, 120, 3), 200, np.uint8)
    assert rd.infer([img, img]).shape == (2, 10, 37)
    assert reader.infer([img]).shape == (1, 10, 37)


def test_real_motorcycle_plate_crop_from_the_field():
    """A real crop from the user's own footage (bike, front view). The bare model said 'JMLZ472' with 98 %."""
    crop_path = Path(__file__).parent / "fixtures" / "real_bike_JML2472.jpg"
    if not crop_path.exists():
        pytest.skip("fixture missing")
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    rd = PlateReader(str(ROOT / "models" / "cct_s_v2_global.onnx"), ["CPUExecutionProvider"], so)
    fr = read_plate(rd, cv2.imread(str(crop_path)), TEMPLATES)
    assert fr.best.text == "JML2472"
