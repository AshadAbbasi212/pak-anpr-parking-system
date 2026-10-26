from app.plates import PlateFormat, display, vote, weighted_distance

F = PlateFormat(True, [r"^[A-Z]{2,4}[0-9]{2,4}$", r"^[A-Z]{2,4}[0-9]{5,6}$"])


def test_valid_plates_pass_unchanged():
    assert F.normalize("LEA-1234") == "LEA1234"
    assert F.normalize("abc 123") == "ABC123"
    assert F.normalize("LEB-21-1234") == "LEB211234"


def test_ocr_confusions_are_repaired():
    assert F.normalize("LEA12S4") == "LEA1254"
    assert F.normalize("1EA1234") == "IEA1234"
    assert F.normalize("L0A1234") == "LOA1234"


def test_garbage_is_rejected():
    assert F.normalize("") is None
    assert F.normalize("12") is None
    assert F.normalize("HELLOWORLD") is None
    assert F.normalize("1234567") is None


def test_validation_can_be_switched_off():
    assert PlateFormat(False, min_length=3).normalize("x-9-y!") == "X9Y"


def test_display():
    assert display("LEA1234") == "LEA-1234"
    assert display("LEB211234") == "LEB-21-1234"
    assert display("") == ""


def test_weighted_distance():
    assert weighted_distance("LEA1234", "LEA1234") == 0
    assert weighted_distance("LEA1234", "LEA1Z34") == 0.5
    assert weighted_distance("LEA1234", "LEA1334") == 1.0
    assert weighted_distance("LEA1234", "LEA123") == 1.0


def test_vote_fixes_single_bad_read():
    reads = [("LEA1234", 0.9), ("LEA1284", 0.6), ("LEA1234", 0.8), ("LEA1234", 0.85)]
    plate, agreement, used = vote(reads)
    assert plate == "LEA1234" and used == 4 and agreement > 0.9


def test_vote_ignores_wrong_length_minority():
    reads = [("LEA1234", 0.9), ("LEA123", 0.5), ("LEA1234", 0.9)]
    assert vote(reads)[0] == "LEA1234"
    assert vote([]) is None
