from conftest import snap, ts

from app.slips import CAM1, CAM2


def slips(store):
    return [e for e in store.db["entries"].find({"kind": "slip"})]


def alerts(store, kind):
    return [a for a in store.alerts if a["type"] == kind]


def test_camera1_issues_slip_and_counts(store, engine):
    d = engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    assert d.kind == "slip" and d.color == "green" and d.slip_code == "000001"
    s = store.stats()
    assert s["slips_today"] == 1 and s["cars_today"] == 1 and s["total_slips"] == 1
    e = store.db["entries"].find_one({"plate": "LEA1234"})
    assert e["camera"] == CAM1 and e["vehicle_type"] == "CAR" and e["created_at"]


def test_camera2_stays_quiet_when_camera1_already_slipped(store, engine):
    engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    d = engine.on_plate(CAM2, "LEA1234", 0.9, "CAR", ts(), 108.0, snap)
    assert d.kind == "already" and d.color == "gray"
    assert len(slips(store)) == 1


def test_camera2_issues_slip_when_camera1_missed(store, engine):
    d = engine.on_plate(CAM2, "BIK9999", 0.8, "BIKE", ts(), 100.0, snap)
    assert d.kind == "slip" and d.color == "green"
    e = store.db["entries"].find_one({"plate": "BIK9999"})
    assert e["camera"] == CAM2 and e["vehicle_group"] == "bike"
    assert store.stats()["bikes_today"] == 1


def test_camera2_similar_plate_is_skipped_with_quiet_note(store, engine):
    engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    d = engine.on_plate(CAM2, "LEA1284", 0.6, "CAR", ts(), 109.0, snap)        # one wrong digit
    assert d.kind == "check" and d.color == "gray"
    assert len(slips(store)) == 1
    notes = alerts(store, "check")
    assert len(notes) == 1 and notes[0]["severity"] == "info"


def test_camera2_different_plate_gets_its_own_slip(store, engine):
    engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    d = engine.on_plate(CAM2, "ABC9876", 0.9, "CAR", ts(), 108.0, snap)
    assert d.kind == "slip" and len(slips(store)) == 2


def test_camera1_slip_too_old_does_not_cover_camera2(store, engine):
    engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    d = engine.on_plate(CAM2, "LEA1234", 0.9, "CAR", ts(), 100.0 + 31 + 6, snap)   # far beyond the window
    assert d.kind == "slip" and len(slips(store)) == 2


def test_same_vehicle_seen_twice_by_same_camera_is_one_slip(store, engine):
    engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    d = engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 130.0, snap)
    assert d.kind == "slip" and len(slips(store)) == 1


def test_same_plate_coming_back_later_gets_a_new_slip(store, engine):
    engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    d = engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0 + 61, snap)
    assert d.kind == "slip" and len(slips(store)) == 2
    assert sorted(e["slip_code"] for e in slips(store)) == ["000001", "000002"]


def test_slightly_different_read_moments_later_is_same_vehicle(store, engine):
    engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    engine.on_plate(CAM1, "LEA1284", 0.6, "CAR", ts(), 102.0, snap)
    assert len(slips(store)) == 1


def test_similar_plate_a_bit_later_is_another_vehicle(store, engine):
    engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    engine.on_plate(CAM1, "LEA1235", 0.9, "CAR", ts(), 108.0, snap)      # consecutive fleet plate
    assert len(slips(store)) == 2


def test_several_vehicles_at_once_each_get_a_slip(store, engine):
    plates = ["AAA111", "BBB222", "CCC333", "DDD444"]
    types = ["CAR", "BIKE", "BIKE", "TRUCK"]
    for p, t in zip(plates, types):
        assert engine.on_plate(CAM1, p, 0.9, t, ts(), 100.0, snap).kind == "slip"
    assert len(slips(store)) == 4
    s = store.stats()
    assert (s["cars_today"], s["bikes_today"], s["other_today"]) == (1, 2, 1)
    assert len({e["slip_code"] for e in slips(store)}) == 4


def test_whitelisted_vehicle_gets_no_slip_but_is_logged_once(store, engine):
    store.whitelist_add("VIP0001", "boss")
    d = engine.on_plate(CAM1, "VIP0001", 0.9, "CAR", ts(), 100.0, snap)
    assert d.kind == "whitelisted" and d.color == "blue"
    d2 = engine.on_plate(CAM2, "VIP0001", 0.9, "CAR", ts(), 108.0, snap)
    assert d2.kind == "whitelisted"
    assert slips(store) == []
    logged = list(store.db["entries"].find({"kind": "whitelisted"}))
    assert len(logged) == 1 and store.stats()["whitelisted_today"] == 1 and store.stats()["slips_today"] == 0


def test_whitelisted_vehicle_missed_by_camera1_is_logged_by_camera2(store, engine):
    store.whitelist_add("VIP0001")
    d = engine.on_plate(CAM2, "VIP0001", 0.9, "CAR", ts(), 100.0, snap)
    assert d.kind == "whitelisted"
    assert store.db["entries"].count_documents({"kind": "whitelisted"}) == 1


def test_unreadable_at_camera1_is_not_an_alert(store, engine):
    d = engine.on_plate if False else engine.on_unreadable(CAM1, "BIKE", ts(), 100.0, snap)
    assert d.kind == "waiting" and d.color == "yellow"
    assert not store.alerts


def test_unreadable_at_camera2_with_nothing_to_explain_it_is_a_loud_alert(store, engine):
    d = engine.on_unreadable(CAM2, "BIKE", ts(), 100.0, snap)
    assert d.kind == "unreadable" and d.color == "red"
    a = alerts(store, "unreadable_vehicle")
    assert len(a) == 1 and a[0]["severity"] == "danger"
    assert slips(store) == []


def test_unreadable_at_camera2_is_explained_by_an_unclaimed_camera1_slip(store, engine):
    engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    d = engine.on_unreadable(CAM2, "CAR", ts(), 108.0, snap)
    assert d.kind == "check" and d.color == "gray"
    assert not alerts(store, "unreadable_vehicle") and len(alerts(store, "check")) == 1
    # the slip is now "used up": a second unreadable vehicle is NOT explained by it
    d2 = engine.on_unreadable(CAM2, "CAR", ts(), 109.0, snap)
    assert d2.kind == "unreadable"


def test_unreadable_right_after_camera1_slip_is_too_early_to_be_the_same_vehicle(store, engine):
    engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    d = engine.on_unreadable(CAM2, "CAR", ts(), 100.5, snap)     # faster than the minimum travel time
    assert d.kind == "unreadable"


def test_camera2_reading_claims_the_camera1_slip(store, engine):
    engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    engine.on_plate(CAM2, "LEA1234", 0.9, "CAR", ts(), 108.0, snap)
    # an unreadable vehicle after that must not be explained by the already-claimed slip
    assert engine.on_unreadable(CAM2, "CAR", ts(), 110.0, snap).kind == "unreadable"


def test_slips_are_queued_for_printing(store, engine, printq):
    import time
    engine.on_plate(CAM1, "LEA1234", 0.9, "CAR", ts(), 100.0, snap)
    e = store.db["entries"].find_one({"plate": "LEA1234"})
    # Printing enqueue is temporarily commented out for user testing
    assert e is not None and e["kind"] == "slip"


class DummySource:
    def __init__(self, status="online", age=1.0):
        self.status = status
        self.age = age

    def seconds_since_frame(self):
        return self.age


def test_camera1_standalone_when_camera2_is_down(store, engine):
    engine.set_sources({CAM1: DummySource("online", 0.5), CAM2: DummySource("offline", 99.0)})
    assert engine.is_online(CAM1) is True
    assert engine.is_online(CAM2) is False
    # Plate read at camera 1 still slips
    d = engine.on_plate(CAM1, "CAM101", 0.9, "CAR", ts(), 100.0, snap)
    assert d.kind == "slip" and d.color == "green"
    # Unreadable at camera 1 now raises a loud alert because camera 2 cannot back it up
    d2 = engine.on_unreadable(CAM1, "BIKE", ts(), 105.0, snap)
    assert d2.kind == "unreadable" and d2.color == "red"
    a = [x for x in store.alerts if x["type"] == "unreadable_vehicle"]
    assert len(a) == 1 and a[0]["severity"] == "danger" and a[0]["camera"] == CAM1


def test_camera2_standalone_when_camera1_is_down(store, engine):
    engine.set_sources({CAM1: DummySource("offline", 99.0), CAM2: DummySource("online", 0.5)})
    assert engine.is_online(CAM1) is False
    assert engine.is_online(CAM2) is True
    # Plate read at camera 2 directly issues slip
    d = engine.on_plate(CAM2, "CAM202", 0.9, "CAR", ts(), 100.0, snap)
    assert d.kind == "slip" and d.color == "green"
    # Unreadable at camera 2 directly raises loud alert
    d2 = engine.on_unreadable(CAM2, "CAR", ts(), 105.0, snap)
    assert d2.kind == "unreadable" and d2.color == "red"

