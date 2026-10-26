import mongomock
import pytest
from conftest import snap, ts

from app.slips import CAM1, SlipEngine
from app.store import Store, StoreError


def test_slip_numbers_are_unique_and_continue_after_restart(cfg):
    client = mongomock.MongoClient(tz_aware=True)
    s = Store(cfg, client=client)
    s.start()
    codes = [s.create_entry("slip", f"AAA{i:03d}", "CAR", CAM1, 0.9, ts())["slip_code"] for i in range(3)]
    assert codes == ["000001", "000002", "000003"]
    s.stop()
    s2 = Store(cfg, client=client)
    s2.start()
    assert s2.create_entry("slip", "ZZZ999", "CAR", CAM1, 0.9, ts())["slip_code"] == "000004"
    s2.stop()


def test_numbering_continues_even_if_mongodb_data_is_gone(cfg):
    s = Store(cfg, client=mongomock.MongoClient(tz_aware=True))
    s.start()
    for i in range(5):
        s.create_entry("slip", f"AAA{i:03d}", "CAR", CAM1, 0.9, ts())
    s.stop()
    s2 = Store(cfg, client=mongomock.MongoClient(tz_aware=True))        # brand-new empty database
    s2.start()
    assert s2.create_entry("slip", "ZZZ999", "CAR", CAM1, 0.9, ts())["slip_no"] == 6
    s2.stop()


def test_void_adjusts_counters_and_queue(store):
    e = store.create_entry("slip", "LEA1234", "BIKE", CAM1, 0.9, ts())
    assert store.stats()["bikes_today"] == 1 and len(store.queue()) == 1
    store.void_entry(e["_id"])
    s = store.stats()
    assert s["bikes_today"] == 0 and s["total_slips"] == 0 and store.queue() == []
    with pytest.raises(StoreError):
        store.void_entry(e["_id"])


def test_handed_over_and_edit_plate(store):
    e = store.create_entry("slip", "LEA1234", "CAR", CAM1, 0.9, ts())
    store.edit_plate(e["_id"], "LEA1235")
    assert store.get_entry(e["_id"])["plate"] == "LEA1235" and "corrected" in store.get_entry(e["_id"])["notes"]
    store.set_handed_over(e["_id"], True)
    assert store.queue() == [] and len(store.queue(include_handed=True)) == 1
    assert store.db["entries"].find_one({"_id": e["_id"]})["handed_over"] is True


def test_changes_are_buffered_when_mongodb_is_down_and_replayed_later(cfg):
    client = mongomock.MongoClient(tz_aware=True)
    s = Store(cfg, client=client)
    s.start()
    eng = SlipEngine(s, None, cfg["matching"])
    s.online = False                                   # MongoDB goes away
    eng.on_plate(CAM1, "AAA111", 0.9, "CAR", ts(), 100.0, snap)
    eng.on_plate(CAM1, "BBB222", 0.9, "BIKE", ts(), 101.0, snap)
    assert s.stats()["slips_today"] == 2 and s.pending_count >= 2
    assert client["t"]["entries"].count_documents({}) == 0
    assert s._connect()
    s.flush_pending()
    assert s.pending_count == 0
    assert {d["plate"] for d in client["t"]["entries"].find()} == {"AAA111", "BBB222"}
    assert client["t"]["meta"].find_one({"_id": "counters"})["slip_counter"] == 2
    s.stop()


def test_restart_while_mongodb_is_down_keeps_whitelist_counters_and_numbering(cfg):
    s = Store(cfg, client=mongomock.MongoClient(tz_aware=True))
    s.start()
    s.whitelist_add("VIP0001")
    s.create_entry("slip", "AAA111", "CAR", CAM1, 0.9, ts())
    s.stop()

    class Dead:
        class admin:
            @staticmethod
            def command(*_a, **_k):
                from pymongo.errors import ServerSelectionTimeoutError
                raise ServerSelectionTimeoutError("down")

    s2 = Store(cfg, client=Dead())
    s2.start()
    assert not s2.online and s2.is_whitelisted("VIP0001")
    assert s2.stats()["total_slips"] == 1
    e = s2.create_entry("slip", "BBB222", "CAR", CAM1, 0.9, ts())
    assert e["slip_no"] == 2 and s2.pending_count >= 1
    rows, total, complete = s2.query_entries()
    assert total == 2 and complete is False
    s2.stop()


def test_history_filters_and_hourly_summary(store):
    store.create_entry("slip", "AAA111", "CAR", CAM1, 0.9, ts())
    store.create_entry("slip", "BBB222", "BIKE", CAM1, 0.9, ts())
    store.create_entry("whitelisted", "VIP0001", "CAR", CAM1, 0.9, ts())
    assert store.query_entries(kind="bike")[1] == 1
    assert store.query_entries(kind="whitelisted")[1] == 1
    assert store.query_entries(q="aaa-1")[1] == 1
    assert store.query_entries(q="000002")[1] == 1
    from app.store import local_day
    d = store.hourly_summary(local_day())
    assert d["totals"] == {"car": 1, "bike": 1, "other": 0, "whitelisted": 1}
    assert sum(h["car"] + h["bike"] for h in d["hours"]) == 2
