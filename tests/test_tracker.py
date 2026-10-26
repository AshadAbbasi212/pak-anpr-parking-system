from app.tracker import Obj, VehicleTracker, iou, vehicle_group


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_vehicle_groups():
    assert vehicle_group("CAR") == "car" and vehicle_group("BIKE") == "bike"
    assert vehicle_group("BUS") == "other" and vehicle_group("VEHICLE") == "other"


def test_two_vehicles_side_by_side_get_two_tracks_and_keep_them():
    t = VehicleTracker(1.0)
    a, _ = t.update([Obj((100, 100, 300, 250), "car", 0.9), Obj((400, 120, 500, 250), "motorcycle", 0.9)], 0.0)
    ids = [x.id for x in a]
    assert len(set(ids)) == 2
    b, _ = t.update([Obj((130, 102, 330, 252), "car", 0.9), Obj((430, 122, 530, 252), "motorcycle", 0.9)], 0.2)
    assert [x.id for x in b] == ids
    assert b[1].vehicle_type == "BIKE" and b[0].vehicle_type == "CAR"


def test_plate_only_observation_joins_the_vehicle_track():
    t = VehicleTracker(1.0)
    a, _ = t.update([Obj((100, 100, 400, 300), "car", 0.9, plate_box=(220, 250, 280, 275))], 0.0)
    plate_obj = Obj((210, 240, 290, 285), None, 0.8, plate_box=(230, 255, 290, 280))
    b, _ = t.update([plate_obj], 0.2)                       # vehicle detector skipped this frame
    assert b[0].id == a[0].id and len(t.tracks) == 1


def test_plate_only_vehicle_gets_upgraded_when_vehicle_box_appears():
    t = VehicleTracker(1.0)
    a, _ = t.update([Obj((210, 240, 290, 285), None, 0.8, plate_box=(220, 250, 280, 275))], 0.0)
    b, _ = t.update([Obj((100, 100, 400, 300), "truck", 0.9, plate_box=(222, 252, 282, 277))], 0.2)
    assert b[0].id == a[0].id and b[0].has_vehicle and b[0].vehicle_type == "TRUCK"


def test_fast_vehicle_is_followed_with_velocity_prediction():
    t = VehicleTracker(1.0)
    x = 0
    ids = set()
    for i in range(6):
        tr, _ = t.update([Obj((x, 100, x + 100, 180), "car", 0.9)], i * 0.2)
        ids.add(tr[0].id)
        x += 160                                        # moves 1.6 car-widths per frame
    assert len(ids) == 1


def test_track_is_lost_after_timeout():
    t = VehicleTracker(1.0)
    a, _ = t.update([Obj((100, 100, 200, 180), "car", 0.9)], 0.0)
    _, lost = t.update([], 0.5)
    assert lost == []
    _, lost = t.update([], 1.6)
    assert [x.id for x in lost] == [a[0].id] and t.tracks == []


def test_nested_duplicate_vehicle_boxes_are_dropped_but_bikes_are_kept():
    from app.pipeline import CameraWorker
    big = Obj((0, 0, 400, 300), "car", 0.9)
    dup = Obj((50, 40, 300, 280), "car", 0.8)
    truck_in_car = Obj((60, 60, 250, 250), "truck", 0.7)
    bike_in_front = Obj((100, 120, 200, 290), "motorcycle", 0.8)
    other = Obj((500, 0, 700, 200), "car", 0.9)
    kept = CameraWorker._drop_nested([big, dup, truck_in_car, bike_in_front, other])
    assert big in kept and other in kept and bike_in_front in kept
    assert dup not in kept and truck_in_car not in kept


def test_fragment_of_a_handled_vehicle_does_not_raise_a_second_alert():
    import time
    from types import SimpleNamespace
    from app.pipeline import CameraWorker
    w = CameraWorker.__new__(CameraWorker)               # only the fragment logic is needed
    w._done_boxes = [((100, 100, 500, 400), time.monotonic())]
    w._heavy_dt = 0.0
    w.tracker = VehicleTracker()
    frag = SimpleNamespace(box=(120, 150, 200, 380))
    far = SimpleNamespace(box=(900, 150, 1100, 380))
    assert w._is_fragment(frag, time.monotonic()) is True
    assert w._is_fragment(far, time.monotonic()) is False
