Put your real test videos / photos here, plus `labels.csv`:

    file,plate,type
    bike_rear_day_01.mp4,JML-2472,bike
    gate_night_03.mp4,LEA-1234,car

Then run (see docs/IMPROVEMENT_GUIDE.md):

    python tools/evaluate.py footage --labels footage/labels.csv --camera camera2 --out eval_out
