# Instructions for AI coding assistants

Before changing anything in this project read **docs/IMPROVEMENT_GUIDE.md** (sections 1-5 and 10 at least).

Short version:
* Measure first and after: `python tools/evaluate.py footage --labels footage/labels.csv --camera camera2 --out eval_out [--compare eval_out/baseline.json]`
* Diagnose a wrong plate with `python tools/debug_plate.py <crop.jpg> --truth <PLATE>`
* All tests must pass: `python -m pytest -q tests`. Never edit a test just to make it pass.
* Plate reading lives in `app/plate_reader.py` + `app/pk_formats.py`; decisions in `app/pipeline.py`;
  two-camera slip rules in `app/slips.py` (do not touch for reading problems).
* Never miss a vehicle (red alert instead of silence), never repeat slip numbers, keep it working offline.
* Record every improvement in `docs/CHANGELOG.md` with before/after numbers.
