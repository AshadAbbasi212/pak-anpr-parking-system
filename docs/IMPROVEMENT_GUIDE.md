# Improvement guide - how to make the plate reading better, step by step

**Audience:** the owner of this project and any AI coding assistant (for example Antigravity) that is asked to improve it
after a real-world test. Read this file completely before changing anything.

**Goal of the project:** two cameras at a parking entrance. Every motor vehicle gets a printed slip with its plate.
Camera 1 (main gate, sees the FRONT) has the first chance; camera 2 (about 20 m down the road, sees the REAR) is the
backup. Plates are Pakistani: cars, motorcycles (often small, rear plate only), rickshaws, vans, trucks, government and
special plates, one-row and square two-row plates.

**The single most important rule:** *never judge a change by feeling. Measure it with `tools/evaluate.py` on real
footage before and after, and keep it only if the numbers improve.*

---------------------------------------------------------------------------------------------------------------------

## 1. Golden rules (do not break these)

1. **Never miss a vehicle.** A vehicle without a readable plate must end in a loud red alert for the operator
   (`SlipEngine.on_unreadable`), never in silence. Do not add code paths that drop vehicles quietly.
2. **Slip numbers never repeat** (`Store.create_entry`, `data/slip_counter.txt`). Do not touch the numbering.
3. **Works offline / when MongoDB is down.** All gate decisions use the in-memory state in `app/store.py`; writes go
   through `Store._write` (buffer in `data/pending.jsonl`). Do not call MongoDB directly from the pipeline.
4. **The two-camera logic** (`app/slips.py`) is separate from reading. Improving the reader must not require changing it.
   Change it only if the requirement changes, and update `tests/test_engine.py` in the same step.
5. **All tests must pass** after every change: `python -m pytest -q tests`. Do NOT edit or delete a test to make it pass
   unless the requirement itself changed - and then say so in the changelog.
6. **One change at a time.** Change one thing, evaluate, keep or revert. Several changes at once make results unreadable.
7. **Keep `config.yaml` backward compatible.** New settings need a default in code (`cfg.get("x", default)`) so an old
   config file still works.
8. Never commit/zip `venv/`, `data/`, `eval_out/`, `training_export/`.

---------------------------------------------------------------------------------------------------------------------

## 2. Map of the code - what lives where

| Area | File | What it does | Edit it when... |
|---|---|---|---|
| Entry point | `app/main.py` | builds everything, `--check` self-test, `build_plate_format()` | a new component must be started |
| Settings | `config.yaml`, `app/config.py` | every tunable number, with comments | tuning without code |
| Camera input | `app/camera.py` | RTSP/phone/video/webcam reader, reconnects | a camera type does not connect |
| Per-camera loop | `app/pipeline.py` `CameraWorker` | motion gate, detectors, tracking, reading, decisions, drawing | detection / tracking / when to decide |
| **Plate reading** | `app/plate_reader.py` | polarity fix, variants, two-row merge, OCR call, evidence fusion | image preparation, OCR usage |
| **Pakistani rules** | `app/pk_formats.py` | plate layouts (`TEMPLATES`), grammar-constrained decoding | a new plate layout, priors, look-alike rules |
| Plate text helpers | `app/plates.py` | `PlateFormat` (validate/repair text), `display()`, `weighted_distance`, `vote` | how plates are shown/validated/compared |
| Vehicle tracking | `app/tracker.py` | one `Track` per vehicle, matching between frames | vehicles get split/merged between frames |
| Motion | `app/motion.py` | frame-difference motion gate, "is this vehicle moving?" | parked cars counted / moving cars ignored |
| Models | `app/recognizer.py` | loads plate detector, vehicle detector, OCR (CPU / CUDA / DirectML) | a new model file or GPU problem |
| Two-camera slips | `app/slips.py` `SlipEngine` | slip / already-slipped / check / unreadable / whitelist | the business rules change |
| Data | `app/store.py` | MongoDB + offline buffer, slips, whitelist, alerts, counters | new data fields |
| Printing | `app/printing.py` | slip picture, ESC/POS, Windows printers, print queue | printer problems |
| Station screen | `app/web.py`, `app/templates/index.html` | web UI + API | UI changes |
| Models (files) | `models/*.onnx` | plate detector (YOLOv9), OCR (CCT, 10 positions x 37 symbols), vehicle detector (RF-DETR COCO) | replacing a model |
| Third-party code | `third_party/` | the three open-source projects (fast-alpr, open-image-models, fast-plate-ocr) | almost never |
| Tools | `tools/` | `evaluate.py`, `debug_plate.py`, `make_synthetic_plates.py`, `export_training_set.py` | new measuring tools |
| Tests | `tests/` | unit tests + `e2e_smoke.py` | every change |

Local customisations already present in this copy (keep them): DirectML/CUDA GPU provider selection in
`recognizer.py`, adaptive night-time preprocessing (`LOW_LIGHT_THRESHOLD` in `pipeline.py`), phone/IP-camera stream
options in `camera.py`, camera-1/camera-2 failover rules in `slips.py`.

---------------------------------------------------------------------------------------------------------------------

## 3. How one plate is read (data flow)

```
camera frame
 -> CameraWorker.process_frame()        motion gate: nothing moves -> nothing is analysed (saves CPU)
 -> _heavy()                            vehicle detector (RF-DETR)  +  plate detector (YOLOv9) on the whole frame
      -> _second_stage_plate()          vehicles WITHOUT a plate: search inside the vehicle crop at full resolution
                                        (motorcycles first, lower confidence limit, shape/position sanity checks)
 -> VehicleTracker.update()             one Track per vehicle (cars, bikes, ...) across frames
 -> _read_plate()                       crop the plate -> plate_reader.read_plate():
        text_is_light()                   light-on-dark plate (black / green) -> inverted
        build_variants()                  raw, contrast-enhanced, (square plate) both rows merged into one line
        PlateReader.infer()               ONE batched OCR call -> probabilities (10 positions x 37 symbols) per variant
        Evidence(weight)                  weight = quality^2 x size factor (clear + large + well formed counts more)
 -> track.evidence                      evidence of ALL frames of this vehicle
 -> _try_finalize() / _fused_decision() fuse() all evidence, then decode_templates():
        for every Pakistani layout pick the best LETTER at letter positions and the best DIGIT at digit positions,
        add log-probabilities + layout prior, keep the best layout.  ("JMLZ472" -> "JML2472")
        ready when: >= min_reads reads, same result twice in a row, margin >= min_margin
        (or max_reads reached, or the vehicle left view with >= min_reads_on_loss reads)
 -> _issue_plate() -> SlipEngine.on_plate()   slip / already slipped / similar / whitelist (see app/slips.py)
 no usable reading -> _decide_unreadable() -> SlipEngine.on_unreadable()   red alert (camera 2) / wait (camera 1)
 steady reading that fits NO layout -> _try_finalize_unusual() -> slip flagged "unusual plate format - please check"
```

Key numbers: **quality** (0-1) = geometric-mean probability over the 10 positions after temperature calibration
(`ocr.temperature`, default 2.0, because the OCR model is over-confident: it said "JMLZ472" with 98 %).
**margin** = how clear-cut the reading is (gap to the next layout and to the runner-up character at the least certain
position). Typical good reading: quality 0.5-0.8, margin > 1.

---------------------------------------------------------------------------------------------------------------------

## 4. The improvement loop (the procedure to follow after every real-world test)

### Step 0 - one-time setup on the development PC
```
setup.bat                                   (creates venv, installs requirements)
venv\Scripts\activate
pip install -r requirements-dev.txt         (adds pytest + mongomock)
python -m pytest -q tests                   (must say: all passed)
```

### Step 1 - collect real footage (this is what makes the system better)
Record from the REAL cameras in the REAL positions. Aim for a mix, at least 40 vehicles in total at first:
* both cameras; day, evening and night; sun glare and shade; rain if possible;
* **motorcycles**: front view (camera 1) AND rear view (camera 2), single rider and two riders, slow and fast;
* cars of every plate type you can find: white, black (commercial), green (government), ICT blue, two-row plates,
  plates with a suffix letter, dirty / bent plates;
* several vehicles side by side and close behind each other.

Put the files in `footage/` (any name). Short clips (10-60 s) are best; one clip per situation.

### Step 2 - write the answers (`footage/labels.csv`)
Watch each clip yourself and list every vehicle that SHOULD get a slip:
```
file,plate,type
bike_rear_day_01.mp4,JML-2472,bike
bike_rear_day_01.mp4,LEA-1234,car
gate_night_03.mp4,LES-18-1573,car
```
`type` is car / bike / other (optional). Dashes and spaces do not matter. Write only what you can read with your own
eyes; leave out vehicles whose plate is truly unreadable (they should produce a red alert, not a guess).
Photos work too (`.jpg`) and are replayed as short clips.

### Step 3 - baseline measurement (before changing anything)
```
python tools/evaluate.py footage --labels footage/labels.csv --camera camera2 --out eval_out
copy eval_out\report.json eval_out\baseline.json
```
Use `--camera camera1` for clips from the main gate. Read the summary: *exact* (plate fully right), *near* (1-2
characters wrong), *missed*, *extra slips* (slips for vehicles not in labels), red alerts, speed.
Debug photos of every slip/alert are saved in `eval_out/<clip name>/`.

### Step 4 - find out WHY each failure happened
For every NEAR / MISSED / EXTRA line:
1. Look at the saved photos in `eval_out/<clip>/` (the `*_plate.jpg` is the crop the reader got).
2. Run the debugger on that crop:
   `python tools/debug_plate.py eval_out\bike_rear_day_01\camera2_JML2472_xxxx_plate.jpg --truth JML2472`
   It prints, per variant, what was read, the top layouts, and the 3 most likely symbols per position.
3. Classify the failure with the table in section 5 and apply the matching fix. Write the diagnosis down.

### Step 5 - change ONE thing
Prefer in this order: (a) a `config.yaml` value, (b) a small code change in the file named in the table,
(c) new logic. Keep the change small and commented.

### Step 6 - measure again and decide
```
python tools/evaluate.py footage --labels footage/labels.csv --camera camera2 --out eval_out --compare eval_out\baseline.json
```
**Keep the change only if** `exact_rate` goes up (or stays equal while `missed` / `extra_slips` / red alerts go down),
nothing important gets worse, and `avg_ms_per_analysed_frame` stays acceptable. Otherwise revert it.
For reader-only changes the quick test on plate crops is enough and much faster:
`python tools/evaluate.py <folder with crops> --crops --labels <labels.csv>`.

### Step 7 - lock the improvement in
1. Copy the failing crop to `tests/fixtures/` and add a test in `tests/test_plate_reader.py` that expects the right text
   (see `test_real_motorcycle_plate_crop_from_the_field`). Real examples beat synthetic ones.
2. `python -m pytest -q tests` - everything must pass.
3. Add an entry to `docs/CHANGELOG.md` (what failed, what you changed, numbers before -> after).
4. Make this run the new baseline (`copy eval_out\report.json eval_out\baseline.json`).

Repeat from Step 1 with new footage. Every cycle the system gets better and the test set (`tests/fixtures`) grows.

---------------------------------------------------------------------------------------------------------------------

## 5. Symptom -> probable cause -> where to fix

| Symptom | Probable cause | What to try (file / setting) |
|---|---|---|
| Motorcycle passes, **no plate found** at all (red alert, no slip) | plate too small for the full-frame detector (4K/1080p frame is shrunk to 640 px) | `vehicles.bike_plate_confidence` lower (0.25 -> 0.18); `models.plate_confidence` lower; `_second_stage_plate()` search box / shape checks in `pipeline.py`; zoom camera 2 in; test `second_stage_max_vehicles` |
| Bike plate **found but unreadable** (low quality) | plate < ~60 px wide, motion blur | camera: faster shutter, closer/zoomed view (plate >= 120 px wide); more frames: `recognition.process_fps` up; `ocr.min_quality` lower (0.3 -> 0.25) only if wrong slips stay rare |
| One wrong character (O/0, S/5, B/8, 1/I, Z/2...) | OCR over-confident | `ocr.temperature` up (2.0 -> 2.5); more frames (`min_reads` 2 -> 3); check layout priors in `pk_formats.TEMPLATES` |
| Letters read as digits or digits as letters in the wrong place | wrong layout chosen | `pk_formats.TEMPLATES` priors; remove a layout that never occurs (`validation.disabled_templates`); add the missing layout (`validation.extra_templates`) |
| Plate with year / two rows ("LES-18 / 1573") read as only 6-7 characters | square plate squashed, or merged line wrong | `ocr.merged_weight` (try 0 / 1 / 2) and `find_row_split()` / `merge_two_rows()` in `plate_reader.py`; run `debug_plate.py` and look at the debug picture |
| Black / green / dark plates read badly | polarity not detected | `text_is_light()` thresholds in `plate_reader.py` (0.36 / 0.64) |
| Night: plates unreadable | low light, glare | `LOW_LIGHT_THRESHOLD` and `clahe_enhance(clip=4.0)` in `plate_reader.py` / `pipeline.py`; IR illumination; lower exposure time |
| A real plate gets the flag "unusual plate format" | layout missing | add it to `pk_formats.TEMPLATES` (see 6) |
| Vehicle gets a slip with a **slightly wrong plate** | decided too early | `recognition.min_reads` up, `ocr.min_margin` up (0.5 -> 1.0); camera 2 then reports "please check" |
| **Duplicate slips** for one vehicle | tracker splits one vehicle into two tracks | `tracker.py` `_score()` distances; `recognition.track_max_missing_s`; `matching.fuzzy_repeat_s` |
| Two different vehicles got **one** slip | repeat protection too wide | `matching.repeat_protection_s` down; `fuzzy_repeat_s` down |
| **Parked / standing cars** get slips | motion gate | `vehicles.must_be_moving` true; `motion.moving_fraction` up; draw `roi:` over the road lane |
| People / signs / billboards produce slips | false plate detections | `models.plate_confidence` up; use `roi:`; unusual plates need a vehicle box (already enforced) |
| Vehicle type wrong (rickshaw shown as truck) | COCO has no rickshaw class | `tracker.VEHICLE_TYPE` mapping; later fine-tune the vehicle detector |
| Too slow (few "fps" on the screen) | CPU too weak | `models.device: gpu` (DirectML), `vehicles.every_n_frames` 2 -> 3, `recognition.process_fps` down, `second_stage_max_vehicles` down, `ocr.fast_accept_quality` |
| Camera not connecting | RTSP address / network | `check.bat`; `camera.py` options; try the address in VLC first |

---------------------------------------------------------------------------------------------------------------------

## 6. Adding a new plate layout

1. Find real examples. Write the layout as one letter per position: `L` = letter, `N` = digit
   (e.g. `LLNNNNNNL` = 2 letters + 2 digit year + 4 digits + suffix letter, printed `LE-16-5471A`). Maximum 10 positions.
2. Quick way (no code): `config.yaml`
   ```yaml
   validation:
     extra_templates:
       - {slots: "LLLNN", groups: [3, 2], prior: -1.5}
   ```
3. Permanent way: add a line to `TEMPLATES` in `app/pk_formats.py`:
   `Template("LLLNN", (3, 2), -1.4, "3 letters + 2 digits (older Balochistan)")`.
   `prior` 0 = common, more negative = rarer. A rare layout needs a more negative prior, otherwise it will "steal" readings
   from common layouts (the 4-letter layouts have -4.5 for exactly this reason).
4. Add the layout to the list in `tests/test_pk_formats.py::test_clear_reading_of_every_common_layout`.
5. Evaluate (Step 6 above) - make sure no common plate got worse.

---------------------------------------------------------------------------------------------------------------------

## 7. Training data and fine-tuning (the big accuracy step)

The bundled OCR model is general-purpose (many countries). The largest remaining gain comes from fine-tuning it on
Pakistani plates - and the system is already collecting the data.

**7.1 Collecting data automatically.** Every slip saves a crop of the plate (`data/snapshots/.../*_plate.jpg`) and the
plate text. When an operator uses **Fix plate**, the slip is marked `plate_corrected` and the original wrong reading is kept.
Corrected slips are *exactly* the cases the reader got wrong = the most valuable training examples.

**7.2 Export** (MongoDB must be running):
```
python tools/export_training_set.py                    (all slips)
python tools/export_training_set.py --only-corrected   (only plates a human fixed = trusted labels)
```
Result: `training_export/images/*.jpg` + `training_export/labels.csv` (`source` = corrected / auto).
Check "auto" labels by eye before using them for training.

**7.3 Measure the current reader on that set** (this is your before-number):
`python tools/evaluate.py training_export/images --crops --labels training_export/labels.csv`

**7.4 Fine-tune.** The OCR comes from github.com/ankandrew/fast-plate-ocr (MIT). Follow the training section of its README:
the dataset format is a CSV with `image_path,plate_text`, the model config is the one in `models/*_plate_config.yaml`
(alphabet `0-9A-Z_`, 10 positions, 64x128 input). Start from the pretrained weights, train on the Pakistani set
(a few thousand labelled crops is a good start, several hundred already help), keep 15 % aside for validation, then
export to ONNX. Check the exact command names in the README of the version in `third_party/`.
**Do not** change the alphabet or the number of positions - `pk_formats.py` assumes 10 positions x 37 symbols
(`ALPHABET`, `MAX_SLOTS`); if you ever must, change both there.

**7.5 Use the new model.** Copy the new `.onnx` (and its `*_plate_config.yaml`) into `models/`, set `models.ocr_model` /
`models.ocr_config` in `config.yaml`, run `python tools/evaluate.py ... --crops` and the full clip evaluation, compare with
the baseline, keep it only if better. Keep the old model file so you can go back.

**7.6 Plate detector for motorcycles.** If section 5 shows that bike plates are not FOUND (not just misread), collect
bike images from your footage, label the plate boxes, and fine-tune a plate detector. The detector in `models/` is YOLOv9
exported by open-image-models; any replacement must output the same format or `recognizer.py` needs a new backend.
Mind the licence of the training tool (some are AGPL).

**7.7 Synthetic plates** (`tools/make_synthetic_plates.py`) are only for checking that the LOGIC works
(polarity, two-row merge, grammar) and for regression tests. They are NOT a substitute for real footage and must not be
used to report real-world accuracy.

---------------------------------------------------------------------------------------------------------------------

## 8. Camera and installation advice (often worth more than any code change)

* **Plate width in the picture matters most.** Aim for **>= 120 px**; below ~60 px reading is unreliable (synthetic
  tests: single frame 57 % at 60 px, 89 % at 130 px; fusing 4 frames lifts 60 px to 76 %). Zoom in / mount closer.
* Shutter 1/500 s or faster for moving vehicles (no motion blur). Night: infrared illumination, no headlight glare.
* Camera angle: plate seen within about 30 degrees of straight on; camera height about 1-2 m.
* Camera 2 must see the REAR of motorcycles (their plate is usually only at the back); make it the best-placed camera.
* Use the main (high) stream from the camera if the PC can cope; sub-streams are often 640x360 = too small.
* Draw `roi:` in `config.yaml` over the road lane so parked cars and footpaths are ignored.

---------------------------------------------------------------------------------------------------------------------

## 9. Command cheat-sheet (Windows)

```
python -m pytest -q tests                                  all tests
python tests\e2e_smoke.py both                             two-camera scenario with the real models (also: cam2only, noplate)
python tools\evaluate.py footage --labels footage\labels.csv --camera camera2 --out eval_out
python tools\evaluate.py crops_folder --crops --labels labels.csv
python tools\debug_plate.py crop.jpg --truth LEA1234
python tools\make_synthetic_plates.py --count 60 --out synthetic_plates
python tools\export_training_set.py --only-corrected
check.bat      run.bat      setup.bat
```
Logs: `data\parking.log`. Snapshots: `data\snapshots\`. Slip pictures: `data\slips\`.

---------------------------------------------------------------------------------------------------------------------

## 10. Rules for an AI assistant making changes

1. Read section 1, 2 and 3 first. Then read the code of the files you will touch.
2. Always run the baseline evaluation BEFORE editing and the comparison AFTER. Quote both numbers in your summary.
3. Do not change `app/slips.py`, `app/store.py` numbering or the offline buffer for a reading problem.
4. Do not weaken `must_be_moving`, the red alert for unreadable vehicles, or the "never print two slips for one vehicle"
   protections to get a better number - those protect the business.
5. Do not delete/modify tests to make them pass; add tests for every fix (real crops in `tests/fixtures`).
6. Keep changes small, commented, and add them to `docs/CHANGELOG.md`.
7. If you are unsure whether a number is "good enough", prefer the safer behaviour: an extra "please check" note is
   better than a wrong slip, and a loud alert is better than a missed vehicle.
8. When a result depends on hardware (speed, GPU/DirectML), say that it was measured on the owner's PC or not at all;
   never claim a speed measured on a different machine.

---------------------------------------------------------------------------------------------------------------------

## 11. Known limits (be honest with the owner)

* Plates narrower than ~60 px are unreliable whatever the software does. The cure is the camera (zoom, position, shutter).
* The bundled OCR is not trained on Pakistani plates; the grammar and fusion reduce but do not remove its mistakes.
  Fine-tuning (section 7) is the way to go further.
* Standard COCO vehicle detection has no rickshaw / qingqi class; they appear as car / truck / motorcycle.
* Urdu-script plates and unusual special plates are not read; they end as "unusual format" or red alerts.
* Everything measured on synthetic plates only shows that the logic works.
