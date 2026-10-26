# Parking Entrance System

Two cameras watch the road into the parking lot. Every car, bike, rickshaw, van, truck or bus that drives in
gets a printed **slip** (slip number, plate, date/time, vehicle type, QR code). The operator at the station at the
end of the road hands the slips to the drivers. It works completely **offline** (no internet needed while running).

```
 main gate                                   ~20 m                          station
 [camera 1: FRONT plate]  -->  turn  -->  [camera 2: REAR plate]  -->  ...  [operator + printer]
```

![Station Dashboard Preview](docs/images/dashboard.png)

## 1. First-time setup (Windows)

1. Install **Python 3.10 or newer** from python.org (tick "Add python.exe to PATH") and make sure the **MongoDB** service is running.
2. Double-click **`setup.bat`** (needs internet, this one time only, to install the Python packages).
3. Open **`config.yaml`** in Notepad and put in the two camera addresses:
   `rtsp://user:password@192.168.1.10:554/stream1` (the exact address is in your CCTV camera's manual / admin page).
4. Double-click **`check.bat`**. It tests MongoDB, the models, the printer and both cameras and prints OK / PROBLEM for each.
5. Double-click **`run.bat`**. The station screen opens in your browser (http://localhost:5000).
6. In the screen: **Settings** tab -> choose your printer -> **Print a test slip**.

Everything you may want to change is in `config.yaml` (every line is explained there).

## 2. How a vehicle is handled

| Situation | What happens | Box colour |
|---|---|---|
| Camera 1 reads the plate | Slip is printed at once. Camera 2 will not print for this vehicle. | green |
| Camera 1 cannot read the plate | Nothing yet - camera 2 is the backup. | yellow |
| Camera 2 reads a plate camera 1 did not slip | Camera 2 prints the slip. | green |
| Camera 2 reads the same plate camera 1 already slipped | No slip. | gray |
| Camera 2 reads a plate *very similar* to one camera 1 slipped | No slip, quiet "please check" note with photo (could be a misread). | gray |
| Neither camera can read the plate, nothing explains it | **Loud alert** with sound + photo: give a manual slip. | red |
| Vehicle without a readable plate right after a camera-1 slip | Assumed to be that vehicle; quiet "please check" note. | gray |
| Plate is on the whitelist | No slip. Logged quietly, counted separately. | blue |
| Same plate comes again later (after about a minute) | New slip. | green |

Several vehicles at the same time are tracked separately - each gets its own slip.
Parked or standing vehicles are ignored (a vehicle must move to count).

## 3. The station screen

* **Live**: both cameras, today's counters, the list of slips to hand over (photo, plate, time, type) with
  *Handed over / Reprint / Fix plate / Void* buttons. Click **"Sound off"** once at the top to switch alert sounds on.
* **History**: search by plate or slip number, filter by date / cars / bikes / other / whitelisted, photos, **export to Excel or CSV**.
* **Summary**: entries per hour for any day.
* **Whitelist**: add / remove plates, or **import a whole list** from an Excel/CSV file (first column plate, second column note).
* **Alerts**: everything that needs attention (red = act now, blue = please check).
* **Settings**: choose and test the slip printer.

Manual slips (for the red alerts) are given by you outside the system, so they are not in the counters.

## 4. Printer

`Settings` tab -> pick the mode:

* **Windows printer - normal driver**: works with any printer installed in Windows. Best to start with.
* **Windows thermal printer - ESC/POS**: faster, and cuts the paper automatically (for receipt printers).
* **Network thermal printer**: for printers with a network cable (enter the IP address, port 9100).
* **No printer yet**: slips are only saved as pictures in `data/slips` (this is the default until you set up a printer).

Choose 80 mm or 58 mm paper. If a test slip comes out as strange characters, switch to "normal driver" mode.
If a slip fails to print, you get a red alert and a **Reprint** button.

## 5. If something goes wrong

* **MongoDB stops**: the system keeps working. Changes are saved on disk (`data/pending.jsonl`) and sent to MongoDB
  automatically when it is back. The screen shows "database OFFLINE - saving locally". Slip numbers never repeat.
* **A camera stops**: red alert, and the system reconnects by itself.
* **The computer restarts**: just run `run.bat` again; numbering continues.

## 6. Tuning for your road (config.yaml)

* **Wrong or missed plates**: make sure the camera sees plates at least ~120 pixels wide and not blurred (fast shutter / good lighting at night).
* **Cars outside the lane getting slips**: draw the road area with `roi:` for each camera. The screen draws the area on the live picture so you can check it.
* **Camera 1 -> camera 2 timing**: `travel_time_min_s` / `travel_time_max_s` (default 2 to 30 seconds).
* **Computer too slow** (live picture shows few "fps"): lower `process_fps`, raise `every_n_frames`, or try `device: "gpu"`
  (needs `onnxruntime-gpu` and CUDA; your older Quadro M1200 may not be supported - CPU mode is the safe default).
* **Plate format**: edit `validation > patterns` (regular expressions). Plates that do not match are ignored.


## 6b. Reading Pakistani plates (new in this version)

* **All common Pakistani layouts** are known (`app/pk_formats.py`): 3 letters + 3 or 4 digits (`LEA-123`, `JML-2472`),
  2 letters + 3 or 4 digits, year formats (`LEB-21-1234`, `LES-18-1573`), suffix letters (`LE-16-5471A`), short old plates,
  ICT / provincial series, government and special plates. Every reading is forced into the best-fitting layout, so a
  confidently wrong "Z" in the digit part becomes "2".
* **Motorcycles**: the plate is searched inside every motorcycle at full camera resolution (lower confidence limit, shape and
  position checks); square two-row plates are read as they are and as one merged line; several frames of the same
  vehicle are combined before the slip is printed.
* **Dark plates** (black commercial, green government) are inverted automatically.
* A plate that is read steadily but fits no known layout still gets a slip, marked "unusual plate format - please check"
  (switch off with `validation.allow_unusual: false`).
* Wrong plate on a slip? **Fix plate** on the screen. The correction is remembered and becomes training data.

## 6c. Making it better with your own footage

Read **`docs/IMPROVEMENT_GUIDE.md`** - it explains the whole procedure (record real clips -> write the answers ->
measure -> find the cause -> change one thing -> measure again -> keep or revert) and where in the code every kind of
problem is fixed. Tools:

| Command | What it does |
|---|---|
| `python tools/evaluate.py footage --labels footage/labels.csv --camera camera2` | accuracy on your clips / photos |
| `python tools/evaluate.py crops --crops --labels labels.csv` | accuracy of the reader alone on plate crops |
| `python tools/debug_plate.py crop.jpg --truth LEA1234` | why was this plate read wrong? |
| `python tools/export_training_set.py --only-corrected` | your corrected slips as a labelled training set |
| `python tools/make_synthetic_plates.py` | fake plates to test the logic (not for accuracy figures) |

AI coding assistants: see `AGENTS.md`.

## 7. Real Testing Proof & Captured Numberplates

The repository contains real-world testing footage results proving the ANPR pipeline in live traffic:

### Live Capture & Slip Generation Examples

| Vehicle Scene Detection | Extracted Plate Crop | Auto-Generated Parking Slip | Plate Read | Vehicle Type |
|---|---|---|---|---|
| <img src="footage/VID_20261007_172952_snapshots/camera1_DEN895_190232_895.jpg" width="280"> | <img src="footage/VID_20261007_172952_snapshots/camera1_DEN895_190232_895_plate.jpg" width="140"> | <img src="footage/VID_20261007_172952_slips/Slip_000001_DEN-895.png" width="140"> | `DEN-895` | Car |
| <img src="footage/VID_20261007_172952_snapshots/camera1_WH965_190240_613.jpg" width="280"> | <img src="footage/VID_20261007_172952_snapshots/camera1_WH965_190240_613_plate.jpg" width="140"> | <img src="footage/VID_20261007_172952_slips/Slip_000002_WH-965.png" width="140"> | `WH-965` | Car |
| <img src="footage/VID_20261007_172952_snapshots/camera1_ARC774_190245_495.jpg" width="280"> | <img src="footage/VID_20261007_172952_snapshots/camera1_ARC774_190245_495_plate.jpg" width="140"> | <img src="footage/VID_20261007_172952_slips/Slip_000003_ARC-774.png" width="140"> | `ARC-774` | Car |
| <img src="footage/VID_20261007_172952_snapshots/camera1_BSJ827_190528_657.jpg" width="280"> | <img src="footage/VID_20261007_172952_snapshots/camera1_BSJ827_190528_657_plate.jpg" width="140"> | <img src="footage/VID_20261007_172952_slips/Slip_000004_BSJ-827.png" width="140"> | `BSJ-827` | Car |

* **Interactive verification report**: [`footage/VID_20261007_172952_slips_review.html`](footage/VID_20261007_172952_slips_review.html)
* **Cars dataset batch OCR report (264 plate crops)**: [`footage/Cars_review.html`](footage/Cars_review.html)
* **Ground-truth Pakistani license plates dataset (53 plates)**: [`dataset/real_plates/`](dataset/real_plates/)

## 8. Folders

* `data/snapshots` - photos of every vehicle that got a decision, `data/slips` - slip pictures,
  `data/parking.log` - log file, `data/settings.json` - printer choice saved from the screen.
* `models/` - the plate detector, plate reader and vehicle detector (from the ankandrew fast-alpr / open-image-models / fast-plate-ocr projects).
* `third_party/` - the code of those three projects. `tests/` - automatic tests (`python -m pytest tests`).

