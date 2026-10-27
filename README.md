# 🚗 Pak-ANPR: Intelligent Dual-Camera Parking Entrance System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-green.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Database](https://img.shields.io/badge/Storage-MongoDB%20%2B%20Disk%20Buffer-brightgreen.svg?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Printing](https://img.shields.io/badge/Printer-ESC%2FPOS%20%26%20Windows%20GDI-orange.svg)]()
[![Offline First](https://img.shields.io/badge/Deployment-100%25%20Offline-success.svg)]()
[![Platform](https://img.shields.io/badge/OS-Windows%20%7C%20Linux-lightgrey.svg?logo=windows&logoColor=white)]()

A robust, production-ready, **offline-first Automatic Number Plate Recognition (ANPR / ALPR) parking entrance system** engineered for high-throughput vehicle entry control. 

Designed specifically for **Pakistani vehicle plates** (cars, motorcycles, rickshaws, vans, trucks, and buses), the system employs a dual-camera synchronized failover pipeline, automated thermal receipt ticket printing, local real-time operator dashboard, and fault-tolerant local disk fallback buffering.

```
 main gate (~0 m)                           ~20 m down the road                   station (~30 m)
 [Camera 1: FRONT Plate]  ------->  vehicle travels  ------->  [Camera 2: REAR Plate]  ----->  [Operator + Thermal Printer]
 (First chance to issue slip)                                   (Backup for bikes/misses)      (Hands printed slip to driver)
```

---

## 📸 Station Operator Dashboard

The station operator monitors live feeds, real-time vehicle entry counters, recent slip issuance queues, audio alerts, and manual slip interventions directly from the embedded web interface (**http://localhost:5000**):

![Station Dashboard Preview](docs/images/dashboard.png)

---

## 🌟 Key Highlights

* **Dual-Camera Synchronized Architecture**: Camera 1 (front plate) captures entry immediately. If missed (e.g. motorcycles with rear-only plates, heavy glare, or obscured plates), Camera 2 (rear backup ~20m down) issues the ticket automatically without double-issuance.
* **Pakistani Grammar-Constrained OCR Engine**: Grammar-driven beam search decoding tailored to provincial standards (Punjab, Sindh, ICT, KPK, Balochistan), two-row square bike plates, year-format series (`LEB-21-1234`), inverted commercial/government plates, and legacy serials.
* **Offline-First Resilience**: Runs 100% locally with zero internet dependency. If MongoDB is temporarily stopped, transactions buffer safely to disk (`data/pending.jsonl`) with guaranteed sequential, non-repeating slip numbers.
* **Automated Slip Printing**: Instant ESC/POS thermal ticket generation (80mm & 58mm) with auto-cut, support for network raw printers, Windows GDI drivers, or image archiving.
* **Live Inspection & Retraining Loop**: Web station UI provides instant "Fix Plate" operator overrides that feed directly into the labeled dataset loop.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Entrance[" 🛣️ Physical Road & Cameras "]
        C1["📹 Camera 1 (Front View - Gate)"]
        C2["📹 Camera 2 (Rear View - 20m Down)"]
    end

    subgraph Perception[" 🧠 Perception & Tracking Pipeline "]
        MG["Motion Gate (Frame-Difference)"]
        VD["Vehicle Detector (RF-DETR COCO)"]
        PD["Plate Detector (YOLOv9-t End2End)"]
        TR["Multi-Vehicle Tracker (SORT)"]
    end

    subgraph Recognition[" 🇵🇰 Pakistani Plate OCR Engine "]
        PP["Crop Preprocessing & Polarity Inverter"]
        OCR["CCT OCR Recognition Model"]
        GF["Pakistani Grammar Constraints (pk_formats.py)"]
    end

    subgraph CoreEngine[" ⚙️ Decision & Storage Engine "]
        SE{"SlipEngine (Two-Camera Rules)"}
        ST[("Store (MongoDB + Disk Buffer)")]
    end

    subgraph Peripherals[" 🖥️ Peripherals & Operator Interface "]
        PR["🖨️ Thermal Printer (ESC/POS / GDI)"]
        UI["💻 Flask Station Web Dashboard"]
        AL["🚨 Red Audio/Visual Operator Alert"]
    end

    C1 --> MG
    C2 --> MG
    MG --> VD
    VD --> PD
    PD --> TR
    TR --> PP
    PP --> OCR
    OCR --> GF
    GF --> SE

    SE -->|"Valid Plate"| PR
    SE -->|"Transaction Record"| ST
    SE -->|"Live Stream & Stats"| UI
    SE -->|"Unreadable Plate"| AL
```

---

## 🔄 Two-Camera Decision Flow

```mermaid
flowchart TD
    Start(["🚗 Vehicle Drives Into Main Gate"]) --> Cam1["Camera 1 evaluates FRONT"]
    
    Cam1 --> Read1{"Front Plate Read?"}
    
    Read1 -->|"Yes"| CheckWL1{"Whitelisted?"}
    CheckWL1 -->|"Yes"| SilentPass["Quiet Log (Gate Opens / No Slip)"]
    CheckWL1 -->|"No"| IssueSlip1["🖨️ Print Slip Instantly (Camera 1)"]
    
    IssueSlip1 --> Record1["Record Entry & Track Vehicle ID"]
    Record1 --> Cam2Skip["Camera 2 recognizes vehicle within travel window -> IGNORE"]
    
    Read1 -->|"No (Motorcycle / Miss)"| WaitCam2["Camera 2 evaluates REAR (~20m down)"]
    WaitCam2 --> Read2{"Rear Plate Read?"}
    
    Read2 -->|"Yes"| CheckWL2{"Whitelisted?"}
    CheckWL2 -->|"Yes"| SilentPass
    CheckWL2 -->|"No"| IssueSlip2["🖨️ Print Slip (Camera 2 Backup)"]
    
    Read2 -->|"No (Both Cameras Missed)"| RedAlert["🚨 LOUD RED ALERT: Operator Hands Manual Slip"]
```

---

## 🔤 Pakistani License Plate Decoding Pipeline

```mermaid
flowchart LR
    RawCrop["Plate Crop"] --> PolCheck{"Plate Style?"}
    PolCheck -->|"Dark Plate (Gov / Commercial)"| Invert["Invert Polarity"]
    PolCheck -->|"Standard Plate (White / Yellow)"| Normal["Direct Feed"]
    
    Invert --> OCR["CCT OCR Neural Network"]
    Normal --> OCR
    
    OCR --> Formats["Grammar Templates (pk_formats.py)"]
    
    subgraph Rules[" Layout Rules "]
        direction TB
        R1["Standard: 3 Letters + 3-4 Digits (e.g. LEA-1234)"]
        R2["Year Format: 2-3 Letters - YY - 3-4 Digits (e.g. LEB-21-1234)"]
        R3["Suffix Series: 2 Letters - YY - Digits + Letter (e.g. LE-16-5471A)"]
        R4["Motorcycle 2-Row Square Plates (Merged & Direct)"]
        R5["ICT, Official, Special & Vintage Series"]
    end
    
    Formats --> Rules
    Rules --> BeamSearch["Constrained Beam Search & Prior Rescoring"]
    BeamSearch --> FinalPlate["Normalized Plate (e.g. 'DEN-895')"]
```

---

## 🎯 Real Testing Proof & Field Verifications

The repository includes real test runs proving detection, character recognition, and slip generation accuracy under actual traffic conditions:

### Live Capture & Automatic Slip Issuance Samples

| Vehicle Scene Detection | Extracted Plate Crop | Auto-Generated Parking Slip | Recognized Plate | Vehicle Type |
|:---:|:---:|:---:|:---:|:---:|
| <img src="footage/VID_20261007_172952_snapshots/camera1_DEN895_190232_895.jpg" width="280"> | <img src="footage/VID_20261007_172952_snapshots/camera1_DEN895_190232_895_plate.jpg" width="140"> | <img src="footage/VID_20261007_172952_slips/Slip_000001_DEN-895.png" width="140"> | `DEN-895` | Car |
| <img src="footage/VID_20261007_172952_snapshots/camera1_WH965_190240_613.jpg" width="280"> | <img src="footage/VID_20261007_172952_snapshots/camera1_WH965_190240_613_plate.jpg" width="140"> | <img src="footage/VID_20261007_172952_slips/Slip_000002_WH-965.png" width="140"> | `WH-965` | Car |
| <img src="footage/VID_20261007_172952_snapshots/camera1_ARC774_190245_495.jpg" width="280"> | <img src="footage/VID_20261007_172952_snapshots/camera1_ARC774_190245_495_plate.jpg" width="140"> | <img src="footage/VID_20261007_172952_slips/Slip_000003_ARC-774.png" width="140"> | `ARC-774` | Car |
| <img src="footage/VID_20261007_172952_snapshots/camera1_BSJ827_190528_657.jpg" width="280"> | <img src="footage/VID_20261007_172952_snapshots/camera1_BSJ827_190528_657_plate.jpg" width="140"> | <img src="footage/VID_20261007_172952_slips/Slip_000004_BSJ-827.png" width="140"> | `BSJ-827` | Car |

* **Interactive Slips & Verification Report**: [`footage/VID_20261007_172952_slips_review.html`](footage/VID_20261007_172952_slips_review.html)
* **Cars Dataset Batch Evaluation (264 plate crops)**: [`footage/Cars_review.html`](footage/Cars_review.html)
* **Ground-Truth Pakistani License Plates Dataset (53 plates)**: [`dataset/real_plates/`](dataset/real_plates/)

---

## ⚡ Performance & Model Benchmarks

| Pipeline Stage | Model Architecture | Resolution | Precision | Inference Time (CPU) |
|---|---|:---:|:---:|:---:|
| **Vehicle Detection** | RF-DETR Nano COCO | 384x384 | INT8 / FP32 | ~28 ms |
| **Plate Detection** | YOLOv9-t License Plates End2End | 640x640 | FP32 | ~18 ms |
| **Plate Recognition (OCR)** | Compact Convolutional Transformer (CCT) | 100x48 | FP32 | ~4 ms |
| **Grammar Constrained Decoding** | Pakistani Beam-Search Decoder | Dynamic | Rule Engine | < 1 ms |
| **Total Pipeline per Frame** | Multi-threaded Worker | 1080p stream | — | **~45 ms (~22 FPS)** |

---

## 🚀 Quick Start Guide

### 1. Requirements & Prerequisites
* **Operating System**: Windows 10/11 or Linux
* **Python**: Version 3.10 or newer (ensure `python` is added to your PATH)
* **MongoDB**: Community Server (runs locally as a service; offline disk fallback handles downtime)

### 2. Setup (One-Click)
Double-click `setup.bat` (or execute manually in PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Cameras
Edit `config.yaml` to specify your camera feeds (RTSP stream, IP phone camera, webcam, or video test file):
```yaml
cameras:
  camera1:
    source: "rtsp://admin:password@192.168.1.10:554/stream1"  # Main gate (Front)
    loop_video: true
    roi: []  # Optional: [[x1, y1], [x2, y2], ...]
  camera2:
    source: "rtsp://admin:password@192.168.1.11:554/stream1"  # Backup road (Rear)
    loop_video: true
    roi: []
```

### 4. Run Diagnostics
Run `check.bat` (or `python -m app.main --check`) to verify:
* MongoDB connection
* Model weight integrity
* Printer driver availability
* Camera 1 & Camera 2 frame reception

### 5. Launch the System
Double-click **`run.bat`**:
```powershell
python -m app.main
```
The operator station screen opens automatically at **`http://localhost:5000`**.

---

## 🖨️ Printer Configuration

Set up your receipt printer via the **Settings** tab on the web dashboard:

| Mode | Use Case | Features |
|---|---|---|
| **Windows GDI Driver** | Any USB/desktop printer installed in Windows | Universal compatibility; ideal for testing. |
| **ESC/POS Thermal (Raw)** | Direct thermal receipt printers (USB/COM) | Blazing fast printing with automatic paper cut. |
| **Network Thermal** | Network-attached thermal printer | Sends raw ESC/POS commands directly to IP:9100. |
| **File Mode (Default)** | Offline / virtual testing | Saves high-res slip PNGs directly into `data/slips/`. |

---

## 🛠️ Developer Tools & Measurement

Never guess whether an algorithm change helps—measure it using the included evaluation suite:

| Tool Command | Purpose |
|---|---|
| `python tools/evaluate.py footage --labels footage/labels.csv --camera camera2` | Measure end-to-end detection and reading accuracy on video footage |
| `python tools/evaluate.py crops --crops --labels labels.csv` | Benchmark the OCR reader standalone against plate crops |
| `python tools/debug_plate.py crop.jpg --truth LEA1234` | Detailed breakdown of why a particular plate was read incorrectly |
| `python tools/export_training_set.py --only-corrected` | Export human-corrected slips into a formatted training dataset |
| `python tools/process_video.py footage/test.mp4 --out footage/output.mp4` | Annotate video with bounding boxes and recognized plates |
| `python -m pytest tests` | Run automated test suite (`tests/`) |

---

## 📂 Project Directory Structure

```text
├── app/                      # Main application source code
│   ├── camera.py             # RTSP / IP webcam / video frame capture & reconnection
│   ├── config.py             # YAML configuration parser and validator
│   ├── main.py               # Application entry point & diagnostic self-check
│   ├── motion.py             # Frame-difference motion gate
│   ├── pipeline.py           # Per-camera worker, tracking, and detection loop
│   ├── pk_formats.py         # Pakistani plate templates & grammar constraint engine
│   ├── plate_reader.py       # Preprocessing, polarity inversion & OCR inference
│   ├── plates.py             # Plate string normalization, comparison & formatting
│   ├── printing.py           # Slip image builder, ESC/POS & Windows GDI drivers
│   ├── recognizer.py         # ONNX Runtime model loader (CPU / CUDA / DirectML)
│   ├── slips.py              # Two-camera failover decision engine (SlipEngine)
│   ├── store.py              # In-memory transaction store, MongoDB & disk buffer
│   ├── tracker.py            # Vehicle multi-object tracking (SORT)
│   ├── web.py                # Flask station dashboard backend & streaming endpoints
│   └── templates/            # Station screen HTML5 dashboard & live view templates
├── dataset/                  # Labeled ground-truth dataset
│   └── real_plates/          # 53 real-world captured Pakistani vehicle plates
├── docs/                     # Guides, architecture docs & image assets
│   ├── CHANGELOG.md          # Version history and measured improvements
│   ├── IMPROVEMENT_GUIDE.md  # Step-by-step performance optimization manual
│   └── images/               # Station dashboard previews & documentation diagrams
├── footage/                  # Field test videos, crops, results & HTML reports
├── models/                   # ONNX model files (managed via Git LFS)
│   ├── cct_s_v2_global.onnx
│   ├── rf-detr-nano-384-coco.onnx
│   └── yolo-v9-t-640-license-plates-end2end.onnx
├── tests/                    # Automated unit, integration & smoke tests
├── tools/                    # Benchmarking, evaluation & dataset processing tools
├── check.bat                 # One-click diagnostic self-check script
├── config.yaml               # System configuration file
├── run.bat                   # One-click system launcher script
└── setup.bat                 # One-click virtual environment setup script
```

---

## 📄 License & Credits

Built upon open-source vision frameworks:
* Fast-ALPR, Open-Image-Models & Fast-Plate-OCR
* YOLOv9 & RF-DETR object detection architectures

Developed and maintained by [AshadAbbasi212](https://github.com/AshadAbbasi212).
