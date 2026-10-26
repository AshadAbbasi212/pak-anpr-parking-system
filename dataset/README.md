# Pakistani License Plates Real Dataset

This directory contains real-world captured license plate images from the parking entrance system test runs.

- **Total Images:** 53 plate samples
- **Format:** High-resolution crops / full frame captures of Pakistani vehicle plates
- **Naming Convention:** `<PLATE_NUMBER>.JPG` (e.g. `LEA-2363.JPG`, `MN-199.JPG`, `LED-2236.jpg`, `BRT-3461.JPG`)
- **Use Cases:**
  - Ground-truth evaluation using `tools/evaluate.py`
  - Reader debugging with `tools/debug_plate.py <path> --truth <PLATE>`
  - Layout / character recognition tuning in `app/pk_formats.py` and `app/plate_reader.py`
