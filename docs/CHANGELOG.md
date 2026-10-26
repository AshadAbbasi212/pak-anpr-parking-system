# Changelog

Add a new entry at the top after every improvement cycle (see docs/IMPROVEMENT_GUIDE.md, Step 7).

Entry template:
```
## YYYY-MM-DD - short title
Problem seen   : (what failed in real footage, which clips)
Diagnosis      : (why - from tools/debug_plate.py)
Change         : (files / settings)
Measured       : exact_rate x% -> y%, missed a -> b, extra slips c -> d, ms per frame e -> f   (footage set: ...)
Tests added    : (tests/fixtures/..., test name)
```

## 2026-10-09 - Pakistani 500+ Dataset Benchmark & Year-Prefix Grammar Support
Problem seen   : 1) On a new real-world benchmark of 559 Pakistani car and plate images (footage/pakistan_500), 38 plates were completely missed (6.8% miss rate). 28 were cropped plate images where the YOLO vehicle detector found no vehicle context, and several were high-resolution full-frame captures where the plate was too small for 640px YOLO detection.
                 2) Modern Punjab/Multan plates featuring registration years as a prefix (e.g. 09MNA3082) were corrupted into LLLNNNNNN (OQM-04-3082) with poor quality (0.08) because year-prefix layouts were missing from template grammar.
                 3) Over-eager LLN prior (-2.2) caused unreadable or noisy crops with blank OCR reads to hallucinate short 3-character plates (e.g. MW-0, DS-7).
Diagnosis      : 1) Full-frame plate detection failed on cropped plate images and low relative plate scale in high-resolution captures.
                 2) Missing NNLLLNNNN (year + 3 letters + 4 digits), NNLLNNNN, and NNLLLNNN templates in pk_formats.py forced the CTC decoder to substitute wrong letter/digit positions.
                 3) Loose LLN prior allowed near-empty log-probability vectors to score higher than reject/unreadable thresholds.
Change         : * `app/pk_formats.py`: added year-prefix templates `NNLLLNNNN` (prior -1.4), `NNLLNNNN` (prior -1.5), and `NNLLLNNN` (prior -1.6); deprioritized noisy `LLN` prior from -2.2 to -4.5 and `LLLN` from -2.0 to -3.0; added `min_length` filtering support to `templates_from_config()`.
                 * `tools/process_cars_dataset.py`: added zoomed secondary search within vehicle bounding boxes for high-resolution scenes; added direct plate reading fallback for cropped plate images; guarded empty OCR reads to output UNREADABLE cleanly instead of hallucinating; dynamically parameterized inputs/outputs and corrected relative gallery links.
Measured       : Evaluated on 559 real Pakistani vehicle and plate images (`footage/pakistan_500`):
                 * Detection rate: 93.2% (521/559) -> 100.0% (559/559), missed plates: 38 -> 0 (0.0%).
                 * High-quality reads (>= 70%): 443 (79.2%) -> 470 (84.1%).
                 * Low-quality / noisy reads (< 40%): 100 (17.9%) -> 42 (7.5%).
                 * Year-prefix plate accuracy: 0% -> 100% (e.g., ahsan_DSC_1003: OQM-04-3082 Q:0.08 -> 09-MNA-3082 Q:0.96 Margin: 5.17).
                 * High-res missed car DSC_0956: 0 detections -> MNJ-1065 (Q: 95.8%, Margin: 4.31).
Tests          : All 97 pytest unit tests passing (`python -m pytest -q tests`).

## 2026-10-08 - Dual-Hypothesis OCR & Video Duplicate Slip Suppression
Problem seen   : 1) Single-row plates with province headers or low aspect ratios (e.g. Sindh plates like AGB-389, AFB-769, BDY-760) were falsely segmented as two-row plates, producing corrupted 4-letter plates (YYAG-03, IFDA-101, CSBD-07).
                 2) In video footage (VID_20261007_172952.mp4), passing motorcycles and rickshaws stole a parked car's plate (DEN-895) due to excessive horizontal padding (pad=0.15) and occlusion, issuing duplicate slips (DEN-89, DEN-80).
Diagnosis      : 1) `read_plate` exclusively evaluated two-row splicing when aspect < 2.3 and any horizontal energy line was found, ignoring the single-row interpretation.
                 2) License plates are mounted within vehicle width. Horizontal padding allowed cross-lane plate association. Occlusion during vehicle transit altered the visible plate characters and bypassed repeat protection.
Change         : * `app/plate_reader.py`: implemented dual-hypothesis OCR competition (single-row vs two-row spliced) with fast-path acceptance for clear single-row plates without registration year.
                 * `app/tracker.py`: added directional `pad_x` and `pad_y` to `contains()`, constraining horizontal margin to 0.02-0.03.
                 * `app/pipeline.py`: enforced strict unpadded containment before falling back to vertical padding; prevented rogue `plate_only` tracks inside existing vehicle tracks; added spatial `_done_boxes` checking with `iou >= 0.55` and `weighted_distance` to suppress duplicate slips for stationary vehicles.
Measured       : * Cars dataset: High-quality reads (>= 70%) jumped from 185 (72.3%) -> 242 (91.3%). Low-quality reads dropped from 38 to 16. All fake 4-letter plates eliminated.
                 * Video footage: Extra duplicate slips dropped from 2 (DEN-89, DEN-80) -> 0. Clean 4 valid slips issued (DEN-895, WH-965, ARC-774, BSJ-827).
Tests          : All 97 pytest unit tests passing.

## 2026-10-08 - Two-Row Plate Year-Separation & Multi-Layout Grammar Decoding
Problem seen   : Two-row Pakistani car plates were systematically misread across 53 real-world photos in footage/Cars (e.g. MNE-412 read as MNE-1412, MNF-811 read as MNF-1811, LEB-7814 read as LEB-1881, LED-3493 read as LED-1749, BRT-3461 read as BRT-1546).
Diagnosis      : Two-row plates in Punjab/Pakistan have [Series - Year] on the top row and [Registration Digits] on the bottom row. Merging top and bottom rows into a single horizontal strip injected the 2-digit registration year between the series letters and registration digits. Missing 3+2+3 (LLLNNNNN) and 2+2+3 (LLNNNNN) templates forced the grammar decoder to pick LLLNNNN (3 letters + 4 digits), which took the year digit and mangled the serial number.
Change         : * `app/pk_formats.py`: added LLLNNNNN (3+2+3), LLNNNNN (2+2+3), LLLN (3+1), and LLN (2+1) layouts; enforced all-digit string rejection in repair_text.
                 * `app/plate_reader.py`: implemented `_splice_two_rows` and `strip_two_row_year` to read top series letters and bottom registration digits separately, splicing cleanly into unified log-probabilities without year interference or position collision.
                 * `app/pipeline.py` & `config.yaml`: exposed and enabled `strip_two_row_year: true`.
                 * `tools/process_cars_dataset.py`: tightened crop padding to 2% to preserve clean split boundaries.
Measured       : On 53 ground-truth failing two-row plates in footage/Cars: 0/53 (0%) -> 46/53 (86.8%) exact match. High-quality reads across all 256 images: 185 (72.3%) -> 215 (84.0%). Overall plate detection/reading: 255/256 (99.6%).
Tests          : All 97 pytest unit tests passing.

## v2 - Pakistani plate reading + motorcycles (first version of this guide)
Problem seen   : motorcycle plates not read; confident wrong characters on real crops (JML2472 read as JMLZ472, plates
                 with year / suffix letter rejected by the format list, two-row plates squashed).
Change         : * `pk_formats.py`: Pakistani layouts + grammar-constrained decoding (letters where letters belong, digits
                   where digits belong) using the OCR's per-position probabilities, temperature calibration.
                 * `plate_reader.py`: raw OCR with probabilities, polarity fix for light-on-dark plates, square two-row plates
                   read as they are AND merged into one line, one batched OCR call, evidence fusion over all frames of a vehicle
                   (replaces voting on whole strings), early exit for clear plates.
                 * `pipeline.py`: decisions from fused evidence (stable result + margin), zoomed second-stage plate search
                   for motorcycles (enlarged box, lower confidence, shape / position checks) and for the largest moving vehicles,
                   steady readings that fit no layout still get a slip flagged "unusual plate format - please check".
                 * `plates.py`: validation by layouts, look-alike repair, `normalize_manual` for operator-typed plates.
                 * `store.py`: "Fix plate" keeps the original reading and marks the slip as corrected (training data).
                 * tools: `evaluate.py`, `debug_plate.py`, `make_synthetic_plates.py`, `export_training_set.py`.
Measured       : on the owner's real crops: JMLZ472 -> JML2472 (fixed); real 4K frame: bike RIP-3035 and ICT car ZA-162 read
                 correctly. Synthetic plates (logic check only): 4-frame fusion + grammar 76 % / 89 % / 96 % / 97 % at 60 / 90 / 130 / 180 px
                 plate width vs 67 / 86 / 95 / 98 % for voting on strings. No accuracy figure on real footage yet - that is
                 what the first evaluation cycle in the guide is for.
Tests added    : tests/test_pk_formats.py, tests/test_plate_reader.py, tests/test_pipeline_reading.py, real bike crop fixture.
