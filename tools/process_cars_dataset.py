"""Batch process a folder of car photos, recognize license plates,
save annotated images, plate crops, and generate an interactive HTML verification gallery.

Usage:
    python tools/process_cars_dataset.py
"""
import sys
import os
import json
import csv
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.main import build_plate_format
from app.recognizer import build_recognizers
from app.pk_formats import decode_templates, decode_free, to_logp
from app.plate_reader import read_plate, fuse
from app.plates import display


def annotate_image(img: np.ndarray, plates_data: list[dict]) -> np.ndarray:
    """Draw high-visibility bounding boxes and stylish labels on detected plates."""
    vis = img.copy()
    h, w = vis.shape[:2]
    
    # Scale text and borders according to image resolution
    base_scale = max(0.6, min(2.5, w / 1200.0))
    line_th = max(2, int(w / 400.0))
    
    if not plates_data:
        # Draw a small warning tag in the top left
        cv2.rectangle(vis, (20, 20), (min(w - 20, 480), 80), (0, 0, 180), -1)
        cv2.putText(vis, "NO PLATE DETECTED", (35, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, base_scale * 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return vis

    for p in plates_data:
        x1, y1, x2, y2 = p["box"]
        plate_txt = p["display_plate"]
        q_pct = int(p["quality"] * 100)
        c_pct = int(p["confidence"] * 100)
        
        # Color coding: Green for good quality, Yellow for medium/unusual, Red/Orange for low
        box_color = (0, 210, 0) if p["quality"] >= 0.50 else (0, 180, 255)
        
        # Plate rectangle
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, line_th)
        
        # Label text
        label_main = f"{plate_txt}"
        label_sub = f"Q:{q_pct}% C:{c_pct}%"
        
        # Measure text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = base_scale * 0.8
        (tw_main, th_main), _ = cv2.getTextSize(label_main, font, font_scale, 2)
        (tw_sub, th_sub), _ = cv2.getTextSize(label_sub, font, font_scale * 0.7, 1)
        
        pad = 8
        label_w = max(tw_main, tw_sub) + pad * 2
        label_h = th_main + th_sub + pad * 3
        
        # Position label above plate (or below if too close to top edge)
        ly1 = max(10, y1 - label_h - 4) if y1 - label_h - 4 > 10 else min(h - label_h - 10, y2 + 4)
        ly2 = ly1 + label_h
        lx1 = max(10, x1)
        lx2 = min(w - 10, lx1 + label_w)
        
        # Draw dark background badge
        cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (15, 23, 42), -1)
        cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), box_color, 2)
        
        # Text inside badge
        text_y_main = ly1 + pad + th_main
        cv2.putText(vis, label_main, (lx1 + pad, text_y_main),
                    font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        
        text_y_sub = text_y_main + pad + th_sub
        cv2.putText(vis, label_sub, (lx1 + pad, text_y_sub),
                    font, font_scale * 0.7, (200, 220, 255), 1, cv2.LINE_AA)
                    
    return vis


def build_html_report(results: list[dict], html_path: Path, processed_dir: Path, crops_dir: Path) -> None:
    """Build a modern, interactive web review gallery."""
    total = len(results)
    detected = sum(1 for r in results if r["plates"])
    high_q = sum(1 for r in results if r["plates"] and r["plates"][0]["quality"] >= 0.70)
    med_q = sum(1 for r in results if r["plates"] and 0.40 <= r["plates"][0]["quality"] < 0.70)
    low_or_none = total - high_q - med_q
    
    det_rate = (detected / total * 100) if total else 0
    high_rate = (high_q / total * 100) if total else 0
    
    cards_html = []
    for r in results:
        fname = r["filename"]
        has_plate = bool(r["plates"])
        p = r["plates"][0] if has_plate else None
        
        plate_str = p["display_plate"] if p else "NO PLATE"
        q_val = p["quality"] if p else 0.0
        c_val = p["confidence"] if p else 0.0
        margin_val = p["margin"] if p else 0.0
        layout_str = p["layout"] if p else "-"
        bare_str = p["bare_ocr"] if p else "-"
        
        status_tag = "high" if q_val >= 0.70 else ("med" if q_val >= 0.40 else ("low" if has_plate else "none"))
        badge_bg = "#10b981" if status_tag == "high" else ("#f59e0b" if status_tag == "med" else ("#ef4444" if status_tag == "low" else "#64748b"))
        
        crop_rel = os.path.relpath(crops_dir / p['crop_file'], html_path.parent).replace('\\', '/') if (p and p.get("crop_file")) else ""
        proc_rel = os.path.relpath(processed_dir / fname, html_path.parent).replace('\\', '/')
        
        card = f"""
        <div class="card" data-status="{status_tag}" data-plate="{plate_str.lower()}" data-file="{fname.lower()}">
            <div class="card-header">
                <div class="plate-title">
                    <span class="badge" style="background:{badge_bg}">{plate_str}</span>
                    <span class="filename">{fname}</span>
                </div>
                <span class="conf-pill">Conf: {int(c_val*100)}%</span>
            </div>
            <div class="card-body">
                <div class="img-preview" onclick="openModal('{proc_rel}')">
                    <img src="{proc_rel}" loading="lazy" alt="{fname}">
                    <div class="zoom-hint">Click to enlarge</div>
                </div>
                <div class="details-panel">
                    <div class="crop-box">
                        <div class="detail-label">ZOOMED PLATE</div>
                        {f'<img src="{crop_rel}" alt="crop">' if crop_rel else '<div class="no-crop">No Crop</div>'}
                    </div>
                    <div class="meta-rows">
                        <div class="meta-row">
                            <span class="detail-label">Quality Score</span>
                            <span class="meta-val" style="color: {badge_bg}">{q_val:.2f} ({int(q_val*100)}%)</span>
                        </div>
                        <div class="meta-row">
                            <span class="detail-label">Margin to 2nd</span>
                            <span class="meta-val">{margin_val:.2f}</span>
                        </div>
                        <div class="meta-row">
                            <span class="detail-label">Matched Layout</span>
                            <span class="meta-val code">{layout_str}</span>
                        </div>
                        <div class="meta-row">
                            <span class="detail-label">Bare OCR Read</span>
                            <span class="meta-val code">{bare_str}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Car License Plate Recognition Verification Gallery</title>
    <style>
        :root {{
            --bg: #090d16;
            --surface: #131b2e;
            --surface-border: #22304d;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --primary: #3b82f6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 28px;
            min-height: 100vh;
        }}
        .header {{
            max-width: 1400px;
            margin: 0 auto 28px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--surface-border);
        }}
        .title-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 20px;
        }}
        h1 {{ font-size: 26px; font-weight: 700; color: #fff; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
        }}
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 10px;
            padding: 14px 18px;
        }}
        .stat-val {{ font-size: 24px; font-weight: 700; color: #fff; }}
        .stat-lbl {{ font-size: 13px; color: var(--text-muted); margin-top: 4px; }}
        
        .controls {{
            max-width: 1400px;
            margin: 0 auto 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 14px;
        }}
        .filters {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .btn {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            color: var(--text-main);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .btn:hover, .btn.active {{
            background: var(--primary);
            border-color: var(--primary);
            color: #fff;
        }}
        .search-box {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            color: #fff;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 14px;
            width: 280px;
        }}
        .search-box:focus {{ outline: none; border-color: var(--primary); }}
        
        .gallery {{
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.15s, border-color 0.15s;
        }}
        .card:hover {{
            transform: translateY(-2px);
            border-color: #3b82f6;
        }}
        .card-header {{
            padding: 12px 16px;
            background: rgba(0,0,0,0.25);
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--surface-border);
        }}
        .plate-title {{ display: flex; align-items: center; gap: 10px; }}
        .badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 16px;
            color: #fff;
            letter-spacing: 0.5px;
        }}
        .filename {{ font-size: 13px; color: var(--text-muted); }}
        .conf-pill {{ font-size: 12px; color: var(--text-muted); background: #1e293b; padding: 2px 8px; border-radius: 999px; }}
        
        .card-body {{
            padding: 14px;
            display: grid;
            grid-template-columns: 1fr 170px;
            gap: 14px;
            align-items: start;
        }}
        .img-preview {{
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--surface-border);
            cursor: pointer;
            background: #000;
            aspect-ratio: 4/3;
        }}
        .img-preview img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }}
        .zoom-hint {{
            position: absolute;
            bottom: 6px;
            right: 6px;
            background: rgba(0,0,0,0.7);
            color: #cbd5e1;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
            opacity: 0;
            transition: opacity 0.2s;
        }}
        .img-preview:hover .zoom-hint {{ opacity: 1; }}
        
        .details-panel {{ display: flex; flex-direction: column; gap: 10px; }}
        .crop-box {{
            background: #0b0f19;
            border: 1px solid var(--surface-border);
            border-radius: 8px;
            padding: 8px;
            text-align: center;
        }}
        .crop-box img {{
            max-width: 100%;
            max-height: 80px;
            border-radius: 4px;
            display: block;
            margin: 4px auto 0;
            background: #000;
        }}
        .no-crop {{ color: var(--text-muted); font-size: 12px; padding: 16px 0; }}
        .detail-label {{ font-size: 10px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; }}
        .meta-rows {{ display: flex; flex-direction: column; gap: 6px; }}
        .meta-row {{ display: flex; flex-direction: column; }}
        .meta-val {{ font-size: 13px; font-weight: 600; color: #f1f5f9; }}
        .meta-val.code {{ font-family: monospace; font-size: 12px; color: #93c5fd; }}
        
        /* Modal Lightbox */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0; top: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.88);
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .modal img {{
            max-width: 95%;
            max-height: 95%;
            border-radius: 8px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            border: 1px solid #334155;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title-row">
            <div>
                <h1>Vehicle Plate Recognition Verification</h1>
                <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">
                    Source: <code>footage/Cars/</code> &bull; Total Images Processed: <strong>{total}</strong>
                </p>
            </div>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-val">{total}</div>
                <div class="stat-lbl">Total Pictures</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: var(--success)">{detected} <span style="font-size:14px">({det_rate:.1f}%)</span></div>
                <div class="stat-lbl">Plates Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: #60a5fa">{high_q} <span style="font-size:14px">({high_rate:.1f}%)</span></div>
                <div class="stat-lbl">High Quality (&ge; 70%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: var(--warning)">{med_q}</div>
                <div class="stat-lbl">Medium Quality (40-69%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: var(--danger)">{low_or_none}</div>
                <div class="stat-lbl">Low / Missed</div>
            </div>
        </div>
    </div>
    
    <div class="controls">
        <div class="filters">
            <button class="btn active" onclick="filterStatus('all')">All ({total})</button>
            <button class="btn" onclick="filterStatus('high')">High Quality ({high_q})</button>
            <button class="btn" onclick="filterStatus('med')">Medium ({med_q})</button>
            <button class="btn" onclick="filterStatus('low')">Low Quality ({sum(1 for r in results if r['plates'] and r['plates'][0]['quality'] < 0.40)})</button>
            <button class="btn" onclick="filterStatus('none')">No Plate ({total - detected})</button>
        </div>
        <input type="text" class="search-box" id="search" placeholder="Search plate (e.g. LEF, 818, DSC)..." oninput="searchCards()">
    </div>
    
    <div class="gallery" id="gallery">
        {''.join(cards_html)}
    </div>
    
    <div class="modal" id="modal" onclick="closeModal()">
        <img id="modal-img" src="" alt="Full preview">
    </div>
    
    <script>
        let currentFilter = 'all';
        function filterStatus(status) {{
            currentFilter = status;
            document.querySelectorAll('.filters .btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            applyFilters();
        }}
        function searchCards() {{
            applyFilters();
        }}
        function applyFilters() {{
            const q = document.getElementById('search').value.toLowerCase().trim();
            document.querySelectorAll('.card').forEach(card => {{
                const st = card.getAttribute('data-status');
                const plate = card.getAttribute('data-plate');
                const file = card.getAttribute('data-file');
                
                const matchesStatus = (currentFilter === 'all') || (st === currentFilter);
                const matchesSearch = !q || plate.includes(q) || file.includes(q);
                
                card.style.display = (matchesStatus && matchesSearch) ? 'flex' : 'none';
            }});
        }}
        function openModal(src) {{
            document.getElementById('modal-img').src = src;
            document.getElementById('modal').style.display = 'flex';
        }}
        function closeModal() {{
            document.getElementById('modal').style.display = 'none';
        }}
    </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Process car photos dataset")
    parser.add_argument("--input", default=str(ROOT / "footage" / "Cars"), help="Input directory")
    parser.add_argument("--out-annotated", default=None, help="Output directory for annotated photos")
    parser.add_argument("--out-crops", default=None, help="Output directory for plate crops")
    parser.add_argument("--csv", default=None, help="Output CSV path")
    parser.add_argument("--json", default=None, help="Output JSON path")
    parser.add_argument("--html", default=None, help="Output HTML review path")
    parser.add_argument("--plate-conf", type=float, default=0.35, help="Plate detection confidence threshold")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="Inference device")
    args = parser.parse_args()

    cfg = load_config(None)
    cfg["models"]["device"] = args.device
    cfg["models"]["cpu_threads"] = 4
    
    cfg["models"]["plate_confidence"] = args.plate_conf
    fmt = build_plate_format(cfg)
    rec = build_recognizers(cfg["models"], 0.40)
    temp = float(cfg.get("ocr", {}).get("temperature", 2.0))
    
    cars_dir = Path(args.input)
    if not cars_dir.is_dir():
        print(f"Error: Directory not found: {cars_dir}")
        return 1
        
    stem = cars_dir.name
    out_annotated = Path(args.out_annotated) if args.out_annotated else cars_dir.parent / f"{stem}_processed"
    out_crops = Path(args.out_crops) if args.out_crops else cars_dir.parent / f"{stem}_crops"
    csv_path = Path(args.csv) if args.csv else cars_dir.parent / f"{stem}_results.csv"
    json_path = Path(args.json) if args.json else cars_dir.parent / f"{stem}_results.json"
    html_path = Path(args.html) if args.html else cars_dir.parent / f"{stem}_review.html"

    # Get all unique image files
    files = sorted(list({f for f in cars_dir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")}))
    total_files = len(files)
    print(f"Starting batch recognition on {total_files} car photos from '{cars_dir}'...")
    
    # Destination directories
    out_annotated.mkdir(parents=True, exist_ok=True)
    out_crops.mkdir(parents=True, exist_ok=True)
    
    results = []
    t0 = time.time()
    strip_two_row_year = bool(cfg.get("ocr", {}).get("strip_two_row_year", True))
    
    pbar = tqdm(files, unit="photo", desc="Processing plates")
    for f in pbar:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Plate detection via YOLOv9
        detected_plates = rec.plates.predict(img)
        
        plates_info = []
        for pl in detected_plates:
            b = pl.bounding_box
            x1, y1, x2, y2 = max(0, int(b.x1)), max(0, int(b.y1)), min(w, int(b.x2)), min(h, int(b.y2))
            pw, ph = x2 - x1, y2 - y1
            if pw < 15 or ph < 8:
                continue
                
            pad_x, pad_y = int(pw * 0.02), int(ph * 0.02)
            crop = img[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)]
            
            fr = read_plate(rec.reader, crop, fmt.templates, temperature=temp, strip_two_row_year=strip_two_row_year)
            if fr is not None and fr.free_text.strip():
                cands = decode_templates(fr.logp, fmt.templates, top=2)
                best_txt = cands[0].text if cands else fr.free_text
                best_q = cands[0].quality if cands else fr.free_conf
                best_margin = cands[0].margin if cands else 0.0
                best_layout = cands[0].template.slots if cands else "FREE"
                disp_plate = display(best_txt)
            else:
                best_txt, best_q, best_margin, best_layout = "", 0.0, 0.0, "UNKNOWN"
                disp_plate = "UNREADABLE"
                
            crop_name = f"{f.stem}_plate_{x1}_{y1}.jpg"
            cv2.imwrite(str(out_crops / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            plates_info.append({
                "box": (x1, y1, x2, y2),
                "confidence": float(pl.confidence),
                "plate": best_txt,
                "display_plate": disp_plate,
                "quality": float(best_q),
                "margin": float(best_margin),
                "layout": best_layout,
                "bare_ocr": fr.free_text if fr else "",
                "crop_file": crop_name,
            })
            
        # Fallback 1: Zoomed search inside detected vehicle box (for high-res photos where full-frame misses)
        if not plates_info and min(w, h) >= 350:
            v_dets = rec.vehicles.predict(img)
            valid_vehicles = [v for v in v_dets if v.label in ("car", "truck", "bus", "motorcycle") and v.confidence >= 0.35]
            valid_vehicles.sort(key=lambda v: (v.bounding_box.x2 - v.bounding_box.x1) * (v.bounding_box.y2 - v.bounding_box.y1), reverse=True)
            for v in valid_vehicles[:2]:
                vb = v.bounding_box
                vx1, vy1, vx2, vy2 = max(0, int(vb.x1)), max(0, int(vb.y1)), min(w, int(vb.x2)), min(h, int(vb.y2))
                vcrop = img[vy1:vy2, vx1:vx2]
                if vcrop.size == 0 or vcrop.shape[0] < 30 or vcrop.shape[1] < 50:
                    continue
                v_plates = rec.plates.predict(vcrop)
                for pl in v_plates:
                    pb = pl.bounding_box
                    x1, y1 = vx1 + max(0, int(pb.x1)), vy1 + max(0, int(pb.y1))
                    x2, y2 = min(w, vx1 + int(pb.x2)), min(h, vy1 + int(pb.y2))
                    pw, ph = x2 - x1, y2 - y1
                    if pw < 15 or ph < 8:
                        continue
                    pad_x, pad_y = int(pw * 0.02), int(ph * 0.02)
                    crop = img[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)]
                    fr = read_plate(rec.reader, crop, fmt.templates, temperature=temp, strip_two_row_year=strip_two_row_year)
                    if fr is not None and fr.free_text.strip():
                        cands = decode_templates(fr.logp, fmt.templates, top=2)
                        best_txt = cands[0].text if cands else fr.free_text
                        best_q = cands[0].quality if cands else fr.free_conf
                        best_margin = cands[0].margin if cands else 0.0
                        best_layout = cands[0].template.slots if cands else "FREE"
                        disp_plate = display(best_txt)
                    else:
                        best_txt, best_q, best_margin, best_layout = "", 0.0, 0.0, "UNKNOWN"
                        disp_plate = "UNREADABLE"
                    crop_name = f"{f.stem}_plate_{x1}_{y1}.jpg"
                    cv2.imwrite(str(out_crops / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    plates_info.append({
                        "box": (x1, y1, x2, y2),
                        "confidence": float(pl.confidence),
                        "plate": best_txt,
                        "display_plate": disp_plate,
                        "quality": float(best_q),
                        "margin": float(best_margin),
                        "layout": best_layout,
                        "bare_ocr": fr.free_text if fr else "",
                        "crop_file": crop_name,
                    })
                if plates_info:
                    break

        # Fallback 2: Direct plate crop reading (for images that are already cropped plates)
        if not plates_info and (1.2 <= (w / max(1, h)) <= 5.5) and (w <= 800 and h <= 400):
            fr = read_plate(rec.reader, img, fmt.templates, temperature=temp, strip_two_row_year=strip_two_row_year)
            if fr is not None:
                cands = decode_templates(fr.logp, fmt.templates, top=2)
                best_txt = cands[0].text if cands else fr.free_text
                best_q = cands[0].quality if cands else fr.free_conf
                best_margin = cands[0].margin if cands else 0.0
                best_layout = cands[0].template.slots if cands else "FREE"
                if best_q >= 0.30 and len(best_txt) >= 4:
                    crop_name = f"{f.stem}_plate_0_0.jpg"
                    cv2.imwrite(str(out_crops / crop_name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    plates_info.append({
                        "box": (0, 0, w, h),
                        "confidence": 0.90,
                        "plate": best_txt,
                        "display_plate": display(best_txt),
                        "quality": float(best_q),
                        "margin": float(best_margin),
                        "layout": best_layout,
                        "bare_ocr": fr.free_text if fr else "",
                        "crop_file": crop_name,
                    })

        # Sort plates by area (largest plate first, usually the primary vehicle)
        plates_info.sort(key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]), reverse=True)
        
        # Save annotated image
        annotated = annotate_image(img, plates_info)
        cv2.imwrite(str(out_annotated / f.name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        results.append({
            "filename": f.name,
            "width": w,
            "height": h,
            "plates": plates_info,
        })
        
        top_txt = plates_info[0]["display_plate"] if plates_info else "none"
        pbar.set_postfix_str(f"Last: {top_txt}")
        
    pbar.close()
    elapsed = time.time() - t0
    
    # Save CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "detected", "plate", "display_plate", "quality", "confidence", "margin", "layout", "bare_ocr", "box"])
        for r in results:
            if r["plates"]:
                for p in r["plates"]:
                    writer.writerow([r["filename"], 1, p["plate"], p["display_plate"], f"{p['quality']:.3f}", f"{p['confidence']:.3f}", f"{p['margin']:.2f}", p["layout"], p["bare_ocr"], str(p["box"])])
            else:
                writer.writerow([r["filename"], 0, "", "NO_PLATE", 0.0, 0.0, 0.0, "", "", ""])
                
    # Save JSON
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    # Generate HTML review page
    build_html_report(results, html_path, out_annotated, out_crops)
    
    det_count = sum(1 for r in results if r["plates"])
    print(f"\nFinished in {elapsed:.1f}s ({elapsed/max(1, len(results)):.2f}s/photo).")
    print(f"Recognized plates in {det_count} of {len(results)} photos ({det_count/len(results)*100:.1f}%).")
    print(f"Annotated images saved to: '{out_annotated}'")
    print(f"Plate crops saved to:      '{out_crops}'")
    print(f"CSV report:                '{csv_path}'")
    print(f"Visual review page:        '{html_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
