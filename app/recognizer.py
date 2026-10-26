"""Builds the vehicle detector, plate detector and plate reader from the bundled model files.
Everything is loaded from disk - no internet needed."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort

from . import ROOT  # noqa: F401  (makes third_party importable)
from .plate_reader import PlateReader
from fast_alpr.default_ocr import DefaultOCR
from open_image_models import create_detector
from open_image_models.detection.core.coco import COCO_CLASSES

log = logging.getLogger("recognizer")


def _providers(device: str) -> list[str]:
    if str(device).lower() in ("gpu", "cuda", "directml", "dml"):
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            log.info("Using GPU (CUDA).")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "DmlExecutionProvider" in avail:
            log.info("Using GPU (DirectML hardware acceleration).")
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        log.warning("device: gpu was requested but no GPU execution provider is available - using CPU instead.")
    return ["CPUExecutionProvider"]


@dataclass
class Recognizers:
    plates: object      # license plate detector
    ocr: object         # plate text reader
    vehicles: object    # COCO vehicle detector
    is_gpu: bool = False
    reader: object = None   # PlateReader: OCR with per-position probabilities (used by the pipeline)


def build_recognizers(models_cfg: dict, vehicle_conf: float = 0.45) -> Recognizers:
    """One set per camera, so the two cameras can be analysed in parallel."""
    for key in ("plate_detector", "ocr_model", "ocr_config", "vehicle_detector"):
        if not Path(models_cfg[key]).is_file():
            raise FileNotFoundError(f"Model file missing: {models_cfg[key]}  (check the 'models' section of config.yaml)")

    providers = _providers(models_cfg.get("device", "cpu"))

    def session_options() -> ort.SessionOptions:
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(models_cfg.get("cpu_threads", 2))
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        return so

    plates = create_detector(
        models_cfg["plate_detector"], backend="yolo_v9", class_labels=("License Plate",),
        conf_thresh=float(models_cfg.get("plate_confidence", 0.4)),
        providers=providers, sess_options=session_options())
    vehicles = create_detector(
        models_cfg["vehicle_detector"], backend="rf_detr", class_labels=COCO_CLASSES,
        conf_thresh=float(vehicle_conf), providers=providers, sess_options=session_options())
    ocr = DefaultOCR(
        hub_ocr_model=None, device="cuda" if providers[0].startswith("CUDA") else "cpu",
        providers=providers, sess_options=session_options(),
        model_path=models_cfg["ocr_model"], config_path=models_cfg["ocr_config"])
    is_gpu = providers[0] != "CPUExecutionProvider"
    # Same OCR model, but giving the raw probabilities of every character position (needed for the Pakistani rules).
    reader = PlateReader(models_cfg["ocr_model"], providers, session_options(), fixed_batch=4 if is_gpu else 0)
    return Recognizers(plates=plates, ocr=ocr, vehicles=vehicles, is_gpu=is_gpu, reader=reader)
