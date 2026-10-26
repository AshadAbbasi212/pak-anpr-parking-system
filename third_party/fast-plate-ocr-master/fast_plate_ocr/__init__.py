"""
fast-plate-ocr package.
"""

from fast_plate_ocr.core.types import PlatePrediction
from fast_plate_ocr.inference.plate_recognizer import LicensePlateRecognizer

__all__ = ["LicensePlateRecognizer", "PlatePrediction"]
