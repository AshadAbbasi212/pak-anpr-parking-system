"""
Default OCR module.
"""

import os
from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
from fast_plate_ocr import LicensePlateRecognizer
from fast_plate_ocr.inference.hub import OcrModel

from fast_alpr.base import BaseOCR, OcrResult


class DefaultOCR(BaseOCR):
    """
    Default OCR class for license plate recognition using `fast-plate-ocr` models.
    """

    def __init__(
        self,
        hub_ocr_model: OcrModel | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions | None = None,
        model_path: str | os.PathLike | None = None,
        config_path: str | os.PathLike | None = None,
        force_download: bool = False,
    ) -> None:
        """
        Initialize the DefaultOCR with the specified parameters. Uses `fast-plate-ocr`'s
        `LicensePlateRecognizer`

        Parameters:
            hub_ocr_model: The name of the OCR model from the model hub.
            device: The device to run the model on. Options are "cuda", "cpu", or "auto". Defaults
             to "auto".
            providers: The execution providers to use in ONNX Runtime. If None, the default
             providers are used.
            sess_options: Custom session options for ONNX Runtime. If None, default session options
             are used.
            model_path: Path to a custom OCR model file. If None, the model is downloaded from the
             hub or cache.
            config_path: Path to a custom configuration file. If None, the default configuration is
             used.
            force_download: If True, forces the download of the model and overwrites any existing
             files.
        """
        self.ocr_model = LicensePlateRecognizer(
            hub_ocr_model=hub_ocr_model,
            device=device,
            providers=providers,
            sess_options=sess_options,
            onnx_model_path=model_path,
            plate_config_path=config_path,
            force_download=force_download,
        )

    def predict(self, cropped_plate: np.ndarray) -> OcrResult | None:
        """
        Perform OCR on a cropped license plate image.

        Parameters:
            cropped_plate: The cropped image of the license plate in BGR format.

        Returns:
            OcrResult: An object containing the recognized text and per-character confidence.
        """
        if cropped_plate is None:
            return None
        if self.ocr_model.config.image_color_mode == "grayscale":
            cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        elif self.ocr_model.config.image_color_mode == "rgb":
            cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
        prediction = self.ocr_model.run_one(cropped_plate, return_confidence=True)

        char_probs = prediction.char_probs
        confidence: float | list[float] = (
            0.0 if char_probs is None else [float(x) for x in char_probs.tolist()]
        )
        return OcrResult(
            text=prediction.plate,
            confidence=confidence,
            region=prediction.region,
            region_confidence=prediction.region_prob,
        )
