"""Cheap frame-difference motion detector.

Used for two things:
  * "motion gate": while the road is empty the heavy plate/vehicle detectors are not run at all (saves CPU);
  * "is this vehicle really moving?": parked or standing cars in view never get a slip.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, width: int = 192, pixel_threshold: int = 22, history: int = 3):
        self.width = width
        self.pixel_threshold = pixel_threshold
        self._hist: deque = deque(maxlen=history)
        self.mask: np.ndarray | None = None
        self.fraction = 0.0
        self._sx = 1.0
        self._sy = 1.0

    def update(self, frame: np.ndarray, roi_mask: np.ndarray | None = None) -> float:
        """Feed a new frame. Returns the share (0-1) of the picture that changed."""
        h, w = frame.shape[:2]
        sh = max(1, int(round(h * self.width / w)))
        small = cv2.resize(frame, (self.width, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        self._sx, self._sy = self.width / w, sh / h
        if self._hist:
            diff = cv2.absdiff(gray, self._hist[0])
            # Adaptive threshold: use the higher of the configured minimum and scene noise floor.
            # This prevents false triggers from moving shadows on sunny days while still catching
            # slow vehicles at night when pixel differences are small.
            noise_floor = float(np.percentile(diff, 90))
            effective_thresh = max(self.pixel_threshold, int(noise_floor * 1.3))
            mask = (diff > effective_thresh).astype(np.uint8)
            if roi_mask is not None:
                rm = cv2.resize(roi_mask, (self.width, sh), interpolation=cv2.INTER_NEAREST)
                mask = mask * rm
                denom = max(1, int(rm.sum()))
                self.fraction = float(mask.sum()) / denom
            else:
                self.fraction = float(mask.mean())
            self.mask = mask
        else:
            self.mask, self.fraction = None, 0.0
        self._hist.append(gray)
        return self.fraction

    def box_fraction(self, box: tuple[int, int, int, int]) -> float:
        """Share (0-1) of the pixels inside `box` (full-resolution coordinates) that changed."""
        if self.mask is None:
            return 0.0
        mh, mw = self.mask.shape[:2]
        x1 = max(0, min(mw, int(box[0] * self._sx)))
        x2 = max(0, min(mw, int(np.ceil(box[2] * self._sx))))
        y1 = max(0, min(mh, int(box[1] * self._sy)))
        y2 = max(0, min(mh, int(np.ceil(box[3] * self._sy))))
        region = self.mask[y1:y2, x1:x2]
        return float(region.mean()) if region.size else 0.0
