"""
Helpers to load annotations CSV files used for OCR training workflows.
"""

import os
import warnings

import pandas as pd

ALLOWED_COLUMNS = {"image_path", "plate_text", "plate_region"}
"""Allowed columns in annotations CSVs."""


def read_annotations_csv(annotations_file: str | os.PathLike) -> pd.DataFrame:
    """
    Read an annotations CSV with consistent `plate_text` handling.

    :param annotations_file: Path to annotations CSV.
    :return: DataFrame with annotations.
    """
    annotations = pd.read_csv(annotations_file, dtype={"plate_text": str})
    annotations["plate_text"] = annotations["plate_text"].fillna("")
    unexpected_columns = sorted(set(annotations.columns) - ALLOWED_COLUMNS)
    if unexpected_columns:
        warnings.warn(
            f"Unexpected column(s) in annotations CSV {annotations_file}: {unexpected_columns}.",
            stacklevel=2,
        )
    return annotations
