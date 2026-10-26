"""
Dataset module.
"""

import math
import os
import warnings

import albumentations as A
import numpy as np
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset

from fast_plate_ocr.core.process import read_and_resize_plate_image
from fast_plate_ocr.train.data.annotations import read_annotations_csv
from fast_plate_ocr.train.model.config import PlateConfig
from fast_plate_ocr.train.utilities import utils


class PlateRecognitionPyDataset(PyDataset):
    """
    Custom PyDataset for OCR license plate recognition.
    """

    def __init__(
        self,
        annotations_file: str | os.PathLike,
        plate_config: PlateConfig,
        batch_size: int,
        transform: A.Compose | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Load annotations
        annotations = read_annotations_csv(annotations_file)
        annotations["image_path"] = (
            os.path.dirname(os.path.realpath(annotations_file)) + os.sep + annotations["image_path"]
        )
        # Check that plate lengths do not exceed max_plate_slots.
        assert (annotations["plate_text"].str.len() <= plate_config.max_plate_slots).all(), (
            "Plates are longer than max_plate_slots specified param. Change the parameter."
        )
        self.annotations = annotations

        self.plate_config = plate_config
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Region recognition support is active only if both config and column exist
        has_plate_region_col = "plate_region" in self.annotations.columns
        regions_defined = self.plate_config.has_region_recognition
        self.region_recognition = has_plate_region_col and regions_defined

        if has_plate_region_col and not regions_defined:
            warnings.warn(
                "plate_region column found in annotations, but plate_config.plate_regions "
                "is None or empty. plate_region labels will be ignored.",
                stacklevel=2,
            )

        # Build mapping for regions-recognition if active
        if self.region_recognition:
            if not self.plate_config.plate_regions:
                raise ValueError("plate_regions must be defined and non-empty when region recognition is enabled.")
            # Validate there are no missing values
            if self.annotations["plate_region"].isna().any():
                raise ValueError("Found NaN values in 'plate_region' column.")
            # Validate regions present in CSV against config
            missing = set(self.annotations["plate_region"].unique()) - set(self.plate_config.plate_regions)
            if missing:
                raise ValueError(f"Found region labels in CSV not present in config.plate_regions: {sorted(missing)}")
            self.region_to_idx = {c: i for i, c in enumerate(self.plate_config.plate_regions)}
            self.num_regions = len(self.plate_config.plate_regions)
            self._region_eye = np.eye(self.num_regions, dtype=np.float32)

        # Shuffle once at initialization if `shuffle=True`
        self._shuffle_data()

    def __len__(self) -> int:
        return math.ceil(len(self.annotations) / self.batch_size)

    def __getitem__(self, idx: int):
        # Determine the idx-es of current batch
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.annotations))
        batch_df = self.annotations.iloc[low:high]

        batch_x, batch_y_plate = [], []
        region_idx = (
            batch_df["plate_region"].map(self.region_to_idx).to_numpy(dtype=np.int32, copy=False)
            if self.region_recognition
            else None
        )

        for row in batch_df.itertuples(index=False):
            image_path = row.image_path
            plate_text = row.plate_text
            # Read and process image
            x = read_and_resize_plate_image(
                image_path=image_path,  # type: ignore[arg-type]
                img_height=self.plate_config.img_height,
                img_width=self.plate_config.img_width,
                image_color_mode=self.plate_config.image_color_mode,
                keep_aspect_ratio=self.plate_config.keep_aspect_ratio,
                interpolation_method=self.plate_config.interpolation,
                padding_color=self.plate_config.padding_color,
            )
            # Transform target for plate
            y_plate = utils.target_transform(
                plate_text=plate_text,  # type: ignore[arg-type]
                max_plate_slots=self.plate_config.max_plate_slots,
                alphabet=self.plate_config.alphabet,
                pad_char=self.plate_config.pad_char,
            )
            # Apply augmentation if provided
            if self.transform:
                x = self.transform(image=x)["image"]
            batch_x.append(x)
            batch_y_plate.append(y_plate)

        batch_x_np = np.array(batch_x)
        batch_y_plate_np = np.array(batch_y_plate)

        if self.region_recognition:
            assert region_idx is not None
            batch_y_region_np = self._region_eye[region_idx]
            y_dict = {"plate": batch_y_plate_np, "region": batch_y_region_np}
        else:
            y_dict = {"plate": batch_y_plate_np}

        return batch_x_np, y_dict

    def _shuffle_data(self) -> None:
        if self.shuffle:
            self.annotations = self.annotations.sample(frac=1.0, random_state=None).reset_index(drop=True)

    def on_epoch_begin(self) -> None:
        # Optionally shuffle the dataset at the start of each epoch
        self._shuffle_data()
