"""
License Plate OCR config. This config file defines how license plate images and text should be
preprocessed for OCR model training and inference.
"""

from pathlib import Path
from typing import Annotated, TypeAlias

import annotated_types
import yaml
from pydantic import (
    BaseModel,
    PositiveInt,
    StringConstraints,
    computed_field,
    model_validator,
)

from fast_plate_ocr.core.types import ImageColorMode, ImageInterpolation, PathLike

UInt8: TypeAlias = Annotated[int, annotated_types.Ge(0), annotated_types.Le(255)]
"""
An integer in the range [0, 255], used for color channel values.
"""


class PlateConfig(BaseModel, extra="forbid", frozen=True):
    """
    Model License Plate config.
    """

    max_plate_slots: PositiveInt
    """
    Max number of plate slots supported. This represents the number of model classification heads.
    """
    alphabet: str
    """
    All the possible character set for the model output.
    """
    pad_char: Annotated[str, StringConstraints(min_length=1, max_length=1)]
    """
    Padding character for plates which length is smaller than MAX_PLATE_SLOTS.
    """
    img_height: PositiveInt
    """
    Image height which is fed to the model.
    """
    img_width: PositiveInt
    """
    Image width which is fed to the model.
    """
    keep_aspect_ratio: bool = False
    """
    Keep aspect ratio of the input image.
    """
    interpolation: ImageInterpolation = "linear"
    """
    Interpolation method used for resizing the input image.
    """
    image_color_mode: ImageColorMode = "grayscale"
    """
    Input image color mode. Use 'grayscale' for single-channel input or 'rgb' for 3-channel input.
    """
    padding_color: tuple[UInt8, UInt8, UInt8] | UInt8 = (114, 114, 114)
    """
    Padding color used when keep_aspect_ratio is True. For grayscale images, this should be a single
    integer and for RGB images, this must be a tuple of three integers.
    """
    plate_regions: list[str] | None = None
    """
    Optional list specifying the regions/countries whose license plates the model can recognize.
    """

    @computed_field  # type: ignore[misc]
    @property
    def vocabulary_size(self) -> int:
        return len(self.alphabet)

    @computed_field  # type: ignore[misc]
    @property
    def pad_idx(self) -> int:
        return self.alphabet.index(self.pad_char)

    @computed_field  # type: ignore[misc]
    @property
    def num_channels(self) -> int:
        return 3 if self.image_color_mode == "rgb" else 1

    @computed_field  # type: ignore[misc]
    @property
    def has_region_recognition(self) -> bool:
        return bool(self.plate_regions)

    @model_validator(mode="after")
    def check_alphabet_and_pad(self) -> "PlateConfig":
        # `pad_char` must be in alphabet
        if self.pad_char not in self.alphabet:
            raise ValueError("Pad character must be present in model alphabet.")
        # all chars in alphabet must be unique
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("Alphabet must not contain duplicate characters.")
        return self

    @model_validator(mode="after")
    def check_plate_regions(self) -> "PlateConfig":
        # Ensure there are no duplicate characters in plate_regions
        if self.plate_regions and len(set(self.plate_regions)) != len(self.plate_regions):
            raise ValueError("Plate regions must not contain duplicate characters.")
        return self


def load_plate_config_from_yaml(yaml_path: PathLike) -> PlateConfig:
    """
    Reads and parses a YAML file containing the plate configuration.

    Args:
        yaml_path: Path to the YAML file containing the plate config.

    Returns:
        PlateConfig: Parsed and validated plate configuration.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    if not Path(yaml_path).is_file():
        raise FileNotFoundError(f"Plate config '{yaml_path}' doesn't exist.")
    with open(yaml_path, encoding="utf-8") as f_in:
        yaml_content = yaml.safe_load(f_in)
    config = PlateConfig(**yaml_content)
    return config
