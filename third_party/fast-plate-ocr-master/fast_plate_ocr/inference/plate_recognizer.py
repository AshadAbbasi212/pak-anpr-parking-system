"""
ONNX Runtime inference for license plate recognition.
"""

import logging
import pathlib
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

try:
    import onnxruntime as ort
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "ONNX Runtime is not installed. Run: pip install 'fast-plate-ocr[onnx]' (or [onnx-gpu], etc.)"
    ) from e
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fast_plate_ocr.core.process import (
    postprocess_output,
    preprocess_image,
    read_and_resize_plate_image,
    resize_image,
)
from fast_plate_ocr.core.types import BatchArray, BatchOrImgLike, ImgLike, PathLike, PlatePrediction
from fast_plate_ocr.core.utils import measure_time
from fast_plate_ocr.inference import hub
from fast_plate_ocr.inference.config import PlateConfig
from fast_plate_ocr.inference.hub import OcrModel


def _frame_from(item: ImgLike, cfg: PlateConfig) -> BatchArray:
    """
    Converts a single image-like input into a normalized (H, W, C) NumPy array ready for model
    inference. It handles both file paths and in-memory images. If input is a file path, the image
    is read and resized using the configuration provided. If it's a NumPy array, it is validated and
    resized accordingly.
    """
    # If it's a path, read and resize
    if isinstance(item, (str | pathlib.PurePath)):
        return read_and_resize_plate_image(
            item,
            img_height=cfg.img_height,
            img_width=cfg.img_width,
            image_color_mode=cfg.image_color_mode,
            keep_aspect_ratio=cfg.keep_aspect_ratio,
            interpolation_method=cfg.interpolation,
            padding_color=cfg.padding_color,
        )

    # Otherwise it must be a numpy array
    if not isinstance(item, np.ndarray):
        raise TypeError(f"Unsupported element type: {type(item)}")

    # If it has (N, H, W, C) shape we assume it's ready for inference
    if item.ndim == 4:
        return item

    # If it's a single frame resize accordingly
    return resize_image(
        item,
        cfg.img_height,
        cfg.img_width,
        image_color_mode=cfg.image_color_mode,
        keep_aspect_ratio=cfg.keep_aspect_ratio,
        interpolation_method=cfg.interpolation,
        padding_color=cfg.padding_color,
    )


def _load_image_from_source(source: BatchOrImgLike, cfg: PlateConfig) -> BatchArray:
    """
    Converts an image input or batch of inputs into a 4-D NumPy array (N, H, W, C).

    This utility supports a wide range of input formats, including single images or batches, file
    paths or NumPy arrays. It ensures the result is always a model-ready batch.

    Supported input formats:
    - Single path (`str` or `PathLike`) -> image is read and resized
    - List or tuple of paths -> each image is read and resized
    - Single 2D or 3D NumPy array -> resized and wrapped in a batch
    - List or tuple of NumPy arrays -> each image is resized and batched
    - Single 4D NumPy array with shape (N, H, W, C) -> returned as is

    Args:
        source: A single image or batch of images in path or NumPy array format.
        cfg: The configuration object that defines image preprocessing parameters.

    Returns:
        A 4D NumPy array of shape (N, H, W, C), dtype uint8, ready for model inference.
    """
    if isinstance(source, np.ndarray) and source.ndim == 4:
        return source

    items: Sequence[ImgLike] = (
        source
        if isinstance(source, Sequence) and not isinstance(source, (str | pathlib.PurePath | np.ndarray))
        else [source]
    )

    frames: list[BatchArray] = [
        frame
        for item in items
        for frame in (
            _frame_from(item, cfg)  # type: ignore[attr-defined]
            if isinstance(item, np.ndarray) and item.ndim == 4
            else [_frame_from(item, cfg)]
        )
    ]

    return np.stack(frames, axis=0, dtype=np.uint8)


class LicensePlateRecognizer:
    """
    ONNX Runtime inference class for license plate recognition.
    """

    def __init__(
        self,
        hub_ocr_model: OcrModel | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions | None = None,
        onnx_model_path: PathLike | None = None,
        plate_config_path: PathLike | None = None,
        force_download: bool = False,
    ) -> None:
        """
        Initializes the `LicensePlateRecognizer` with the specified model and inference device.

        The current OCR models available from the HUB are:

        - `cct-s-v2-global-model`: OCR model trained with **global** plates data. Based on Compact
            Convolutional Transformer (CCT) architecture. This is the recommended **S** default.
        - `cct-xs-v2-global-model`: OCR model trained with **global** plates data. Based on Compact
            Convolutional Transformer (CCT) architecture. This is the **XS** variant of the v2
            family.
        - `cct-s-v1-global-model`: OCR model trained with **global** plates data. Based on Compact
            Convolutional Transformer (CCT) architecture. This is the **S** variant.
        - `cct-xs-v1-global-model`: OCR model trained with **global** plates data. Based on Compact
            Convolutional Transformer (CCT) architecture. This is the **XS** variant.
        - `argentinian-plates-cnn-model`: OCR for **Argentinian** license plates. Uses fully conv
            architecture.
        - `argentinian-plates-cnn-synth-model`: OCR for **Argentinian** license plates trained with
            synthetic and real data. Uses fully conv architecture.
        - `european-plates-mobile-vit-v2-model`: OCR for **European** license plates. Uses
            MobileVIT-2 for the backbone.
        - `global-plates-mobile-vit-v2-model`: OCR for **global** license plates (+65 countries).
            Uses MobileVIT-2 for the backbone.

        Args:
            hub_ocr_model: Name of the OCR model to use from the HUB.
            device: Device type for inference. Should be one of ('cpu', 'cuda', 'auto'). If
                'auto' mode, the device will be deduced from
                `onnxruntime.get_available_providers()`.
            providers: Optional sequence of providers in order of decreasing precedence. If not
                specified, all available providers are used based on the device argument.
            sess_options: Advanced session options for ONNX Runtime.
            onnx_model_path: Path to ONNX model file to use (In case you want to use a custom one).
            plate_config_path: Path to config file to use (In case you want to use a custom one).
            force_download: Force and download the model, even if it already exists.
        Returns:
            None.
        """
        self.logger = logging.getLogger(__name__)

        if providers is not None:
            self.providers = providers
            self.logger.info("Using custom providers: %s", providers)
        else:
            if device == "cuda":
                self.providers = ["CUDAExecutionProvider"]
            elif device == "cpu":
                self.providers = ["CPUExecutionProvider"]
            elif device == "auto":
                self.providers = ort.get_available_providers()
            else:
                raise ValueError(f"Device should be one of ('cpu', 'cuda', 'auto'). Got '{device}'.")

            self.logger.info("Using device '%s' with providers: %s", device, self.providers)

        if onnx_model_path and plate_config_path:
            onnx_model_path = pathlib.Path(onnx_model_path)
            plate_config_path = pathlib.Path(plate_config_path)
            if not onnx_model_path.exists() or not plate_config_path.exists():
                raise FileNotFoundError("Missing model/config file!")
            self.model_name = onnx_model_path.stem
        elif hub_ocr_model:
            self.model_name = hub_ocr_model
            onnx_model_path, plate_config_path = hub.download_model(
                model_name=hub_ocr_model, force_download=force_download
            )
        else:
            raise ValueError("Either provide a model from the HUB or a custom model_path and config_path")

        self.config = PlateConfig.from_yaml(plate_config_path)
        self.model = ort.InferenceSession(onnx_model_path, providers=self.providers, sess_options=sess_options)
        self.output_names = [output.name for output in self.model.get_outputs()]
        self.plate_output_name = "plate" if "plate" in self.output_names else self.output_names[0]
        self.region_output_name = "region" if "region" in self.output_names else None
        self.has_region_head = self.region_output_name is not None and self.config.has_region_recognition

        if self.region_output_name is None and self.config.has_region_recognition:
            self.logger.warning(
                "Plate config declares regions but the loaded model has no 'region' output. "
                "Region predictions will be disabled."
            )
        if self.region_output_name is not None and not self.config.has_region_recognition:
            self.logger.warning(
                "Model exports a 'region' output but the plate config has no region list. "
                "Region predictions will be disabled."
            )
        self._fetch_outputs = (
            [self.plate_output_name] if not self.has_region_head else [self.plate_output_name, self.region_output_name]
        )

    def benchmark(
        self,
        n_iter: int = 2_500,
        batch_size: int = 1,
        include_processing: bool = False,
        warmup: int = 250,
    ) -> None:
        """
        Run an inference benchmark and pretty print the results.

        It reports the following metrics:

        * **Average latency per batch** (milliseconds)
        * **Throughput** in *plates / second* (PPS), i.e., how many plates the pipeline can process
          per second at the chosen ``batch_size``.

        Args:
            n_iter: The number of iterations to run the benchmark. This determines how many times
                the inference will be executed to compute the average performance metrics.
            batch_size: Batch size to use for the benchmark.
            include_processing: Indicates whether the benchmark should include preprocessing and
                postprocessing times in the measurement.
            warmup: Number of warmup iterations to run before the benchmark.
        """
        x = np.random.randint(
            0,
            256,
            size=(
                batch_size,
                self.config.img_height,
                self.config.img_width,
                self.config.num_channels,
            ),
            dtype=np.uint8,
        )

        # Warm-up
        for _ in range(warmup):
            if include_processing:
                self.run(x)
            else:
                self.model.run(None, {"input": x})

        # Timed loop
        cum_time = 0.0
        for _ in range(n_iter):
            with measure_time() as time_taken:
                if include_processing:
                    self.run(x)
                else:
                    self.model.run(None, {"input": x})
            cum_time += time_taken()

        avg_time_ms = cum_time / n_iter if n_iter else 0.0
        pps = (1_000 / avg_time_ms) * batch_size if n_iter else 0.0

        console = Console()
        model_info = Panel(
            Text(f"Model: {self.model_name}\nProviders: {self.providers}", style="bold green"),
            title="Model Information",
            border_style="bright_blue",
            expand=False,
        )
        console.print(model_info)
        table = Table(title=f"Benchmark for '{self.model_name}'", border_style="bright_blue")
        table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
        table.add_column("Value", justify="center", style="magenta")

        table.add_row("Batch size", str(batch_size))
        table.add_row("Warm-up iters", str(warmup))
        table.add_row("Timed iterations", str(n_iter))
        table.add_row("Average Time / batch (ms)", f"{avg_time_ms:.4f}")
        table.add_row("Plates per Second (PPS)", f"{pps:.4f}")
        console.print(table)

    def run(
        self,
        source: str | list[str] | npt.NDArray | list[npt.NDArray],
        return_confidence: bool = False,
        remove_pad_char: bool = True,
    ) -> list[PlatePrediction]:
        """
        Run plate recognition on one or more images.

        Args:
            source: One or more image inputs, which can be:

                - A file path (``str`` or ``PathLike``) to an image.
                - A list of file paths.
                - A NumPy array for a single image with shape ``(H, W)``, ``(H, W, 1)`` or ``(H, W, 3)``.
                - A list of NumPy arrays, each representing a single image.
                - A 4D NumPy array of shape ``(N, H, W, C)`` already batched and ready for inference.

                Inputs are resized/converted as needed according to the loaded `PlateConfig` (color mode,
                aspect ratio handling, interpolation, and padding). For in-memory NumPy arrays, the library
                assumes the arrays already use the expected color mode from the config: grayscale arrays for
                grayscale models, and RGB arrays for RGB models. Arrays are expected in ``channels_last``
                layout and are cast to ``uint8`` before inference.

            return_confidence: If ``True``, include per-character confidence scores in each prediction
                (`PlatePrediction.char_probs`).
            remove_pad_char: If ``True`` (default), remove trailing configured padding characters
                from decoded plate text.

        Returns:
            A list of `PlatePrediction`, one per input image (or batch element).

            - `plate` is always present.
            - `char_probs` is present only when ``return_confidence=True``.
            - `region` is populated automatically when the loaded model exports a ``region`` output and
                the config provides region labels; otherwise it is ``None``.
            - `region_prob` is populated only when both region prediction is available and
                ``return_confidence=True``.
        """

        x = _load_image_from_source(source, self.config)
        x = preprocess_image(x)

        fetch_outputs = self._fetch_outputs if self.has_region_head else [self.plate_output_name]
        raw_outputs: list[npt.NDArray] = self.model.run(fetch_outputs, {"input": x})
        outputs = dict(zip(fetch_outputs, raw_outputs, strict=True))

        return postprocess_output(
            model_output=outputs[self.plate_output_name],
            max_plate_slots=self.config.max_plate_slots,
            model_alphabet=self.config.alphabet,
            pad_char=self.config.pad_char,
            remove_pad_char=remove_pad_char,
            return_confidence=return_confidence,
            return_region=self.has_region_head,
            region_output=(
                outputs[self.region_output_name]
                if self.has_region_head and self.region_output_name is not None
                else None
            ),
            region_labels=self.config.plate_regions if self.has_region_head else None,
        )

    def run_one(
        self,
        source: str | npt.NDArray,
        return_confidence: bool = False,
        remove_pad_char: bool = True,
    ) -> PlatePrediction:
        """
        Convenience wrapper around `run()` for single-image inference.

        Args:
            source: A single image input (path or NumPy array).
            return_confidence: Whether to include per-character confidences.
            remove_pad_char: Whether to remove trailing configured padding characters from decoded
                plate text.

        Returns:
            A single `PlatePrediction`. Region fields are populated automatically when supported.

        Raises:
            ValueError: If the input resolves to anything other than exactly one sample.
        """
        results = self.run(source, return_confidence=return_confidence, remove_pad_char=remove_pad_char)
        if len(results) != 1:
            raise ValueError(f"Expected exactly 1 result, got {len(results)}.")
        return results[0]
