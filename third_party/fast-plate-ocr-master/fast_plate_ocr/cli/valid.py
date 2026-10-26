"""
Script for validating trained OCR models.
"""

import json
import pathlib

import click

from fast_plate_ocr.train.data.dataset import PlateRecognitionPyDataset
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities.utils import load_keras_model


def evaluate_model_by_region(model, val_dataset: PlateRecognitionPyDataset) -> dict[str, dict[str, float | int]]:
    """
    Evaluate metrics grouped by `plate_region`.
    """
    region_metrics: dict[str, dict[str, float | int]] = {}
    original_annotations = val_dataset.annotations

    try:
        for region, group in original_annotations.groupby("plate_region", sort=True):
            val_dataset.annotations = group.reset_index(drop=True)
            metrics = model.evaluate(val_dataset, return_dict=True, verbose=0)
            region_metrics[str(region)] = {
                "num_samples": len(group),
                **{name: float(value) for name, value in metrics.items()},
            }
    finally:
        val_dataset.annotations = original_annotations

    return region_metrics


@click.command(context_settings={"max_content_width": 120})
@click.option(
    "-m",
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path to the saved .keras model.",
)
@click.option(
    "--plate-config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="Path pointing to the model license plate OCR config.",
)
@click.option(
    "-a",
    "--annotations",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Annotations file used for validation.",
)
@click.option(
    "-b",
    "--batch-size",
    default=1,
    show_default=True,
    type=int,
    help="Batch size.",
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=int,
    help="Number of worker threads/processes for parallel data loading via PyDataset.",
)
@click.option(
    "--use-multiprocessing/--no-use-multiprocessing",
    default=False,
    show_default=True,
    help="Whether to use multiprocessing for data loading.",
)
@click.option(
    "--max-queue-size",
    default=10,
    show_default=True,
    type=int,
    help="Maximum number of batches to prefetch for the dataset.",
)
@click.option(
    "--evaluate-by-region/--no-evaluate-by-region",
    default=False,
    show_default=True,
    help=(
        "If enabled, also evaluate each `plate_region` group separately. "
        "Requires region recognition (`plate_region` column and `plate_config.plate_regions`)."
    ),
)
@click.option(
    "--region-metrics-output",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Optional output path to save per-region metrics as JSON. Requires `--evaluate-by-region`.",
)
def valid(
    model_path: pathlib.Path,
    plate_config_file: pathlib.Path,
    annotations: pathlib.Path,
    batch_size: int,
    workers: int,
    use_multiprocessing: bool,
    max_queue_size: int,
    evaluate_by_region: bool,
    region_metrics_output: pathlib.Path | None,
) -> None:
    """
    Validate the trained OCR model on a labeled set.
    """
    plate_config = load_plate_config_from_yaml(plate_config_file)
    model = load_keras_model(model_path, plate_config)
    val_dataset = PlateRecognitionPyDataset(
        annotations_file=annotations,
        plate_config=plate_config,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
    )

    if evaluate_by_region and not val_dataset.region_recognition:
        raise click.UsageError(
            "`--evaluate-by-region` requires region recognition. "
            "Make sure annotations include `plate_region` and the config defines `plate_regions`."
        )
    if region_metrics_output is not None and not evaluate_by_region:
        raise click.UsageError("`--region-metrics-output` requires `--evaluate-by-region`.")

    if not val_dataset.region_recognition and "region" in model.output_names:
        model.compile(optimizer=model.optimizer, loss={"plate": model.loss["plate"]}, jit_compile=False)

    model.evaluate(val_dataset)

    if evaluate_by_region:
        region_metrics = evaluate_model_by_region(model=model, val_dataset=val_dataset)
        if region_metrics_output is not None:
            region_metrics_output.parent.mkdir(parents=True, exist_ok=True)
            with open(region_metrics_output, "w", encoding="utf-8") as f_out:
                json.dump(region_metrics, f_out, indent=2, sort_keys=True)
            click.echo(f"\nPer-region metrics saved to: {region_metrics_output}")
        else:
            click.echo("\nPer-region metrics:")
            click.echo(json.dumps(region_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    valid()
