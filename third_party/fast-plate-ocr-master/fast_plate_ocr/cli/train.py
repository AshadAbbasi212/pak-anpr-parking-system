"""
Script for training the License Plate OCR models.
"""

import json
import pathlib
import shutil
from datetime import datetime
from typing import Literal

import albumentations as A
import click
import keras
from keras.src.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    SwapEMAWeights,
    TensorBoard,
    TerminateOnNaN,
)
from keras.src.optimizers import AdamW

from fast_plate_ocr.cli.utils import print_params, print_train_details
from fast_plate_ocr.cli.validate_dataset import (
    DEFAULT_MIN_HEIGHT,
    DEFAULT_MIN_WIDTH,
    console,
    rich_report,
    run_dataset_validation,
)
from fast_plate_ocr.train.data.annotations import read_annotations_csv
from fast_plate_ocr.train.data.augmentation import (
    default_train_augmentation,
)
from fast_plate_ocr.train.data.dataset import PlateRecognitionPyDataset
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.model.loss import cce_loss, focal_cce_loss
from fast_plate_ocr.train.model.metric import (
    cat_acc_metric,
    plate_acc_metric,
    plate_len_acc_metric,
    top_3_k_metric,
)
from fast_plate_ocr.train.model.model_builders import build_model
from fast_plate_ocr.train.model.model_schema import load_model_config_from_yaml

# ruff: noqa: PLR0913
# pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
# pylint: disable=too-many-branches, too-many-statements


EVAL_METRICS: dict[str, Literal["max", "min", "auto"]] = {
    "val_plate_acc": "max",
    "val_plate_char_acc": "max",
    "val_plate_top3_acc": "max",
    "val_plate_len_acc": "max",
    "val_loss": "min",
    "val_plate_loss": "min",
    "val_region_acc": "max",
    "val_region_top3_acc": "max",
    "val_region_macro_f1": "max",
    "val_region_loss": "min",
}
"""Eval metric to monitor."""
ValidationMode = Literal["off", "warn", "error"]
"""Validation mode to use when training."""


def resolve_metric_name_for_logs(requested_metric: str, has_region_head: bool) -> str:
    """
    Map a logical early-stopping metric name to the actual logs key.

    :param requested_metric: Logical metric name.
    :param has_region_head: Whether the dataset has a region head.
    :return: Actual metric key logged.
    """
    region_metrics = {
        "val_region_acc",
        "val_region_top3_acc",
        "val_region_macro_f1",
        "val_region_loss",
    }

    if not has_region_head and requested_metric in region_metrics:
        raise ValueError(
            f"Early-stopping metric '{requested_metric}' requires region recognition, "
            "but the dataset/model does not have a region head."
        )

    if not has_region_head:
        single_head_metric_map = {
            "val_plate_acc": "val_acc",
            "val_plate_char_acc": "val_char_acc",
            "val_plate_top3_acc": "val_top3_acc",
            "val_plate_len_acc": "val_len_acc",
            "val_plate_loss": "val_loss",
        }
        return single_head_metric_map.get(requested_metric, requested_metric)

    return requested_metric


def validate_datasets_before_training(
    plate_config,
    annotations: pathlib.Path,
    val_annotations: pathlib.Path,
    mode: ValidationMode,
) -> None:
    if mode == "off":
        return

    def validate_one(label: str, csv_path: pathlib.Path) -> bool:
        df_annots = read_annotations_csv(csv_path)
        csv_root = csv_path.parent
        df_annots["image_path"] = df_annots["image_path"].apply(lambda p: str((csv_root / p).resolve()))
        errors, warnings, _ = run_dataset_validation(
            df_annots,
            plate_config,
            DEFAULT_MIN_HEIGHT,
            DEFAULT_MIN_WIDTH,
        )
        console.print(f"\n[bold]Dataset validation ({label})[/]")
        rich_report(errors, warnings)
        return bool(errors)

    train_has_errors = validate_one("train", annotations)
    val_has_errors = validate_one("val", val_annotations)

    if (train_has_errors or val_has_errors) and mode == "error":
        raise ValueError("Dataset validation failed. Fix errors or use --validate-dataset=warn to proceed.")


@click.command(context_settings={"max_content_width": 120})
@click.option(
    "--model-config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path to the YAML config that describes the model architecture.",
)
@click.option(
    "--plate-config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path to the plate YAML config.",
)
@click.option(
    "--annotations",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path pointing to the train annotations CSV file.",
)
@click.option(
    "--val-annotations",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path pointing to the train validation CSV file.",
)
@click.option(
    "--validate-dataset",
    default="off",
    show_default=True,
    type=click.Choice(["off", "warn", "error"], case_sensitive=False),
    help="Validate train/val CSVs before training. 'warn' prints issues, 'error' aborts on errors.",
)
@click.option(
    "--validation-freq",
    default=1,
    show_default=True,
    type=int,
    help="Frequency (in epochs) at which to evaluate the validation data.",
)
@click.option(
    "--augmentation-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="YAML file pointing to the augmentation pipeline saved with Albumentations.save(...)",
)
@click.option(
    "--lr",
    default=0.001,
    show_default=True,
    type=float,
    help="Initial learning rate.",
)
@click.option(
    "--final-lr-factor",
    default=1e-2,
    show_default=True,
    type=float,
    help="Final learning rate factor for the cosine decay scheduler. It's the fraction of"
    " the initial learning rate that remains after decay.",
)
@click.option(
    "--warmup-fraction",
    default=0.05,
    show_default=True,
    type=float,
    help="Fraction of total training steps to linearly warm up.",
)
@click.option(
    "--weight-decay",
    default=0.01,
    show_default=True,
    type=float,
    help="Weight decay for the AdamW optimizer.",
)
@click.option(
    "--clipnorm",
    default=1.0,
    show_default=True,
    type=float,
    help="Gradient clipping norm value for the AdamW optimizer.",
)
@click.option(
    "--plate-loss",
    default="cce",
    type=click.Choice(["cce", "focal_cce"], case_sensitive=False),
    show_default=True,
    help="Loss function to use during training.",
)
@click.option(
    "--plate-focal-alpha",
    default=0.25,
    show_default=True,
    type=float,
    help="Alpha parameter for plate focal loss. Applicable only when '--plate-loss' is 'focal_cce'.",
)
@click.option(
    "--plate-focal-gamma",
    default=2.0,
    show_default=True,
    type=float,
    help="Gamma parameter for plate focal loss. Applicable only when '--plate-loss' is 'focal_cce'.",
)
@click.option(
    "--region-loss",
    default="cce",
    type=click.Choice(["cce", "focal_cce"], case_sensitive=False),
    show_default=True,
    help="Loss function for region recognition.",
)
@click.option(
    "--region-focal-alpha",
    default=0.25,
    show_default=True,
    type=float,
    help="Alpha parameter for region focal loss. Applicable only when '--plate-loss' is 'focal_cce'.",
)
@click.option(
    "--region-focal-gamma",
    default=2.0,
    show_default=True,
    type=float,
    help="Gamma parameter for region focal loss. Applicable only when '--plate-loss' is 'focal_cce'.",
)
@click.option(
    "--label-smoothing",
    default=0.01,
    show_default=True,
    type=float,
    help="Amount of label smoothing to apply.",
)
@click.option(
    "--plate-loss-weight",
    default=0.9,
    show_default=True,
    type=float,
    help="Weight for the plate recognition loss.",
)
@click.option(
    "--region-loss-weight",
    default=0.1,
    show_default=True,
    type=float,
    help="Weight for the region recognition loss (when enabled).",
)
@click.option(
    "--mixed-precision-policy",
    default=None,
    type=click.Choice(["mixed_float16", "mixed_bfloat16", "float32"]),
    help=(
        "Optional mixed precision policy for training. Choose one of: mixed_float16, "
        "mixed_bfloat16, or float32. If not provided, Keras uses its default global policy."
    ),
)
@click.option(
    "--batch-size",
    default=64,
    show_default=True,
    type=int,
    help="Batch size for training.",
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=int,
    help="Number of worker threads/processes for parallel data loading.",
)
@click.option(
    "--use-multiprocessing/--no-use-multiprocessing",
    default=False,
    show_default=True,
    help="Use multiprocessing for data loading.",
)
@click.option(
    "--max-queue-size",
    default=10,
    show_default=True,
    type=int,
    help="Maximum queue size for dataset workers.",
)
@click.option(
    "--output-dir",
    default="./trained_models",
    type=click.Path(dir_okay=True, file_okay=False, path_type=pathlib.Path),
    help="Output directory where model will be saved.",
)
@click.option(
    "--epochs",
    default=150,
    show_default=True,
    type=int,
    help="Number of training epochs.",
)
@click.option(
    "--tensorboard",
    "-t",
    is_flag=True,
    help="Whether to use TensorBoard visualization tool.",
)
@click.option(
    "--tensorboard-dir",
    "-l",
    default="tensorboard_logs",
    show_default=True,
    type=click.Path(path_type=pathlib.Path),
    help="The path of the directory where to save the TensorBoard log files.",
)
@click.option(
    "--early-stopping-patience",
    default=100,
    show_default=True,
    type=int,
    help="Stop training when the early stopping metric doesn't improve for X epochs.",
)
@click.option(
    "--early-stopping-metric",
    default="val_plate_acc",
    show_default=True,
    type=click.Choice(list(EVAL_METRICS), case_sensitive=False),
    help="Metric to monitor for early stopping.",
)
@click.option(
    "--weights-path",
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="Path to the pretrained model weights file.",
)
@click.option(
    "--use-ema/--no-use-ema",
    default=True,
    show_default=True,
    help=(
        "Whether to use exponential moving averages in the AdamW optimizer. "
        "Defaults to True; use --no-use-ema to disable."
    ),
)
@click.option(
    "--wd-ignore",
    default="bias,layer_norm",
    show_default=True,
    type=str,
    help="Comma-separated list of variable substrings to exclude from weight decay.",
)
@click.option(
    "--seed",
    type=int,
    help="Sets all random seeds (Python, NumPy, and backend framework, e.g. TF).",
)
@print_params(table_title="CLI Training Parameters", c1_title="Parameter", c2_title="Details")
def train(  # noqa: PLR0912, PLR0915
    model_config_file: pathlib.Path,
    plate_config_file: pathlib.Path,
    annotations: pathlib.Path,
    val_annotations: pathlib.Path,
    validate_dataset: ValidationMode,
    validation_freq: int,
    augmentation_path: pathlib.Path | None,
    lr: float,
    final_lr_factor: float,
    warmup_fraction: float,
    weight_decay: float,
    clipnorm: float,
    plate_loss: str,
    plate_focal_alpha: float,
    plate_focal_gamma: float,
    region_loss: str,
    region_focal_alpha: float,
    region_focal_gamma: float,
    label_smoothing: float,
    plate_loss_weight: float,
    region_loss_weight: float,
    mixed_precision_policy: str | None,
    batch_size: int,
    workers: int,
    use_multiprocessing: bool,
    max_queue_size: int,
    output_dir: pathlib.Path,
    epochs: int,
    tensorboard: bool,
    tensorboard_dir: pathlib.Path,
    early_stopping_patience: int,
    early_stopping_metric: str,
    weights_path: pathlib.Path | None,
    use_ema: bool,
    wd_ignore: str,
    seed: int | None,
) -> None:
    """
    Train the License Plate OCR model.
    """
    if seed is not None:
        keras.utils.set_random_seed(seed)

    if mixed_precision_policy is not None:
        keras.mixed_precision.set_global_policy(mixed_precision_policy)

    plate_config = load_plate_config_from_yaml(plate_config_file)
    model_config = load_model_config_from_yaml(model_config_file)

    validate_datasets_before_training(
        plate_config=plate_config,
        annotations=annotations,
        val_annotations=val_annotations,
        mode=validate_dataset,
    )
    train_augmentation = (
        A.load(augmentation_path, data_format="yaml")
        if augmentation_path
        else default_train_augmentation(img_color_mode=plate_config.image_color_mode)
    )
    print_train_details(train_augmentation, plate_config.model_dump())

    train_dataset = PlateRecognitionPyDataset(
        annotations_file=annotations,
        transform=train_augmentation,
        plate_config=plate_config,
        batch_size=batch_size,
        shuffle=True,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
    )

    val_dataset = PlateRecognitionPyDataset(
        annotations_file=val_annotations,
        plate_config=plate_config,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
    )

    if val_dataset.region_recognition != train_dataset.region_recognition:
        raise ValueError(
            "Mismatch between training and validation datasets: region labels available in only one of them."
        )

    has_region_head = train_dataset.region_recognition

    # Map the logical metric name to the actual logs key Keras will emit.
    monitor_metric_name = resolve_metric_name_for_logs(early_stopping_metric, has_region_head=has_region_head)

    monitor_mode = EVAL_METRICS[early_stopping_metric]

    # Train
    model = build_model(model_config, plate_config, enable_region_head=has_region_head)

    if weights_path:
        model.load_weights(weights_path, skip_mismatch=True)

    total_steps = epochs * len(train_dataset)
    warmup_steps = int(warmup_fraction * total_steps)

    cosine_decay = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0 if warmup_steps > 0 else lr,
        decay_steps=total_steps - warmup_steps,
        alpha=final_lr_factor,
        warmup_steps=warmup_steps,
        warmup_target=lr if warmup_steps > 0 else None,
    )

    optimizer = AdamW(cosine_decay, weight_decay=weight_decay, clipnorm=clipnorm, use_ema=use_ema)
    optimizer.exclude_from_weight_decay(var_names=[name.strip() for name in wd_ignore.split(",") if name.strip()])

    if plate_loss == "cce":
        plate_loss_fn = cce_loss(vocabulary_size=plate_config.vocabulary_size, label_smoothing=label_smoothing)
    elif plate_loss == "focal_cce":
        plate_loss_fn = focal_cce_loss(
            vocabulary_size=plate_config.vocabulary_size,
            alpha=plate_focal_alpha,
            gamma=plate_focal_gamma,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(f"Unsupported plate loss type: {plate_loss}")

    if region_loss == "cce":
        region_loss_fn = keras.losses.CategoricalCrossentropy()
    elif region_loss == "focal_cce":
        region_loss_fn = keras.losses.CategoricalFocalCrossentropy(alpha=region_focal_alpha, gamma=region_focal_gamma)
    else:
        raise ValueError(f"Unsupported region loss type: {region_loss}")

    base_metrics = [
        cat_acc_metric(max_plate_slots=plate_config.max_plate_slots, vocabulary_size=plate_config.vocabulary_size),
        plate_acc_metric(max_plate_slots=plate_config.max_plate_slots, vocabulary_size=plate_config.vocabulary_size),
        top_3_k_metric(vocabulary_size=plate_config.vocabulary_size),
        plate_len_acc_metric(
            max_plate_slots=plate_config.max_plate_slots,
            vocabulary_size=plate_config.vocabulary_size,
            pad_token_index=plate_config.pad_idx,
        ),
    ]

    if train_dataset.region_recognition:
        loss_config = {"plate": plate_loss_fn, "region": region_loss_fn}
        loss_weights = {"plate": plate_loss_weight, "region": region_loss_weight}
        metrics_config = {
            "plate": base_metrics,
            "region": [
                keras.metrics.CategoricalAccuracy(name="acc"),
                keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
                keras.metrics.F1Score(average="macro", name="macro_f1"),
            ],
        }
    else:
        loss_config = {"plate": plate_loss_fn}
        loss_weights = None
        metrics_config = {"plate": base_metrics}

    model.compile(
        loss=loss_config,
        loss_weights=loss_weights,
        jit_compile=False,
        optimizer=optimizer,
        metrics=metrics_config,
    )

    output_dir /= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file_path = output_dir / "best.keras"

    # Save params and configs used for training
    shutil.copy(model_config_file, output_dir / "model_config.yaml")
    shutil.copy(plate_config_file, output_dir / "plate_config.yaml")
    A.save(train_augmentation, output_dir / "train_augmentation.yaml", "yaml")
    with open(output_dir / "hyper_params.json", "w", encoding="utf-8") as f_out:
        json.dump(
            {k: v for k, v in locals().items() if k in click.get_current_context().params},
            f_out,
            indent=4,
            default=str,
        )

    callbacks = [
        # Stop training when early_stopping_metric doesn't improve for X epochs
        EarlyStopping(
            monitor=monitor_metric_name,
            patience=early_stopping_patience,
            mode=monitor_mode,
            restore_best_weights=False,
            verbose=1,
        ),
        # To save model checkpoint with EMA weights, we need to place this before `ModelCheckpoint`
        *([SwapEMAWeights(swap_on_epoch=True)] if use_ema else []),
        # We don't use EarlyStopping restore_best_weights=True because it won't restore the best
        # weights when it didn't manage to EarlyStop but finished all epochs
        ModelCheckpoint(output_dir / "last.keras", save_weights_only=False, save_best_only=False),
        ModelCheckpoint(
            model_file_path,
            monitor=monitor_metric_name,
            mode=monitor_mode,
            save_weights_only=False,
            save_best_only=True,
            verbose=1,
        ),
        TerminateOnNaN(),
        CSVLogger(str(output_dir / "training_log.csv")),
    ]

    if tensorboard:
        run_dir = tensorboard_dir / datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(TensorBoard(log_dir=run_dir))

    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        validation_freq=validation_freq,
    )


if __name__ == "__main__":
    train()
