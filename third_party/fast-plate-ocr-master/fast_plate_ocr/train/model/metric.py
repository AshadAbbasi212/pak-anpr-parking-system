"""
Evaluation metrics for license plate recognition models.
"""

from keras import metrics, ops


def cat_acc_metric(max_plate_slots: int, vocabulary_size: int):
    """
    Categorical accuracy metric (character-level).
    """

    def char_acc(y_true, y_pred):
        """
        Per-character categorical accuracy averaged over the plate.
        """
        y_true = ops.reshape(y_true, newshape=(-1, max_plate_slots, vocabulary_size))
        y_pred = ops.reshape(y_pred, newshape=(-1, max_plate_slots, vocabulary_size))
        return ops.mean(metrics.categorical_accuracy(y_true, y_pred))

    return char_acc


def plate_acc_metric(max_plate_slots: int, vocabulary_size: int):
    """
    Plate accuracy metric (exact match).
    """

    def acc(y_true, y_pred):
        """
        1 if the whole plate is correct, else 0.
        """
        y_true = ops.reshape(y_true, newshape=(-1, max_plate_slots, vocabulary_size))
        y_pred = ops.reshape(y_pred, newshape=(-1, max_plate_slots, vocabulary_size))
        y_pred = ops.cast(y_pred, dtype="float32")
        et = ops.equal(ops.argmax(y_true, axis=-1), ops.argmax(y_pred, axis=-1))
        return ops.mean(ops.cast(ops.all(et, axis=-1, keepdims=False), dtype="float32"))

    return acc


def top_3_k_metric(vocabulary_size: int):
    """
    Top-3 accuracy metric (character-level).
    """

    def top3_acc(y_true, y_pred):
        """
        True character is in top-3 predictions.
        """
        y_true = ops.reshape(y_true, newshape=(-1, vocabulary_size))
        y_pred = ops.reshape(y_pred, newshape=(-1, vocabulary_size))
        y_pred = ops.cast(y_pred, dtype="float32")
        return ops.mean(metrics.top_k_categorical_accuracy(y_true, y_pred, k=3))

    return top3_acc


def plate_len_acc_metric(
    max_plate_slots: int,
    vocabulary_size: int,
    pad_token_index: int,
):
    """
    Plate-length accuracy metric.
    """

    def len_acc(y_true, y_pred):
        """
        Proportion of plates whose predicted length matches the ground-truth length.
        """
        y_true = ops.reshape(y_true, (-1, max_plate_slots, vocabulary_size))
        y_pred = ops.reshape(ops.cast(y_pred, "float32"), (-1, max_plate_slots, vocabulary_size))
        true_idx = ops.argmax(y_true, axis=-1)
        pred_idx = ops.argmax(y_pred, axis=-1)
        true_len = ops.sum(ops.cast(ops.not_equal(true_idx, pad_token_index), "int32"), axis=-1)
        pred_len = ops.sum(ops.cast(ops.not_equal(pred_idx, pad_token_index), "int32"), axis=-1)
        return ops.mean(ops.cast(ops.equal(true_len, pred_len), dtype="float32"))

    return len_acc
