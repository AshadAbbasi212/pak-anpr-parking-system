"""
Model builder functions for supported architectures.
"""

from collections.abc import Sequence

import keras
import numpy as np
from keras import layers

from fast_plate_ocr.train.model.config import PlateConfig
from fast_plate_ocr.train.model.layers import (
    PatchExtractor,
    PositionEmbedding,
    SequencePooling,
    TokenReducer,
    TransformerBlock,
    VocabularyProjection,
)
from fast_plate_ocr.train.model.model_schema import AnyModelConfig, CCTModelConfig, LayerConfig


def _build_stem_from_config(specs: Sequence[LayerConfig]) -> keras.Sequential:
    return keras.Sequential([spec.to_keras_layer() for spec in specs], name="conv_stem")


def _build_cct_model(
    cfg: CCTModelConfig,
    input_shape: tuple[int, int, int],
    plate_cfg: PlateConfig,
    enable_region_head: bool,
) -> keras.Model:
    # 1. Input
    inputs = layers.Input(shape=input_shape)

    # 2. Rescale & conv stem
    data_rescale = cfg.rescaling.to_keras_layer()
    x = _build_stem_from_config(cfg.tokenizer.blocks)(data_rescale(inputs))

    # 3. Patch extraction: (B, H, W, C) -> (B, num_patches, C*patch_size**2)
    x = PatchExtractor(patch_size=cfg.tokenizer.patch_size)(x)

    # 5. Optional patch MLP
    if cfg.tokenizer.patch_mlp is not None:
        x = cfg.tokenizer.patch_mlp.to_keras_layer()(x)

    # 6. Positional embeddings
    if cfg.tokenizer.positional_emb:
        seq_len = keras.ops.shape(x)[1]
        x = x + PositionEmbedding(sequence_length=seq_len, name="pos_emb")(x)

    # 7. N x TransformerBlock's
    dpr = list(np.linspace(0.0, cfg.transformer_encoder.stochastic_depth, cfg.transformer_encoder.layers))
    for i, rate in enumerate(dpr, 1):
        x = TransformerBlock(
            projection_dim=cfg.transformer_encoder.projection_dim,
            num_heads=cfg.transformer_encoder.heads,
            mlp_units=cfg.transformer_encoder.units,
            attention_layout=cfg.transformer_encoder.attention_layout,
            attention_dropout=cfg.transformer_encoder.attention_dropout,
            mlp_dropout=cfg.transformer_encoder.mlp_dropout,
            drop_path_rate=rate,
            norm_type=cfg.transformer_encoder.normalization,
            activation=cfg.transformer_encoder.activation,
            name=f"transformer_block_{i}",
        )(x)

    # 8. Reduce to a fixed number of tokens
    token_features = TokenReducer(
        num_tokens=plate_cfg.max_plate_slots,
        projection_dim=cfg.transformer_encoder.projection_dim,
        num_heads=cfg.transformer_encoder.token_reducer_heads,
        attention_layout=cfg.transformer_encoder.attention_layout,
        attention_dropout=cfg.transformer_encoder.attention_dropout,
        use_query_residual=cfg.transformer_encoder.token_reducer_use_query_residual,
        use_output_norm=cfg.transformer_encoder.token_reducer_use_output_norm,
        norm_type=cfg.transformer_encoder.normalization,
    )(x)

    # 9. Add N transformer blocks AFTER TokenReduce (use same settings as other blocks)
    post_reduce_layers = cfg.transformer_encoder.post_token_reducer_layers
    x = token_features
    for i in range(1, post_reduce_layers + 1):
        x = TransformerBlock(
            projection_dim=cfg.transformer_encoder.projection_dim,
            num_heads=cfg.transformer_encoder.heads,
            mlp_units=cfg.transformer_encoder.units,
            attention_layout=cfg.transformer_encoder.attention_layout,
            attention_dropout=cfg.transformer_encoder.attention_dropout,
            mlp_dropout=cfg.transformer_encoder.mlp_dropout,
            drop_path_rate=0.0,
            norm_type=cfg.transformer_encoder.normalization,
            activation=cfg.transformer_encoder.activation,
            name=f"post_reduce_transformer_block_{i}",
        )(x)

    # 10. Project reduced tokens to vocab
    plate_logits = VocabularyProjection(
        vocabulary_size=plate_cfg.vocabulary_size,
        dropout_rate=cfg.transformer_encoder.head_mlp_dropout,
        name="plate",
    )(x)

    if enable_region_head:
        if not plate_cfg.plate_regions:
            raise ValueError("Region head requested, but no regions are defined in the plate config.")

        # 11. Add N transformer blocks before SeqPool for region branch
        region_x = x
        region_pre_seqpool_layers = cfg.transformer_encoder.region_pre_seqpool_layers
        for i in range(1, region_pre_seqpool_layers + 1):
            region_x = TransformerBlock(
                projection_dim=cfg.transformer_encoder.projection_dim,
                num_heads=cfg.transformer_encoder.heads,
                mlp_units=cfg.transformer_encoder.units,
                attention_layout=cfg.transformer_encoder.attention_layout,
                attention_dropout=cfg.transformer_encoder.attention_dropout,
                mlp_dropout=cfg.transformer_encoder.mlp_dropout,
                drop_path_rate=0.0,
                norm_type=cfg.transformer_encoder.normalization,
                activation=cfg.transformer_encoder.activation,
                name=f"region_pre_pool_transformer_block_{i}",
            )(region_x)

        pooled_tokens = SequencePooling(name="region_seq_pool")(region_x)
        region_logits = layers.Dense(len(plate_cfg.plate_regions), activation="softmax", name="region")(pooled_tokens)
        outputs = {"plate": plate_logits, "region": region_logits}
    else:
        outputs = {"plate": plate_logits}

    return keras.Model(inputs, outputs, name="CCT_OCR")


def build_model(
    model_cfg: AnyModelConfig,
    plate_cfg: PlateConfig,
    enable_region_head: bool = False,
) -> keras.Model:
    """
    Build a Keras model based on the specified model and plate configuration.
    """
    if model_cfg.model == "cct":
        return _build_cct_model(
            cfg=model_cfg,
            input_shape=(plate_cfg.img_height, plate_cfg.img_width, plate_cfg.num_channels),
            plate_cfg=plate_cfg,
            enable_region_head=enable_region_head,
        )
    raise ValueError(f"Unsupported model type: {model_cfg.model!r}")
