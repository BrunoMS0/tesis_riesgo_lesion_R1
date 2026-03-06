"""
model.py – Bi-LSTM + Temporal Attention architecture for fatigue estimation.

Architecture
------------
Input (batch, 14, N_features)
  → Bi-LSTM 64 units (return_sequences) → Dropout 0.3
  → Bi-LSTM 32 units (return_sequences) → TemporalAttention
  → Dense 32 ReLU → Dropout 0.2
  → Dense 1 Sigmoid  ∈ [0, 1]

The custom ``TemporalAttention`` layer exposes per-time-step attention
weights so they can be extracted for thesis interpretability analysis.

Public API
----------
TemporalAttention           – Keras layer
build_fatigue_model(cfg)    – factory function returning compiled Model
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers

from .config import FatigueConfig

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Custom attention layer
# ────────────────────────────────────────────────────────────

class TemporalAttention(layers.Layer):
    """
    Learnable additive attention over the time axis.

    Given hidden states ``H`` of shape ``(batch, T, d)``, the layer
    computes:

    .. math::

        e_t      = \\tanh(H_t W + b)  \\cdot  v
        \\alpha   = \\text{softmax}(e)
        context  = \\sum_t \\alpha_t \\, H_t

    The attention weights ``α`` are stored as an attribute so they can
    be retrieved after a forward pass for interpretability.

    Parameters
    ----------
    units : int
        Dimensionality of the internal projection (defaults to input dim).
    """

    def __init__(self, units: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._units = units

    def build(self, input_shape):
        d = int(input_shape[-1])
        u = self._units or d
        self.W = self.add_weight(
            name="att_W", shape=(d, u),
            initializer="glorot_uniform", trainable=True,
        )
        self.b = self.add_weight(
            name="att_b", shape=(u,),
            initializer="zeros", trainable=True,
        )
        self.v = self.add_weight(
            name="att_v", shape=(u,),
            initializer="glorot_uniform", trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, return_attention=False):
        # inputs: (batch, T, d)
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b)
        # score: (batch, T, u)
        score = tf.tensordot(score, self.v, axes=[[2], [0]])
        # score: (batch, T)
        alpha = tf.nn.softmax(score, axis=1)  # (batch, T)
        context = tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), axis=1)
        # context: (batch, d)

        # Store for external retrieval
        self._last_attention_weights = alpha

        if return_attention:
            return context, alpha
        return context

    def get_config(self):
        cfg = super().get_config()
        cfg["units"] = self._units
        return cfg


# ────────────────────────────────────────────────────────────
# Model factory
# ────────────────────────────────────────────────────────────

def build_fatigue_model(
    n_features: int,
    window_size: int = 14,
    cfg: Optional[FatigueConfig] = None,
) -> tf.keras.Model:
    """
    Build and **compile** the Bi-LSTM + Attention fatigue model.

    Parameters
    ----------
    n_features : int
        Number of input features per time step.
    window_size : int
        Length of the lookback window (time steps).
    cfg : FatigueConfig, optional
        Hyper-parameters; defaults used when *None*.

    Returns
    -------
    tf.keras.Model
        Compiled model ready for ``model.fit()``.
    """
    if cfg is None:
        cfg = FatigueConfig()

    l2 = regularizers.l2(cfg.l2_reg)

    inp = layers.Input(shape=(window_size, n_features), name="time_series")

    # --- Bi-LSTM stack -----------------------------------------------
    x = layers.Bidirectional(
        layers.LSTM(cfg.lstm1_units, return_sequences=True,
                    kernel_regularizer=l2),
        name="bi_lstm_1",
    )(inp)
    x = layers.Dropout(cfg.dropout_lstm, name="drop_lstm_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(cfg.lstm2_units, return_sequences=True,
                    kernel_regularizer=l2),
        name="bi_lstm_2",
    )(x)

    # --- Temporal Attention ------------------------------------------
    x = TemporalAttention(name="attention")(x)

    # --- Dense head --------------------------------------------------
    x = layers.Dense(cfg.dense_units, activation="relu",
                     kernel_regularizer=l2, name="dense_1")(x)
    x = layers.Dropout(cfg.dropout_dense, name="drop_dense")(x)
    out = layers.Dense(1, activation="sigmoid", name="dfi_output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="fatigue_bilstm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    logger.info("Model built: %d params (%d trainable)",
                model.count_params(),
                sum(w.numpy().size for w in model.trainable_weights))
    model.summary(print_fn=logger.info)

    return model
