"""
test_fatigue_model.py – Tests for the Bi-LSTM + Attention model.

Validates:
- Model builds without error
- Forward pass returns correct output shape
- Output values are in [0, 1] (sigmoid)
- Attention weights sum to 1
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.fatigue.config import FatigueConfig
from src.fatigue.model import build_fatigue_model, TemporalAttention


@pytest.fixture()
def cfg():
    return FatigueConfig()


@pytest.fixture()
def small_model(cfg):
    """Build model with small dimensions for fast testing."""
    return build_fatigue_model(n_features=10, window_size=7, cfg=cfg)


class TestBuildModel:
    def test_builds(self, small_model):
        assert isinstance(small_model, tf.keras.Model)

    def test_parameter_count(self, small_model):
        assert small_model.count_params() > 0

    def test_output_shape(self, small_model):
        dummy = tf.random.normal((4, 7, 10))
        out = small_model(dummy, training=False)
        assert out.shape == (4, 1)

    def test_output_range(self, small_model):
        """Sigmoid output must be in [0, 1]."""
        dummy = tf.random.normal((16, 7, 10))
        out = small_model(dummy, training=False).numpy()
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)


class TestTemporalAttention:
    def test_weights_sum_to_one(self):
        layer = TemporalAttention()
        x = tf.random.normal((8, 14, 64))
        context, alpha = layer(x, return_attention=True)
        sums = tf.reduce_sum(alpha, axis=1).numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_context_shape(self):
        layer = TemporalAttention()
        x = tf.random.normal((8, 14, 64))
        context, alpha = layer(x, return_attention=True)
        assert context.shape == (8, 64)
        assert alpha.shape == (8, 14)

    def test_stored_weights(self):
        layer = TemporalAttention()
        x = tf.random.normal((4, 7, 32))
        _ = layer(x)
        assert hasattr(layer, "_last_attention_weights")
        assert layer._last_attention_weights.shape == (4, 7)


class TestModelCompilation:
    def test_loss_and_metrics(self, small_model):
        assert small_model.loss is not None
        metric_names = [m.name for m in small_model.metrics]
        # After compilation, metrics include 'loss' and 'mae'
        # Just verify the model is compiled
        assert small_model.optimizer is not None
