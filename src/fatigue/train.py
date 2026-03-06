"""
train.py – Training loop for the R4 Fatigue model.

Provides callbacks, reproducibility seeds and a single function to
execute a full training run.

Public API
----------
train_fatigue_model(model, bundle, cfg) -> tf.keras.callbacks.History
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from .config import FatigueConfig
from .dataset import FatigueDatasetBundle

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    """Set global seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seeds set to %d", seed)


def _build_callbacks(cfg: FatigueConfig) -> list:
    """Create the standard callback list for training."""
    model_dir = Path(cfg.output_path) / "fatigue_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=cfg.lr_patience,
            factor=cfg.lr_factor,
            min_lr=cfg.lr_min,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_weights.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(
            str(model_dir / "training_log.csv"),
            separator=",",
            append=False,
        ),
    ]
    return callbacks


def _save_hyperparams(cfg: FatigueConfig) -> None:
    """Persist hyper-parameters alongside the model checkpoint."""
    model_dir = Path(cfg.output_path) / "fatigue_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    hp_path = model_dir / "hyperparameters.json"
    hp_path.write_text(json.dumps(asdict(cfg), indent=2, default=str),
                       encoding="utf-8")
    logger.info("Hyperparameters saved to %s", hp_path)


def train_fatigue_model(
    model: tf.keras.Model,
    bundle: FatigueDatasetBundle,
    cfg: Optional[FatigueConfig] = None,
) -> tf.keras.callbacks.History:
    """
    Run training with early stopping, LR scheduling and checkpointing.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled model from :func:`model.build_fatigue_model`.
    bundle : FatigueDatasetBundle
        Train / val datasets from :func:`dataset.build_fatigue_datasets`.
    cfg : FatigueConfig, optional

    Returns
    -------
    tf.keras.callbacks.History
    """
    if cfg is None:
        cfg = FatigueConfig()

    _set_seeds(cfg.seed)
    _save_hyperparams(cfg)
    callbacks = _build_callbacks(cfg)

    logger.info("Starting training — max %d epochs, batch %d, "
                "train=%d, val=%d samples",
                cfg.max_epochs, cfg.batch_size,
                bundle.n_train, bundle.n_val)

    history = model.fit(
        bundle.train,
        validation_data=bundle.val,
        epochs=cfg.max_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    logger.info("Training finished after %d epochs  "
                "(best val_loss=%.5f)",
                len(history.history["loss"]),
                min(history.history["val_loss"]))

    return history
