"""
lstm.py – LSTM-based injury risk model for R5.

Architecture
------------
Input  (batch, window, n_features)
  → LSTM(64, return_sequences=True) → Dropout(0.3)
  → LSTM(32, return_sequences=False) → Dropout(0.2)
  → Dense(16, relu)
  → Dense(1, sigmoid)

The model receives time-windowed sequences of ``window_size`` consecutive
days per athlete and predicts ``injury_next{N}d`` at the last timestep.

Public API
----------
make_sequences(df, feature_cols, target_col, window_size, athlete_col)
    → X (n_seq, window, n_feat), y (n_seq,), meta (n_seq,)

build_lstm_model(n_features, window_size, cfg)  → compiled Keras model
train_lstm(model, X_train, y_train, X_val, y_val, cfg)  → history
evaluate_lstm(model, X, y, meta, cfg)  → dict of metrics
loao_lstm(X_all, y_all, meta_all, cfg, window_size)  → LOSOResult
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Suppress TF/Keras verbosity unless explicitly set
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ────────────────────────────────────────────────────────────
# Sequence dataset construction
# ────────────────────────────────────────────────────────────

def make_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    window_size: int = 14,
    athlete_col: str = "participant_id",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Create overlapping sliding-window sequences per athlete.

    For each athlete, consecutive rows are turned into windows of
    ``window_size`` days.  The label for window ending at day *t* is
    the target value at day *t* (last step of the window).

    Rows with NaN in features or target are dropped before windowing.
    Windows that straddle athlete boundaries are never created.

    Parameters
    ----------
    df          : DataFrame with athlete_col, date, feature_cols, target_col.
    feature_cols: Feature columns to use (in order).
    target_col  : Binary target column.
    window_size : Number of consecutive days per sequence.
    athlete_col : Participant ID column.

    Returns
    -------
    X    : float32 array of shape (n_sequences, window_size, n_features)
    y    : int32 array of shape (n_sequences,)
    meta : DataFrame with columns [athlete_col, 'date', 'seq_end_date']
           indexed 0..n_sequences-1.
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_rows: List[dict] = []

    needed_cols = list(feature_cols) + [target_col, athlete_col]
    if "date" in df.columns:
        needed_cols.append("date")
    sub = df[needed_cols].dropna(subset=list(feature_cols) + [target_col])
    sub = sub.sort_values([athlete_col, "date"] if "date" in sub.columns else [athlete_col])

    n_feat = len(feature_cols)

    for pid, grp in sub.groupby(athlete_col, sort=False):
        feat_arr = grp[feature_cols].values.astype(np.float32)  # (T, F)
        tgt_arr = grp[target_col].values.astype(np.int32)        # (T,)
        dates = grp["date"].values if "date" in grp.columns else np.arange(len(grp))

        T = len(grp)
        if T < window_size:
            logger.debug("Athlete %s: only %d rows < window %d — skipped", pid, T, window_size)
            continue

        for start in range(T - window_size + 1):
            end = start + window_size - 1
            X_list.append(feat_arr[start : start + window_size])  # (window, F)
            y_list.append(int(tgt_arr[end]))
            meta_rows.append({
                athlete_col: pid,
                "date": dates[end],
                "seq_end_date": dates[end],
            })

    if not X_list:
        return (
            np.empty((0, window_size, n_feat), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            pd.DataFrame(columns=[athlete_col, "date", "seq_end_date"]),
        )

    X = np.stack(X_list, axis=0).astype(np.float32)      # (N, W, F)
    y = np.array(y_list, dtype=np.int32)                  # (N,)
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)

    logger.info(
        "make_sequences: %d sequences (window=%d, features=%d), "
        "%d positives (%.1f%%)",
        len(y), window_size, n_feat, int(y.sum()), 100.0 * y.mean(),
    )
    return X, y, meta


# ────────────────────────────────────────────────────────────
# Model factory
# ────────────────────────────────────────────────────────────

def build_lstm_model(
    n_features: int,
    window_size: int,
    lstm_units: int = 64,
    lstm_units_2: int = 32,
    dense_units: int = 16,
    dropout: float = 0.3,
    dropout_2: float = 0.2,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """
    Build and compile a 2-layer LSTM classifier.

    Parameters
    ----------
    n_features   : Number of input features per timestep.
    window_size  : Sequence length (days).
    lstm_units   : Units in first LSTM layer.
    lstm_units_2 : Units in second LSTM layer.
    dense_units  : Units in intermediate Dense layer.
    dropout      : Dropout after first LSTM.
    dropout_2    : Dropout after second LSTM.
    learning_rate: Adam learning rate.

    Returns
    -------
    Compiled Keras model.
    """
    import keras
    from keras import layers as kl

    inputs = keras.Input(shape=(window_size, n_features), name="input_seq")
    x = kl.LSTM(lstm_units, return_sequences=True, name="lstm_1")(inputs)
    x = kl.Dropout(dropout, name="drop_1")(x)
    x = kl.LSTM(lstm_units_2, return_sequences=False, name="lstm_2")(x)
    x = kl.Dropout(dropout_2, name="drop_2")(x)
    x = kl.Dense(dense_units, activation="relu", name="dense_1")(x)
    outputs = kl.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="InjuryLSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    logger.info(
        "Built InjuryLSTM (window=%d, features=%d, lstm=%d/%d, dense=%d)",
        window_size, n_features, lstm_units, lstm_units_2, dense_units,
    )
    return model


# ────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────

def train_lstm(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 10,
    class_weight: Optional[dict] = None,
) -> "keras.callbacks.History":
    """
    Train the LSTM with early stopping on val loss.

    Parameters
    ----------
    model      : Compiled Keras model.
    X_train    : (N_train, window, features)
    y_train    : (N_train,)
    X_val      : (N_val, window, features)
    y_val      : (N_val,)
    epochs     : Maximum epochs.
    batch_size : Mini-batch size.
    patience   : EarlyStopping patience (val_loss).
    class_weight: Optional dict {0: w0, 1: w1}.  If None, computed from
                  y_train prevalence to handle imbalance.

    Returns
    -------
    Keras History object.
    """
    import keras

    if class_weight is None:
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        if n_pos > 0 and n_neg > 0:
            class_weight = {0: 1.0, 1: n_neg / n_pos}
        else:
            class_weight = {0: 1.0, 1: 1.0}

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )
    actual_epochs = len(history.history["loss"])
    val_auc = history.history.get("val_auc", [float("nan")])[-1]
    logger.info(
        "LSTM training: %d epochs (patience=%d), val_AUC=%.4f, "
        "class_weight={0:%.2f, 1:%.2f}",
        actual_epochs, patience, val_auc,
        class_weight[0], class_weight[1],
    )
    return history


# ────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────

def evaluate_lstm(
    model,
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    athlete_col: str = "participant_id",
) -> dict:
    """
    Evaluate a trained LSTM on a sequence dataset.

    Returns
    -------
    dict with keys: roc_auc, pr_auc, f1, brier_score, n_samples, n_pos
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, brier_score_loss,
    )

    if len(y) == 0 or int(y.sum()) == 0:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "f1": float("nan"), "brier_score": float("nan"),
                "n_samples": len(y), "n_pos": 0}

    y_prob = model.predict(X, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        roc = roc_auc_score(y, y_prob)
    except Exception:
        roc = float("nan")
    try:
        pr = average_precision_score(y, y_prob)
    except Exception:
        pr = float("nan")

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, y_prob)),
        "n_samples": len(y),
        "n_pos": int(y.sum()),
    }


# ────────────────────────────────────────────────────────────
# LOAO cross-validation for LSTM
# ────────────────────────────────────────────────────────────

@dataclass
class LSTMFoldResult:
    participant_id: str
    roc_auc: float
    pr_auc: float
    f1: float
    n_samples: int
    n_injuries: int
    skipped: bool


@dataclass
class LSTMLoaoResult:
    folds: List[LSTMFoldResult]
    mean_roc_auc: float
    std_roc_auc: float
    mean_pr_auc: float
    std_pr_auc: float
    mean_f1: float
    std_f1: float
    n_skipped_folds: int


def loao_lstm(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    window_size: int = 14,
    athlete_col: str = "participant_id",
    lstm_units: int = 64,
    lstm_units_2: int = 32,
    dense_units: int = 16,
    dropout: float = 0.3,
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 10,
    seed: int = 42,
) -> LSTMLoaoResult:
    """
    Leave-One-Athlete-Out cross-validation for the LSTM model.

    For each athlete:
    1. Hold out that athlete as the test fold.
    2. Build sequences from all other athletes for training.
    3. Use 10 % of training sequences as internal validation (by athlete).
    4. Build and train a fresh LSTM from scratch.
    5. Evaluate on the held-out athlete's sequences.

    Parameters
    ----------
    df_all       : Full dataset (all athletes) with feature_cols, target_col.
    feature_cols : Features to use as input.
    target_col   : Binary target column.
    window_size  : Sequence length (days).
    athlete_col  : Participant ID column.

    Returns
    -------
    LSTMLoaoResult with per-fold stats and mean±std.
    """
    import keras

    rng = np.random.RandomState(seed)
    athletes = list(df_all[athlete_col].unique())
    n_feat = len(feature_cols)
    folds: List[LSTMFoldResult] = []

    for i, test_pid in enumerate(athletes):
        logger.info("LSTM LOAO fold %d/%d [%s]", i + 1, len(athletes), test_pid)

        # ── Held-out test athlete ──────────────────────────────
        df_test = df_all[df_all[athlete_col] == test_pid].copy()
        X_test_seq, y_test_seq, meta_test_seq = make_sequences(
            df_test, feature_cols, target_col, window_size, athlete_col,
        )

        if int(y_test_seq.sum()) == 0 or len(X_test_seq) == 0:
            logger.info("  SKIPPED — 0 injury sequences for athlete %s", test_pid)
            folds.append(LSTMFoldResult(
                participant_id=test_pid,
                roc_auc=float("nan"), pr_auc=float("nan"), f1=float("nan"),
                n_samples=len(X_test_seq), n_injuries=0, skipped=True,
            ))
            continue

        # ── Training pool (all other athletes) ────────────────
        df_train_pool = df_all[df_all[athlete_col] != test_pid].copy()
        train_athletes = [p for p in athletes if p != test_pid]

        # Use last 10 % of athletes (by list order) as internal val
        n_val_athletes = max(1, len(train_athletes) // 10)
        rng.shuffle(train_athletes)
        val_athletes = train_athletes[:n_val_athletes]
        fit_athletes = train_athletes[n_val_athletes:]

        df_fit = df_train_pool[df_train_pool[athlete_col].isin(fit_athletes)]
        df_iVal = df_train_pool[df_train_pool[athlete_col].isin(val_athletes)]

        X_fit, y_fit, _ = make_sequences(
            df_fit, feature_cols, target_col, window_size, athlete_col,
        )
        X_ival, y_ival, _ = make_sequences(
            df_iVal, feature_cols, target_col, window_size, athlete_col,
        )

        if len(X_fit) == 0 or int(y_fit.sum()) == 0:
            logger.warning("  SKIPPED — empty/no-positive training fold for %s", test_pid)
            folds.append(LSTMFoldResult(
                participant_id=test_pid,
                roc_auc=float("nan"), pr_auc=float("nan"), f1=float("nan"),
                n_samples=len(X_test_seq), n_injuries=int(y_test_seq.sum()),
                skipped=True,
            ))
            continue

        if len(X_ival) == 0:
            X_ival, y_ival = X_fit[:batch_size], y_fit[:batch_size]

        # ── Train fresh model ──────────────────────────────────
        keras.utils.set_random_seed(seed)
        model = build_lstm_model(
            n_features=n_feat,
            window_size=window_size,
            lstm_units=lstm_units,
            lstm_units_2=lstm_units_2,
            dense_units=dense_units,
            dropout=dropout,
        )
        train_lstm(model, X_fit, y_fit, X_ival, y_ival,
                   epochs=epochs, batch_size=batch_size, patience=patience)

        # ── Evaluate ───────────────────────────────────────────
        metrics = evaluate_lstm(model, X_test_seq, y_test_seq, meta_test_seq)
        logger.info(
            "  ROC-AUC=%.4f, PR-AUC=%.4f, F1=%.4f "
            "(n=%d, injuries=%d)",
            metrics["roc_auc"], metrics["pr_auc"], metrics["f1"],
            metrics["n_samples"], metrics["n_pos"],
        )
        folds.append(LSTMFoldResult(
            participant_id=test_pid,
            roc_auc=metrics["roc_auc"],
            pr_auc=metrics["pr_auc"],
            f1=metrics["f1"],
            n_samples=metrics["n_samples"],
            n_injuries=metrics["n_pos"],
            skipped=False,
        ))

        # Free memory after each fold
        del model
        keras.backend.clear_session()

    # ── Aggregate ─────────────────────────────────────────────
    valid = [f for f in folds if not f.skipped]
    aucs = [f.roc_auc for f in valid]
    pr_aucs = [f.pr_auc for f in valid]
    f1s = [f.f1 for f in valid]

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
    mean_pr = float(np.mean(pr_aucs)) if pr_aucs else 0.0
    std_pr = float(np.std(pr_aucs, ddof=1)) if len(pr_aucs) > 1 else 0.0
    mean_f1 = float(np.mean(f1s)) if f1s else 0.0
    std_f1 = float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0

    logger.info(
        "LSTM LOAO complete — mean ROC-AUC=%.4f (±%.4f), "
        "mean PR-AUC=%.4f  [%d valid folds, %d skipped]",
        mean_auc, std_auc, mean_pr,
        len(valid), len(folds) - len(valid),
    )

    return LSTMLoaoResult(
        folds=folds,
        mean_roc_auc=mean_auc,
        std_roc_auc=std_auc,
        mean_pr_auc=mean_pr,
        std_pr_auc=std_pr,
        mean_f1=mean_f1,
        std_f1=std_f1,
        n_skipped_folds=len(folds) - len(valid),
    )
