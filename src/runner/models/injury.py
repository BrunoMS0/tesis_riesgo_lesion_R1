"""
injury.py — M2: Constructor del dataset y configuración del clasificador de lesión.

Segunda etapa del pipeline predictivo M1→M2. Prepara el dataset del Runner Dataset
para el clasificador de lesión, combinando las features GPS (de transform.py) con
la predicción de fatiga de M1 (fatigue_score_predicted).

Produce el mismo formato InjuryDatasetBundle que src/injury/dataset.py, permitiendo
reutilizar directamente loso_cross_validation, train_injury_model y evaluate_model.

Estrategia de split
-------------------
  Estratificado 70/10/20 a nivel de atleta por presencia de lesión (seed=42).
  Tanto atletas lesionados como no lesionados aparecen en todas las particiones.

Módulo anterior en el pipeline
---------------------------------
  src/runner/models/fatigue.py → runner_fatigue_predictions_loao.csv
  (el fatigue_score_predicted se une al dataset antes de llamar a build_runner_datasets)

Dependencias compartidas de src/injury/
-----------------------------------------
  InjuryConfig         — dataclass de configuración del modelo
  InjuryDatasetBundle  — contenedor estandarizado de splits train/val/test
  apply_normalizer     — aplica PowerTransformer ajustado
  fit_normalizer       — ajusta PowerTransformer sobre el conjunto de entrenamiento

Public API
----------
    build_runner_datasets(csv_path, feature_cols, save_processed) -> InjuryDatasetBundle
    make_runner_injury_config(feature_cols)                        -> InjuryConfig
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.injury.config import InjuryConfig
from src.injury.dataset import InjuryDatasetBundle
from src.injury.normalize import apply_normalizer, fit_normalizer

from ..config import (
    AUGMENTATION_METHOD,
    PMDATA_COMMON_FEATURES,
    RF_CLASS_WEIGHT,
    RF_MAX_FEATURES,
    RF_N_ESTIMATORS,
    RUNNER_COMMON_FEATURES,
    RUNNER_CSV,
    RUNNER_FEATURE_COLUMNS,
    RUNNER_OUTPUT_CSV,
    SEED,
    SMOTE_K_NEIGHBORS,
    TARGET_COL,
    TARGET_RATIO,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from ..etl.extract import load_runner_csv
from ..etl.transform import compute_features

logger = logging.getLogger(__name__)


# ─── Split estratificado ──────────────────────────────────────────────────────

def _stratified_split(
    pids: List[str],
    injury_flags: List[int],
    seed: int = SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split estratificado 70 / 10 / 20 a nivel de atleta.

    Estratificación por injury_flag (0 = sin lesión, 1 = ≥1 lesión) para
    garantizar representación proporcional de ambos grupos en todos los splits.
    """
    train_ids, temp_ids, _, temp_flags = train_test_split(
        pids, injury_flags,
        test_size=1.0 - TRAIN_SPLIT,
        random_state=seed,
        stratify=injury_flags,
    )
    # val = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT) del temp
    val_ratio = VAL_SPLIT / (1.0 - TRAIN_SPLIT)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=1.0 - val_ratio,
        random_state=seed,
        stratify=temp_flags,
    )
    return list(train_ids), list(val_ids), list(test_ids)


# ─── Constructor principal del dataset ───────────────────────────────────────

def build_runner_datasets(
    csv_path: str = RUNNER_CSV,
    feature_cols: Optional[List[str]] = None,
    save_processed: bool = True,
) -> InjuryDatasetBundle:
    """
    Pipeline completo de preparación del dataset del Runner Dataset:
      extract → feature engineering → split estratificado → normalización Yeo-Johnson.

    Parameters
    ----------
    csv_path      : Ruta a day_approach_maskedID_timeseries.csv.
    feature_cols  : Override de la lista de features (por defecto: RUNNER_FEATURE_COLUMNS).
    save_processed: Si True, guarda el CSV procesado en RUNNER_OUTPUT_CSV.

    Returns
    -------
    InjuryDatasetBundle compatible con los componentes de src/injury/
    (loso_cross_validation, train_injury_model, evaluate_model, etc.).

    Notas
    -----
    - target_col = 'injury' (ya prospectiva — no se llama create_prospective_target)
    - dfi_predicted no está presente (el Runner Dataset no tiene sensor Fitbit)
    - Los participant_id son strings: 'runner_0' … 'runner_73'
    """
    if feature_cols is None:
        feature_cols = RUNNER_FEATURE_COLUMNS

    # ── 1. Extracción ──────────────────────────────────────────────────────────
    raw_df = load_runner_csv(csv_path)

    # ── 2. Ingeniería de features ──────────────────────────────────────────────
    df = compute_features(raw_df)

    # ── 3. Guardar CSV procesado (opcional) ───────────────────────────────────
    if save_processed:
        Path(RUNNER_OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RUNNER_OUTPUT_CSV, index=False)
        logger.info("Dataset procesado guardado → %s (%d filas)", RUNNER_OUTPUT_CSV, len(df))

    # ── 4. Split estratificado de atletas ─────────────────────────────────────
    pids = sorted(df["participant_id"].unique())
    injury_per_athlete = df.groupby("participant_id")["injury"].max()
    flags = [int(injury_per_athlete[p]) for p in pids]

    train_pids, val_pids, test_pids = _stratified_split(pids, flags, seed=SEED)

    n_inj = lambda plist: int(sum(injury_per_athlete[p] for p in plist))
    logger.info(
        "Split de atletas — train: %d (%d lesionados) | val: %d (%d) | test: %d (%d)",
        len(train_pids), n_inj(train_pids),
        len(val_pids),   n_inj(val_pids),
        len(test_pids),  n_inj(test_pids),
    )

    # ── 5. Construcción de la matriz de features ───────────────────────────────
    avail = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(avail)
    if missing:
        logger.warning("Features no encontradas en el dataset procesado (omitidas): %s", missing)

    X    = df[avail]
    y    = df[TARGET_COL].astype(int)
    meta = df[["participant_id", "date"]].copy()

    def _subset(pids_list: List[str]):
        mask = meta["participant_id"].isin(pids_list)
        return (
            X.loc[mask].reset_index(drop=True),
            y.loc[mask].reset_index(drop=True),
            meta.loc[mask].reset_index(drop=True),
        )

    X_train, y_train, meta_train = _subset(train_pids)
    X_val,   y_val,   meta_val   = _subset(val_pids)
    X_test,  y_test,  meta_test  = _subset(test_pids)

    logger.info(
        "Split de filas — train: %d (%d+) | val: %d (%d+) | test: %d (%d+)",
        len(X_train), int(y_train.sum()),
        len(X_val),   int(y_val.sum()),
        len(X_test),  int(y_test.sum()),
    )

    # ── 6. Normalización Yeo-Johnson (ajustar solo sobre entrenamiento) ────────
    X_train_raw = X_train.copy()
    X_val_raw   = X_val.copy()
    X_test_raw  = X_test.copy()

    normalizer = fit_normalizer(X_train)
    X_train    = apply_normalizer(X_train, normalizer)
    X_val      = apply_normalizer(X_val,   normalizer)
    if len(X_test) > 0:
        X_test = apply_normalizer(X_test, normalizer)

    logger.info("Normalización Yeo-Johnson aplicada (ajustada solo sobre el split de entrenamiento)")

    return InjuryDatasetBundle(
        X_train=X_train, y_train=y_train, meta_train=meta_train,
        X_val=X_val,     y_val=y_val,     meta_val=meta_val,
        X_test=X_test,   y_test=y_test,   meta_test=meta_test,
        train_pids=train_pids, val_pids=val_pids, test_pids=test_pids,
        feature_columns=avail,
        normalizer=normalizer,
        X_train_raw=X_train_raw, X_val_raw=X_val_raw, X_test_raw=X_test_raw,
    )


# ─── Fábrica de InjuryConfig para el Runner Dataset ──────────────────────────

def make_runner_injury_config(
    feature_cols: Optional[List[str]] = None,
) -> InjuryConfig:
    """
    Devolver un InjuryConfig pre-configurado para el Runner Dataset.

    Este config se pasa a loso_cross_validation, build_model y
    augment_training_data — todos de src.injury — para reutilizarlos directamente.

    Overrides clave vs. InjuryConfig por defecto:
      - feature_columns : RUNNER_FEATURE_COLUMNS (sin dfi_predicted)
      - target_col      : 'injury'
      - model_type      : 'rf'
      - augmentation    : SMOTE con target_ratio reducido (1.36% prevalencia base)
      - train/val/test  : listas vacías → el split externo lo maneja build_runner_datasets
    """
    return InjuryConfig(
        feature_columns=list(feature_cols or RUNNER_FEATURE_COLUMNS),
        target_col=TARGET_COL,
        use_prospective_target=False,       # injury ya es prospectiva
        train_participants=[],              # splits manejados por build_runner_datasets
        val_participants=[],
        test_participants=[],
        model_type="rf",
        rf_n_estimators=RF_N_ESTIMATORS,
        rf_max_features=RF_MAX_FEATURES,
        rf_class_weight=RF_CLASS_WEIGHT,
        augmentation_method=AUGMENTATION_METHOD,
        target_ratio=TARGET_RATIO,
        smote_k_neighbors=SMOTE_K_NEIGHBORS,
        seed=SEED,
    )
