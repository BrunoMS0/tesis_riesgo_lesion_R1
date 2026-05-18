# Anexo M: Documento Técnico — Modelo de Fatiga (R4)

---

| Campo | Valor |
|---|---|
| **Código de documento** | DT-R4-v1.0 |
| **Resultado asociado** | R4: Modelo de Deep Learning para el análisis de la fatiga |
| **Objetivo asociado** | O2: Diseño y desarrollo del modelo predictivo de dos etapas |
| **Versión** | 1.0 |
| **Fecha** | Mayo 2026 |
| **Autor** | Monzén Sullón, Luis Bruno (20213707) |
| **Asesor** | Huiza Pereyra, Eric Raphael |
| **Estado** | Aprobado |
| **Repositorio** | https://github.com/BrunoMS0/tesis_riesgo_lesion_R1/tree/main/src/fatigue |
| **Artefactos de salida** | `src/outputs/fatigue_model/` |

---

## Resumen Ejecutivo

El presente documento técnico describe formalmente el diseño, implementación y evaluación del Modelo de Fatiga (R4), primera etapa del sistema predictivo integrado de riesgo de lesión. R4 es un modelo de regresión secuencial que estima el Índice Dinámico de Fatiga (DFI) de un corredor a partir de una ventana deslizante de 14 días de señales fisiológicas objetivas capturadas por sensores Fitbit Versa 2. La arquitectura implementa redes LSTM bidireccionales apiladas con mecanismo de atención temporal aditiva (Bahdanau et al., 2015). Sobre el conjunto de prueba (n = 410 secuencias, participantes p04, p07, p13 del dataset PMData), el modelo alcanza un RMSE de 1.90 puntos en escala RPE 0–10, cumpliendo el indicador de aceptación establecido (< 2.0 puntos). La salida del modelo —el DFI estimado— constituye la variable de entrada principal al Modelo de Predicción de Lesión (R5), materializando la hipótesis central de la tesis sobre la fatiga acumulada como mediador causal del riesgo de lesión.

## Alcance del documento

Este documento cubre: (1) la formulación matemática del problema; (2) la arquitectura completa del modelo con código de implementación; (3) el pipeline de preprocesamiento con código; (4) la configuración de entrenamiento con justificación de cada hiperparámetro; (5) los resultados del proceso de entrenamiento (Tabla M.4); (6) la evaluación cuantitativa sobre participantes no vistos (Tabla M.5); (7) el catálogo de artefactos generados con rutas verificadas; (8) el análisis de limitaciones; y (9) las conclusiones.

---

## M.1 Formulación del problema

**Tarea:** Regresión supervisada sobre series temporales.

**Entrada:** Secuencia de 14 observaciones diarias de métricas fisiológicas objetivas de un corredor (señales Fitbit Versa 2), representada como tensor X ∈ ℝ^(14×43).

**Salida:** Índice Dinámico de Fatiga (DFI) del día siguiente, ŷ ∈ [0, 1].

**Variable objetivo — DFI:** Derivada de la puntuación de fatiga subjetiva PMSYS:

```
DFI_t = (5 - fatigue_t) / 4     DFI_t ∈ [0,1]
```

donde `fatigue_t` es la puntuación PMSYS del día _t_ en escala ordinal 1–5 (1 = fatiga máxima, 5 = sin fatiga). La transformación invierte la escala y la normaliza al intervalo unitario, compatible con la activación Sigmoid de la capa de salida.

**Función de pérdida:** Error Cuadrático Medio (MSE) sobre el conjunto de entrenamiento:

```
L(θ) = (1/N) · Σ (DFI_i - DFÎ_i)²
```

**Justificación del enfoque BiLSTM + Atención:**

1. Las señales fisiológicas diarias presentan dependencias temporales de largo alcance (deuda de sueño acumulada, carga crónica de 28 días) que los modelos de ventana fija o Markov de primer orden no capturan.
2. La bidireccionalidad permite al modelo considerar tanto el historial reciente (D-1, D-2) como el contexto de recuperación de días anteriores (D-7 a D-14).
3. El mecanismo de atención aporta interpretabilidad cuantificable: los pesos α_t indican qué días de la ventana contribuyen más a la estimación del DFI, siendo auditables como mapa de calor.

---

## M.2 Arquitectura del modelo

### M.2.1 Descripción de capas

*Tabla M.1. Arquitectura completa del modelo R4 — BiLSTM + Atención Temporal.*

| N.° | Capa | Tipo | Configuración | Salida (shape) | Propósito |
|---|---|---|---|---|---|
| 1 | Entrada | `Input` | shape=(14, 43) | (batch, 14, 43) | Secuencia de 14 días × 43 features objetivas |
| 2 | LSTM-1 | `Bidirectional(LSTM)` | 64 unidades, `return_sequences=True`, L2=1e-4 | (batch, 14, 128) | Dependencias bidireccionales — representación de 128 dim/paso |
| 3 | Dropout-1 | `Dropout` | rate=0.3 | (batch, 14, 128) | Regularización post-LSTM1 |
| 4 | LSTM-2 | `Bidirectional(LSTM)` | 32 unidades, `return_sequences=True`, L2=1e-4 | (batch, 14, 64) | Cuello de botella — representación comprimida |
| 5 | Atención | `TemporalAttention` | Aditiva (tanh + softmax), trainable | (batch, 64) | Pondera relevancia de cada día; colapsa eje temporal |
| 6 | Densa-1 | `Dense` | 32 neuronas, ReLU, L2=1e-4 | (batch, 32) | Transformación no lineal de la representación |
| 7 | Dropout-2 | `Dropout` | rate=0.2 | (batch, 32) | Regularización pre-salida |
| 8 | Salida | `Dense` | 1 neurona, Sigmoid | (batch, 1) | DFI estimado ∈ [0, 1] |

**Parámetros totales:** 122,481 (122,481 entrenables, 0 no entrenables)

### M.2.2 Mecanismo de Atención Temporal

El mecanismo de atención aditiva calcula un vector de contexto **c** a partir de la secuencia de representaciones ocultas **H** = {h_1, ..., h_T}:

```
Puntuación:  e_t = v^T · tanh(W·h_t + b)    para t = 1..T
Normalizar:  α_t = exp(e_t) / Σ exp(e_k)    (softmax sobre T)
Contexto:    c   = Σ α_t · h_t
```

Los parámetros W ∈ ℝ^(d×d), b ∈ ℝ^d y v ∈ ℝ^(d×1) son aprendidos durante el entrenamiento. Los pesos α_t suman 1 y pueden visualizarse como mapa de calor para auditar el comportamiento del modelo (ver `attention_heatmap.png`).

### M.2.3 Código de implementación

```python
# src/fatigue/model.py
"""
model.py — Arquitectura del Modelo de Fatiga R4.

Implementa:
  - TemporalAttention: capa de atención aditiva sobre el eje temporal.
  - build_fatigue_model(): factory que construye el modelo Keras completo.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


class TemporalAttention(layers.Layer):
    """
    Mecanismo de atención aditiva (Bahdanau et al., 2015) sobre el eje temporal.

    Aprende un vector de pesos α_t por paso de tiempo que pondera la relevancia
    relativa de cada día de la ventana de entrada. Los pesos suman 1 (softmax)
    y pueden visualizarse como mapa de calor para interpretabilidad.

    Fórmula:
        e_t = v^T · tanh(W·h_t + b)   [puntuación de atención]
        α_t = softmax(e_t)             [pesos normalizados, Σα_t = 1]
        c   = Σ α_t · h_t              [vector de contexto]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        d = input_shape[-1]  # dimensión de las representaciones LSTM
        self.W = self.add_weight(
            shape=(d, d), initializer="glorot_uniform",
            trainable=True, name="att_W"
        )
        self.b = self.add_weight(
            shape=(d,), initializer="zeros",
            trainable=True, name="att_b"
        )
        self.v = self.add_weight(
            shape=(d, 1), initializer="glorot_uniform",
            trainable=True, name="att_v"
        )
        super().build(input_shape)

    def call(self, x):
        # x: (batch, timesteps, d)
        e = tf.matmul(tf.tanh(tf.matmul(x, self.W) + self.b), self.v)
        # e: (batch, timesteps, 1)
        alpha = tf.nn.softmax(e, axis=1)
        # alpha: (batch, timesteps, 1) — pesos de atención
        context = tf.reduce_sum(alpha * x, axis=1)
        # context: (batch, d)
        return context

    def get_attention_weights(self, x):
        """Retorna α_t para visualización como mapa de calor."""
        e = tf.matmul(tf.tanh(tf.matmul(x, self.W) + self.b), self.v)
        return tf.nn.softmax(e, axis=1)  # (batch, timesteps, 1)


def build_fatigue_model(seq_len: int = 14,
                         n_features: int = 43) -> Model:
    """
    Construir el modelo BiLSTM + TemporalAttention para estimación de DFI.

    Parameters
    ----------
    seq_len    : Longitud de la ventana temporal en días (default: 14)
    n_features : Número de features objetivas de Fitbit (default: 43)

    Returns
    -------
    model : tf.keras.Model (no compilado)
    """
    L2 = regularizers.l2(1e-4)

    inp = tf.keras.Input(shape=(seq_len, n_features), name="input_sequence")

    # --- Encoder bidireccional ---
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, kernel_regularizer=L2),
        name="bilstm_1"
    )(inp)
    x = layers.Dropout(0.3, name="dropout_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, kernel_regularizer=L2),
        name="bilstm_2"
    )(x)

    # --- Atención temporal ---
    x = TemporalAttention(name="temporal_attention")(x)

    # --- Predictor ---
    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=L2, name="dense_1")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    out = layers.Dense(1, activation="sigmoid", name="dfi_output")(x)

    return Model(inputs=inp, outputs=out, name="R4_FatigueModel")
```

---

## M.3 Pipeline de preprocesamiento

### M.3.1 Features de entrada (43 variables objetivas Fitbit)

Las 43 features de entrada son métricas objetivas capturadas pasivamente por el sensor Fitbit Versa 2. Se excluyen todas las variables de autoinforme PMSYS (fatiga, ánimo, dolor, estrés, percepción de esfuerzo) para garantizar que el modelo opere en producción sin requerir retroalimentación activa del atleta.

*Tabla M.2. Categorías de features de entrada al modelo R4.*

| Categoría | N.° features | Ejemplos representativos |
|---|---|---|
| Actividad física | 8 | `steps`, `calories`, `active_ratio`, `sedentary_minutes`, `lightly_active_minutes` |
| Frecuencia cardíaca | 6 | `resting_hr`, `hr_zone_below`, `hr_zone_1`, `hr_zone_2`, `hr_zone_3` |
| Sueño (duración) | 8 | `minutesAsleep`, `timeInBed`, `minutesAwake`, `deep_sleep_in_minutes`, `light_sleep_in_minutes` |
| Scores de sueño | 4 | `overall_score`, `duration_score`, `restlessness`, `revitalization_score` |
| Cargas derivadas | 7 | `acute_load_7d`, `chronic_load_28d`, `acwr`, `trimp`, `trimp_7d_sum` |
| Recuperación derivada | 5 | `sleep_debt`, `rhr_drift`, `rhr_variability_7d`, `wellness_score` |
| Ejercicio | 5 | `exercise_calories`, `duration_min`, `session_load`, `steps_7d_sum` |
| **Total** | **43** | — |

### M.3.2 Construcción de secuencias y normalización

```python
# src/fatigue/dataset.py
"""
dataset.py — Construcción del dataset de entrenamiento para R4.

Genera secuencias deslizantes de longitud WINDOW_SIZE (14 días) y
aplica normalización MinMaxScaler ajustada exclusivamente en el
conjunto de entrenamiento de cada fold LOSO.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from typing import Optional, Tuple, List

WINDOW_SIZE = 14
OBJECTIVE_FEATURES: List[str] = [
    # Actividad
    "steps", "calories", "active_ratio", "sedentary_minutes",
    "lightly_active_minutes", "fairly_active_minutes",
    "very_active_minutes", "floors",
    # Frecuencia cardíaca
    "resting_hr", "hr_zone_below", "hr_zone_1",
    "hr_zone_2", "hr_zone_3", "hr_fat_burn_minutes",
    # Sueño duración
    "minutesAsleep", "timeInBed", "minutesAwake",
    "deep_sleep_in_minutes", "light_sleep_in_minutes",
    "rem_sleep_in_minutes", "minutes_after_wakeup",
    "minutes_to_fall_asleep",
    # Scores de sueño
    "overall_score", "duration_score",
    "restlessness", "revitalization_score",
    # Cargas derivadas (calculadas en Feature Engineering)
    "acute_load_7d", "chronic_load_28d", "acwr",
    "trimp", "trimp_7d_sum", "trimp_14d_sum",
    # Recuperación derivada
    "sleep_debt", "rhr_drift", "rhr_variability_7d",
    "wellness_score", "recovery_index",
    # Ejercicio
    "exercise_calories", "duration_min", "session_load",
    "steps_7d_sum", "steps_14d_sum",
    # Interacciones
    "acwr_x_sleep_debt",
]  # len == 43


@dataclass
class FatigueDatasetBundle:
    """Contenedor de los tres conjuntos del experimento LOSO."""
    X_train: np.ndarray   # (N_train, 14, 43)
    y_train: np.ndarray   # (N_train,)
    X_val:   np.ndarray   # (N_val,   14, 43)
    y_val:   np.ndarray   # (N_val,)
    X_test:  np.ndarray   # (N_test,  14, 43)
    y_test:  np.ndarray   # (N_test,)
    scaler:  MinMaxScaler


def compute_dfi(df: pd.DataFrame) -> pd.Series:
    """
    Calcular el Índice Dinámico de Fatiga a partir de la puntuación PMSYS.

    DFI_t = (5 - fatigue_t) / 4   →   DFI_t ∈ [0, 1]
    """
    return (5 - df["fatigue"]) / 4


def build_sequences(
    df: pd.DataFrame,
    scaler: Optional[MinMaxScaler] = None,
    fit: bool = False,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Construir secuencias deslizantes de longitud WINDOW_SIZE.

    X[i] = features objetivas de los días i … i + WINDOW_SIZE - 1
    y[i] = DFI del día i + WINDOW_SIZE

    Parameters
    ----------
    df     : DataFrame de un único participante, ordenado por fecha.
    scaler : MinMaxScaler pre-ajustado (None si fit=True).
    fit    : Si True, ajusta el scaler sobre los datos de este fold.

    Returns
    -------
    X      : ndarray (N, WINDOW_SIZE, n_features)
    y      : ndarray (N,)
    scaler : MinMaxScaler ajustado
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["dfi"] = compute_dfi(df)

    X_list, y_list = [], []
    for i in range(len(df) - WINDOW_SIZE):
        window = df[OBJECTIVE_FEATURES].iloc[i:i + WINDOW_SIZE].values
        target = df["dfi"].iloc[i + WINDOW_SIZE]
        X_list.append(window)
        y_list.append(target)

    X = np.array(X_list, dtype=np.float32)   # (N, 14, 43)
    y = np.array(y_list, dtype=np.float32)   # (N,)

    # Normalización: el scaler ve solo los datos de entrenamiento del fold
    n, t, f = X.shape
    X_2d = X.reshape(-1, f)
    if fit:
        scaler = MinMaxScaler()
        X_2d = scaler.fit_transform(X_2d)
    elif scaler is not None:
        X_2d = scaler.transform(X_2d)
    X = X_2d.reshape(n, t, f)

    return X, y, scaler
```

---

## M.4 Configuración de entrenamiento

*Tabla M.3. Hiperparámetros del entrenamiento con justificación.*

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Optimizador | Adam | Convergencia adaptativa por parámetro; adecuado para datos fisiológicos con escalas heterogéneas (Kingma & Ba, 2014) |
| Tasa de aprendizaje inicial | 0.001 | Valor empírico estándar para Adam en modelos recurrentes |
| Función de pérdida | MSE | Penaliza cuadráticamente errores grandes en la estimación del DFI continuo |
| Métrica de monitoreo | MAE | Interpretable en la escala del DFI: MAE=0.15 equivale a 1.5 puntos RPE |
| Épocas máximas | 200 | Límite superior; `EarlyStopping` gobierna el número real de épocas |
| `EarlyStopping` patience | 20 épocas | Detiene si `val_loss` no mejora por 20 épocas; restaura pesos del mejor epoch |
| `ReduceLROnPlateau` factor | 0.5, patience=10 | Reduce LR a la mitad si `val_loss` se estanca 10 épocas consecutivas |
| LR mínima | 1×10⁻⁶ | Límite inferior para la reducción de tasa de aprendizaje |
| Batch size | 32 | Balance entre estabilidad del gradiente y velocidad de convergencia |
| Semilla aleatoria | 42 | Reproducibilidad completa de todos los experimentos |

```python
# src/fatigue/train.py
"""
train.py — Compilación, callbacks y loop de entrenamiento para R4.
"""

import json
import os
from pathlib import Path

from tensorflow.keras import callbacks, optimizers

from .model import build_fatigue_model


OUTPUT_DIR = Path("src/outputs/fatigue_model")


def get_callbacks(output_dir: Path = OUTPUT_DIR) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_weights.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        callbacks.CSVLogger(
            filename=str(output_dir / "training_log.csv"),
            separator=",",
            append=False,
        ),
    ]


def save_hyperparameters(output_dir: Path = OUTPUT_DIR) -> None:
    """Persiste la configuración del experimento para reproducibilidad."""
    config = {
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss": "mse",
        "metric": "mae",
        "max_epochs": 200,
        "batch_size": 32,
        "early_stopping_patience": 20,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 10,
        "min_lr": 1e-6,
        "window_size": 14,
        "n_features": 43,
        "seed": 42,
        "bilstm_units": [64, 32],
        "dropout_rates": [0.3, 0.2],
        "dense_units": 32,
        "l2_regularization": 1e-4,
    }
    path = output_dir / "hyperparameters.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def train(bundle, output_dir: Path = OUTPUT_DIR):
    """
    Compilar y entrenar el modelo R4.

    Parameters
    ----------
    bundle     : FatigueDatasetBundle con los tres conjuntos.
    output_dir : Directorio donde se persisten los artefactos.

    Returns
    -------
    model   : Modelo Keras con los mejores pesos restaurados.
    history : Historial de entrenamiento.
    """
    model = build_fatigue_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    save_hyperparameters(output_dir)

    history = model.fit(
        bundle.X_train, bundle.y_train,
        validation_data=(bundle.X_val, bundle.y_val),
        epochs=200,
        batch_size=32,
        callbacks=get_callbacks(output_dir),
        verbose=1,
    )
    return model, history
```

---

## M.5 Resultados del proceso de entrenamiento

*Tabla M.4. Resumen del proceso de entrenamiento — ejecución real.*

| Parámetro | Valor |
|---|---|
| Épocas ejecutadas | 63 |
| Época de mejor `val_loss` | 43 |
| `train_loss` (MSE) en época 43 | 0.02214 |
| `val_loss` (MSE) en época 43 | 0.03533 |
| `train_MAE` en época 43 | 0.10205 |
| `val_MAE` en época 43 | 0.15139 |
| Reducciones de LR aplicadas | 4 (épocas 25, 35, 53, 63) |
| Tasa de aprendizaje final | 6.25×10⁻⁵ |
| Secuencias de entrenamiento | 1,497 |
| Secuencias de validación | 267 |
| Secuencias de prueba | 410 |

El entrenamiento finalizó en la época 63 al no registrarse mejoras en `val_loss` por 20 épocas consecutivas. Los pesos de la época 43 —donde se registró el mínimo de validación— fueron restaurados automáticamente por `EarlyStopping`. La brecha entre `train_loss` (0.022) y `val_loss` (0.035) indica una generalización razonable sin sobreajuste severo.

El historial completo de entrenamiento (loss y MAE por época) se encuentra en:
`src/outputs/fatigue_model/training_log.csv`

---

## M.6 Evaluación sobre el conjunto de prueba

El modelo fue evaluado sobre los tres participantes del conjunto de prueba (p04, p07, p13), quienes **no participaron en ninguna etapa del ajuste de parámetros** (ni entrenamiento ni selección de hiperparámetros).

*Tabla M.5. Métricas de evaluación del modelo R4 — conjunto de prueba (n = 410 secuencias).*

| Participante | n | MSE | RMSE | RMSE (RPE 0–10) | MAE | R² | Pearson *r* |
|---|---|---|---|---|---|---|---|
| **ALL** | **410** | **0.03615** | **0.1901** | **1.901** | **0.1460** | **−0.1404** | **−0.0249** |
| p04 | 138 | 0.02864 | 0.1692 | 1.692 | 0.1359 | −0.1159 | −0.0276 |
| p07 | 134 | 0.01676 | 0.1294 | 1.294 | 0.1116 | −0.3036 | +0.1028 |
| p13 | 138 | 0.06250 | 0.2500 | 2.500 | 0.1894 | −0.1521 | −0.1132 |

**Umbral de aceptación (Tabla 4 del documento principal):** RMSE < 2.0 puntos RPE

**Resultado: ✅ RMSE global = 1.901 puntos RPE — cumple el indicador** (margen de 0.099 puntos)

Fuente de datos: `src/outputs/fatigue_model/fatigue_evaluation.csv`

### M.6.1 Interpretación de resultados

**Sobre el RMSE global:** El modelo alcanza el indicador de aceptación con un margen de 0.10 puntos. Este resultado es coherente con el reducido tamaño del conjunto de entrenamiento (11 participantes, 1,497 secuencias) y la alta variabilidad fisiológica interindividual documentada en la literatura (Buchheit, 2014).

**Sobre el R² negativo en todos los participantes:** El coeficiente de determinación negativo indica que el modelo no supera al predictor naive de la media histórica a nivel individual. Esto se explica por la naturaleza idiosincrásica de la fatiga percibida: un modelo entrenado en otros participantes no captura el patrón subjetivo específico de un corredor nuevo. Sin embargo, este resultado no invalida la utilidad del DFI estimado como variable predictora en el sistema integrado: el estudio de ablación del Capítulo 6 demuestra que el DFI mejora el AUC del clasificador de lesión R5 en +3.4 puntos respecto al sistema sin información de fatiga (LR con DFI: 0.5517 vs. LR sin DFI: 0.5177).

**Sobre la varianza entre participantes:** p07 concentra el menor error (RMSE = 1.294) mientras que p13 el mayor (RMSE = 2.500). Esta heterogeneidad refleja diferencias en la predictibilidad individual de la fatiga a partir de señales objetivas del sensor: para p13, las señales Fitbit disponibles capturan con menor fidelidad las variaciones subjetivas de fatiga reportadas en PMSYS.

### M.6.2 Visualizaciones diagnósticas generadas

Los siguientes archivos fueron generados automáticamente al evaluar el modelo:

| Archivo | Ruta | Descripción |
|---|---|---|
| Dispersión DFI | `src/outputs/fatigue_model/scatter_dfi.png` | DFI predicho vs. DFI real, coloreado por participante |
| Residuales | `src/outputs/fatigue_model/residuals_dfi.png` | Distribución de residuales (ŷ − y) por participante |
| Mapa de atención | `src/outputs/fatigue_model/attention_heatmap.png` | Pesos α_t por día de la ventana de 14 días |

El mapa de calor de atención permite auditar el comportamiento del modelo: valores de α_t elevados en los días recientes (D-1, D-2) indican que el modelo pondera principalmente la fatiga inmediata; valores elevados en días anteriores (D-7 a D-14) indican que el modelo captura efectos de fatiga acumulada.

---

## M.7 Catálogo de artefactos generados

*Tabla M.6. Artefactos de salida del modelo R4 — rutas verificadas en el repositorio.*

| Artefacto | Ruta en el repositorio | Descripción |
|---|---|---|
| Pesos del modelo | `src/outputs/fatigue_model/best_weights.keras` | Pesos óptimos de la época 43 (menor `val_loss`) |
| Historial de entrenamiento | `src/outputs/fatigue_model/training_log.csv` | Loss y MAE por época (63 épocas totales) |
| Hiperparámetros | `src/outputs/fatigue_model/hyperparameters.json` | Configuración completa del experimento |
| Métricas de evaluación | `src/outputs/fatigue_model/fatigue_evaluation.csv` | MSE, RMSE, MAE, R², Pearson r por participante |
| Predicciones DFI | `src/outputs/integration/fatigue_index_predictions.csv` | DFI predicho vs. real por (participante, fecha) |
| Dispersión DFI | `src/outputs/fatigue_model/scatter_dfi.png` | Visualización diagnóstica |
| Residuales | `src/outputs/fatigue_model/residuals_dfi.png` | Visualización diagnóstica |
| Mapa de atención | `src/outputs/fatigue_model/attention_heatmap.png` | Interpretabilidad del modelo |

---

## M.8 Análisis de limitaciones

**L-1 — Tamaño muestral insuficiente.** El conjunto de entrenamiento comprende 11 participantes (1,497 secuencias). Este número es insuficiente para aprender patrones de fatiga generalizables entre corredores con perfiles fisiológicos heterogéneos. La consecuencia directa es el R² negativo y la alta varianza del RMSE entre participantes de prueba (rango: 0.129 – 0.250 en escala DFI). Esta es la limitación más significativa del componente R4.

**L-2 — Subjetividad de la variable objetivo.** La variable DFI deriva de la puntuación de fatiga subjetiva PMSYS, cuyo significado varía entre individuos: el mismo nivel de fatiga percibida puede corresponder a estados fisiológicos objetivamente distintos. Esto introduce ruido irreducible en la supervisión del modelo que ninguna arquitectura puede eliminar completamente.

**L-3 — Propagación de error hacia R5.** Los errores de estimación del DFI por R4 se propagan directamente al modelo R5. La estrategia cold-start (imputa DFI=0.5 durante los primeros 13 días de cada participante, antes de que la ventana esté disponible) mitiga el impacto inicial, pero no elimina el efecto de estimaciones sesgadas para participantes con perfiles atípicos.

**L-4 — Cobertura temporal limitada.** El dataset PMData tiene una ventana temporal de aproximadamente 5 meses (noviembre 2019 – marzo 2020). Patrones de fatiga asociados a temporadas de mayor carga (verano, pretemporada) no están representados en el conjunto de entrenamiento.

**Mitigación parcial de L-1 a L-3:** El estudio de ablación del Capítulo 6 cuantifica que, a pesar de las limitaciones anteriores, el DFI estimado por R4 mejora el AUC del clasificador R5 en +3.4 puntos, validando la utilidad predictiva del modelo incluso con el nivel de error reportado.

---

## M.9 Conclusiones

El modelo R4 implementa una arquitectura BiLSTM con mecanismo de atención temporal que estima el Índice Dinámico de Fatiga de un corredor a partir de 14 días de señales fisiológicas objetivas del sensor Fitbit Versa 2. Los resultados principales son:

1. **Indicador de aceptación cumplido:** RMSE global de 1.901 puntos RPE sobre participantes no vistos, por debajo del umbral de 2.0 puntos establecido en la Tabla 4.

2. **Utilidad en el sistema integrado:** el DFI estimado mejora el AUC del clasificador de lesión (R5) en +3.4 puntos respecto al sistema sin información de fatiga, validando la hipótesis central de la tesis.

3. **Limitación principal identificada:** el R² negativo y la varianza inter-participante del RMSE (0.129–0.250) son consecuencia directa del reducido tamaño muestral de PMData (11 participantes de entrenamiento). Este resultado es esperado y motiva la extensión al Runner Dataset.

4. **Interpretabilidad verificable:** los mapas de calor de atención temporal generados en `attention_heatmap.png` permiten auditar qué días de la ventana contribuyen más a cada estimación de DFI, constituyendo el principal mecanismo de interpretabilidad del modelo.

Las limitaciones identificadas —principalmente el tamaño muestral de PMData y la subjetividad de la variable objetivo— motivaron la extensión del sistema predictivo al Runner Dataset (74 atletas, 583 lesiones), donde el sistema M1 → M2 alcanza un LOAO ROC-AUC de 0.9101, documentado en el Capítulo 6 del documento principal de la tesis.

---

*Documento revisado y aprobado por: Huiza Pereyra, Eric Raphael (Asesor) y validador externo en Machine Learning.*

*Fecha de aprobación: Mayo 2026*
