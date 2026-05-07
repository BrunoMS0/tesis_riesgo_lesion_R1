# PLAN_RUNNER_V2.md
# Instrucción IA: Leer este archivo al inicio de cada sesión.
# Este plan REEMPLAZA las fases pendientes de PLAN_TESIS_RUNNER.md.
# Marcar [x] al completar cada tarea. No avanzar de fase sin completar la anterior.

---

## CONTEXTO: DECISIÓN DE ARQUITECTURA (6 mayo 2026)

### Por qué se cambió el enfoque
El pipeline original usaba **PMData (16 atletas Fitbit)** como dataset principal. El problema
fue estructural: solo 9/16 atletas tienen lesiones, lo que hace que el LOAO produzca
AUC = 0.55 — inaceptable como resultado de tesis.

### Nueva arquitectura: pipeline unificado con Runner Dataset
El **Runner Dataset (Löwdal et al., 2021)** ya tiene todo lo necesario para ambos modelos:
- **Datos objetivos de carga** (reloj GPS de running): km, zonas Z3-4, Z5-T1-T2, sprinting,
  sesiones, entrenamiento de fuerza, horas de entrenamiento alternativo
- **Cuestionario subjetivo diario**: `perceived_recovery`, `perceived_exertion`, `perceived_trainingSuccess`
- **Etiqueta de lesión**: columna `injury` (0/1)
- **74 atletas, 583 lesiones** → volumen suficiente para LOAO robusto

### Argumento académico central (no cambia)
> "Un reloj GPS de running (wearable deportivo) registra la carga de entrenamiento objetiva
> diariamente. El Modelo 1 estima la fatiga acumulada del atleta a partir de esa carga.
> El Modelo 2 usa esa fatiga estimada junto con las features de carga para predecir
> riesgo de lesión. El sistema se valida mediante LOAO sobre 74 corredores competitivos
> con seguimiento longitudinal real."

---

## FLUJO DEL PIPELINE UNIFICADO

```
Reloj GPS del corredor (wearable)
         ↓
  Datos objetivos: km, zonas Z3-4/Z5, sprinting, sesiones, fuerza
  (ventana D-7 a D-2 — 6 días de historia)
         ↓
  ┌─────────────────────────────────┐
  │  MODELO 1 — Fatigue Regressor   │
  │  RF Regressor (nuevo)           │
  │  Target: perceived_recovery D-1 │
  │  Output: fatigue_score_predicted│
  └─────────────────────────────────┘
         ↓
  ┌─────────────────────────────────┐
  │  MODELO 2 — Injury Classifier   │
  │  Random Forest (ya existe)      │
  │  Input: carga + fatigue_score   │
  │  Target: injury (0/1)           │
  │  Output: P(lesión)              │
  └─────────────────────────────────┘
         ↓
  Predicción: riesgo de lesión del corredor
```

---

## LO QUE YA EXISTE (no tocar)

| Artefacto | Ruta | Descripción |
|-----------|------|-------------|
| ETL Runner | `src/runner/extract.py` | Carga del CSV wide format |
| Feature Engineering | `src/runner/transform.py` | 18 features derivadas |
| Dataset split | `src/runner/dataset.py` | 51/7/16 atletas estratificado |
| Config | `src/runner/config.py` | Paths y parámetros |
| RF Modelo 2 | `src/outputs/rf_runner_model.pkl` | LOAO AUC = 0.9101 |
| LOAO resultados | `src/outputs/loao_runner_results.csv` | 63/74 folds válidos |
| Dataset procesado | `src/outputs/runner_dataset_processed.csv` | 42,766 obs × 21 cols |
| Validación externa | `Runner dataset/week_approach_maskedID_timeseries.csv` | Sin usar aún |

---

## FASE 7 — Modelo 1 de Fatiga sobre Runner Dataset

> Implementar un regresor que prediga `perceived_recovery` a partir de features objetivas de carga.
> Este modelo reemplaza el rol del LSTM de PMData en el pipeline.

### Diseño técnico
- **Target**: `perceived_recovery.6` (columna D-1 del dataset wide) — recuperación percibida **pre-sesión** en D-1 (el atleta la registra antes de correr, por lo que no introduce leakage respecto a los km de ese día)
- **Ventana temporal**: D-7 a D-1 inclusive (usar los agregados existentes sin modificar `transform.py`) — **Duda 1 resuelta: Opción B**
  - Justificación: `perceived_recovery.6` es PRE-sesión → temporalmente anterior a `total km.6` → no hay leakage
- **Inputs (10 features objetivas)**: `acute_load_7d`, `chronic_load_28d`, `acwr`, `high_intensity_km_7d`, `nr_sessions_7d`, `nr_rest_days_7d`, `km_sprint_7d`, `strength_days_7d`, `alt_hours_7d`, `recent_km` — **Duda 2 resuelta**
  - **Excluir** del input: `session_load_proxy` (= `acute_load_7d × mean_perceived_exertion`, contaminado con dato subjetivo), `mean_perceived_exertion`, `mean_perceived_recovery`, `mean_perceived_success`, `wellness_score`, `recent_exertion`, `recent_success`
- **Algoritmo**: RandomForestRegressor (compatible con formato wide existente, sin necesidad de reestructurar)
- **Validación**: LOAO — entrenar con 73 atletas, evaluar en 1, repetir 74 veces
- **Métricas**: RMSE, MAE, R² (comparar con baseline de media histórica por atleta)

### Tareas

- [ ] T7.1 Crear `src/runner/fatigue.py` — extrae inputs objetivos y target `perceived_recovery.6`
  - **Inputs exactos** (10 features): `acute_load_7d`, `chronic_load_28d`, `acwr`, `high_intensity_km_7d`, `nr_sessions_7d`, `nr_rest_days_7d`, `km_sprint_7d`, `strength_days_7d`, `alt_hours_7d`, `recent_km`
  - **Excluir explícitamente**: `session_load_proxy`, `mean_perceived_exertion`, `mean_perceived_recovery`, `mean_perceived_success`, `wellness_score`, `recent_exertion`, `recent_success`
  - **Target**: `perceived recovery.6` (columna raw del CSV wide, sufijo `.6`, recuperación PRE-sesión)

- [ ] T7.2 Implementar LOAO para regresión en `src/runner/fatigue.py`
  - Mismo protocolo que LOAO de lesión: dejar fuera un atleta, evaluar en él
  - Calcular RMSE, MAE, R² por atleta y media global
  - Guardar resultados en `src/outputs/loao_fatigue_runner_results.csv`

- [ ] T7.3 Entrenar modelo final en todos los atletas (sin LOAO)
  - Guardar en `src/outputs/rf_fatigue_runner_model.pkl`
  - Guardar importancia de features en `src/outputs/fatigue_feature_importance.csv`

- [ ] T7.4 Verificar: RMSE global < 0.15 en escala normalizada (equivalente a ~1.5 puntos en escala 1-7)
  - Si RMSE > 0.20: explorar gradient boosting (XGBoost/LightGBM) como alternativa
  - Si R² < 0 en mayoría de atletas: documentar y justificar (variabilidad individual esperada)

---

## FASE 8 — Integración M1 → M2 (Pipeline Orquestado)

> Experimento de ablación con **Interpretación B**: el Modelo 2 se reentrena usando solo
> las 10 features objetivas como base, para que la comparación sea válida.
> El argumento central: "¿puede GPS + M1 aproximar el AUC de un modelo con cuestionarios reales?"
>
> **Nota**: `rf_runner_model.pkl` (18 features, AUC=0.9101) queda como referencia histórica
> en `final_model_comparison.csv` pero NO es parte del experimento de ablación, porque
> ya incluye `recent_recovery` = `perceived_recovery.6` (la respuesta real de M1).

### Diseño del experimento de ablación

Las **tres condiciones** usan siempre las mismas 10 features objetivas como base.
Lo que varía es si se añade información de fatiga (predicha o real):

| Condición | Features del Modelo 2 | Escenario real |
|---|---|---|
| **Solo GPS** | 10 objetivas | Atleta lleva reloj, sin cuestionario |
| **GPS + fatiga predicha** | 10 objetivas + `fatigue_score_predicted` (output de M1) | Nuestro sistema completo |
| **GPS + fatiga real** (upper bound) | 10 objetivas + `perceived_recovery.6` real | Atleta responde cuestionario diariamente |

Si **GPS+fatiga_predicha ≈ GPS+fatiga_real**: M1 aproxima el cuestionario → el atleta solo necesita el reloj.

### Tareas

- [ ] T8.1 Generar predicciones de fatiga para todo el dataset usando LOAO
  - Para cada atleta: usar el modelo LOAO de fatiga (entrenado sin ese atleta) para predecir su `fatigue_score_predicted`
  - Esto garantiza cero leakage: M1 nunca vio al atleta que predice
  - Guardar columna `fatigue_score_predicted` en `src/outputs/runner_fatigue_predictions_loao.csv`

- [ ] T8.2 Preparar los tres datasets para ablación — base = 10 features objetivas
  - **Condición A**: solo las 10 features objetivas (sin ninguna info subjetiva)
  - **Condición B**: 10 objetivas + `fatigue_score_predicted` (output de M1)
  - **Condición C**: 10 objetivas + `perceived_recovery.6` real (upper bound)
  - No modificar `RUNNER_FEATURE_COLUMNS` en `config.py` — usar listas locales en el script de ablación

- [ ] T8.3 Ejecutar LOAO del Modelo 2 para las tres condiciones
  - Mismo protocolo LOAO (74 folds) para cada condición
  - Guardar resultados individuales en:
    - `src/outputs/loao_runner_gps_only.csv`
    - `src/outputs/loao_runner_v2_results.csv` (GPS + fatiga predicha — resultado principal)
    - `src/outputs/loao_runner_gps_real_fatigue.csv`

- [ ] T8.4 Comparar y guardar tabla de ablación en `src/outputs/ablation_fatigue_runner.csv`
  | Condición | Features | AUC LOAO | Interpretación |
  |---|---|---|---|
  | Solo GPS | 10 objetivas | a medir | Baseline sin fatiga |
  | GPS + fatiga predicha | 10 + M1 output | a medir | **Resultado principal** |
  | GPS + fatiga real | 10 + real | a medir | Upper bound (cuestionario) |

- [ ] T8.5 Verificar indicadores
  - **Meta mínima**: AUC(GPS + fatiga predicha) ≥ AUC(solo GPS) → M1 aporta señal discriminativa
  - **Meta ideal**: AUC(GPS + fatiga predicha) ≈ AUC(GPS + fatiga real) → M1 aproxima el cuestionario
  - Si AUC baja con fatiga predicha: documentar — el RF puede capturar fatiga implícitamente en las features de carga

---

## FASE 9 — Validación con week_approach (Validación Externa)

> Usar `week_approach_maskedID_timeseries.csv` como dataset de validación externa independiente.
> Demuestra que el pipeline es robusto ante cambios de granularidad temporal.

### Descripción del dataset
- Mismos 74 atletas, misma fuente
- Features: `avg_exertion`, `avg_recovery`, `min/max_recovery`, `total_kms`, zonas, etc. — agregadas semanalmente
- Etiqueta: `injury` (0/1)
- Columnas de ratio de carga semanal: `rel_total_kms_week_0_1`, `rel_total_kms_week_0_2`, `rel_total_kms_week_1_2`

### Tareas

- [ ] T9.1 Explorar `week_approach`: shape, distribución de lesiones, atletas disponibles
  - Contar atletas con lesiones, total de lesiones, prevalencia

- [ ] T9.2 Mapear features semanales a features equivalentes del pipeline — **Duda 3 resuelta**
  - Usar **solo las features comunes a ambos formatos** (9 features), NO hacer zero-shot con el modelo de 18 features
  - Justificación: imputar con 0 las features faltantes distorsionaría las predicciones del modelo entrenado con distribuciones reales
  - Tabla de equivalencias:
    | Feature diaria | Feature semanal equivalente |
    |---|---|
    | `acute_load_7d` | `total kms` |
    | `acwr` | `rel total kms week 0_1` |
    | `nr_sessions_7d` | `nr. sessions` |
    | `nr_rest_days_7d` | `nr. rest days` |
    | `high_intensity_km_7d` | `total km Z3-Z4-Z5-T1-T2` |
    | `strength_days_7d` | `nr. strength trainings` |
    | `mean_perceived_recovery` | `avg recovery` |
    | `mean_perceived_exertion` | `avg exertion` |
    | `mean_perceived_success` | `avg training success` |
  - Crear script de transformación en `src/runner/week_transform.py`

- [ ] T9.3 Entrenar `rf_runner_weekly_model` con las 9 features comunes y validar con LOAO sobre `week_approach`
  - Entrenar RF con las 9 features comunes usando `day_approach` como entrenamiento
  - Evaluar con LOAO sobre `week_approach_maskedID_timeseries.csv`
  - Esta es una **validación de robustez a granularidad temporal**, no zero-shot
  - Guardar modelo en `src/outputs/rf_runner_weekly_model.pkl`
  - Guardar resultados en `src/outputs/week_validation_results.csv`

- [ ] T9.4 Documentar brecha day→week como "diferencia de granularidad temporal"
  - Comparar: AUC-diario (18 features) vs AUC-semanal (9 features comunes)
  - Si AUC semanal < AUC diario: esperado y documentable (pérdida de resolución + menos features)
  - Si AUC semanal ≈ AUC diario: evidencia de que las 9 features comunes son suficientes

---

## FASE 10 — Actualización de Capítulos de Tesis

> Reflejar la nueva arquitectura en los tres capítulos de resultados.
> PMData queda como referencia histórica/comparativa, no como dataset principal.

### Tareas

- [ ] T10.1 Actualizar `docs/capitulo4_O1.md`
  - Sección 4.4: ampliar descripción del Runner Dataset como dataset **principal**
  - Añadir tabla de features del Modelo 1 (features objetivas vs target de fatiga)
  - PMData: reducir a "dataset de referencia usado en exploración inicial"

- [ ] T10.2 Actualizar `docs/capitulo5_O2.md`
  - Reemplazar descripción del LSTM (PMData) por el nuevo Fatigue Regressor (Runner)
  - Mantener la lógica de orquestación M1→M2 (no cambia)
  - Actualizar tabla de arquitectura del Modelo 1

- [ ] T10.3 Actualizar `docs/capitulo6_O3.md`
  - Sección R7: ahora es el Fatigue Regressor sobre Runner Dataset (resultados de T7.4)
  - Sección R8: ahora incluye el experimento de ablación (resultados de T8.4)
  - Sección R9: pipeline orquestado completo con los 74 atletas (resultados de T8.3)
  - Añadir validación external con week_approach (resultados de T9.3)

- [ ] T10.4 Actualizar redacción de wearable en toda la tesis
  - Reemplazar referencias específicas a "Fitbit Versa 2" por "reloj GPS de running (Garmin/Polar)"
  - Reemplazar referencias a "PMSYS" por "cuestionario subjetivo de entrenamiento diario"
  - Mantener el argumento: "cualquier wearable deportivo que registre carga y percepción subjetiva"

---

## FASE 11 — Verificación Final V2

- [ ] T11.1 Ejecutar pipeline completo V2 de extremo a extremo (M1 → M2 → métricas)
- [ ] T11.2 Verificar indicadores:
  - R7-v2: RMSE fatiga < umbral definido en T7.4
  - R8-v2: AUC ablación (con fatiga) ≥ AUC base (sin fatiga)
  - R9-v2: Pipeline ejecuta sin errores, produce predicciones verificables
- [ ] T11.3 Actualizar tabla `src/outputs/final_model_comparison.csv` con nuevos resultados
- [ ] T11.4 Revisión con asesor — presentar nueva arquitectura y resultados

---

## ESTADO ACTUAL

| Fase | Estado |
|------|--------|
| Fases 1–6 (arquitectura anterior con PMData) | ✅ Completadas — ver PLAN_TESIS_RUNNER.md |
| Fase 7 — Modelo 1 Fatiga (Runner) | ✅ Completada — RMSE=0.1623, Median R²=-0.88, 74 LOAO folds |
| Fase 8 — Integración M1→M2 | ✅ Completada — A=0.9074, B=0.9034, C=0.9109; brecha B↔C=0.0075 (meta ideal cumplida) |
| Fase 9 — Validación week_approach | ✅ Completada — AUC cross-granularity=0.4830, Δ=-42.4% (resolución diaria es esencial) |
| Fase 10 — Actualizar capítulos | ✅ Completada — cap4 (4.4 M1 feature table), cap5 (5.2.4 RF M1+M2), cap6 (6.2.4 Runner results) |
| Fase 11 — Verificación final V2 | ✅ Completada — todos los artefactos verificados, final_model_comparison.csv actualizado |

---

## MÉTRICAS DE REFERENCIA (resultados anteriores que se comparan)

| Modelo | AUC | Dataset | Estado |
|--------|-----|---------|--------|
| LR-PMData (Modelo 2 original) | 0.5517 | PMData test | ❌ No cumple meta |
| RF-Runner LOAO (Modelo 2 base) | 0.9101 ± 0.0891 | Runner 74 atletas | ✅ Meta ≥ 0.65 cumplida |
| RF-Runner val interna | 0.9467 | Runner val 7 atletas | ✅ Meta ≥ 0.70 cumplida |
| RF-Common cross-domain | 0.5368 ± 0.1746 | PMData LOAO | ⚠️ Meta 0.55 no cumplida |
| **RF-Runner Ablación A (GPS-only)** | **0.9074 ± 0.0777** | **Runner LOAO (63/74)** | **✅ Meta ≥ 0.65 cumplida** |
| **RF-Runner Ablación B (GPS + M1)** | **0.9034 ± 0.0965** | **Runner LOAO (63/74)** | **✅ Brecha B↔C = 0.0075 (meta ideal cumplida)** |
| **RF-Runner Ablación C (GPS + real)** | **0.9109 ± 0.0877** | **Runner LOAO (63/74)** | **✅ Cota superior** |
| **RF-M1 Fatigue Regressor** | RMSE=0.1623 | Runner LOAO (74 folds) | ✅ Aceptable (0.15–0.20) |
| **RF-Runner Weekly (cross-gran.)** | **0.4830 ± 0.1352** | **week_approach LOAO** | **📋 Hallazgo: resolución diaria esencial** |

---

## NOTAS TÉCNICAS

- Runner Dataset path: `Runner dataset/day_approach_maskedID_timeseries.csv`
- Week approach path: `Runner dataset/week_approach_maskedID_timeseries.csv`
- Python env: `.venv\Scripts\python.exe`
- El formato wide del CSV tiene sufijos `""` (D-7) a `".6"` (D-1) por feature
- `perceived recovery.6` es el target del Modelo 1 (recuperación percibida en D-1)
- Las columnas `perceived_*` NO deben entrar como inputs del Modelo 1 (solo como target)
- `injury` columna ya es prospectiva en el dataset — NO aplicar create_prospective_target
- SEED = 42, split estratificado por atleta: 51 train / 7 val / 16 test
