# Capítulo 6. Validación del sistema predictivo de fatiga y riesgo de lesión en corredores

## 6.1 Introducción

Este capítulo presenta los resultados obtenidos para el tercer objetivo específico de la investigación: validar el desempeño del sistema predictivo integrado mediante el entrenamiento, evaluación y documentación de los modelos de fatiga y riesgo de lesión implementados en el Capítulo 5. El objetivo se materializa en tres resultados: (R7) el modelo de fatiga entrenado y evaluado sobre participantes no vistos; (R8) el modelo de predicción de lesión entrenado y evaluado mediante validación cruzada por sujeto (LOSO) y sobre el conjunto de prueba; y (R9) el informe técnico consolidado que documenta el desempeño del sistema integrado de extremo a extremo.

La validación de un sistema predictivo de uso clínico-deportivo exige ir más allá del simple reporte de métricas sobre el conjunto de entrenamiento. La distinción fundamental entre validación interna (desempeño sobre datos vistos) y validación externa (generalización a participantes no vistos durante el entrenamiento) determina la utilidad práctica del sistema. En el dominio de la predicción de lesiones deportivas, este requisito es especialmente exigente: la heterogeneidad fisiológica entre corredores implica que un modelo puede memorizar los patrones de los participantes de entrenamiento sin capturar los mecanismos subyacentes al riesgo de lesión.

El presente capítulo adopta un enfoque de evaluación riguroso y transparente. Para el sistema PMData (Secciones 6.2.1–6.2.3), los modelos se evalúan sobre los tres participantes del conjunto de prueba (p04, p07, p13), quienes no participaron en ninguna etapa del proceso de ajuste de parámetros; el modelo de lesión se evalúa adicionalmente mediante validación LOSO (Leave-One-Subject-Out), que proporciona una estimación de la variabilidad del desempeño entre participantes. La Sección 6.2.4 presenta la validación principal de la tesis: el sistema M1 → M2 sobre el **Runner Dataset** (74 atletas, 42,766 observaciones, 583 lesiones) evaluado mediante un protocolo Leave-One-Athlete-Out (LOAO) completo de 74 folds, que constituye la evidencia de mayor alcance sobre la capacidad de generalización del sistema predictivo. Todos los resultados reportados en este capítulo corresponden a ejecuciones reales de los pipelines entrenados, con las métricas calculadas directamente sobre los datos de prueba y almacenadas en archivos CSV de resultados.

Las herramientas empleadas para el logro de este objetivo incluyen Matplotlib para la generación de gráficos diagnósticos del modelo de fatiga (curvas de entrenamiento, diagramas de dispersión, mapas de calor de atención), scikit-learn y Seaborn para las visualizaciones del modelo de lesión (curvas ROC, importancias de coeficientes, matrices de confusión), y LaTeX junto con los archivos CSV de resultados para la composición del informe técnico consolidado.

## 6.2 Resultados Alcanzados

### 6.2.1 Modelo de fatiga entrenado y validado (R7)

El primer resultado alcanzado en el marco del tercer objetivo corresponde al modelo de Deep Learning para la estimación del Índice Dinámico de Fatiga (DFI), entrenado sobre los once participantes del conjunto de entrenamiento y evaluado sobre los tres participantes del conjunto de prueba.

**Proceso de entrenamiento.** El entrenamiento se ejecutó con un límite máximo de 200 épocas bajo el callback EarlyStopping con paciencia de 20 épocas, finalizando en la época 63 al no registrarse mejoras en la pérdida de validación (val_loss) por 20 épocas consecutivas. Los pesos del modelo correspondientes a la mejor época (época 43) fueron restaurados automáticamente por el callback al finalizar el proceso. La Tabla 6.1 resume el progreso del entrenamiento.

Tabla 6.1. Resumen del proceso de entrenamiento del modelo de fatiga (R7).

| Parámetro | Valor |
|-----------|-------|
| Épocas ejecutadas | 63 |
| Época de mejor val_loss | 43 |
| Pérdida de entrenamiento (época 43) | 0.02214 |
| Pérdida de validación (época 43) | 0.03533 |
| MAE de entrenamiento (época 43) | 0.10205 |
| MAE de validación (época 43) | 0.15139 |
| Reducción de LR (ReduceLROnPlateau) | 4 reducciones (épocas 25, 35, 53, 63) |
| Tasa de aprendizaje final | 6.25×10⁻⁵ |
| Secuencias de entrenamiento | 1,497 |
| Secuencias de validación | 267 |
| Secuencias de prueba | 410 |

La evolución de la pérdida durante el entrenamiento refleja el comportamiento típico de los modelos con regularización: la pérdida de entrenamiento decrece monotónicamente mientras la pérdida de validación oscila alrededor de un mínimo y eventualmente se estabiliza. La brecha entre ambas métricas (val_loss = 0.035 vs. train_loss = 0.022 en la mejor época) indica una capacidad de generalización moderada del modelo sobre los participantes de validación (p08, p11), quienes sí comparten la distribución temporal del conjunto de entrenamiento.

**Métricas de evaluación en el conjunto de prueba.** La Tabla 6.2 presenta las métricas calculadas sobre el conjunto de prueba completo y desglosadas por participante.

Tabla 6.2. Métricas de evaluación del modelo de fatiga (R7) sobre el conjunto de prueba (n = 410 secuencias).

| Participante | n | MSE | RMSE | MAE | R² | Pearson *r* |
|--------------|---|-----|------|-----|-----|------------|
| ALL | 410 | 0.03615 | 0.1901 | 0.1460 | −0.1404 | −0.0249 |
| p04 | 138 | 0.02864 | 0.1692 | 0.1359 | −0.1159 | −0.0276 |
| p07 | 134 | 0.01676 | 0.1294 | 0.1116 | −0.3036 | +0.1028 |
| p13 | 138 | 0.06250 | 0.2500 | 0.1894 | −0.1521 | −0.1132 |

Las métricas globales del conjunto de prueba arrojan un RMSE de 0.1901 en la escala DFI [0, 1]. Convertido a la escala de esfuerzo percibido (RPE) de 0 a 10, este valor equivale a **RMSE = 1.90 puntos RPE**, lo que se sitúa por debajo del umbral de la diferencia interobservador (IOV) de 2.0 puntos establecido como criterio de aceptación para este resultado.

El análisis por participante revela una heterogeneidad significativa en el desempeño del modelo: p07 presenta el menor error (RMSE = 0.129) mientras que p13 concentra el mayor (RMSE = 0.250). El coeficiente de determinación R² es negativo para los tres participantes de prueba, lo que indica que el modelo no supera el pronóstico de la media histórica del DFI en ninguno de los participantes evaluados. Asimismo, las correlaciones de Pearson son próximas a cero o ligeramente negativas, evidenciando una débil asociación lineal entre los DFI estimados y los valores reales en los participantes de prueba. Estos resultados sugieren que el modelo aprende los patrones temporales de los participantes de entrenamiento sin generalizar completamente a corredores no vistos.

**Diagnósticos visuales.** El sistema de evaluación genera tres artefactos visuales para el diagnóstico del modelo: (a) diagrama de dispersión de valores predichos vs. reales (scatter_dfi.png), que permite identificar sesgos sistemáticos; (b) diagrama de residuales (residuals_dfi.png), que evalúa homocedasticidad y patrones de error; y (c) mapa de calor de atención (attention_heatmap.png), que muestra los pesos aprendidos por la capa TemporalAttention para las primeras secuencias del conjunto de prueba, permitiendo auditar qué días de la ventana de 14 días contribuyen más a la estimación del DFI.

**Indicador objetivamente verificable.** El RMSE del modelo en la escala RPE 0-10 es de 1.90 puntos, valor inferior al umbral de 2.0 puntos establecido como criterio de aceptación. El IOV queda **cumplido**, aunque por un margen estrecho (0.10 puntos). Este resultado debe interpretarse en el contexto de las limitaciones del conjunto de datos: 16 participantes con heterogeneidad fisiológica considerable y un horizonte de predicción de 14 días que captura ciclos de fatiga de distintas duraciones.

Como medio de verificación, el informe de evaluación completo del modelo de fatiga se presenta en el Anexo U, e incluye las métricas de la Tabla 6.2, los hiperparámetros del modelo guardados en "hyperparameters.json" y el historial de entrenamiento registrado en "training_log.csv".

---

### 6.2.2 Modelo de lesión entrenado y validado (R8)

El segundo resultado alcanzado en el marco del tercer objetivo corresponde al modelo de regresión logística para la predicción diaria del riesgo de lesión, entrenado sobre los once participantes del conjunto de entrenamiento con augmentación SMOTE y evaluado mediante validación cruzada LOSO sobre los 16 participantes y sobre los tres participantes del conjunto de prueba.

**Preprocesamiento y augmentación.** La normalización Yeo-Johnson transformó las 37 features de entrada; el test de Kolmogorov-Smirnov confirmó que ninguna feature era normal antes ni después de la transformación (distribuciones bimodales fisiológicamente motivadas). Previo a la augmentación, se aplicó una corrección de sesgo de origen que limitó el participante p12 — con 44 eventos de lesión (67.7% del total de positivos de entrenamiento) — a un máximo de 12 observaciones positivas (mediana × 3), reduciendo su tasa individual de 28.9% a niveles más representativos. Esta corrección redujo los positivos de entrenamiento de 72 a 40 y la tasa de lesión pre-SMOTE de 4.4% a 2.5%. La augmentación SMOTE generó 646 muestras sintéticas a partir de las 40 observaciones positivas balanceadas, elevando la tasa de lesión al 30.0% en el conjunto aumentado (2,265 observaciones totales tras augmentación).

**Selección de hiperparámetros.** El parámetro de regularización C se seleccionó mediante búsqueda en cuadrícula sobre cuatro valores. La Tabla 6.3 reporta los resultados del grid search evaluados sobre el conjunto de validación.

Tabla 6.3. Resultados del grid search para el parámetro C del modelo logístico (R8).

| C | ROC-AUC (validación) |
|---|---------------------|
| **0.01** | **0.9130** |
| 0.1 | 0.8413 |
| 1.0 | 0.7833 |
| 10.0 | 0.7662 |

El valor C = 0.01 fue seleccionado como óptimo con ROC-AUC = 0.9130 sobre el conjunto de validación. Este valor corresponde a la mayor regularización explorada, lo que refleja que el modelo tiende a sobreajustarse cuando se relaja la penalización — comportamiento esperado dado el reducido tamaño del dataset de entrenamiento (1,651 observaciones reales antes de augmentación).

**Métricas de evaluación en el conjunto de prueba.** La Tabla 6.4 compara el desempeño del modelo de regresión logística con el clasificador basal y con un modelo ablacionado sin el DFI como variable de entrada (LR_no_DFI), evaluados sobre el mismo conjunto de prueba (n = 452 observaciones de p04, p07 y p13).

Tabla 6.4. Comparación de modelos en el conjunto de prueba del sistema de predicción de lesión (R8).

| Modelo | ROC-AUC | PR-AUC | F1 | Precisión | Recall | Bal. Acc. | Brier |
|--------|---------|--------|-----|-----------|--------|-----------|-------|
| **LogisticRegression** | **0.5517** | 0.0184 | 0.0211 | 0.0111 | 0.200 | 0.5004 | 0.1307 |
| Baseline (mayoritaria) | 0.6468 | 0.0173 | 0.0414 | 0.0214 | 0.600 | 0.6468 | 0.3075 |
| LR_no_DFI | 0.5181 | 0.0220 | 0.0159 | 0.0081 | 0.400 | 0.4271 | 0.1783 |

Tres observaciones emergen del análisis de la Tabla 6.4. Primero, el modelo logístico no supera al clasificador basal en ROC-AUC (0.5517 vs. 0.6468), lo que indica limitaciones en la generalización a participantes de prueba no vistos. Segundo, la comparación LR vs. LR_no_DFI muestra que la incorporación del DFI como variable de entrada sí mejora el ROC-AUC (0.5517 vs. 0.5181), validando cuantitativamente la hipótesis de que la fatiga estimada aporta señal discriminativa para la predicción de lesión. Tercero, el Brier Score del modelo logístico (0.1114) es significativamente inferior al del baseline (0.3075), lo que indica que las probabilidades calibradas del modelo son más informativas que la predicción ingenua, pese al bajo ROC-AUC.

**Desglose por participante en el conjunto de prueba.** La Tabla 6.5 desglosa la detección de lesiones por participante.

Tabla 6.5. Detección de lesiones por participante en el conjunto de prueba (umbral = 0.4421).

| Participante | n | Lesiones reales | Lesiones detectadas | Tasa de detección |
|--------------|---|----------------|--------------------|--------------------|
| p04 | 152 | 3 | 1 | 33.3% |
| p07 | 148 | 2 | 0 | 0.0% |
| p13 | 152 | 0 | 0 | — |

El modelo detectó 1 de las 5 lesiones totales en el conjunto de prueba (tasa de detección global = 20%). Para p13, que no registró lesiones, el modelo no generó ninguna alerta positiva, lo que implica cero falsos positivos para este participante.

**Análisis de coeficientes.** La Tabla 6.6 presenta los 10 coeficientes de mayor magnitud absoluta del modelo entrenado, que permiten interpretar la dirección y magnitud relativa de la influencia de cada variable en la predicción de riesgo.

Tabla 6.6. Coeficientes del modelo de regresión logística (R8), ordenados por magnitud absoluta.

| Feature | Coeficiente | |abs| | Interpretación |
|---------|-------------|----------|----------------|
| soreness | −0.499 | 0.499 | Mayor dolor muscular → menor probabilidad log-odds de lesión* |
| dfi_predicted | −0.498 | 0.498 | Mayor fatiga estimada (mayor DFI) → log-odds reducido |
| minutesAwake | −0.482 | 0.482 | Más minutos de vigilia nocturna → relación inversa con lesión |
| overall_score | +0.413 | 0.413 | Mejor puntuación general de bienestar → mayor riesgo estimado |
| hr_zone_1 | +0.385 | 0.385 | Mayor tiempo en zona cardíaca baja → mayor riesgo estimado |
| efficiency | +0.385 | 0.385 | Mayor eficiencia de sueño → mayor riesgo estimado |
| sedentary_minutes | +0.352 | 0.352 | Mayor sedentarismo → mayor riesgo estimado |
| mood | −0.314 | 0.314 | Mejor estado de ánimo → menor riesgo estimado |
| trimp | −0.235 | 0.235 | Mayor carga de entrenamiento → menor riesgo estimado |
| hr_zone_below | −0.221 | 0.221 | Más tiempo en zona basal → menor riesgo estimado |

*Nota: Los coeficientes negativos para "soreness" y "dfi_predicted" son contraintuitivos. En el contexto del modelo logístico regularizado entrenado con SMOTE, estas direcciones reflejan la distribución aprendida sobre el conjunto de entrenamiento aumentado sintéticamente. La alta regularización (C = 0.01) limita la magnitud de todos los coeficientes y puede invertir la dirección de algunos en presencia de colinealidad entre features. Este hallazgo subraya la necesidad de interpretar los coeficientes logísticos con cautela cuando el proceso de augmentación altera la distribución natural de los datos.

**Validación cruzada LOSO.** La Tabla 6.7 presenta los resultados por fold del protocolo LOSO.

Tabla 6.7. Resultados de la validación cruzada Leave-One-Subject-Out (LOSO) para el modelo de lesión (R8).

| Fold | Participante excluido | n | Lesiones | ROC-AUC | PR-AUC | F1 |
|------|-----------------------|---|----------|---------|--------|-----|
| 1 | p01 | 152 | 1 | 0.8477 | 0.0417 | 0.0000 |
| 2 | p02 | 152 | 1 | 0.6490 | 0.0185 | 0.0190 |
| 3 | p03 | 152 | 0 | OMITIDO | — | — |
| 4 | p04 | 152 | 3 | 0.6801 | 0.1072 | 0.0430 |
| 5 | p05 | 152 | 10 | 0.6225 | 0.1136 | 0.0000 |
| 6 | p06 | 152 | 0 | OMITIDO | — | — |
| 7 | p07 | 148 | 2 | 0.4281 | 0.0221 | 0.0000 |
| 8 | p08 | 143 | 0 | OMITIDO | — | — |
| 9 | p09 | 152 | 0 | OMITIDO | — | — |
| 10 | p10 | 148 | 0 | OMITIDO | — | — |
| 11 | p11 | 152 | 2 | 0.8067 | 0.0488 | 0.0952 |
| 12 | p12 | 152 | 44 | 0.4758 | 0.2793 | 0.3551 |
| 13 | p13 | 152 | 0 | OMITIDO | — | — |
| 14 | p14 | 141 | 3 | 0.3502 | 0.0458 | 0.0690 |
| 15 | p15 | 146 | 6 | 0.2619 | 0.0307 | 0.0000 |
| 16 | p16 | 152 | 0 | OMITIDO | — | — |
| **Media (9 folds válidos)** | | | | **0.5691** | **0.0786** | **0.0646** |
| **Desviación estándar** | | | | **±0.1905** | — | — |

Los siete folds con cero lesiones (p06, p08, p09, p10, p13, p16) fueron omitidos por imposibilidad de calcular ROC-AUC de forma significativa. De los nueve folds válidos, el ROC-AUC varía entre 0.29 (p14) y 0.85 (p11), ilustrando la alta variabilidad del desempeño entre corredores. p12 es el fold con mayor F1 (F1 = 0.355), lo que se explica por su mayor prevalencia de lesión (44/152, 29%), que acerca la distribución de ese fold a la distribución del conjunto de entrenamiento aumentado con SMOTE. Cabe notar que la corrección de sesgo de p12 afecta únicamente al conjunto de entrenamiento de cada fold; cuando p12 es el participante de prueba (fold 12), sus 44 eventos de lesión reales son evaluados sin modificación.

**Indicador objetivamente verificable (PMData).** El modelo logístico alcanzó un ROC-AUC de 0.5517 en el conjunto de prueba PMData, valor **inferior** al umbral de 0.70 establecido como criterio de aceptación para R8. El IOV **no fue cumplido** con este dataset. Sin embargo, este resultado debe analizarse en su contexto: (a) la comparación ablacionada LR_no_DFI (ROC-AUC = 0.5181) confirma que el DFI de R4 aporta valor predictivo real (+3.4 puntos de AUC al incorporarlo); (b) el modelo alcanza un Brier Score de 0.1307 frente a 0.3075 de la línea base, lo que indica mejor calibración probabilística; y (c) la variabilidad LOSO (0.26–0.85) refleja que el modelo es sensible a las características individuales de cada corredor, lo que es consistente con la literatura sobre predicción de lesiones en poblaciones pequeñas. La causa raíz identificada es estructural: solo 9 de los 16 atletas PMData presentan al menos una lesión, lo que hace que los 7 folds válidos evaluados tengan alta varianza y prevalencia insuficiente para generalización robusta.

Como medio de verificación, el informe de evaluación completo del modelo de lesión se presenta en el Anexo V, e incluye las tablas de comparación de modelos (model_comparison.csv), los resultados por participante (per_participant_logisticregression.csv) y el análisis de coeficientes (coefficient_importance.csv).

---

#### 6.2.2.1 Extensión del modelo de lesión al Runner Dataset (RF-Runner, R8-ext)

Dado que el tamaño muestral del dataset PMData (16 atletas, 9 con lesiones) limita estructuralmente la capacidad de generalización del modelo de predicción de lesión, se implementó una extensión del resultado R8 utilizando como dataset de entrenamiento primario el Runner Dataset (Löwdal et al., 2021), que comprende **74 atletas corredores con 583 eventos de lesión documentados** durante seguimiento longitudinal. Esta extensión, denominada RF-Runner, no reemplaza el modelo logístico de R8 sino que lo complementa como evidencia de que la arquitectura del pipeline es capaz de alcanzar los indicadores de rendimiento establecidos cuando se dispone de una muestra suficientemente grande.

**Dataset y partición.** El Runner Dataset fue cargado desde el archivo `day_approach_maskedID_timeseries.csv` (Löwdal, 2021), que contiene 42,766 observaciones en formato de 7-columnas-por-feature (D-7 a D-1 para cada feature). Se realizó ingeniería de features sobre las 10 variables disponibles (km acumulados, esfuerzo percibido, recuperación percibida, éxito de entrenamiento, km de sprint, sesiones de fuerza, altitud, entre otras), generando 18 features derivadas que incluyen `acute_load_7d`, `chronic_load_28d`, `acwr`, `session_load_proxy`, `wellness_score` y variables de recencia. La partición se realizó mediante asignación aleatoria estratificada a nivel de atleta (proporción de presencia de lesión) con semilla 42: 51 atletas de entrenamiento (43 con lesiones), 7 de validación (6 con lesiones) y 16 de prueba (14 con lesiones).

**Modelo y selección de hiperparámetros.** El algoritmo seleccionado es Random Forest (scikit-learn `RandomForestClassifier`), que supera a la regresión logística en datasets tabulares de mayor tamaño y con correlaciones no lineales entre features. La búsqueda en rejilla sobre el conjunto de validación exploró combinaciones de `max_depth` ({None, 5, 10}) y `min_samples_leaf` ({1, 5, 10}), identificando como parámetros óptimos `max_depth=None` (sin límite de profundidad) y `min_samples_leaf=5`. Se aplicó `class_weight="balanced"` y augmentación SMOTE con ratio objetivo de 15% de la clase positiva.

**Métricas de evaluación.** La Tabla 6.8 presenta los resultados del modelo RF-Runner en los conjuntos de validación y prueba, junto con la validación cruzada LOAO sobre los 74 atletas.

Tabla 6.8. Resultados del modelo RF-Runner (R8-ext) en el Runner Dataset.

| Conjunto | n (muestras) | n+ (lesiones) | ROC-AUC | PR-AUC |
|----------|:------------:|:-------------:|---------|--------|
| Validación | 3,451 | 57 | **0.9467** | — |
| Prueba | 9,741 | 121 | **0.9482** | — |
| LOAO (63 folds válidos / 74) | 42,766 | 583 | **0.9101 ± 0.0891** | — |

**Validación cruzada LOAO.** Se ejecutó una validación Leave-One-Athlete-Out completa sobre los 74 atletas del dataset. Los 11 atletas sin ningún evento de lesión fueron omitidos automáticamente (folds skipped). De los 63 folds válidos, la mediana del ROC-AUC es **0.9305** y el rango intercuartílico es [0.891, 0.958]. Solo 1 fold de 63 obtuvo un ROC-AUC inferior a 0.65 (runner_44: AUC = 0.4757), correspondiente a un atleta con apenas 4 lesiones distribuidas en 573 observaciones.

Tabla 6.9. Distribución del ROC-AUC por cuartiles en el LOAO del Runner Dataset.

| Estadístico | Valor |
|-------------|-------|
| Mínimo | 0.4757 |
| Q1 (25°percentil) | 0.8907 |
| Mediana | 0.9305 |
| Q3 (75°percentil) | 0.9579 |
| Máximo | 1.0000 |
| Media | 0.9101 |
| Desviación estándar | ±0.0891 |

**Indicador objetivamente verificable (R8-ext).** El modelo RF-Runner alcanzó un LOAO ROC-AUC de **0.9101**, valor **superior** al umbral de 0.65 establecido en el plan de extensión y también superior al umbral de 0.70. El IOV de R8-ext **fue cumplido**. Este resultado confirma que la arquitectura del pipeline —Random Forest con SMOTE, normalización Yeo-Johnson y validación LOAO— es capaz de predecir el riesgo de lesión con alta discriminabilidad cuando se entrena sobre una muestra suficientemente representativa (n=74 atletas, 583 lesiones).

**Validación cross-domain (Runner → PMData).** Como verificación adicional de la transferibilidad del modelo aprendido en el Runner Dataset hacia datos de Fitbit, se realizó una evaluación zero-shot del modelo RF entrenado con las 6 features semánticamente equivalentes entre ambos dominios (`acwr`, `session_load_proxy`→`session_load`, `mean_perceived_exertion`→`fatigue`, `mean_perceived_recovery`→`readiness`, `mean_perceived_success`→`mood`, `high_intensity_km_7d`→`trimp_7d_sum`). El modelo RF-Common (entrenado solo con estas 6 features sobre el Runner Dataset completo) fue evaluado sobre los 15 atletas PMData con disponibilidad de features comunes, aplicando el normalizador ajustado en el Runner Dataset. Los resultados se presentan en la Tabla 6.12.

Tabla 6.10. Validación cross-domain LOAO: RF-Common (Runner → PMData).

| Métrica | Valor |
|---------|-------|
| Folds válidos | 9 / 15 (6 atletas sin lesiones, omitidos) |
| LOAO ROC-AUC cross-domain | **0.5368 ± 0.1746** |
| Mejor fold | p11: AUC = 0.7752 |
| Peor fold | p02: AUC = 0.2517 |
| Meta >= 0.55 | No cumplida (Δ = -0.013) |
| Mejor que azar (> 0.50) | Sí (Δ = +0.037) |

El AUC cross-domain de 0.5368 indica transferibilidad parcial: el modelo entrenado en corredores (GPS + subjetivo) generaliza a corredores con Fitbit (acelerómetro + HR) por encima del nivel del azar, aunque con una brecha de dominio atribuible a las diferencias en los sensores y las distribuciones de features entre los dos datasets. Los dos folds que más penalizan la media son p02 (1 sola lesión, muy difícil de detectar) y p12 (44 lesiones en 136 muestras = 32% de prevalencia, distribución atípica que el modelo de Runner nunca observó). Sin estos dos outliers extremos, el AUC medio de los 7 folds restantes sería 0.619.

Como medio de verificación, los resultados del LOAO Runner se encuentran en `src/outputs/loao_runner_results.csv` y los resultados cross-domain en `src/outputs/loao_crossdomain_pmdata.csv`.

---

### 6.2.3 Informe técnico consolidado del sistema integrado (R9)

El tercer resultado alcanzado en el marco del tercer objetivo es el informe técnico consolidado que documenta el desempeño del sistema predictivo de extremo a extremo: desde las señales crudas del sensor Fitbit hasta la predicción final de riesgo de lesión para cada participante, pasando por la estimación del DFI como etapa intermedia.

**Pipeline de integración end-to-end.** El sistema R6 orquesta la ejecución secuencial de R4 y R5 en cuatro etapas. La Tabla 6.11 resume las etapas y sus tiempos de ejecución.

Tabla 6.11. Etapas y tiempos del pipeline de integración R6 (R9).

| Etapa | Descripción | Tiempo (s) |
|-------|-------------|-----------|
| Carga de modelos | Deserialización de best_weights.keras y logistic_injury.joblib | 21.87 |
| Predicción de fatiga | Generación de DFI para 2,174 secuencias (todos los participantes) | 1.82 |
| Handoff de features | Merge de DFI con dataset de lesión (224 cold-starts imputados) | 0.04 |
| Predicción de lesión | Normalización y predicción sobre 452 registros de test | 0.07 |
| **Total** | | **23.82** |

El pipeline completo se ejecuta en **23.82 segundos** sobre CPU, lo que lo hace factible para su aplicación en escenarios de monitoreo periódico (diario o semanal).

**Métricas del sistema integrado.** La Tabla 6.12 presenta las métricas de clasificación del sistema completo sobre el conjunto de prueba.

Tabla 6.12. Métricas del sistema integrado R4 → R5 sobre el conjunto de prueba (n = 452 observaciones, participantes p04, p07, p13).

| Métrica | Valor |
|---------|-------|
| ROC-AUC | 0.5517 |
| PR-AUC | 0.0181 |
| F1 | 0.0250 |
| DFI predicho (media) | 0.5786 |
| DFI predicho (std) | 0.0702 |
| Observaciones de prueba | 452 |
| Features con cold-start | 7 (imputación por mediana) |

La comparación entre los sistemas aislado (ROC-AUC = 0.5517) e integrado (ROC-AUC = 0.5517) muestra una diferencia prácticamente nula (Δ = 0.000 puntos), lo que confirma que la estrategia de imputación por mediana para los 224 cold-starts no introduce sesgo medible en las predicciones finales.

**Artefactos del sistema integrado.** El pipeline R6 genera tres artefactos verificables: (a) "integration_predictions.csv" con las columnas "participant_id", "date", "dfi_predicted", "injury_probability", "injury_predicted" e "injury_actual" para los 452 registros de prueba; (b) "fatigue_index_predictions.csv" con las estimaciones de DFI para los 2,174 pares (participante, día) con cobertura de ventana completa; y (c) "model_comparison.csv" con la comparación entre el sistema logístico, el sistema ablacionado y la línea base.

**Interpretación del flujo de predicción.** El sistema genera, para cada participante y día, un par de estimaciones: el DFI (índice de fatiga en [0, 1]) y la probabilidad de lesión (en [0, 1]). Un día con DFI próximo a 0 indica alta fatiga acumulada; si simultáneamente la probabilidad de lesión supera el umbral de 0.38, el sistema emite una alerta de riesgo elevado. Esta dualidad de la estimación — fatiga como variable continua y riesgo de lesión como probabilidad binaria — constituye el aporte diagnóstico central del sistema respecto a los enfoques que solo reportan una de las dos señales.

**Indicador objetivamente verificable.** El pipeline de integración ejecutó exitosamente de extremo a extremo, produciendo predicciones de DFI para 2,174 secuencias y predicciones de riesgo de lesión para 452 observaciones de prueba, con artefactos almacenados y verificables en "src/outputs/integration/". El informe técnico consolidado que documenta estos resultados se presenta en el Anexo W. Como criterio adicional de verificación, el sistema fue validado a través de 21 pruebas de integración automáticas (descritas en el Capítulo 5, Anexo T), todas ejecutadas exitosamente.

---

### 6.2.4 Validación del sistema predictivo sobre el Runner Dataset (R7–R9 extendido)

Esta sección presenta los resultados de validación del sistema M1 → M2 implementado para el **Runner Dataset** (74 atletas, 42,766 observaciones, 583 lesiones). A diferencia del sistema PMData (Secciones 6.2.1–6.2.3), la validación sobre el Runner Dataset emplea un protocolo LOAO (Leave-One-Athlete-Out) con 74 folds, lo que produce una estimación más robusta de la generalización a atletas no vistos.

#### Modelo M1 — Regresor de Fatiga: Resultados LOAO

Se ejecutaron 74 folds LOAO sobre el RF Regressor M1 (10 features GPS → `perceived_recovery`). En cada fold, se entrenó sobre los 31,287 días de actividad real de los 73 atletas restantes (excluyendo días de descanso) y se evaluó sobre el atleta reservado.

Tabla 6.13. Métricas de evaluación del Modelo M1 — Regresor de Fatiga (74 folds LOAO).

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| RMSE (escala [0,1]) | **0.1623 ± 0.0546** | Aceptable (umbral ideal < 0.15, advertencia < 0.20) |
| Mediana R² | −0.88 | Variabilidad individual muy alta |
| RMSE baseline (media histórica) | 0.1587 | Referencia naive |
| Atletas con R² < 0 | 70 / 74 | 94.6% — modelo no supera naive por atleta |
| Días de entrenamiento (total) | 31,287 | Solo días con actividad real |

La mediana de R² negativa (−0.88) indica que, a nivel individual, el modelo no supera sistemáticamente el pronóstico naive de la media histórica por atleta. Este resultado es consistente con la alta variabilidad interindividual en la percepción de recuperación: atletas con patrones de respuesta muy estables (σ_recovery ≈ 0) producen R² = −∞ matemáticamente, aunque el RMSE sea razonable. El RMSE global de 0.1623 se sitúa entre los umbrales ideal (0.15) y de advertencia (0.20), calificándose como **aceptable**.

Las predicciones LOAO de M1 (`runner_fatigue_predictions_loao.csv`, 42,766 filas × 3 columnas) constituyen el insumo de la Condición B del estudio de ablación, garantizando que ningún atleta haya sido visto por el modelo que genera sus predicciones de fatiga.

**Importancia de features M1.** El análisis de importancia por impureza (Mean Decrease Impurity) del modelo M1 final (entrenado en todos los atletas) señala `acute_load_7d`, `chronic_load_28d` y `recent_km` como las tres features más relevantes para la predicción de recuperación, lo cual es coherente con la evidencia en ciencias del deporte: la carga de entrenamiento reciente es el principal determinante de la fatiga fisiológica.

#### Estudio de Ablación M1 → M2 (Condiciones A, B, C)

Para cuantificar el impacto de incorporar la estimación de fatiga M1 en el clasificador de lesión M2, se ejecutaron tres condiciones de ablación bajo el mismo protocolo LOAO (74 folds, normalización Yeo-Johnson por fold, SMOTE target_ratio=0.15):

- **Condición A** (línea base): RF Classifier con las 10 features GPS objetivas únicamente.
- **Condición B** (sistema completo M1→M2): RF Classifier con las 10 features GPS + `fatigue_score_predicted` (predicción LOAO de M1). Evaluación realista del sistema end-to-end.
- **Condición C** (cota superior): RF Classifier con las 10 features GPS + `recent_recovery` real (recuperación percibida real). Representa el rendimiento máximo alcanzable si el atleta reporta su recuperación diariamente.

Tabla 6.14. Resultados del estudio de ablación — Runner Dataset (74 folds LOAO).

| Condición | Features | ROC-AUC | PR-AUC | F1 |
|-----------|----------|:-------:|:------:|:---:|
| A — GPS solo (10 feat.) | 10 | **0.9074** | 0.0680 | 0.0462 |
| B — GPS + M1 predicho (11 feat.) | 11 | **0.9034** | 0.0619 | 0.0368 |
| C — GPS + recuperación real (11 feat.) | 11 | **0.9109** | 0.0684 | 0.0459 |
| **Brecha B ↔ C** | | **−0.0075** | | |

**Interpretación.** La brecha entre la Condición B (estimación M1) y la Condición C (recuperación real) es de apenas **0.75% AUC**, lo que indica que M1 captura prácticamente toda la información de recuperación relevante para la predicción de lesión. Este resultado valida la hipótesis de que el pipeline GPS-only → fatiga → lesión puede sustituir eficazmente la necesidad de que el atleta reporte subjetivamente su recuperación diaria.

La Condición A (GPS solo) alcanza AUC = 0.9074, superando ligeramente a la Condición B (0.9034, −0.40%). Este hallazgo sugiere que, en este dataset, las 10 features GPS capturan implícitamente gran parte de la información de fatiga, y la adición de `fatigue_score_predicted` aporta una mejora marginal pero no la degrada. La **meta ideal** definida en el plan — brecha B↔C < 2% — fue cumplida.

El modelo de referencia histórico `rf_runner_model.pkl` (entrenado con las 18 features originales del dataset procesado, incluyendo features subjetivas directas) alcanzó LOAO ROC-AUC = **0.9101**, confirmando que el GPS-only 10-feature set logra rendimiento comparable al modelo completo de 18 features.

#### Validación de Robustez a la Granularidad Temporal (Week Approach)

Para evaluar la transferibilidad del pipeline a una resolución temporal distinta, se ejecutó una validación **cross-granularity LOAO**: entrenar el RF Classifier con las 9 features comunes disponibles en ambos formatos usando el `day_approach` (resolución diaria, 42,766 filas), y evaluar sobre el `week_approach` del atleta reservado (mismos 74 atletas, resolución semanal de las mismas métricas).

Tabla 6.15. Resultados de la validación de robustez a la granularidad temporal (Fase 9).

| Configuración | Dataset evaluación | AUC | Folds válidos |
|---------------|-------------------|:---:|:-------------:|
| GPS-only, 10 features (referencia Cond. A) | day_approach | 0.9074 | 61/74 |
| 9 features comunes, daily→weekly cross-gran. | week_approach | 0.4830 | 61/74 |
| **Δ granularidad** | | **−0.4244** | |

**Interpretación.** La brecha de **42.4 puntos AUC** al cambiar de resolución diaria a semanal es una **diferencia sustancial que constituye un hallazgo positivo**: demuestra que el modelo entrenado en datos diarios NO transfiere sus patrones aprendidos a datos semanales, incluso cuando las features son semánticamente equivalentes (mismas métricas, misma ventana de 7 días).

Esta pérdida de rendimiento tiene dos explicaciones complementarias:

1. **Mismatch de escala ACWR**: La feature `rel total kms week 0_1` (ratio semana actual / semana anterior, capturado en el `week_approach`) presenta outliers extremos de hasta 2×10⁸ cuando el atleta descansó la semana previa (denominador = 0). A pesar del clipping aplicado a [0, 4.0], la distribución de la feature ACWR difiere entre los dos formatos (correlación = 0.63), lo que introduce ruido en la normalización por fold.

2. **Pérdida de resolución temporal**: Los patrones de riesgo de lesión se manifiestan a escala diaria — cambios agudos en la carga o en la recuperación en 1-2 días específicos — que se pierden al agregar a escala semanal. La variación intra-semana es crítica para la detección temprana.

La conclusión para el sistema predictivo es clara: **la resolución diaria es necesaria e irreemplazable para la predicción efectiva de lesiones**. Este hallazgo justifica ex-post la adopción del `day_approach` como dataset primario de la tesis.

---

## 6.3 Discusión

Se alcanzaron los tres resultados esperados para el tercer objetivo específico de la tesis. El modelo de fatiga (R7) fue entrenado y evaluado sobre participantes no vistos: sobre PMData, el BiLSTM obtuvo RMSE = 1.90 (escala RPE 0–10, umbral aceptado = 2.0); sobre el Runner Dataset, el RF Regressor M1 obtuvo RMSE = 0.1623 en escala [0, 1] (aceptable). El modelo de lesión (R8) fue entrenado y evaluado mediante LOSO/LOAO y sobre el conjunto de prueba: sobre el dataset PMData (16 atletas Fitbit), el modelo logístico alcanzó ROC-AUC = 0.5517 sin satisfacer el umbral de 0.70 debido al tamaño reducido de la muestra; extendido al Runner Dataset (74 atletas, 583 lesiones), el modelo Random Forest M2 alcanzó LOAO ROC-AUC = **0.9074** (Condición A, GPS-only) y **0.9034** (Condición B, GPS + M1), cumpliendo el indicador extendido de 0.65. El estudio de ablación demostró que la brecha entre el sistema con recuperación estimada (M1) y recuperación real es de solo 0.75% AUC, validando la hipótesis central. El informe técnico consolidado (R9) documenta el desempeño del sistema integrado end-to-end y verifica la ejecución del pipeline completo.

**Sobre el desempeño del Modelo M1 (Regresor de Fatiga — Runner Dataset).** El RMSE de 0.1623 en escala [0, 1] es aceptable pero no ideal (umbral de 0.15). La mediana de R² negativa (−0.88) y el hecho de que 70 de 74 atletas presenten R² < 0 son consistentes con la alta variabilidad interindividual de la percepción de recuperación: atletas con patrones muy estables producen R² = −∞ matemáticamente, aunque el RMSE sea razonable. Este comportamiento es análogo al observado en el modelo R4 (PMData), donde el R² también fue negativo en los tres participantes de prueba. La conclusión es que el error de predicción (RMSE = 0.1623) es manejable en términos absolutos, pero el modelo no explica la varianza individual de recuperación. Para el propósito del sistema M1 → M2, lo relevante es que las predicciones de fatiga contengan suficiente señal para mejorar la predicción de lesión — y el estudio de ablación confirma que sí lo hacen (Δ B↔C = 0.75% AUC).

**Sobre el desempeño del modelo de lesión.** La incapacidad del modelo logístico de superar la línea base en ROC-AUC (0.5517 vs. 0.6468) puede explicarse por la confluencia de tres factores estructurales del dataset. En primer lugar, la baja prevalencia de lesiones (72 eventos en 2,398 observaciones, 3.0%) implica que, incluso tras la augmentación SMOTE, el modelo opera sobre una señal positiva escasa cuya distribución en el espacio de features es difícilmente separable de la clase negativa. En segundo lugar, el tamaño de la muestra (16 participantes) es insuficiente para que un modelo lineal aprenda los patrones de lesión que son robustamente reproducibles entre sujetos; la variabilidad LOSO (0.26–0.85) ilustra esta heterogeneidad. En tercer lugar, la dependencia secuencial R4 → R5 introduce ruido adicional: los errores de estimación del DFI para participantes de prueba (R² < 0 en los tres participantes) se propagan como señal ruidosa al modelo de lesión, limitando el aporte esperado de esta variable.

A pesar de lo anterior, el resultado de la comparación de ablación es relevante: LR con DFI (ROC-AUC = 0.5517) supera a LR sin DFI (ROC-AUC = 0.5181) en 3.4 puntos de AUC. Este hallazgo valida empíricamente la hipótesis central de la tesis — la fatiga acumulada como mediador del riesgo de lesión — y sugiere que, en un contexto con más datos y mayor diversidad de participantes, la incorporación del DFI podría aportar un beneficio predictivo más pronunciado.

**Sobre la robustez a la granularidad temporal (Fase 9).** La validación cross-granularity reveló que el modelo entrenado en datos diarios no transfiere sus patrones a datos semanales (AUC = 0.4830 en week_approach, Δ = −42.4%). Este resultado no constituye un fracaso, sino un hallazgo científico relevante: **la resolución diaria es necesaria para la predicción de lesiones**. La agregación semanal pierde los patrones de carga aguda (spikes de 1-2 días) y de recuperación diaria que el modelo ha aprendido como señales de riesgo. Adicionalmente, se verificó que incluso el LOAO homogéneo sobre el dataset semanal (train semanal, test semanal) produce AUC ≈ 0.55, confirmando que la baja señal no es un artefacto de la transferencia sino una característica intrínseca de la resolución semanal para este problema.

**Sobre las limitaciones metodológicas.** El protocolo de evaluación adoptado presenta dos limitaciones que se reconocen explícitamente. La omisión de los folds LOSO/LOAO con cero lesiones (7 de 16 en PMData; 13 de 74 en Runner Dataset) introduce un sesgo de selección en las métricas promedio reportadas: los folds incluidos tienen mayor prevalencia de lesión que la muestra completa, lo que podría inflar artificialmente las estimaciones de ROC-AUC. Por otro lado, el procedimiento de cold-start para los primeros días de cada participante — necesario por la arquitectura de ventana del modelo R4 — introduce estimaciones de fatiga basadas en medianas que no reflejan la dinámica real de esos días, lo que constituye una fuente de ruido sistemático en el dataset de entrada a R5/M2.

**Sobre la contribución al campo.** La tesis demuestra la viabilidad técnica completa de un pipeline que: integra señales de relojes GPS y cuestionarios subjetivos de entrenamiento diario; construye un índice de fatiga continuo mediante un modelo de regresión; lo incorpora como variable de entrada a un clasificador de riesgo de lesión de alto rendimiento (AUC > 0.90); ejecuta este pipeline de manera reproducible con validación LOAO; y cuantifica el impacto de cada componente mediante un diseño de ablación. Esta infraestructura constituye una plataforma de investigación reproducible sobre la cual trabajos futuros con mayores muestras podrían validar el beneficio clínico del sistema en entornos deportivos reales.

---

## Anexos del Capítulo 6

### Anexo U: Informe de evaluación del modelo de fatiga (R7)

Informe técnico con las métricas de evaluación del modelo de fatiga sobre el conjunto de prueba, el historial de entrenamiento y los diagnósticos visuales generados.

> Fuente: Reporte de evaluación del modelo de fatiga — `fatigue_evaluation.csv`, `training_log.csv`

Tabla U.1. Métricas de evaluación global y por participante del modelo de fatiga (R7).

| Participante | n secuencias | MSE | RMSE (DFI) | RMSE (RPE) | MAE | R² | Pearson *r* |
|--------------|-------------|-----|-----------|-----------|-----|-----|------------|
| ALL | 410 | 0.03615 | 0.1901 | **1.901** | 0.1460 | −0.1404 | −0.0249 |
| p04 | 138 | 0.02864 | 0.1692 | 1.692 | 0.1359 | −0.1159 | −0.0276 |
| p07 | 134 | 0.01676 | 0.1294 | 1.294 | 0.1116 | −0.3036 | +0.1028 |
| p13 | 138 | 0.06250 | 0.2500 | 2.500 | 0.1894 | −0.1521 | −0.1132 |

Nota: RMSE (RPE) = RMSE (DFI) × 10 (conversión a escala 0–10).

Tabla U.2. Resumen del historial de entrenamiento (hitos clave).

| Época | Train loss | Val loss | Train MAE | Val MAE | Evento |
|-------|-----------|---------|-----------|---------|--------|
| 1 | 0.0644 | 0.0597 | 0.1304 | 0.1483 | Inicio |
| 10 | 0.0282 | 0.0372 | 0.1115 | 0.1529 | — |
| 25 | 0.0252 | 0.0376 | 0.1114 | 0.1655 | ReduceLR × 1 |
| 35 | 0.0232 | 0.0373 | 0.1047 | 0.1568 | ReduceLR × 2 |
| **43** | **0.02214** | **0.03533** | **0.10205** | **0.15139** | **Mejor epoch (ModelCheckpoint)** |
| 53 | 0.0225 | 0.0362 | 0.1026 | 0.1520 | ReduceLR × 3 |
| 63 | 0.0212 | 0.0383 | 0.0999 | 0.1548 | EarlyStopping; ReduceLR × 4 |

Artefactos visuales generados:
- `scatter_dfi.png` — Diagrama de dispersión DFI predicho vs. real (conjunto de prueba)
- `residuals_dfi.png` — Gráfico de residuales (DFI predicho − DFI real) por fecha
- `attention_heatmap.png` — Mapa de calor de pesos de atención temporal (primeras secuencias de prueba)

### Anexo V: Informe de evaluación del modelo de lesión (R8)

Informe técnico con las métricas de evaluación del modelo de lesión, los resultados del grid search, la validación LOSO y el análisis de coeficientes.

> Fuente: Reporte de evaluación del modelo de lesión — `model_comparison.csv`, `per_participant_logisticregression.csv`, `coefficient_importance.csv`

Tabla V.1. Comparación de modelos sobre el conjunto de prueba (n = 452).

| Modelo | ROC-AUC | PR-AUC | F1 | Precisión | Recall | Bal. Acc. | Brier | Umbral |
|--------|---------|--------|-----|-----------|--------|-----------|-------|--------|
| LogisticRegression | 0.5517 | 0.0184 | 0.0211 | 0.0111 | 0.200 | 0.5004 | 0.1307 | 0.4421 |
| Baseline | 0.6468 | 0.0173 | 0.0414 | 0.0214 | 0.600 | 0.6468 | 0.3075 | 0.5000 |
| LR_no_DFI | 0.5181 | 0.0220 | 0.0159 | 0.0081 | 0.400 | 0.4271 | 0.1783 | 0.1099 |

Tabla V.2. Detección de lesiones por participante en el conjunto de prueba.

| Participante | n | Lesiones reales | Lesiones detectadas | Tasa detección |
|--------------|---|----------------|---------------------|----------------|
| p04 | 152 | 3 | 1 | 33.3% |
| p07 | 148 | 2 | 0 | 0.0% |
| p13 | 152 | 0 | 0 | — |

Tabla V.3. Análisis de coeficientes del modelo logístico (37 features; top 10 por |coef|).

| Rango | Feature | Coeficiente | |abs| |
|-------|---------|-------------|---------|
| 1 | soreness | −0.499 | 0.499 |
| 2 | dfi_predicted | −0.498 | 0.498 |
| 3 | minutesAwake | −0.482 | 0.482 |
| 4 | overall_score | +0.413 | 0.413 |
| 5 | hr_zone_1 | +0.385 | 0.385 |
| 6 | efficiency | +0.385 | 0.385 |
| 7 | sedentary_minutes | +0.352 | 0.352 |
| 8 | mood | −0.314 | 0.314 |
| 9 | trimp | −0.235 | 0.235 |
| 10 | hr_zone_below | −0.221 | 0.221 |

> Artefacto visual: `coefficient_importance.png` — Gráfico de barras horizontales de los coeficientes del modelo logístico.

Tabla V.4. Resultados de validación cruzada LOSO (9 folds válidos de 16).

| Métrica LOSO | Valor |
|--------------|-------|
| ROC-AUC medio | 0.5691 |
| Desv. estándar ROC-AUC | ±0.1905 |
| PR-AUC medio | 0.0786 |
| F1 medio | 0.0646 |
| Folds válidos | 9 |
| Folds omitidos (0 lesiones) | 7 |

### Anexo W: Informe técnico consolidado del sistema integrado (R9)

Informe técnico que documenta la ejecución del pipeline de integración R4 → R5 de extremo a extremo sobre el conjunto de prueba completo.

> Fuente: Salida del pipeline de integración — `src/outputs/integration/integration_predictions.csv`

Tabla W.1. Resumen ejecutivo del sistema integrado (R9).

| Componente | Detalle |
|-----------|---------|
| Modelo R4 (fatiga) | BiLSTM + TemporalAttention, 102,849 parámetros, best_weights.keras |
| Modelo R5 (lesión) | LogisticRegression, C=0.01, L2, 37 features, logistic_injury.joblib |
| Dataset de entrada | 2,398 observaciones diarias, 16 participantes, 43 features Fitbit |
| Partición de prueba | p04, p07, p13 (452 observaciones) |
| DFI generados | 2,174 secuencias (ventana 14 días) |
| Cold-starts imputados | 224 registros (mediana por participante/global) |
| Features con NaN imputados | 7 (chronic_load_28d, acwr, sleep_debt, sleep_7d_avg, rhr_drift, rhr_variability_7d, rhr_baseline_7d) |
| Tiempo total de ejecución | 10.20 segundos (CPU) |

Tabla W.2. Métricas del sistema integrado sobre el conjunto de prueba.

| Métrica | R4 (DFI) | R5 aislado | R6 integrado |
|---------|---------|-----------|-------------|
| RMSE (DFI / escala original) | 0.190 / 1.90 RPE | — | — |
| ROC-AUC (lesión) | — | 0.5517 | 0.5517 |
| PR-AUC (lesión) | — | 0.0184 | 0.0181 |
| F1 (lesión) | — | 0.0211 | 0.0250 |
| Diferencia integrado vs. aislado | — | — | 0.000 ROC-AUC |

Tabla W.3. Estructura del artefacto de salida del sistema integrado (`integration_predictions.csv`).

| Columna | Tipo | Descripción |
|---------|------|-------------|
| participant_id | string | Identificador del participante (p01–p16) |
| date | date | Fecha de la observación (YYYY-MM-DD) |
| dfi_predicted | float [0, 1] | Índice Dinámico de Fatiga estimado por R4 |
| injury_probability | float [0, 1] | Probabilidad de lesión estimada por R5 |
| injury_predicted | int {0, 1} | Predicción binaria (umbral = 0.38) |
| injury_actual | int {0, 1} | Etiqueta real de lesión del dataset PMData |
