# Capítulo 5. Implementación de modelos predictivos de fatiga y riesgo de lesión en corredores

## 5.1 Introducción

Este capítulo tiene como finalidad presentar los resultados obtenidos para el segundo objetivo específico planteado, el cual corresponde al desarrollo e implementación de un sistema predictivo compuesto por dos modelos de aprendizaje automático y un pipeline de integración que los orquesta en secuencia, con el propósito de analizar la fatiga y predecir el riesgo de lesión en corredores recreacionales.

El desarrollo de modelos predictivos en el dominio de las ciencias del deporte presenta desafíos específicos que trascienden la selección del algoritmo: la naturaleza temporal de los datos fisiológicos, el severo desbalance de clases inherente a la baja prevalencia de lesiones, la heterogeneidad entre participantes y la necesidad de garantizar la ausencia de fuga de datos entre conjuntos de entrenamiento y evaluación. Estos desafíos condicionan las decisiones metodológicas adoptadas en cada uno de los tres resultados desarrollados en este capítulo.

El insumo principal para las Secciones 5.2.1 a 5.2.3 es el dataset curado producido en el Capítulo 4 (O1), el cual contiene 2,398 observaciones diarias de 16 corredores con 49 variables, incluyendo métricas fisiológicas de sensores Fitbit, registros subjetivos de bienestar (PMSYS) y características derivadas mediante ingeniería de features. Este dataset se consume en dos formatos complementarios: un archivo CSV para el preprocesamiento y particionado, y archivos TFRecord para el entrenamiento eficiente con TensorFlow. Adicionalmente, la Sección 5.2.4 documenta el sistema M1 → M2 implementado sobre el **Runner Dataset** (74 atletas, 42,766 observaciones, 583 eventos de lesión), que constituye el dataset de mayor escala de la tesis; su validación completa se presenta en la Sección 6.2.4 del Capítulo 6.

El enfoque metodológico adoptado comprende tres fases que se materializan en los tres resultados del objetivo: (1) el diseño e implementación de un modelo de Deep Learning basado en redes LSTM bidireccionales con mecanismo de atención, destinado a estimar un índice dinámico de fatiga (DFI) a partir de series temporales de 14 días; (2) el diseño e implementación de un modelo de regresión logística con técnicas de sobremuestreo, normalización y validación cruzada por sujeto, destinado a predecir el riesgo diario de lesión incorporando el DFI estimado como variable de entrada; y (3) la implementación de un pipeline de integración que orquesta la ejecución secuencial de ambos modelos y materializa el sistema predictivo completo.

Las herramientas empleadas para el logro de este objetivo incluyen Python con TensorFlow y Keras para la implementación del modelo de fatiga (R4), Python con scikit-learn e imbalanced-learn para el modelo de predicción de lesión (R5), y Python junto con Git y GitHub para el sistema de integración y control de versiones (R6).

A continuación, se presentan los resultados alcanzados que permitieron lograr el cumplimiento del objetivo, a partir de sus descripciones, medios de verificación y el cumplimiento de los indicadores objetivamente verificables.

## 5.2 Resultados Alcanzados

### 5.2.1 Modelo de Deep Learning implementado para el análisis de la fatiga (R4)

Para el primer resultado alcanzado, se diseñó e implementó un modelo de Deep Learning para la estimación continua de la fatiga fisiológica de un corredor a partir de series temporales de 14 días de métricas de sensores vestibles. El modelo adopta una arquitectura de redes LSTM bidireccionales apiladas con un mecanismo de atención temporal que pondera la contribución relativa de cada día de la secuencia de entrada, mejorando tanto la capacidad predictiva como la interpretabilidad del modelo.

Variable objetivo. La variable a predecir es el Índice Dinámico de Fatiga (DFI, Dynamic Fatigue Index), definido a partir de la escala de fatiga subjetiva reportada diariamente por los participantes en la plataforma PMSYS. La transformación aplicada es:

$$\text{DFI} = \frac{5 - \text{fatigue}}{4}$$

donde "fatigue" corresponde a la puntuación en escala PMSYS (1–5), y el DFI resultante está acotado en el intervalo [0, 1], donde 0 representa mínima fatiga y 1 representa máxima fatiga. Esta formulación invierte la escala original — en PMSYS, 1 indica alta fatiga y 5 indica baja fatiga — y la normaliza a una escala continua interpretable. El DFI constituye la variable de salida del modelo R4 y, como se detallará en el resultado R5, actúa también como variable de entrada al modelo de predicción de lesión.

Arquitectura del modelo. El modelo sigue una arquitectura de tipo Encoder-Predictor compuesta por seis capas funcionales. La Tabla 5.1 describe la arquitectura completa.

Tabla 5.1. Arquitectura del modelo de Deep Learning para estimación de fatiga (R4).

| Capa | Tipo | Configuración | Propósito |
|------|------|---------------|-----------|
| Entrada | Input | shape (14, 43) | Secuencia de 14 días × 43 features |
| 1ª LSTM Bidireccional | Bidirectional LSTM | 64 unidades, return_sequences=True, L2=1e-4 | Captura dependencias temporales en ambas direcciones |
| Dropout 1 | Dropout | rate=0.3 | Regularización post-LSTM1 |
| 2ª LSTM Bidireccional | Bidirectional LSTM | 32 unidades, return_sequences=True, L2=1e-4 | Representación de alto nivel de la secuencia |
| Atención Temporal | TemporalAttention | Aditiva (tanh + softmax), learnable | Pondera la importancia de cada día de la secuencia |
| Capa Densa | Dense | 32 unidades, ReLU, L2=1e-4 | Transformación no lineal de la representación |
| Dropout 2 | Dropout | rate=0.2 | Regularización pre-salida |
| Salida | Dense | 1 unidad, Sigmoid | DFI estimado ∈ [0, 1] |

La capa de Atención Temporal (TemporalAttention) constituye el componente central del modelo desde el punto de vista de la interpretabilidad. Esta capa aprende un vector de atención aditiva mediante una función de puntaje "tanh(W·h + b)" seguida de una normalización "softmax", produciendo pesos por paso de tiempo que reflejan la relevancia relativa de cada día de la ventana de 14 días para la estimación del DFI del día siguiente. Estos pesos pueden visualizarse como un mapa de calor para auditar el comportamiento del modelo.

Preprocesamiento y construcción de secuencias. Para cada participante, se construyen secuencias deslizantes de longitud 14 sobre sus observaciones ordenadas cronológicamente: la entrada X corresponde a los 14 días de features objetivas y la etiqueta y es el DFI del día 15. Se utilizan exclusivamente las 43 features objetivas de Fitbit (actividad, frecuencia cardíaca, sueño, TRIMP, ACWR, cargas acumuladas), excluyendo deliberadamente las variables de autoinforme de PMSYS — excepto "fatigue", que es la fuente del DFI — para que el modelo no acceda a información subjetiva durante la inferencia.

La normalización se realiza mediante MinMaxScaler, ajustado (*fit*) exclusivamente sobre los participantes del conjunto de entrenamiento y aplicado (*transform*) a los conjuntos de validación y prueba, impidiendo la fuga de información de test hacia el proceso de entrenamiento.

Configuración de entrenamiento. La Tabla 5.2 resume los hiperparámetros del proceso de entrenamiento, almacenados en el archivo "hyperparameters.json" del directorio de salida.

Tabla 5.2. Hiperparámetros de entrenamiento del modelo de fatiga (R4).

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Optimizador | Adam | Convergencia adaptativa para series temporales |
| Tasa de aprendizaje inicial | 0.001 | Valor empírico estándar para Adam |
| Función de pérdida | MSE | Tarea de regresión continua [0, 1] |
| Métrica de monitoreo | MAE | Interpretable en la escala del DFI |
| Épocas máximas | 200 | Límite superior con detención anticipada |
| EarlyStopping patience | 20 épocas | Detiene si val_loss no mejora, restaura mejores pesos |
| ReduceLROnPlateau factor | 0.5 (patience=10) | Reduce LR a la mitad si val_loss estanca 10 épocas |
| LR mínima | 1×10⁻⁶ | Límite inferior de la tasa de aprendizaje |
| Batch size | 32 | Balance entre estabilidad y velocidad |
| Semilla aleatoria | 42 | Reproducibilidad |

Los callbacks configurados son: "EarlyStopping" (monitorea "val_loss", restaura los mejores pesos), "ReduceLROnPlateau" (reduce la tasa de aprendizaje en mesetas de validación), "ModelCheckpoint" (guarda los pesos óptimos en "best_weights.keras") y "CSVLogger" (registra el historial de entrenamiento en "training_log.csv").

Métricas de evaluación. El modelo se evalúa sobre el conjunto de prueba mediante cinco métricas de regresión calculadas tanto globalmente como desglosadas por participante: Error Cuadrático Medio (MSE), Raíz del Error Cuadrático Medio (RMSE), Error Absoluto Medio (MAE), Coeficiente de Determinación (R²) y Correlación de Pearson (r). El desglose por participante permite identificar subgrupos con mayor dificultad de predicción, lo cual es relevante dada la heterogeneidad fisiológica entre corredores.

Como medio de verificación de este resultado, el código fuente de la implementación del modelo se encuentra alojado en un repositorio de control de versiones (Anexo L), y el documento técnico con la descripción de la arquitectura, hiperparámetros y métricas de evaluación se presenta en el Anexo M. Como indicador objetivamente verificable, el informe técnico fue revisado y aprobado al 100% por el asesor de tesis y un experto en aprendizaje automático.

### 5.2.2 Modelo de Machine Learning implementado para la predicción del riesgo de lesión (R5)

Para el segundo resultado alcanzado, se diseñó e implementó un modelo de clasificación binaria para la predicción diaria del riesgo de lesión en corredores. El modelo integra el Índice Dinámico de Fatiga estimado por R4 como variable de entrada, estableciendo así la dependencia secuencial entre ambos modelos.

> **Nota sobre extensión de R5:** Adicionalmente al modelo logístico entrenado sobre PMData (16 atletas Fitbit) descrito en esta sección, se implementó una extensión denominada **RF-Runner** que utiliza el Runner Dataset (Löwdal et al., 2021) como dataset de entrenamiento primario, alcanzando LOAO AUC = 0.9101 sobre 74 atletas. Esta extensión se documenta en la Sección 6.2.2.1 del Capítulo 6. El modelo logístico con DFI descrito a continuación constituye el resultado principal de R5 dentro del sistema integrado R4→R5→R6.

Preprocesamiento. Las variables de entrada son normalizadas mediante una transformación Yeo-Johnson seguida de estandarización z-score, implementadas a través de "PowerTransformer" de scikit-learn. Esta combinación reduce la asimetría de las distribuciones no normales y lleva todas las variables a una escala comparable, lo cual mejora la convergencia del optimizador del modelo logístico. Crítico para la validez metodológica, el transformador se ajusta (*fit*) únicamente sobre los datos de entrenamiento de cada fold y se aplica (*transform*) a los datos de validación y prueba, previniendo la fuga de información entre conjuntos.

Incorporación del DFI de R4. El dataset de entrada al modelo R5 contiene 39 variables, incluyendo "dfi_predicted", la estimación de fatiga producida por R4 para cada par (participante, día). Para los días en que no existe una predicción DFI disponible — por ejemplo, los primeros 13 días de un participante, que no cuentan con una ventana de 14 días completa — se aplica una estrategia de imputación de arranque en frío (*cold-start imputation*): primero se rellena con la mediana del DFI del propio participante y, si este valor tampoco está disponible, se utiliza la mediana global del DFI del conjunto de entrenamiento. Esta estrategia garantiza que la columna "dfi_predicted" esté completamente poblada sin introducir valores nulos al modelo.

Corrección de sesgo de origen y preprocesamiento de ACWR. Previo a la augmentación sintética, el pipeline aplica dos correcciones orientadas a garantizar que la distribución de los datos de entrenamiento sea estable y representativa. En primer lugar, la función `_balance_participant_positives()` implementa una corrección del sesgo de participante: si algún corredor concentra un número de eventos positivos desproporcionadamente elevado respecto al resto (definido como más de tres veces la mediana de positivos del grupo), su contribución se limita muestreando aleatoriamente hasta ese umbral. En la práctica, el participante p12 presentaba 44 eventos de lesión (67.7% del total de positivos del conjunto de entrenamiento y una tasa individual del 28.9%), mientras que la mediana del grupo era de ~4 eventos; el umbral de corte (mediana × 3 = 12) redujo sus positivos de 44 a 12, bajando su representación al nivel esperable para un corredor recreacional. En segundo lugar, la columna `acwr` (Acute:Chronic Workload Ratio) — la feature de mayor relevancia biomecánica del dataset — presentaba un 22.3% de valores nulos (534 de 2,398 registros), concentrados en los primeros días de cada participante cuando la carga crónica de 28 días aún no estaba completamente acumulada. Se aplica un relleno hacia adelante (*forward-fill*) y hacia atrás (*backward-fill*) por participante, reduciendo los valores nulos al 6.3% (152 registros residuales sin información previa disponible). Estas correcciones siguen el principio de que la augmentación sintética debe operar sobre una distribución de origen sana: corregir el sesgo y la varianza antes de generar muestras sintéticas evita que SMOTE amplifique artefactos presentes en los datos crudos.

Augmentación de datos. El desbalance de clases de la variable objetivo (prevalencia de lesión del 3.0%, ratio 1:32) se aborda mediante Synthetic Minority Over-sampling Technique (SMOTE), aplicado sobre el conjunto de entrenamiento tras la corrección de sesgo descrita anteriormente. Los parámetros configurados son: ratio de sobremuestreo de 0.30 (la clase minoritaria alcanza el 30% del total tras la augmentación) y k_neighbors=5 (adaptado dinámicamente al tamaño de la clase minoritaria para evitar errores en folds pequeños). La augmentación se realiza *después* de la normalización y *antes* del ajuste del modelo, respetando el orden correcto del pipeline.

Modelo y selección de hiperparámetros. El algoritmo seleccionado es Regresión Logística ("LogisticRegression" de scikit-learn) con regularización L2, lo cual es metodológicamente apropiado dado el tamaño reducido del dataset (n=16 participantes) y la necesidad de interpretabilidad de los coeficientes. La selección del parámetro de regularización C se realiza mediante búsqueda en rejilla (*grid search*) sobre los valores {0.01, 0.1, 1.0, 10.0}, evaluando cada candidato mediante validación cruzada estratificada de 5 folds con métrica ROC-AUC. Se configura "class_weight="balanced"" para que el modelo pondere los errores sobre la clase positiva proporcionalmente a su infrarrepresentación residual tras la augmentación SMOTE.

Ajuste de umbral de decisión (*threshold tuning*). A diferencia del umbral por defecto de 0.5, el umbral de clasificación se determina empíricamente sobre el conjunto de validación, seleccionando el valor p* que maximiza el F1-score calculado a partir de la curva ROC. Este ajuste posterior permite calibrar la sensibilidad del modelo según las preferencias del contexto clínico, donde un falso negativo (no detectar una lesión inminente) puede tener mayor costo que un falso positivo.

Validación cruzada por sujeto (LOSO). Para estimar la capacidad de generalización del modelo a corredores no vistos durante el entrenamiento, se implementó una validación cruzada Leave-One-Subject-Out (LOSO). En cada fold, un participante es reservado como conjunto de prueba y el modelo se entrena con los datos de los demás. Los folds en los que el participante reservado no registra ningún evento de lesión ("n_injuries = 0") son omitidos automáticamente — puesto que la métrica ROC-AUC no está definida para clases con un único valor — y se registran con "skipped=True" en el reporte de resultados. Adicionalmente, los datos sintéticos generados por SMOTE (identificados por el prefijo "synth_*" en el campo "participant_id") son excluidos de los conjuntos de train y test de cada fold LOSO, garantizando que la evaluación se realice exclusivamente sobre datos reales.

Métricas de evaluación. El modelo se evalúa mediante ocho métricas complementarias que cubren distintas perspectivas del rendimiento en clasificación desbalanceada:

- ROC-AUC: área bajo la curva ROC, métrica primaria
- PR-AUC: área bajo la curva Precisión-Recall, más informativa que ROC-AUC en clases desbalanceadas
- F1-score: media armónica de precisión y recall sobre la clase positiva
- Precisión: proporción de predicciones positivas correctas
- Recall (Sensibilidad): proporción de lesiones reales detectadas
- Balanced Accuracy: promedio de sensibilidad y especificidad
- Brier Score: error cuadrático entre probabilidades predichas y etiquetas reales
- Matriz de confusión: desglose de VP, FP, VN, FN

Plan de pruebas. Se diseñó e implementó un plan de 41 pruebas unitarias organizadas en cuatro módulos que cubren cada componente del modelo. La Tabla 5.3 presenta la distribución por módulo.

Tabla 5.3. Distribución de pruebas unitarias del modelo de predicción de lesión (R5).

| Módulo | Componente | N.° de tests | Aspectos verificados |
|--------|------------|:------------:|----------------------|
| test_injury_model | Modelo LR | 6 | Construcción, class_weight, probabilidades válidas, baseline |
| test_injury_train | Entrenamiento | 6 | Smoke test, métricas finitas, modelo guardado, grid search |
| test_injury_augment | SMOTE y Cópula | 12 | Filas aumentadas, clase minoritaria incrementada, columnas preservadas, datos reales preservados, PIDs sintéticos, distribución |
| test_injury_dataset | Dataset | 17 | Integración DFI, cold-start, splits sin solapamiento, normalización (media≈0, std≈1), lekage entre participantes |
| Total | | 41 | |

Como medio de verificación de este resultado, el código fuente del modelo se encuentra alojado en un repositorio de control de versiones (Anexo O), el documento técnico con la descripción metodológica del modelo se presenta en el Anexo P, el catálogo completo de 41 pruebas unitarias se documenta en el Anexo Q, y los resultados de la validación LOSO se presentan en el Anexo R. Como indicador objetivamente verificable, el informe técnico fue revisado y aprobado al 100% por el asesor de tesis y un experto en aprendizaje automático.

### 5.2.3 Sistema predictivo integrado que orquesta la ejecución de los dos modelos en secuencia (R6)

Para el tercer resultado alcanzado, se implementó un pipeline de integración que orquesta la ejecución secuencial del modelo de fatiga (R4) y el modelo de predicción de lesión (R5), materializando el sistema predictivo completo. El pipeline recibe un dataset de entrada en formato CSV, produce estimaciones de DFI mediante R4, transfiere estas estimaciones como feature a R5, y genera como salida un conjunto de predicciones integradas que incluyen tanto el índice de fatiga como la probabilidad diaria de lesión para cada corredor.

Arquitectura del sistema de integración. El pipeline se compone de cuatro etapas ejecutadas secuencialmente, orquestadas por el módulo `src/integration/pipeline.py`. La Figura 5.1 ilustra el flujo de datos entre las etapas.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE INTEGRACIÓN (R6)                         │
│                                                                         │
│  ┌────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │            │  │                  │  │              │  │          │ │
│  │ LoadModels │─▶│FatiguePrediction │─▶│FeatureHandoff│─▶│ Injury   │ │
│  │            │  │                  │  │              │  │Prediction│ │
│  │ R4 Keras   │  │ DFI = R4(X_14d)  │  │ LEFT JOIN    │  │          │ │
│  │ R5 joblib  │  │ por participante │  │ DFI → R5     │  │ LR(X+DFI)│ │
│  │ Normalizer │  │                  │  │ cold-start   │  │ prob[0,1]│ │
│  └────────────┘  └──────────────────┘  └──────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

Figura 5.1. Arquitectura del pipeline de integración R4 → R5.

Etapa 1 — LoadModels. Se cargan en memoria los tres artefactos persistidos en disco: el modelo R4 (archivo `.keras` con la capa personalizada `TemporalAttention`), el modelo R5 (archivo `.joblib` con el LogisticRegression ajustado) y el normalizador PowerTransformer de R5 (archivo `.joblib`). Las rutas de los tres artefactos son configurables mediante variables de entorno, lo que facilita la reutilización del pipeline en diferentes entornos de ejecución.

Etapa 2 — FatiguePrediction. Se invocan las predicciones del modelo R4 ("predict_all()") sobre todos los participantes del dataset de entrada. Para cada participante, se construyen secuencias deslizantes de 14 días y se produce una estimación de DFI para el día siguiente. El resultado es un DataFrame con columnas "[participant_id, date, dfi_predicted, dfi_actual]", que registra las estimaciones de fatiga junto con el valor real para su posterior evaluación. Se registran en el informe el número de predicciones producidas, la media y la desviación estándar del DFI estimado.

Etapa 3 — FeatureHandoff. Se realiza la transferencia en memoria de las estimaciones de DFI hacia el dataset de entrada del modelo R5 mediante un *LEFT JOIN* sobre las claves "(participant_id, date)". Esta estrategia garantiza que el conteo de filas del dataset de R5 se preserve íntegramente. Los días sin cobertura de DFI — correspondientes a los primeros 13 días de cada participante, que carecen de una ventana de 14 días completa para R4 — reciben imputación por la mediana del DFI del propio participante; si esta tampoco está disponible, se utiliza la mediana global. Tras esta etapa, la columna "dfi_predicted" está completamente poblada (0% de valores nulos).

Etapa 4 — InjuryPrediction. Se seleccionan los participantes asignados al conjunto de prueba según la partición determinista de R5, se normalizan sus features con el PowerTransformer pre-ajustado y se ejecuta la predicción del modelo R5 mediante "predict_proba". El resultado es un DataFrame con las siguientes columnas: "participant_id", "date", "dfi_predicted", "injury_probability", "injury_predicted", "injury_actual". Las métricas PR-AUC, ROC-AUC y F1 se calculan sobre estas predicciones y se registran en el informe final del pipeline.

Plan de pruebas. Se diseñó e implementó un plan de 21 pruebas unitarias organizadas en seis clases que cubren el comportamiento del sistema de integración de manera integral. La Tabla 5.4 presenta la distribución por clase.

Tabla 5.4. Distribución de pruebas unitarias del sistema de integración (R6).

| Clase | N.° de tests | Aspectos verificados |
|-------|:------------:|----------------------|
| TestModelLoading | 2 | Carga de R4 (Keras + TemporalAttention) y R5 (joblib); normalizer funcional |
| TestEndToEnd | 5 | Pipeline completo sin errores; DFI ∈ [0, 1]; probabilidades de lesión ∈ [0, 1]; columnas de salida correctas; sin NaN |
| TestHandoffIntegrity | 3 | Columna dfi_predicted inyectada; filas preservadas; features de R5 completas |
| TestColdStartImputation | 2 | Participante sin DFI imputado; sin NaN tras imputación |
| TestErrorHandling | 2 | RuntimeError con modelo R4 faltante; RuntimeError con modelo R5 faltante |
| TestReportCompleteness | 7 | 4 etapas en reporte; nombres correctos; duraciones positivas; métricas finitas; CSV guardado; rutas de modelos registradas |
| Total | 21 | |

Como medio de verificación de este resultado, el script de integración y los módulos asociados se encuentran alojados en un repositorio de control de versiones (Anexo S), y el catálogo completo de 21 pruebas unitarias de integración se documenta en el Anexo T. Como indicador objetivamente verificable, el script de integración "run_integration.py" se ejecuta de manera exitosa y sin errores sobre el conjunto de datos de prueba, produciendo todas las salidas esperadas.

---

### 5.2.4 Sistema predictivo sobre el Runner Dataset: Modelo M1 (Regresor de Fatiga) y Modelo M2 (Clasificador de Lesión RF)

Como extensión central de la arquitectura implementada para PMData, se desarrolló una versión del sistema predictivo adaptada al **Runner Dataset** (74 atletas, 42,766 observaciones), que constituye el dataset primario de la tesis. La arquitectura sigue el mismo principio de orquestación M1 → M2, pero adopta algoritmos y protocolos de validación apropiados para la mayor escala del dataset.

#### Modelo M1 — Regresor de Fatiga (Random Forest Regressor)

**Propósito y diseño.** A diferencia del modelo R4 (BiLSTM), el Modelo M1 estima la percepción de recuperación diaria del atleta (`perceived_recovery.6`, escala [0, 1]) a partir exclusivamente de las 10 features GPS objetivas descritas en la Tabla 4.5 del Capítulo 4. El modelo adopta un **Random Forest Regressor** (RF) en lugar de LSTM por tres razones: (1) las 10 features ya son agregados de 7 días, eliminando la necesidad de modelar dependencias secuenciales explícitas; (2) los Random Forests son más robustos ante el tamaño reducido del conjunto de entrenamiento de cada fold LOAO; y (3) el RF no requiere secuencias de longitud fija, simplificando el pipeline.

**Variables excluidas del entrenamiento.** Los días de descanso (donde `perceived exertion = -0.01`) se excluyen del conjunto de entrenamiento de M1: el modelo solo aprende de días con actividad real. Esto evita que el regresor aprenda el valor constante de descanso (-0.01) como señal de recuperación. El target se extrae directamente del CSV crudo (antes de la imputación por forward-fill de `compute_features()`), preservando los valores NaN en días de descanso.

**Hiperparámetros del modelo.** La Tabla 5.5 resume la configuración del RF Regressor M1.

Tabla 5.5. Hiperparámetros del Random Forest Regressor M1.

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `n_estimators` | 200 | Balance capacidad-varianza |
| `max_depth` | 10 | Evita sobreajuste en folds con pocos datos |
| `min_samples_leaf` | 5 | Regularización implícita |
| `random_state` | 42 | Reproducibilidad |
| `n_jobs` | −1 | Paralelización completa |

**Protocolo de validación LOAO.** Se utiliza **Leave-One-Athlete-Out (LOAO)** con 74 folds (uno por atleta): para cada fold, se entrena en los 73 atletas restantes con normalización Yeo-Johnson por fold, y se evalúa en el atleta reservado. El LOAO simula el escenario de predicción para un atleta nuevo no visto durante el entrenamiento.

**Generación de predicciones para M2.** En el mismo pasada LOAO que sirve para la evaluación de M1, se generan predicciones de recuperación para TODOS los registros del atleta reservado. Esto garantiza que las predicciones de M1 disponibles para M2 sean siempre out-of-sample: el atleta i nunca fue visto por el modelo que genera sus predicciones.

Los artefactos generados incluyen:
- `src/outputs/rf_fatigue_runner_model.pkl` — modelo final entrenado en los 31,287 días de entrenamiento activo (todos los atletas)
- `src/outputs/runner_fatigue_predictions_loao.csv` — predicciones LOAO de recuperación para los 42,766 registros
- `src/outputs/fatigue_feature_importance.csv` — importancia relativa de cada feature GPS

#### Modelo M2 — Clasificador de Lesión (Random Forest Classifier)

**Propósito y diseño.** El Modelo M2 predice la probabilidad de lesión a partir de las 18 features derivadas del Runner Dataset (incluidas en el dataset procesado), con la opción de incorporar el `fatigue_score_predicted` generado por M1 como variable adicional. Se adopta un **Random Forest Classifier** con la configuración de la Tabla 5.6.

Tabla 5.6. Hiperparámetros del Random Forest Classifier M2.

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `n_estimators` | 200 | Estabilidad del ensemble |
| `max_features` | `sqrt` | Descorelación de árboles |
| `class_weight` | `balanced` | Compensa el 1.36% de prevalencia |
| `min_samples_leaf` | 1 | Máxima expresividad |
| `max_depth` | None | Sin poda preestablecida |
| `random_state` | 42 | Reproducibilidad |

**Augmentación SMOTE por fold.** Dentro de cada fold LOAO, se aplica SMOTE (Synthetic Minority Over-sampling Technique) con `target_ratio=0.15` y `k_neighbors=5` sobre los datos de entrenamiento normalizados. Adicionalmente, se aplica downsampling controlado para que ningún atleta aporte más de 21 muestras positivas, previniendo el dominio de atletas con alta tasa de lesión.

**Normalización por fold.** La transformación Yeo-Johnson se ajusta exclusivamente sobre los datos de entrenamiento del fold y se aplica al fold de prueba, evitando la fuga de información entre atletas.

Los artefactos generados incluyen:
- `src/outputs/rf_runner_model.pkl` — clasificador entrenado en los 18 features originales (sin M1)
- `src/outputs/loao_runner_v2_results.csv` — resultados LOAO del M2 con 11 features (10 GPS + fatigue_score_predicted)
- `src/outputs/ablation_fatigue_runner.csv` — tabla de ablación completa (ver Capítulo 6)

**Pipeline de inferencia M1 → M2.** El flujo de inferencia completo aplica M1 sobre las features GPS del día actual para obtener `fatigue_score_predicted`, y luego concatena este valor a las 10 GPS features para producir el vector de 11 features de entrada a M2. Este pipeline reproduciría el flujo de un sistema de alerta en tiempo real donde el atleta no necesita reportar subjetivamente su recuperación.

## 5.3 Discusión

Se llevaron a cabo tres resultados esperados para lograr el objetivo de desarrollar un sistema predictivo integrado de fatiga y riesgo de lesión en corredores. Para el primer resultado (R4), se implementó un modelo de Deep Learning con arquitectura BiLSTM + TemporalAttention que estima el Índice Dinámico de Fatiga a partir de ventanas de 14 días de señales fisiológicas objetivas. Para el segundo resultado (R5), se implementó un modelo de regresión logística que incorpora el DFI estimado por R4 como variable de entrada y aplica técnicas de sobremuestreo, normalización robusta y validación cruzada por sujeto para predecir el riesgo diario de lesión. Para el tercer resultado (R6), se implementó un pipeline de integración que orquesta la ejecución secuencial de ambos modelos, materializando el sistema predictivo completo con 21 pruebas de integración, todas ejecutadas exitosamente.

Cuatro decisiones de diseño merecen particular reflexión. En primer lugar, la incorporación del DFI de R4 como variable de entrada a R5 establece una dependencia explícita entre los dos modelos que refleja la hipótesis biomecánica central del trabajo: la fatiga acumulada es un factor mediador del riesgo de lesión. Esta dependencia implica que los errores de predicción de R4 se propagan a R5; la estrategia de cold-start mitiga este efecto para los días sin cobertura de ventana, pero no lo elimina para los días con estimaciones imprecisas de DFI.

En segundo lugar, la selección de regresión logística para R5 — en lugar de modelos de mayor capacidad como Random Forest o redes neuronales — responde a tres consideraciones: (a) el tamaño reducido del dataset (n=16 participantes, ~2,400 observaciones) limita la ventaja de modelos más complejos; (b) los coeficientes del modelo logístico son directamente interpretables como log-odds, lo cual facilita la validación clínica; y (c) la regularización L2 con selección de C vía grid search proporciona un mecanismo explícito de control de la complejidad.

En tercer lugar, el ajuste del umbral de decisión sobre el conjunto de validación — en lugar del umbral por defecto de 0.5 — reconoce explícitamente que, en el contexto de la predicción de lesiones deportivas, el costo de un falso negativo (no alertar sobre una lesión inminente) supera al de un falso positivo (alertar innecesariamente). El umbral óptimo que maximiza F1 representa un compromiso entre precisión y recall definido por los datos, no por una convención estadística.

En cuarto lugar, la validación LOSO refleja el escenario de despliegue realista del sistema: predecir para corredores no vistos durante el entrenamiento. La omisión automática de folds con cero lesiones — una decisión técnica necesaria para la definición de ROC-AUC — introduce un sesgo potencial de selección que debe reconocerse como limitación del protocolo de evaluación.

El trabajo desarrollado presenta ciertas limitaciones que se reconocen explícitamente. El tamaño de la muestra (16 participantes) restringe la potencia estadística de la validación LOSO y limita la generalización de los resultados a poblaciones de corredores más amplias. El severo desbalance de clases (3% de prevalencia de lesión) significa que, incluso tras la augmentación SMOTE, el modelo opera en condiciones de aprendizaje desfavorables para la clase positiva. La dependencia secuencial R4 → R5 implica que la calidad del sistema integrado está acotada superiormente por la precisión del modelo de fatiga; si R4 comete un error sistemático en la estimación del DFI para un participante particular, este error se trasladará a las predicciones de lesión de R5 para ese mismo participante.

El sistema predictivo desarrollado en este capítulo, junto con el dataset curado construido en el Capítulo 4, constituye el aporte técnico central de la presente tesis, evidenciando la viabilidad de integrar señales de sensores vestibles de consumo masivo con técnicas de aprendizaje automático para la predicción temprana del riesgo de lesión en corredores recreacionales.

---

## Anexos del Capítulo 5

### Anexo L: Código fuente — Modelo de fatiga (R4)

Código fuente completo del modelo de Deep Learning para la estimación del DFI, compuesto por 6 módulos Python alojados en un repositorio de control de versiones.

> Enlace: [src/fatigue/ — Repositorio GitHub](#) *(insertar enlace al repositorio)*

| Módulo | Archivo | Propósito |
|--------|---------|-----------|
| Configuración | config.py | Hiperparámetros, rutas de salida, lista de features objetivas |
| Dataset | dataset.py | Cálculo de DFI, construcción de secuencias, splits por participante, MinMaxScaler |
| Modelo | model.py | Arquitectura BiLSTM + TemporalAttention, compilación |
| Entrenamiento | train.py | Bucle de entrenamiento, callbacks, guardado de pesos |
| Evaluación | evaluate.py | Métricas MSE/RMSE/MAE/R²/Pearson r, desglose por participante |
| Pipeline | pipeline.py | Orquestador: Dataset → Train → Evaluate con reporte de etapas |

### Anexo M: Documento técnico — Modelo de fatiga (R4)

Especificación técnica del modelo de Deep Learning, incluyendo arquitectura, hiperparámetros almacenados y descripción de la capa de atención temporal.

> Fuente: Hiperparámetros del modelo de fatiga — `hyperparameters.json`

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | BiLSTM(64) → Dropout(0.3) → BiLSTM(32) → TemporalAttention → Dense(32, ReLU) → Dropout(0.2) → Sigmoid |
| Regularización L2 | 1×10⁻⁴ (kernels LSTM y Dense) |
| Ventana temporal | 14 días |
| Features de entrada | 43 features objetivas Fitbit |
| Variable objetivo | DFI = (5 − fatigue) / 4 ∈ [0, 1] |
| Optimizador | Adam, LR=0.001 |
| Función de pérdida | MSE |
| Épocas máximas | 200 (con EarlyStopping patience=20) |
| Batch size | 32 |
| ReduceLROnPlateau | factor=0.5, patience=10, min_lr=1×10⁻⁶ |
| Semilla | 42 |
| Salida del modelo | best_weights.keras, training_log.csv |

### Anexo N: Plan de pruebas — Modelo de fatiga (22 tests)

Catálogo documentado de las 22 pruebas unitarias implementadas con pytest para el modelo de fatiga.

> Fuente: Archivos de prueba pytest — `test_fatigue_model.py`, `test_fatigue_train.py`, `test_fatigue_dataset.py`

Tabla N.1. Catálogo de pruebas — Construcción del modelo (8 tests).

| ID | Test | Descripción | Resultado |
|----|------|-------------|:---------:|
| FM-1 | test_build_returns_model | build_model() retorna un objeto Keras Model | ✅ Pass |
| FM-2 | test_output_shape | Salida tiene shape (batch, 1) | ✅ Pass |
| FM-3 | test_output_range | Salida ∈ [0, 1] (activación Sigmoid) | ✅ Pass |
| FM-4 | test_trainable_params | El modelo tiene parámetros entrenables > 0 | ✅ Pass |
| FM-5 | test_attention_layer_present | El modelo incluye la capa TemporalAttention | ✅ Pass |
| FM-6 | test_attention_weights_sum_to_one | Pesos de atención suman 1 (softmax correcto) | ✅ Pass |
| FM-7 | test_attention_output_shape | La atención produce shape (batch, features) | ✅ Pass |
| FM-8 | test_model_compiled | El modelo tiene optimizador y función de pérdida configurados | ✅ Pass |

Tabla N.2. Catálogo de pruebas — Entrenamiento (4 tests).

| ID | Test | Descripción | Resultado |
|----|------|-------------|:---------:|
| FT-1 | test_smoke_train | El modelo entrena al menos 1 época sin error | ✅ Pass |
| FT-2 | test_loss_is_finite | val_loss es un número finito tras entrenamiento | ✅ Pass |
| FT-3 | test_evaluate_returns_metrics | evaluate() retorna MSE, RMSE, MAE, R², Pearson r | ✅ Pass |
| FT-4 | test_per_participant_breakdown | Evaluación produce desglose por participante | ✅ Pass |

Tabla N.3. Catálogo de pruebas — Dataset (10 tests).

| ID | Test | Descripción | Resultado |
|----|------|-------------|:---------:|
| FD-1 | test_dfi_computation | DFI = (5 − fatigue) / 4 calculado correctamente | ✅ Pass |
| FD-2 | test_dfi_range | DFI ∈ [0, 1] para todos los registros | ✅ Pass |
| FD-3 | test_participant_split_no_overlap | Train/Val/Test no comparten participantes | ✅ Pass |
| FD-4 | test_split_ratios | Proporciones de split aproximadas (70/15/15) | ✅ Pass |
| FD-5 | test_sequence_shape | X tiene shape (N, 14, n_features) | ✅ Pass |
| FD-6 | test_target_shape | y tiene shape (N, 1) | ✅ Pass |
| FD-7 | test_target_range | y ∈ [0, 1] (DFI normalizado) | ✅ Pass |
| FD-8 | test_minmax_fitted_on_train | Scaler ajustado solo en train (sin data leakage) | ✅ Pass |
| FD-9 | test_no_participant_leakage | Ningún participante de test aparece en train | ✅ Pass |
| FD-10 | test_build_returns_bundle | build_fatigue_datasets() retorna FatigueDatasetBundle | ✅ Pass |

> Resultado global: 22 de 22 pruebas ejecutadas exitosamente (100%).

### Anexo O: Código fuente — Modelo de predicción de lesión (R5)

Código fuente completo del modelo de Machine Learning para la predicción de riesgo de lesión, compuesto por 7 módulos Python alojados en un repositorio de control de versiones.

> Enlace: [src/injury/ — Repositorio GitHub](#) *(insertar enlace al repositorio)*

| Módulo | Archivo | Propósito |
|--------|---------|-----------|
| Configuración | config.py | Hiperparámetros, C_GRID, SMOTE ratio, rutas |
| Dataset | dataset.py | Integración DFI, cold-start, splits, normalización Yeo-Johnson |
| Augmentación | augment.py | SMOTE y Cópula Gaussiana para sobremuestreo |
| Modelo | model.py | LogisticRegression, baseline naive, compilación |
| Entrenamiento | train.py | Grid search C, threshold tuning, guardado de artefactos |
| Evaluación | evaluate.py | ROC-AUC, PR-AUC, F1, Precision, Recall, Balanced Acc., Brier |
| Validación | validate.py | LOSO cross-validation, skip de folds vacíos |

### Anexo P: Documento técnico — Modelo de predicción de lesión (R5)

Especificación técnica del modelo de regresión logística, incluyendo preprocesamiento, augmentación, selección de hiperparámetros y protocolo de validación.

> Fuente: Configuración del modelo de lesión — `src/injury/config.py`

| Componente | Especificación |
|------------|---------------|
| Clasificador | LogisticRegression (scikit-learn) |
| Penalización | L2 |
| Solver | lbfgs |
| Max iteraciones | 1,000 |
| class_weight | balanced |
| Grid de C | {0.01, 0.1, 1.0, 10.0} |
| Métrica grid search | ROC-AUC (5-fold estratificado) |
| Normalización | PowerTransformer (yeo_johnson) + z-score |
| Augmentación | SMOTE, ratio=0.30, k_neighbors=5 |
| Threshold | Maximización F1 sobre validation set |
| Validación | LOSO (Leave-One-Subject-Out) |
| Features de entrada | 39 (incluyendo dfi_predicted) |
| Variable objetivo | is_injured ∈ {0, 1} |

### Anexo Q: Plan de pruebas — Modelo de predicción de lesión (41 tests)

Catálogo documentado de las 41 pruebas unitarias implementadas con pytest para el modelo de lesión.

> Fuente: Archivos de prueba pytest — `test_injury_model.py`, `test_injury_train.py`, `test_injury_augment.py`, `test_injury_dataset.py`

Tabla Q.1. Catálogo de pruebas — Modelo LR (6 tests).

| ID | Test | Descripción | Resultado |
|----|------|-------------|:---------:|
| IM-1 | test_builds | build_model() retorna un LogisticRegression | ✅ Pass |
| IM-2 | test_class_weight_balanced | class_weight es "balanced" | ✅ Pass |
| IM-3 | test_predictions_are_probabilities | predict_proba retorna valores ∈ [0, 1] | ✅ Pass |
| IM-4 | test_predict_proba_sums_to_one | Suma de probabilidades por fila ≈ 1.0 | ✅ Pass |
| IM-5 | test_builds (baseline) | Baseline naive construye correctamente | ✅ Pass |
| IM-6 | test_predictions_are_probabilities (baseline) | Baseline retorna probabilidades válidas | ✅ Pass |

Tabla Q.2. Catálogo de pruebas — Entrenamiento (6 tests).

| ID | Test | Descripción | Resultado |
|----|------|-------------|:---------:|
| IT-1 | test_smoke_train | Entrenamiento completa sin error | ✅ Pass |
| IT-2 | test_evaluation_metrics_finite | Métricas son números finitos (no inf/NaN) | ✅ Pass |
| IT-3 | test_smoke_train (baseline) | Baseline entrena sin error | ✅ Pass |
| IT-4 | test_returns_fitted_model | train() retorna modelo ajustado | ✅ Pass |
| IT-5 | test_save_creates_file | Artefacto .joblib creado en disco | ✅ Pass |
| IT-6 | test_returns_valid_result | train() retorna TrainResult completo | ✅ Pass |

Tabla Q.3. Catálogo de pruebas — Augmentación (12 tests).

| ID | Test | Descripción | Resultado |
|----|------|-------------|:---------:|
| IA-1 | test_augmented_has_more_rows | SMOTE produce más filas que el original | ✅ Pass |
| IA-2 | test_target_remains_binary | La variable objetivo es binaria tras SMOTE | ✅ Pass |
| IA-3 | test_feature_columns_preserved | Columnas de features preservadas | ✅ Pass |
| IA-4 | test_minority_class_increased | Clase minoritaria aumenta proporcionalmente | ✅ Pass |
| IA-5 | test_smote_method | Método SMOTE ejecuta correctamente | ✅ Pass |
| IA-6 | test_copula_method | Método Cópula Gaussiana ejecuta correctamente | ✅ Pass |
| IA-7 | test_augmented_has_more_rows (copula) | Cópula produce más filas que el original | ✅ Pass |
| IA-8 | test_real_data_preserved | Datos reales no son modificados por SMOTE | ✅ Pass |
| IA-9 | test_synthetic_pids_distinct | PIDs sintéticos tienen prefijo "synth_*" | ✅ Pass |
| IA-10 | test_target_remains_binary (copula) | Target binario tras Cópula | ✅ Pass |
| IA-11 | test_feature_columns_preserved (copula) | Columnas preservadas tras Cópula | ✅ Pass |
| IA-12 | test_returns_pvalues | Método retorna p-values del test KS de distribución | ✅ Pass |

Tabla Q.4. Catálogo de pruebas — Dataset (17 tests).

| ID | Test | Descripción | Resultado |
|----|------|-------------|:---------:|
| ID-1 | test_merge_adds_dfi_column | LEFT JOIN inyecta columna dfi_predicted | ✅ Pass |
| ID-2 | test_cold_start_filled | Cold-start rellena NaN de dfi_predicted | ✅ Pass |
| ID-3 | test_row_count_preserved | Conteo de filas se preserva tras el merge | ✅ Pass |
| ID-4 | test_no_metadata_in_features | Columnas de metadata excluidas de X | ✅ Pass |
| ID-5 | test_target_is_binary | is_injured ∈ {0, 1} | ✅ Pass |
| ID-6 | test_dfi_in_features | dfi_predicted presente como feature de X | ✅ Pass |
| ID-7 | test_all_assigned | Todos los participantes asignados a un split | ✅ Pass |
| ID-8 | test_no_overlap | Train/Val/Test no comparten participantes | ✅ Pass |
| ID-9 | test_deterministic | Splits son deterministas con misma semilla | ✅ Pass |
| ID-10 | test_bundle_shapes | Shapes de X_train, y_train coherentes | ✅ Pass |
| ID-11 | test_no_participant_leakage | Ningún participante de test aparece en train | ✅ Pass |
| ID-12 | test_total_rows | Suma de splits = total de filas del dataset | ✅ Pass |
| ID-13 | test_normalizer_present | Bundle incluye artefacto normalizer | ✅ Pass |
| ID-14 | test_train_mean_near_zero | Media de X_train ≈ 0 tras z-score | ✅ Pass |
| ID-15 | test_train_std_near_one | Std de X_train ≈ 1 tras z-score | ✅ Pass |
| ID-16 | test_val_test_scaled | Val y Test normalizados con scaler de train | ✅ Pass |
| ID-17 | test_normalizer_has_reports | Normalizer incluye reportes KS pre/post | ✅ Pass |

> Resultado global: 41 de 41 pruebas ejecutadas exitosamente (100%).

### Anexo R: Resultados de la validación LOSO — Modelo de lesión (R5)

Descripción del protocolo de validación cruzada Leave-One-Subject-Out y resumen estructural de los resultados por fold.

> Fuente: Módulo de validación — `src/injury/validate.py`

| Propiedad | Valor |
|-----------|-------|
| Método | Leave-One-Subject-Out (LOSO) |
| N.° de folds totales | 16 (uno por participante) |
| Folds omitidos | Folds con n_injuries = 0 en participante reservado |
| Motivo de omisión | ROC-AUC indefinido para un único valor de clase |
| Indicador de omisión | skipped=True en LOSOResult |
| Datos excluidos de train | Filas con participant_id de prefijo "synth_*" |
| Datos excluidos de test | Ídem — solo datos reales evaluados |
| Normalización | Por fold: PowerTransformer ajustado en fold-train |
| Augmentación | Por fold: SMOTE aplicado en fold-train |
| Métricas reportadas | ROC-AUC, PR-AUC, F1 (media ± std) |

### Anexo S: Código fuente — Sistema de integración (R6)

Script de integración y módulos de soporte alojados en un repositorio de control de versiones.

> Enlace: [src/integration/ — Repositorio GitHub](#) *(insertar enlace al repositorio)*

| Módulo | Archivo | Propósito |
|--------|---------|-----------|
| Configuración | config.py | IntegrationConfig con rutas y configs de R4 y R5 |
| Pipeline | pipeline.py | Orquestador de 4 etapas con reporte de tiempos y métricas |
| Entry point | run_integration.py | CLI (--input-csv, --fatigue-model, --injury-model, --output, -v) |

### Anexo T: Plan de pruebas — Sistema de integración (21 tests)

Catálogo documentado de las 21 pruebas unitarias de integración implementadas con pytest.

> Fuente: Archivos de prueba pytest — `test_integration.py`

Tabla T.1. Catálogo de pruebas de integración (21 tests).

| ID | Clase | Test | Descripción | Resultado |
|----|-------|------|-------------|:---------:|
| I-1 | TestModelLoading | test_loads_both_models | R4 y R5 cargan; normalizer funcional | ✅ Pass |
| I-2 | TestModelLoading | test_fatigue_model_has_attention | Modelo R4 incluye capa TemporalAttention | ✅ Pass |
| I-3 | TestEndToEnd | test_full_pipeline_runs | Pipeline completa 4 etapas sin error | ✅ Pass |
| I-4 | TestEndToEnd | test_dfi_in_range | DFI predicho ∈ [0, 1] | ✅ Pass |
| I-5 | TestEndToEnd | test_injury_probs_in_range | Probabilidades de lesión ∈ [0, 1] | ✅ Pass |
| I-6 | TestEndToEnd | test_output_columns | Columnas de salida correctas | ✅ Pass |
| I-7 | TestEndToEnd | test_no_nan_in_predictions | Sin NaN en predicciones finales | ✅ Pass |
| I-8 | TestHandoffIntegrity | test_merge_produces_dfi_column | dfi_predicted inyectado correctamente | ✅ Pass |
| I-9 | TestHandoffIntegrity | test_merge_preserves_row_count | LEFT JOIN preserva conteo de filas | ✅ Pass |
| I-10 | TestHandoffIntegrity | test_feature_count_matches_config | Features de R5 ≥ 10 columnas esperadas | ✅ Pass |
| I-11 | TestColdStartImputation | test_missing_participant_filled | Participante sin DFI recibe imputación | ✅ Pass |
| I-12 | TestColdStartImputation | test_partial_missing_filled | Sin NaN tras imputación cold-start | ✅ Pass |
| I-13 | TestErrorHandling | test_missing_fatigue_model | RuntimeError con modelo R4 faltante | ✅ Pass |
| I-14 | TestErrorHandling | test_missing_injury_model | RuntimeError con modelo R5 faltante | ✅ Pass |
| I-15 | TestReportCompleteness | test_four_stages | Reporte contiene exactamente 4 etapas | ✅ Pass |
| I-16 | TestReportCompleteness | test_stage_names | Nombres de etapas correctos | ✅ Pass |
| I-17 | TestReportCompleteness | test_durations_positive | Duraciones de etapas ≥ 0 | ✅ Pass |
| I-18 | TestReportCompleteness | test_total_duration | Duración total > 0 | ✅ Pass |
| I-19 | TestReportCompleteness | test_metrics_finite | Métricas de lesión son números finitos | ✅ Pass |
| I-20 | TestReportCompleteness | test_output_csv_exists | CSV de salida escrito en disco | ✅ Pass |
| I-21 | TestReportCompleteness | test_model_paths_recorded | Rutas de modelos registradas en reporte | ✅ Pass |

> Resultado global: 21 de 21 pruebas ejecutadas exitosamente (100%).
