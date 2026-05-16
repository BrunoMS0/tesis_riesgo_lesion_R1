# Capítulo 4. Construcción de un dataset multimodal para el análisis de fatiga y lesiones en corredores

## 4.1 Introducción

Este capítulo tiene como finalidad presentar los resultados obtenidos para el primer objetivo específico planteado, el cual corresponde a la construcción de un dataset multimodal y procesado para el análisis de la fatiga y las lesiones en corredores, integrando datos de sensores vestibles, registros de entrenamiento y historiales de lesiones.

La construcción de un dataset de calidad constituye un requisito fundamental para cualquier sistema de predicción basado en aprendizaje automático. En el contexto de la predicción de lesiones deportivas, este proceso adquiere particular complejidad debido a la naturaleza heterogénea de las fuentes de datos involucradas: señales fisiológicas capturadas por dispositivos vestibles, registros subjetivos de bienestar y carga de entrenamiento, y eventos clínicos de lesión reportados por los propios atletas.

Para el desarrollo de este objetivo, se trabajó con el dataset público PMData (Ni et al., 2019), el cual contiene datos recopilados de 16 corredores recreacionales durante un período de aproximadamente 5 meses. Los datos provienen de dos fuentes principales: (1) sensores vestibles Fitbit Versa 2, que registran métricas de actividad física, frecuencia cardíaca, sueño y calorías; y (2) la plataforma PMSYS (Athlete Self-Report Measures), mediante la cual los participantes reportaron diariamente su percepción de esfuerzo (sRPE), bienestar subjetivo y eventos de lesión.

El enfoque metodológico adoptado comprende tres fases secuenciales que se reflejan en los tres resultados alcanzados: (1) un análisis exploratorio que permitió formalizar el modelo de datos y definir las características de ingeniería mediante criterios estadísticos y evidencia de la literatura científica; (2) la implementación de un pipeline de software automatizado para la Extracción, Transformación y Carga (ETL) de los datos; y (3) la generación y validación del dataset final curado, listo para ser consumido por los modelos predictivos.

Las herramientas empleadas para el logro de este objetivo incluyen Jupyter Notebook y Python como entorno de análisis exploratorio, Python con las librerías pandas, scikit-learn y scipy para el procesamiento de datos, y TensorFlow Data junto con el formato TFRecord para la serialización y carga eficiente de los datos en los modelos de aprendizaje automático.

A continuación, se presentan los resultados alcanzados que permitieron lograr el cumplimiento del objetivo, a partir de sus descripciones, medios de verificación y el cumplimiento de los indicadores objetivamente verificables.

## 4.2 Resultados Alcanzados

### 4.2.1 Modelo de datos formalizado y conjunto definido de características de ingeniería (R1)

Para el primer resultado alcanzado, se llevó a cabo un análisis exploratorio exhaustivo de los datos crudos provenientes de las 13 fuentes de información disponibles (10 flujos de datos Fitbit y 3 fuentes PMSYS), con el propósito de comprender la estructura, distribución y relaciones entre las variables, y a partir de ello definir un conjunto formalizado de características de ingeniería (features) respaldadas por la literatura científica en ciencias del deporte.

El análisis exploratorio se desarrolló en un notebook de Jupyter compuesto por 47 celdas ejecutables. El proceso integró los datos a granularidad diaria, resultando en un dataset consolidado de 2,398 observaciones correspondientes a los 16 participantes del estudio. La variable objetivo identificada fue `is_injured`, una variable binaria que indica la presencia o ausencia de lesión en un día determinado, con una prevalencia del 3.0% (72 días con lesión de 2,398 observaciones totales). El notebook completo se encuentra disponible en el Anexo B.

Ingeniería de características. A partir de la revisión de la literatura en ciencias del deporte y fisiología del ejercicio, se definieron siete familias de variables derivadas que capturan dimensiones clave del riesgo de lesión. Cada variable fue diseñada con una formulación explícita, ventanas temporales fundamentadas y respaldo bibliográfico. La Tabla 4.1 presenta un resumen de las principales características creadas.

Tabla 4.1. Características de ingeniería derivadas y su fundamento bibliográfico.

| Variable | Fórmula | Ventana | Fundamento |
|----------|---------|---------|------------|
| Carga aguda (*acute_load_7d*) | Media móvil de *session_load* | 7 días | Gabbett (2016) |
| Carga crónica (*chronic_load_28d*) | Media móvil de *session_load* | 28 días | Gabbett (2016) |
| ACWR | *acute_load_7d* / *chronic_load_28d* | Derivada | Gabbett (2016) |
| TRIMP | Σ(*zona_HR* × *peso_zona*), pesos: {0.5, 1.0, 2.0, 3.0} | Diaria | Edwards (1993) |
| TRIMP acumulado (*trimp_7d_sum*) | Suma móvil de TRIMP | 7 días | Edwards (1993) |
| Deuda de sueño (*sleep_debt*) | *sleep_7d_avg* − *minutesAsleep_t* | 7 días | Milewski et al. (2014) |
| Deriva de FC reposo (*rhr_drift*) | *resting_hr_t* − *rhr_baseline_7d* | 7 días | Buchheit (2013) |
| Variabilidad FC (*rhr_variability_7d*) | Desviación estándar de *resting_hr* | 7 días | Buchheit (2013) |
| Puntaje de bienestar (*wellness_score*) | Media de (fatiga, ánimo, disposición, calidad de sueño) | Diaria | Saw et al. (2015) |
| Ratio de actividad (*active_ratio*) | *minutos_activos* / (*activos* + *sedentarios*) | Diaria | Diseño propio |

La Acute:Chronic Workload Ratio (ACWR) merece particular atención, ya que constituye uno de los indicadores más utilizados en la literatura para la predicción de lesiones deportivas. Según Gabbett (2016), valores de ACWR entre 0.8 y 1.3 representan una "zona segura", mientras que valores superiores a 1.5 se asocian con un riesgo elevado de lesión. De manera complementaria, el Training Impulse (TRIMP) cuantifica la carga fisiológica del entrenamiento mediante la ponderación del tiempo en cada zona de frecuencia cardíaca según los pesos propuestos por Edwards (1993). La deuda de sueño se fundamenta en los hallazgos de Milewski et al. (2014), quienes demostraron que atletas con menos de 8 horas de sueño presentan un riesgo 1.7 veces mayor de sufrir lesiones. El documento técnico completo con la descripción detallada de las 33 variables, sus fórmulas, fuentes de datos, método de normalización y justificación se presenta en el Anexo C.

Análisis estadístico. Se llevaron a cabo cuatro análisis estadísticos complementarios para fundamentar la selección de variables:

En primer lugar, se realizaron tests de normalidad de Shapiro-Wilk sobre las 56 variables numéricas del dataset. Los resultados evidenciaron que la totalidad de las variables (56 de 56) rechazaron la hipótesis nula de normalidad (p < 0.05). Este hallazgo fundamentó la elección de métodos no paramétricos para los análisis subsiguientes. Las estadísticas descriptivas completas y los resultados de los tests de normalidad se encuentran en el Anexo D.

En segundo lugar, se evaluó la correlación de cada variable con la variable objetivo mediante el coeficiente de correlación de Spearman, apropiado dada la no normalidad de los datos. De las 56 variables analizadas, 28 presentaron una correlación estadísticamente significativa con "is_injured" (p < 0.05). Las variables con mayor correlación incluyeron "minutesAwake" (ρ = −0.170, p = 4.8×10⁻¹⁷), "restlessness" (ρ = −0.152, p = 6.9×10⁻¹⁴), "overall_score" (ρ = +0.145, p = 1.1×10⁻¹²) y "duration_score" (ρ = +0.141, p = 3.6×10⁻¹²).

En tercer lugar, se identificaron 9 pares de variables con multicolinealidad elevada (|ρ| > 0.90). Para cada par, se retuvo la variable con mayor correlación absoluta con la variable objetivo. Por ejemplo, del par "session_load" ↔ "duration_min" (ρ = 0.99), se conservó "session_load" por presentar mayor correlación con el target (|ρ| = 0.051 vs. 0.038).

En cuarto lugar, se realizó un test de Kruskal-Wallis para comparar las distribuciones de cada variable entre los grupos lesionado y no lesionado, identificando las variables con mayor capacidad discriminante. Adicionalmente, se realizó un Análisis de Componentes Principales (PCA), donde los tres primeros componentes capturan el 45.22% de la varianza total, dominados por métricas de carga de entrenamiento (PC1), sueño y recuperación (PC2) y actividad (PC3). Los resultados completos de los análisis de correlación, multicolinealidad y PCA se encuentran en el Anexo E.

Selección de variables. El criterio de selección adoptado fue un enfoque conservador de consenso triple: una variable se descarta únicamente si los tres métodos coinciden en su baja relevancia — esto es, si su correlación de Spearman con el target es |ρ| < 0.02, su correlación de Pearson es |r| < 0.02 y su p-valor en el test ANOVA F es superior a 0.05. Este enfoque minimiza el riesgo de eliminar variables potencialmente informativas y garantiza que la reducción de dimensionalidad respete múltiples perspectivas estadísticas.

Como medio de verificación de este resultado, se elaboró un documento técnico de Feature Engineering que define y justifica cada variable creada, con su formulación matemática y su relevancia fundamentada en la literatura científica. Dicho documento se presenta en el Anexo C. Como indicador objetivamente verificable, el documento fue revisado y aprobado al 100% por el asesor de tesis.

### 4.2.2 Pipeline de software para la Extracción, Transformación y Carga (ETL) de datos (R2)

Para el segundo resultado alcanzado, se diseñó e implementó un pipeline ETL modular y reproducible que automatiza la integración, limpieza, ingeniería de características y exportación de los datos provenientes de las múltiples fuentes del estudio PMData. El pipeline fue implementado en Python y se encuentra alojado en un repositorio de control de versiones (GitHub).

Arquitectura del pipeline. El pipeline sigue una arquitectura de tres etapas secuenciales — Extract, Transform y Load — orquestadas por un módulo central que emite reportes estructurados con tiempos de ejecución, conteos de filas y columnas, y metadatos por cada etapa. La Figura 4.1 ilustra el flujo de datos a través de las tres etapas y sus sub-etapas.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE ETL (R2)                              │
│                                                                         │
│  ┌──────────┐    ┌─────────────────────────────────┐    ┌────────────┐ │
│  │          │    │          TRANSFORM               │    │            │ │
│  │ EXTRACT  │───▶│  Clean → Engineer → Select       │───▶│    LOAD    │ │
│  │          │    │                                   │    │            │ │
│  │ 10 Fitbit│    │ • Nulos y duplicados              │    │ • CSV      │ │
│  │ 3 PMSYS  │    │ • 8 features derivadas            │    │ • TFRecord │ │
│  │ ×16 part.│    │ • Selección estadística           │    │ • tf.data  │ │
│  └──────────┘    └─────────────────────────────────┘    └────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

Figura 4.1. Arquitectura del pipeline ETL.

Etapa de Extracción (Extract). Esta etapa se encarga de leer las fuentes de datos crudos de cada uno de los 16 participantes, los cuales comprenden 10 archivos Fitbit (actividad diaria, zonas de frecuencia cardíaca, sueño, ejercicio, entre otros) y 3 archivos PMSYS (lesiones, percepción de esfuerzo y bienestar), en formatos CSV y JSON. La estrategia de integración realiza un *outer join* por las claves "(participant_id, date)", consolidando todos los flujos en un único DataFrame a granularidad diaria. Los participantes con archivos faltantes se registran en el log sin interrumpir la ejecución del pipeline.

Etapa de Transformación (Transform). Esta etapa constituye el núcleo del procesamiento y se compone de tres sub-etapas ejecutadas secuencialmente:

- Clean (Limpieza): Se aplican cuatro reglas deterministas. Primero, las variables de evento ("is_injured", "session_load", "perceived_exertion", "duration_min") se rellenan con cero, bajo el supuesto de que la ausencia de registro equivale a la ausencia de evento. Segundo, se eliminan las columnas con más del 60% de valores nulos, preservando las variables protegidas (identificadores y eventos). Tercero, se realiza una imputación por participante en tres pasos: forward-fill, backward-fill y relleno con la mediana global, respetando así las trayectorias individuales de cada corredor. Cuarto, se eliminan registros duplicados por la clave "(participant_id, date)".

- Engineer (Ingeniería de características): Se calculan las 8 variables derivadas definidas en R1, incluyendo las cargas aguda y crónica con ventanas de 7 y 28 días respectivamente, la ACWR, el TRIMP con ponderación por zonas de frecuencia cardíaca (pesos: zona inferior = 0.5, zona 1 = 1.0, zona 2 = 2.0, zona 3 = 3.0), la deuda de sueño, la deriva y variabilidad de la frecuencia cardíaca en reposo, el puntaje de bienestar y el ratio de actividad.

- Select (Selección de variables): Se aplica el filtro de multicolinealidad (umbral |ρ| > 0.90) y el criterio de consenso triple descrito en el resultado R1. Esta sub-etapa genera un reporte de selección de variables que documenta cada decisión de inclusión o exclusión.

Una decisión de diseño fundamental en esta etapa fue la exclusión intencional de la normalización del pipeline ETL. Dado que cada modelo downstream aplica su propio normalizador ajustado exclusivamente sobre datos de entrenamiento — MinMaxScaler para el modelo de fatiga (R4) y Yeo-Johnson con estandarización z-score para el modelo de lesión (R5) —, incluir una normalización previa en el ETL habría generado una doble normalización e introducido un riesgo de fuga de datos (*data leakage*).

Etapa de Carga (Load). Esta etapa exporta el dataset procesado en dos formatos complementarios: (1) un archivo CSV ("dataset_etl_output.csv") destinado a la inspección y análisis por parte de los investigadores, y (2) tres archivos TFRecord ("train.tfrecord", "val.tfrecord", "test.tfrecord") optimizados para el entrenamiento eficiente con TensorFlow. La partición de datos se realiza por participante — no por fila — asignando el 70% de los participantes al conjunto de entrenamiento, el 15% al de validación y el 15% al de prueba. Esta estrategia garantiza que ningún dato de un mismo corredor aparezca en más de un conjunto, previniendo así la fuga de datos entre particiones.

Plan de pruebas. Se diseñó e implementó un plan de pruebas compuesto por 37 tests unitarios organizados en cuatro módulos, cubriendo cada etapa del pipeline con datos sintéticos generados automáticamente (3 participantes simulados, 60 días de observaciones). La Tabla 4.2 presenta la distribución de pruebas por etapa.

Tabla 4.2. Distribución de pruebas unitarias del pipeline ETL.

| Módulo | Etapa | N.° de tests | Aspectos verificados |
|--------|-------|:------------:|----------------------|
| test_extract | Extract | 9 | Presencia de columnas, manejo de participantes faltantes, tipos de fecha, parseo JSON, agregación diaria |
| test_transform | Transform | 13 | Tratamiento de nulos, eliminación de columnas con alta nulidad, imputación, deduplicación, existencia y rangos de features derivadas, selección de variables |
| test_load | Load | 9 | Exportación CSV, splits de tf.data, formas de batch, roundtrip TFRecord, dimensionalidad |
| test_pipeline | Pipeline | 6 | Estructura del reporte, existencia de archivos, datasets creados, duración positiva, conteo de features |
| Total | | 37 | |

Como medio de verificación de este resultado, el código fuente del pipeline se encuentra alojado en un repositorio GitHub (Anexo F), y el plan de pruebas documentado con la descripción de cada test, la etapa que verifica y su resultado se presenta en el Anexo G. Como indicador objetivamente verificable, el 100% de las 37 pruebas unitarias definidas se ejecutan exitosamente, como se evidencia en la salida de ejecución del framework de testing pytest.

### 4.2.3 Dataset final curado y listo para el modelado (R3)

Para el tercer resultado alcanzado, se generó el dataset final curado mediante la ejecución del pipeline ETL (R2) y se elaboró un informe de calidad de datos que verifica el cumplimiento de la estructura definida en el documento de Feature Engineering (R1).

Descripción del dataset final. El dataset producido contiene 2,398 filas y 49 columnas, correspondientes a las observaciones diarias de los 16 participantes del estudio. Todas las variables presentan 0% de valores nulos tras el proceso de limpieza e imputación. La variable objetivo "is_injured" es binaria con una prevalencia del 3.0%, lo que implica un ratio de desbalance de clases de aproximadamente 1:32 (72 días con lesión frente a 2,326 días sin lesión). En cuanto al formato, el dataset se distribuye en un archivo CSV para análisis directo y en tres archivos TFRecord particionados por participante para el entrenamiento de los modelos de aprendizaje automático.

Un aspecto relevante del diseño del dataset es que los datos se exportan sin normalización previa. Esta decisión responde a que cada modelo predictivo desarrollado en el Capítulo 5 aplica su propio esquema de normalización ajustado exclusivamente sobre los datos de entrenamiento: el modelo de fatiga (R4) utiliza MinMaxScaler con rango [0, 1], mientras que el modelo de predicción de lesión (R5) emplea una transformación Yeo-Johnson seguida de estandarización z-score. Ambos normalizadores se ajustan (fit) únicamente sobre el conjunto de entrenamiento y se aplican (transform) a los conjuntos de validación y prueba, previniendo así la fuga de datos.

Validación cruzada R1 vs. R2. Para garantizar que el pipeline automatizado (R2) reproduce fielmente los cálculos realizados en la fase exploratoria (R1), se ejecutó un test de Kolmogorov-Smirnov (KS) comparando las distribuciones de cada variable entre ambas versiones del dataset. Los resultados, presentados en la Tabla 4.3, demuestran que 28 de las 29 variables evaluadas presentan distribuciones estadísticamente equivalentes (KS = 0.0, p = 1.0), confirmando la fidelidad del pipeline.

Tabla 4.3. Resultados del test de Kolmogorov-Smirnov — validación R1 vs. R2 (resumen).

| Resultado | N.° de variables | KS estadístico | p-valor |
|-----------|:----------------:|:--------------:|:-------:|
| Equivalentes | 28 | 0.000 | 1.000 |
| Divergencia menor | 1 (rhr_variability_7d) | 0.281 | < 0.001 |

La única variable con divergencia fue "rhr_variability_7d", cuya discrepancia se atribuye a una diferencia en el parámetro "min_periods" de la ventana móvil entre la implementación exploratoria (R1) y la de producción (R2). Esta diferencia no afecta la validez del dataset para el modelado, dado que la variable se mantiene dentro de rangos estadísticamente razonables. Los resultados completos del test de validación se presentan en el Anexo I.

Informe de calidad de datos. Se elaboró un informe de calidad (Data Quality Report) que documenta las propiedades del dataset final, incluyendo: dimensiones, tipos de datos, distribución de valores únicos por variable, estadísticas de asimetría (skewness) y curtosis, y la verificación de ausencia de valores nulos. Este informe se presenta en el Anexo H. Adicionalmente, el diccionario de datos completo con la descripción de cada variable, su tipo, conteo de nulos, valores únicos y ejemplos se encuentra en el Anexo J.

Como medio de verificación de este resultado, se presenta el informe de calidad de datos en el Anexo H. Como indicador objetivamente verificable, se confirma que: (1) el dataset cumple con el 100% de la estructura y las variables definidas en el documento de Feature Engineering (R1), conteniendo las 33 variables documentadas; y (2) la variable objetivo "is_injured" presenta un 0% de valores nulos en las 2,398 observaciones.

## 4.3 Dataset primario para el sistema predictivo: Runner Dataset (Löwdal et al., 2021)

Con el propósito de superar la limitación de tamaño muestral del dataset PMData (16 atletas, 9 con lesiones), se adoptó como dataset principal para el entrenamiento y validación del sistema predictivo el Runner Dataset publicado por Löwdal et al. (2021), disponible públicamente en el repositorio asociado a su trabajo. A diferencia del PMData — que emplea sensores vestibles Fitbit Versa 2 y la plataforma PMSYS —, el Runner Dataset proviene de relojes GPS de running (Garmin/Polar) y cuestionarios subjetivos de entrenamiento diario, lo que lo hace más representativo del perfil de corredor recreacional con tecnología accesible.

Descripción del dataset. El Runner Dataset comprende datos longitudinales de 74 corredores seguidos durante períodos de entre 7 meses y 2 años, con 583 eventos de lesión documentados y una prevalencia del 1.36% (583 eventos positivos de 42,766 observaciones totales). Los datos se presentan en formato tabular de 73 columnas mediante un esquema de *7-day window wide format*: cada fila representa el resumen de 7 días anteriores de entrenamiento y bienestar para un atleta en una fecha dada. Las 10 variables base disponibles son: "total km" (distancia total diaria), "km Z3-4" (km en zona umbral anaeróbico), "km Z5-T1-T2" (km en zona de alta intensidad/intervalos), "km sprinting" (km de sprint), "nr. sessions" (número de sesiones), "strength training" (sesión de fuerza, binaria), "hours alternative" (horas de entrenamiento cruzado), "perceived exertion", "perceived recovery" y "perceived trainingSuccess". Cada variable tiene 7 columnas representando los días D-7 a D-1 (sufijos vacío, ".1", ".2", ..., ".6"). La variable objetivo "injury" es ya prospectiva (lesión que ocurrirá en los próximos días), por lo que no requiere desplazamiento temporal adicional.

Proceso ETL aplicado. Se implementaron los módulos "src/runner/extract.py", "src/runner/transform.py" y "src/runner/dataset.py" para procesar el Runner Dataset. El proceso de ingeniería de features genera 18 variables derivadas, descritas en la Tabla 4.4.

Tabla 4.4. Features derivadas del Runner Dataset.

| Feature | Descripción | Analogía PMData |
|---------|-------------|-----------------|
| `acute_load_7d` | Suma de km de los últimos 7 días | `acute_load_7d` |
| `chronic_load_28d` | Media rolling 28 días reconstruida por atleta | `chronic_load_28d` |
| `acwr` | `acute_load_7d / (chronic_load_28d / 4)`, clipado a [0, 5.0] | `acwr` |
| `high_intensity_km_7d` | Suma de km en zonas de alta intensidad (Z3-Z4 + Z5-T1-T2) 7 días | `trimp_7d_sum` (proxy) |
| `session_load_proxy` | `acute_load_7d × mean_perceived_exertion` | `session_load` |
| `nr_sessions_7d` | Número de sesiones de entrenamiento 7 días | — |
| `nr_rest_days_7d` | Días de descanso (exertion = -0.01) | — |
| `km_sprint_7d` | Total km sprint 7 días | — |
| `strength_days_7d` | Sesiones de fuerza 7 días | — |
| `alt_hours_7d` | Horas de entrenamiento alternativo (cruzado) 7 días | — |
| `mean_perceived_exertion` | Media esfuerzo percibido (excl. días de descanso) | `fatigue` |
| `mean_perceived_recovery` | Media recuperación percibida | `readiness` |
| `mean_perceived_success` | Media éxito de entrenamiento percibido | `mood` |
| `wellness_score` | `(mean_recovery + mean_success) / 2` | `wellness_score` |
| `recent_exertion` | Esfuerzo del día D-1 | — |
| `recent_recovery` | Recuperación del día D-1 | — |
| `recent_success` | Éxito del día D-1 | — |
| `recent_km` | km del día D-1 | — |

Partición estratificada. La partición se realizó por atleta con estratificación por presencia de lesión (semilla 42, proporciones 70/10/20): 51 atletas de entrenamiento (43 con lesiones, 29,574 filas), 7 de validación (6 con lesiones, 3,451 filas) y 16 de prueba (14 con lesiones, 9,741 filas). Esta estrategia garantiza que los tres conjuntos tengan proporciones similares de atletas con lesiones y previene la fuga de datos.

Features del Modelo M1 (Regresor de Fatiga). El sistema predictivo incorpora un modelo intermedio M1 que estima la percepción de recuperación del atleta (`perceived_recovery.6`, día D-1) a partir exclusivamente de features objetivas del GPS — sin acceso a variables subjetivas. La Tabla 4.5 describe las 10 features de entrada y la variable objetivo del modelo M1.

Tabla 4.5. Features del Modelo M1 — Regresor de Fatiga (10 entradas GPS → recuperación percibida).

| Feature GPS (entrada) | Descripción | Ventana |
|-----------------------|-------------|---------|
| `acute_load_7d` | Suma de km de los últimos 7 días | 7 días |
| `chronic_load_28d` | Media rolling de km en 28 días | 28 días |
| `acwr` | Ratio agudo:crónico clipado a [0, 5] | Derivada |
| `high_intensity_km_7d` | Km totales en zonas de alta intensidad (Z3-Z4 + Z5-T1-T2) | 7 días |
| `nr_sessions_7d` | Número de sesiones de entrenamiento | 7 días |
| `nr_rest_days_7d` | Días de descanso (exertion = -0.01) | 7 días |
| `km_sprint_7d` | Km de sprint | 7 días |
| `strength_days_7d` | Sesiones de entrenamiento de fuerza | 7 días |
| `alt_hours_7d` | Horas de entrenamiento alternativo (cruzado) | 7 días |
| `recent_km` | Km del día D-1 | 1 día |
| `recent_recovery` | Recuperación percibida D-1 (objetivo) | D-1 |

La separación entre features objetivas (entrada) y subjetivas (objetivo) garantiza que el modelo M1 sea operable sin retroalimentación del atleta durante la inferencia, lo que lo hace aplicable en tiempo real con datos de reloj GPS únicamente.

El dataset procesado se exporta a "src/outputs/runner_dataset_processed.csv" (42,766 filas × 21 columnas) y el modelo entrenado se guarda en "src/outputs/rf_runner_model.pkl".

## 4.4 Discusión

Se llevaron a cabo tres resultados esperados para lograr el objetivo de construir un dataset multimodal para el análisis de fatiga y lesiones en corredores. Para el primer resultado (R1), se desarrolló un modelo de datos formalizado mediante un análisis exploratorio que definió 33 variables — 16 originales provenientes de las fuentes de datos y 17 características derivadas mediante ingeniería de features —, cada una respaldada por evidencia de la literatura científica en ciencias del deporte. Los análisis estadísticos realizados (normalidad, correlación, multicolinealidad y PCA) proporcionaron la fundamentación cuantitativa necesaria para las decisiones de selección de variables.

Para el segundo resultado (R2), se implementó un pipeline ETL automatizado que traduce las decisiones del análisis exploratorio en un proceso reproducible, modular y verificable. El pipeline fue validado mediante un plan de pruebas de 37 tests unitarios, todos ejecutados exitosamente. Para el tercer resultado (R3), se generó y validó el dataset final mediante un informe de calidad y una comparación estadística con los cálculos exploratorios, confirmando la fidelidad del proceso automatizado.

Tres decisiones de diseño merecen particular reflexión. En primer lugar, la normalización fue deliberadamente excluida del pipeline ETL. Esta decisión, aunque poco convencional en pipelines ETL tradicionales, responde a la necesidad de que cada modelo predictivo ajuste su propio normalizador exclusivamente sobre datos de entrenamiento, evitando la doble normalización y la fuga de datos. En segundo lugar, la selección de variables adoptó un criterio conservador de consenso triple (Spearman, Pearson y ANOVA F-test), lo cual reduce el riesgo de descartar variables potencialmente informativas a costa de retener un espacio de features ligeramente mayor. En tercer lugar, la partición de datos se realizó por participante en lugar de por observación individual, lo cual simula un escenario realista de despliegue en el que el modelo debe predecir para corredores no vistos durante el entrenamiento.

El trabajo desarrollado presenta ciertas limitaciones que se reconocen explícitamente. El tamaño de la muestra de PMData (16 participantes) restringe la generalización de los resultados del sistema exploratorio y limita la potencia estadística de la validación por participante. El desbalance de clases (3% de prevalencia de lesión en PMData; 1.36% en el Runner Dataset) constituye un reto significativo para el entrenamiento de modelos predictivos, lo cual es abordado mediante técnicas de sobremuestreo en el Capítulo 5. Asimismo, las variables de bienestar subjetivo de PMData (fatiga, ánimo, dolor, estrés) no lograron normalidad tras la transformación Yeo-Johnson, lo cual se atribuye a su naturaleza ordinal (escala Likert 1-7); sin embargo, la literatura respalda que esta condición no invalida su uso en modelos de regresión logística y redes neuronales.

Ambos datasets construidos en este capítulo — el dataset PMData curado (R3, Sección 4.2.3) y el Runner Dataset procesado (Sección 4.3) — constituyen los insumos directos para el Capítulo 5, donde se desarrollarán los modelos predictivos de fatiga (R4, M1) y riesgo de lesión (R5, M2), así como el sistema integrado que los orquesta en secuencia.

---

## Anexos del Capítulo 4

### Anexo C: Notebook R1 — Feature Engineering Exploratorio

Notebook de Jupyter con 47 celdas ejecutables que documenta el análisis exploratorio completo, incluyendo la integración de datos, cálculo de features derivadas, análisis estadísticos y selección de variables.

> Enlace: [R1_Feature_Engineering.ipynb — Repositorio GitHub](#) *(insertar enlace al repositorio)*

### Anexo D: Documento técnico de Feature Engineering

Tabla técnica que define y justifica las 33 variables del dataset, incluyendo: nombre de la variable, tipo (original/derivada), fuente de datos, fórmula matemática, método de normalización y justificación basada en la literatura.

> Fuente: Documento técnico de Feature Engineering — `R3_feature_engineering_document.csv`

| Variable | Tipo | Fuente | Fórmula | Normalización | Justificación |
|----------|------|--------|---------|---------------|---------------|
| is_injured | Target | PMSYS injury.csv | Binaria: 1 si lesión, 0 si no | — | Variable objetivo |
| perceived_exertion | Original | PMSYS srpe.csv | Escala Borg sRPE (1–10) | Yeo-Johnson | Carga subjetiva de entrenamiento |
| fatigue | Original | PMSYS wellness.csv | Escala Likert (1–7) | Yeo-Johnson | Saw et al. (2015) |
| mood | Original | PMSYS wellness.csv | Escala Likert (1–7) | Yeo-Johnson | Indicador psicológico |
| readiness | Original | PMSYS wellness.csv | Escala Likert (1–7) | Yeo-Johnson | Disposición para entrenar |
| soreness | Original | PMSYS wellness.csv | Escala Likert (1–7) | Yeo-Johnson | Dolor muscular percibido |
| overall_score | Original | Fitbit sleep_score.csv | Índice 0–100 | Yeo-Johnson | Calidad de sueño global |
| deep_sleep_in_minutes | Original | Fitbit sleep_score.csv | Minutos | Yeo-Johnson | Recuperación fisiológica |
| restlessness | Original | Fitbit sleep_score.csv | Índice | Yeo-Johnson | Inquietud durante el sueño |
| minutesAwake | Original | Fitbit sleep.json | Minutos despierto en cama | Yeo-Johnson | Calidad de sueño |
| timeInBed | Original | Fitbit sleep.json | Minutos totales en cama | Yeo-Johnson | Oportunidad de sueño |
| hr_zone_below | Original | Fitbit HR zones | Minutos en zona inferior | Yeo-Johnson | Actividad de baja intensidad |
| hr_zone_1 | Original | Fitbit HR zones | Minutos en zona 1 | Yeo-Johnson | Actividad aeróbica base |
| exercise_calories | Original | Fitbit exercise.json | Calorías | Yeo-Johnson | Gasto energético |
| acute_load_7d | Derivada | Calculada | mean(session_load, 7d) | Yeo-Johnson | Gabbett (2016) |
| chronic_load_28d | Derivada | Calculada | mean(session_load, 28d) | Yeo-Johnson | Gabbett (2016) |
| acwr | Derivada | Calculada | acute / chronic | Yeo-Johnson | Gabbett (2016), zona segura 0.8–1.3 |
| trimp | Derivada | Calculada | Σ(zona × peso) | Yeo-Johnson | Edwards (1993) |
| trimp_7d_sum | Derivada | Calculada | sum(trimp, 7d) | Yeo-Johnson | Carga fisiológica acumulada |
| sleep_debt | Derivada | Calculada | sleep_7d_avg − minutesAsleep | Yeo-Johnson | Milewski et al. (2014) |
| rhr_drift | Derivada | Calculada | resting_hr − baseline_7d | Yeo-Johnson | Buchheit (2013) |
| rhr_variability_7d | Derivada | Calculada | std(resting_hr, 7d) | Yeo-Johnson | Buchheit (2013) |
| wellness_score | Derivada | Calculada | mean(fatiga, ánimo, disposición, sueño) | Yeo-Johnson | Saw et al. (2015) |
| active_ratio | Derivada | Calculada | activos / (activos + sedentarios) | Yeo-Johnson | Balance actividad/inactividad |

*(Tabla parcial — el documento completo contiene las 33 variables)*

### Anexo E: Estadísticas descriptivas y tests de normalidad

Resumen de las estadísticas descriptivas (media, desviación estándar, coeficiente de variación, asimetría y curtosis) y los resultados del test de Shapiro-Wilk para las 56 variables numéricas del dataset pre-transformación.

> Fuentes:
> - Estadísticas descriptivas del dataset — `estadistica_descriptiva.csv`
> - Resultados del test de Shapiro-Wilk — `test_normalidad.csv`

Test de Shapiro-Wilk (muestra de resultados):

| Variable | Estadístico W | p-valor | Asimetría | Normal (α=0.05) |
|----------|:------------:|:-------:|:---------:|:---------------:|
| is_injured | 0.225 | 1.5×10⁻⁷³ | 5.51 | No |
| hr_zone_3 | 0.407 | 1.1×10⁻⁶⁶ | 4.32 | No |
| rhr_variability_7d | 0.811 | 9.9×10⁻⁶⁴ | 2.34 | No |
| acwr | 0.832 | 4.5×10⁻⁵⁴ | 1.28 | No |
| trimp | 0.963 | 2.1×10⁻²⁰ | 0.58 | No |

*Resultado: 56 de 56 variables rechazaron la hipótesis nula de normalidad.*

### Anexo F: Análisis de correlación, multicolinealidad y PCA

F.1 Correlación Spearman con la variable objetivo — `correlacion_target.csv`

| Variable | ρ Spearman | p-valor | Significativa |
|----------|:----------:|:-------:|:------------:|
| minutesAwake | −0.170 | 4.8×10⁻¹⁷ | Sí |
| restlessness | −0.152 | 6.9×10⁻¹⁴ | Sí |
| overall_score | +0.145 | 1.1×10⁻¹² | Sí |
| duration_score | +0.141 | 3.6×10⁻¹² | Sí |
| hr_zone_below | −0.131 | 1.2×10⁻¹⁰ | Sí |
| trimp | −0.093 | 4.6×10⁻⁰⁶ | Sí |
| acwr | +0.072 | 4.5×10⁻⁰⁴ | Sí |

*28 de 56 variables presentaron correlación significativa (p < 0.05).*

F.2 Pares multicolineales (|ρ| > 0.90) — `multicolinealidad.csv`

| Par de variables | ρ Pearson | Variable retenida | Criterio |
|------------------|:---------:|-------------------|----------|
| session_load ↔ duration_min | 0.992 | session_load | Mayor |ρ(target)| |
| steps ↔ distance | 0.989 | steps | Mayor |ρ(target)| |
| minutesAsleep ↔ timeInBed | 0.977 | timeInBed | Mayor |ρ(target)| |
| steps_7d_sum ↔ distance_7d_sum | 0.988 | steps_7d_sum | Mayor |ρ(target)| |
| sedentary_minutes ↔ active_ratio | −0.901 | sedentary_minutes | Mayor |ρ(target)| |

*9 pares identificados; se eliminó 1 variable de cada par.*

F.3 Análisis PCA — Varianza explicada — `pca_varianza.csv`

| Componente | Varianza explicada (%) | Varianza acumulada (%) |
|:----------:|:---------------------:|:---------------------:|
| PC1 | 22.18 | 22.18 |
| PC2 | 12.38 | 34.56 |
| PC3 | 10.66 | 45.22 |
| PC4 | 7.91 | 53.13 |
| PC5 | 6.73 | 59.86 |
| ... | ... | ... |
| PC10 | 3.63 | 77.49 |
| PC18 | — | 85.00 |
| PC22 | — | 90.00 |
| PC28 | — | 100.00 |

### Anexo G: Código fuente del pipeline ETL

Código fuente completo del pipeline ETL, compuesto por 5 módulos Python alojados en un repositorio de control de versiones:

> Enlace: [src/etl/ — Repositorio GitHub](#) *(insertar enlace al repositorio)*

| Módulo | Archivo | Propósito |
|--------|---------|-----------|
| Configuración | config.py | Parámetros centralizados (umbrales, rutas, ventanas temporales) |
| Extracción | extract.py | Lectura de PMData (10 fuentes Fitbit + 3 PMSYS × 16 participantes) |
| Transformación | transform.py | Limpieza, ingeniería de features, selección de variables |
| Carga | load.py | Exportación CSV, TFRecord y tf.data.Dataset |
| Orquestador | pipeline.py | Ejecuta Extract → Transform → Load con reportes de cada etapa |

### Anexo H: Plan de pruebas del pipeline ETL

Catálogo documentado de las 37 pruebas unitarias implementadas con el framework pytest. Cada prueba utiliza datos sintéticos generados automáticamente (3 participantes, 60 días).

> Fuente: Archivos de prueba pytest — `test_extract.py`, `test_transform.py`, `test_load.py`, `test_pipeline.py`

Tabla H.1. Catálogo de pruebas — Etapa Extract (9 tests).

| ID | Test | Descripción | Resultado |
|----|------|-------------|:---------:|
| E-1 | test_returns_dataframe | Extract retorna un DataFrame válido | ✅ Pass |
| E-2 | test_required_columns | Columnas obligatorias presentes (participant_id, date, is_injured) | ✅ Pass |
| E-3 | test_participant_count | Se detectan los 3 participantes sintéticos | ✅ Pass |
| E-4 | test_date_column_type | Columna date es datetime64 | ✅ Pass |
| E-5 | test_no_duplicate_dates | Sin duplicados por (participant_id, date) | ✅ Pass |
| E-6 | test_fitbit_columns_present | Columnas de Fitbit presentes (steps, calories, etc.) | ✅ Pass |
| E-7 | test_pmsys_columns_present | Columnas PMSYS presentes (fatigue, mood, etc.) | ✅ Pass |
| E-8 | test_missing_participant_handled | Participante faltante no interrumpe el pipeline | ✅ Pass |
| E-9 | test_daily_aggregation | Datos agregados a granularidad diaria | ✅ Pass |

Tabla H.2. Catálogo de pruebas — Etapa Transform (13 tests).

| ID | Test | Descripción | Resultado |
|----|------|-------------|:---------:|
| T-1 | test_event_vars_no_nulls | Variables de evento sin NaN tras limpieza | ✅ Pass |
| T-2 | test_high_null_columns_dropped | Columnas con >60% nulos eliminadas | ✅ Pass |
| T-3 | test_no_numeric_nulls | Sin nulos en columnas numéricas tras imputación | ✅ Pass |
| T-4 | test_no_duplicates | Sin duplicados (participant_id, date) | ✅ Pass |
| T-5 | test_acwr_exists | Columna ACWR presente | ✅ Pass |
| T-6 | test_acwr_finite | ACWR finito donde chronic_load > 0 | ✅ Pass |
| T-7 | test_trimp_exists | Columna TRIMP presente | ✅ Pass |
| T-8 | test_sleep_debt_exists | Columna Sleep Debt presente | ✅ Pass |
| T-9 | test_rhr_drift_exists | Columna RHR Drift presente | ✅ Pass |
| T-10 | test_wellness_score_range | Wellness Score ∈ [1, 7] | ✅ Pass |
| T-11 | test_features_reduced_or_equal | Features seleccionadas ≤ features totales | ✅ Pass |
| T-12 | test_target_preserved | Variable is_injured preservada tras selección | ✅ Pass |
| T-13 | test_returns_transform_result | Pipeline completo retorna TransformResult | ✅ Pass |

Tabla H.3. Catálogo de pruebas — Etapa Load (9 tests).

| ID | Test | Descripción | Resultado |
|----|------|-------------|:---------:|
| L-1 | test_file_exists | CSV exportado existe en disco | ✅ Pass |
| L-2 | test_row_count_matches | Filas del CSV coinciden con DataFrame | ✅ Pass |
| L-3 | test_returns_bundle | build_tf_datasets retorna DatasetBundle | ✅ Pass |
| L-4 | test_splits_cover_all_rows | Train + Val + Test = total de filas | ✅ Pass |
| L-5 | test_batch_shapes | Batch shapes correctos (features, labels) | ✅ Pass |
| L-6 | test_files_exist | 3 archivos TFRecord escritos en disco | ✅ Pass |
| L-7 | test_roundtrip_row_count | Roundtrip TFRecord: filas leídas = escritas | ✅ Pass |
| L-8 | test_feature_dimensionality | Dimensionalidad de features correcta en TFRecord | ✅ Pass |
| L-9 | test_returns_load_result | Función load() retorna LoadResult completo | ✅ Pass |

Tabla H.4. Catálogo de pruebas — Pipeline completo (6 tests).

| ID | Test | Descripción | Resultado |
|----|------|-------------|:---------:|
| P-1 | test_returns_report | Pipeline retorna PipelineReport | ✅ Pass |
| P-2 | test_three_stages | Reporte contiene exactamente 3 etapas | ✅ Pass |
| P-3 | test_csv_exists | Archivo CSV de salida existe | ✅ Pass |
| P-4 | test_tf_datasets_created | Datasets de TensorFlow creados correctamente | ✅ Pass |
| P-5 | test_positive_duration | Duración de cada etapa > 0 | ✅ Pass |
| P-6 | test_features_positive | Conteo de features > 0 | ✅ Pass |

> Resultado global: 37 de 37 pruebas ejecutadas exitosamente (100%).

### Anexo I: Informe de calidad del dataset (Data Quality Report)

Informe que resume las propiedades del dataset final producido por el pipeline ETL.

> Fuente: Especificación del dataset final — `R3_dataset_specification.csv`

| Propiedad | Valor |
|-----------|-------|
| Filas totales | 2,398 |
| Columnas totales | 49 |
| Participantes | 16 (p01–p16) |
| Rango temporal | ~5 meses (Nov 2019 – Mar 2020) |
| Variables numéricas | 47 (float64) |
| Variables identificadoras | 2 (participant_id: str, date: datetime) |
| Variable objetivo | is_injured (binaria) |
| Prevalencia de lesión | 3.0% (72 / 2,398) |
| Valores nulos totales | 0 |
| Nulos en variable objetivo | 0 (0%) |
| Registros duplicados | 0 |
| Formato CSV | dataset_etl_output.csv |
| Formato TFRecord | train.tfrecord, val.tfrecord, test.tfrecord |
| Split Train | 70% de participantes |
| Split Validación | 15% de participantes |
| Split Test | 15% de participantes |

### Anexo J: Validación cruzada R1 vs. R2 — Test de Kolmogorov-Smirnov

Resultados del test KS comparando las distribuciones de cada variable entre el dataset generado por el notebook exploratorio (R1) y el dataset producido por el pipeline ETL (R2).

> Fuente: Resultados de validación cruzada KS — `R3_validacion_R1_vs_R2.csv`

| Variable | KS estadístico | p-valor | Equivalencia |
|----------|:-------------:|:-------:|:------------:|
| perceived_exertion | 0.000 | 1.000 | ✅ Equivalente |
| fatigue | 0.000 | 1.000 | ✅ Equivalente |
| mood | 0.000 | 1.000 | ✅ Equivalente |
| readiness | 0.000 | 1.000 | ✅ Equivalente |
| soreness | 0.000 | 1.000 | ✅ Equivalente |
| overall_score | 0.000 | 1.000 | ✅ Equivalente |
| acwr | 0.000 | 1.000 | ✅ Equivalente |
| trimp | 0.000 | 1.000 | ✅ Equivalente |
| sleep_debt | 0.000 | 1.000 | ✅ Equivalente |
| rhr_drift | 0.000 | 1.000 | ✅ Equivalente |
| wellness_score | 0.000 | 1.000 | ✅ Equivalente |
| ... | ... | ... | ... |
| rhr_variability_7d | 0.281 | < 0.001 | ⚠️ Divergencia menor |

*28 de 29 variables: distribución equivalente. 1 variable con divergencia menor atribuida a diferencia en el parámetro min_periods.*

### Anexo K: Diccionario de datos del dataset final

Diccionario que describe cada variable del dataset final, incluyendo su nombre, tipo de dato, conteo de valores nulos, número de valores únicos y ejemplo de valor.

> Fuente: Diccionario de variables del dataset — `diccionario_datos.csv`

| Variable | Tipo | Nulos | Valores únicos | Ejemplo |
|----------|------|:-----:|:--------------:|---------|
| participant_id | str | 0 | 16 | p01 |
| date | datetime | 0 | 153 | 2019-11-01 |
| is_injured | float64 | 0 | 2 | 0.0 |
| perceived_exertion | float64 | 0 | 11 | 7.0 |
| fatigue | float64 | 0 | 7 | 2.0 |
| session_load | float64 | 0 | 127 | 210.0 |
| overall_score | float64 | 0 | 56 | 82.0 |
| deep_sleep_in_minutes | float64 | 0 | 146 | 67.0 |
| acwr | float64 | 0 | 1,842 | 1.15 |
| trimp | float64 | 0 | 1,556 | 797.0 |
| wellness_score | float64 | 0 | 25 | 4.25 |

*(Tabla parcial — el diccionario completo contiene las 49 variables)*
