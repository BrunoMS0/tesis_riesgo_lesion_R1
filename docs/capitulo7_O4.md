# Capítulo 7. Interpretabilidad y validación comparativa del sistema predictivo de lesiones en corredores

## 7.1 Introducción

Este capítulo presenta los resultados obtenidos para el cuarto y último objetivo específico de la investigación: realizar un análisis de interpretabilidad del modelo de predicción de lesiones y validar comparativamente su rendimiento frente a líneas base. El objetivo se materializa en dos resultados: (R10) un análisis de la importancia de características mediante el método SHAP (SHapley Additive exPlanations), que explica qué variables GPS contribuyen más al riesgo de lesión predicho por el modelo; y (R11) una validación comparativa del sistema integrado M1 → M2 frente a un clasificador ingenuo, al modelo GPS-exclusivo (Condición A), al modelo con datos subjetivos reales (Condición C) y al clasificador de regresión logística entrenado sobre PMData.

La interpretabilidad de un sistema de predicción de riesgo de lesión es, en sí misma, un requisito de uso clínico-deportivo. Un entrenador o fisioterapeuta que recibe una alerta de riesgo elevado necesita saber qué variables del atleta desencadenaron esa predicción para poder tomar decisiones de carga o recuperación fundamentadas. El método SHAP, derivado de la teoría de valores de Shapley de la teoría de juegos cooperativos, asigna a cada característica una contribución marginal exacta sobre la predicción, manteniendo propiedades de eficiencia, simetría y linealidad que lo diferencian de métodos de importancia basados en impureza (Lundberg y Lee, 2017). Su aplicación a modelos de bosque aleatorio mediante `TreeExplainer` garantiza el cómputo exacto de los valores SHAP sin necesidad de aproximaciones de muestreo del espacio de características.

La validación comparativa, por su parte, cierra el ciclo de evaluación de la tesis al contextualizar el rendimiento del sistema frente a alternativas de referencia. El protocolo Leave-One-Athlete-Out (LOAO) de 74 folds, empleado en el Capítulo 6, asegura que la comparación entre condiciones se efectúe sobre exactamente los mismos participantes de prueba en cada fold, lo cual permite aplicar la prueba de Wilcoxon de rangos signados para determinar si las diferencias observadas en el AUC-ROC son estadísticamente significativas.

Las herramientas empleadas para el logro de este objetivo incluyen SHAP (SHapley Additive exPlanations) para el análisis de interpretabilidad, scikit-learn para el clasificador DummyClassifier de referencia, scipy para la prueba de hipótesis de Wilcoxon, y Matplotlib junto con la API de plots de SHAP para la generación de visualizaciones diagnósticas.

## 7.2 Resultados Alcanzados

### 7.2.1 Análisis de interpretabilidad SHAP sobre el modelo M2 Condición A (R10)

El primer resultado del cuarto objetivo corresponde al análisis de interpretabilidad del modelo de predicción de lesión (M2) entrenado exclusivamente sobre las 10 características GPS objetivas que componen la Condición A. Este análisis permite identificar cuáles son las variables de carga de entrenamiento que más contribuyen a la predicción del riesgo de lesión y en qué dirección operan, respondiendo a la pregunta de por qué el modelo clasifica una observación determinada como de alto riesgo.

**Protocolo de análisis.** El análisis SHAP se ejecutó sobre el modelo Random Forest de la Condición A (200 árboles, `max_features=sqrt`, `class_weight=balanced`, entrenado con SMOTE sobre 34,316 observaciones de 51 atletas). Se empleó `shap.TreeExplainer`, que calcula valores SHAP exactos aprovechando la estructura de árbol sin muestreo del espacio de características. Para hacer computacionalmente viable el análisis sobre el conjunto de prueba (9,741 observaciones, 16 atletas), se extrajo una muestra estratificada de 2,000 observaciones preservando la proporción de clase de lesión (25 positivos, 1,975 negativos), práctica estándar en publicaciones de interpretabilidad de modelos de aprendizaje automático. Los valores SHAP se calculan para la clase positiva (lesión = 1), de modo que valores positivos indican contribución al incremento del riesgo predicho y valores negativos a su reducción.

**Ranking de importancia global.** La importancia global de cada característica se calculó como el valor absoluto medio de sus valores SHAP sobre las 2,000 observaciones de la muestra de análisis. La Tabla 7.1 presenta el ranking completo de las 10 características GPS.

Tabla 7.1. Ranking de importancia global SHAP para el modelo M2 Condición A (n = 2,000 observaciones, muestra estratificada del conjunto de prueba).

| Rango | Característica | Media \|SHAP\| | Interpretación |
|-------|---------------|----------------|----------------|
| 1 | `acwr` | 0.1956 | Razón de carga aguda:crónica (7d/28d) |
| 2 | `chronic_load_28d` | 0.1103 | Carga acumulada de distancia en 28 días |
| 3 | `nr_sessions_7d` | 0.0368 | Número de sesiones de carrera en 7 días |
| 4 | `high_intensity_km_7d` | 0.0357 | Kilómetros de alta intensidad en 7 días |
| 5 | `acute_load_7d` | 0.0270 | Carga acumulada de distancia en 7 días |
| 6 | `strength_days_7d` | 0.0253 | Días de entrenamiento de fuerza en 7 días |
| 7 | `nr_rest_days_7d` | 0.0246 | Días de descanso en 7 días |
| 8 | `recent_km` | 0.0223 | Kilómetros de carrera del día anterior |
| 9 | `alt_hours_7d` | 0.0151 | Horas de actividades de baja intensidad en 7 días |
| 10 | `km_sprint_7d` | 0.0104 | Kilómetros de sprint en 7 días |

El análisis revela una jerarquía interpretable desde la perspectiva biomecánica y fisiológica del deporte. La razón de carga aguda:crónica (`acwr`) emerge como la variable dominante con una importancia media de 0.1956, valor que prácticamente duplica la importancia de la segunda característica (`chronic_load_28d` = 0.1103). Esta jerarquía es coherente con la teoría de la carga de entrenamiento: el ACWR captura el desequilibrio entre la carga reciente (7 días) y la capacidad adaptada (28 días), y valores superiores a 1.5 están sistemáticamente asociados al riesgo de lesión en la literatura (Gabbett, 2016; Hulin et al., 2016). La segunda posición de `chronic_load_28d` sugiere que el modelo considera igualmente relevante el nivel absoluto de carga acumulada, no solo el desequilibrio relativo: atletas con cargas crónicas elevadas presentan un perfil de riesgo diferente a atletas con cargas bajas incluso ante igual ACWR.

Las posiciones 3–7 corresponden a variables que operan como modificadores del riesgo: `nr_sessions_7d` y `high_intensity_km_7d` representan la densidad y la calidad del estímulo de entrenamiento, mientras que `strength_days_7d` y `nr_rest_days_7d` capturan la estrategia de recuperación y periodización del atleta. La presencia de `recent_km` en posición 8 confirma que el historial de carga inmediata del día anterior contiene señal predictiva incremental sobre las ventanas de 7 y 28 días.

La brecha de importancia entre las dos primeras características y el resto (0.1956 y 0.1103 vs. ≤0.0368) indica que el modelo concentra su capacidad discriminativa en el balance de carga aguda:crónica y en el nivel de acondicionamiento crónico, empleando el resto de variables como señales de ajuste de segundo orden. Este patrón es robusto: se reproduce de forma consistente en la visualización de beeswarm (Figura 7.1), donde `acwr` y `chronic_load_28d` dominan el espectro de dispersión.

**Visualizaciones diagnósticas.** Se generaron cuatro tipos de visualizaciones SHAP para el informe técnico de interpretabilidad:

*(a) Beeswarm global (shap_beeswarm.png).* Cada punto representa una observación de la muestra de análisis. El eje horizontal indica el valor SHAP (contribución al log-odds de lesión), y el color codifica el valor de la característica (azul = bajo, rojo = alto). Para `acwr`, los valores altos (rojos) se desplazan hacia la derecha (incrementan el riesgo predicho), mientras que los valores bajos (azules) se desplazan hacia la izquierda (reducen el riesgo). Este patrón confirma que el modelo captura la relación dosis-respuesta no lineal: valores de ACWR superiores a ~1.3 generan contribuciones SHAP positivas y crecientes. Para `chronic_load_28d`, el patrón es invertido: valores bajos (baja capacidad aeróbica adaptada) incrementan el riesgo, reflejando la vulnerabilidad de atletas desacondicionados ante cargas agudas similares.

*(b) Bar plot global (shap_bar.png).* Presenta el ranking de importancia media absoluta de la Tabla 7.1 en formato de gráfico de barras horizontales, facilitando la comparación visual directa entre características. Este gráfico constituye la visualización de resumen del IOV del resultado R10.

*(c) Dependence plots para top-3 características (shap_dependence_acwr.png, shap_dependence_chronic_load_28d.png, shap_dependence_nr_sessions_7d.png).* Cada gráfico muestra la relación entre el valor de la característica (eje x), el valor SHAP asociado (eje y) y la interacción con la segunda característica de mayor correlación (color). Los gráficos de dependencia permiten identificar umbrales de riesgo y efectos de interacción: el gráfico de `acwr` revela un umbral aproximado en ACWR ≈ 1.3, a partir del cual los valores SHAP se vuelven consistentemente positivos, y un efecto de interacción con `chronic_load_28d` donde atletas con baja carga crónica exhiben valores SHAP más altos para el mismo nivel de ACWR.

*(d) Waterfall plots para 3 casos de alto riesgo (shap_waterfall_case1.png, shap_waterfall_case2.png, shap_waterfall_case3.png).* Cada gráfico presenta la explicación SHAP individual para una observación de verdadero positivo (observación con lesión real) de alta probabilidad predicha (probabilidades de 0.65, 0.73 y 0.83 respectivamente). El gráfico parte del valor base (probabilidad media del modelo, base_value ≈ 0.50) y descompone la desviación hacia la probabilidad final en contribuciones individuales de cada característica. Estos casos ilustran cómo el modelo integra múltiples señales de riesgo simultáneamente: en los tres casos, `acwr` elevado y `chronic_load_28d` bajo aparecen como los impulsores primarios de la predicción positiva.

**Indicador objetivamente verificable.** El análisis SHAP identifica y rankea las 10 variables con mayor impacto predictivo en el modelo de lesión, presentando los valores de importancia media absoluta, las direcciones de efecto y las visualizaciones diagnósticas. El IOV queda **cumplido**: se identifican y rankean las 10 variables GPS con su importancia cuantificada (Tabla 7.1) y se generan 7 artefactos visuales de interpretabilidad almacenados en `src/outputs/plots/`. El informe de análisis SHAP completo con todas las figuras se presenta en el Anexo V.

---

### 7.2.2 Validación comparativa del modelo integrado M1 → M2 (R11)

El segundo resultado del cuarto objetivo corresponde a la validación comparativa del sistema integrado M1 → M2 frente a múltiples referencias, mediante el protocolo LOAO de 74 folds implementado en el Capítulo 6. Este análisis permite cuantificar la contribución de cada componente del sistema y situar su rendimiento en el contexto de modelos de referencia clínicamente relevantes.

**Diseño del experimento comparativo.** La comparación se estructura en cuatro ejes:

1. **Vs. clasificador ingenuo (DummyClassifier):** Cuantifica la ganancia total del sistema sobre la predicción aleatoria, constituyendo el umbral mínimo de utilidad clínica.
2. **Vs. Condición A (GPS-solo, sin predicción de fatiga M1):** Aísla la contribución marginal del modelo M1 al sistema integrado, al comparar M1 → M2 (Condición B) con un M2 entrenado directamente sobre las mismas 10 características GPS.
3. **Vs. Condición C (GPS + variables subjetivas reales):** Contextualiza el coste de reemplazar variables subjetivas (RPE, bienestar reportado) por la predicción del modelo M1, cuantificando la penalización por eliminar la recopilación manual de datos.
4. **Vs. LR-PMData (regresión logística sobre dataset de prueba de concepto):** Evalúa la ganancia de generalización al pasar del dataset de prueba de concepto (16 atletas, PMData) al dataset primario de la tesis (74 atletas, Runner Dataset).

Todos los modelos del Runner Dataset fueron evaluados mediante el protocolo LOAO completo de 74 folds (Leave-One-Athlete-Out), donde en cada fold un atleta es retirado como conjunto de prueba y los 73 restantes conforman el entrenamiento. Las métricas reportadas son el promedio y la desviación estándar del AUC-ROC sobre los folds válidos (folds en los que el atleta de prueba presentó al menos un evento de lesión). El modelo DummyClassifier fue evaluado con el mismo protocolo, generando predicciones de probabilidad uniformemente distribuidas en [0, 1] para cada observación.

**Resultados de la validación comparativa.** La Tabla 7.2 presenta las métricas consolidadas de todos los modelos comparados.

Tabla 7.2. Validación comparativa del sistema M1 → M2 frente a líneas base y condiciones alternativas (protocolo LOAO, Runner Dataset, n = 74 atletas).

| Modelo | AUC-ROC | DE | Folds válidos | Δ vs. base | Δ% vs. base |
|--------|---------|-----|---------------|------------|-------------|
| DummyClassifier LOAO | 0.5009 | ±0.0188 | 63 / 74 | — | — |
| LR PMData (prueba de concepto) | 0.5517 | — | — | +0.0508 | +5.08% |
| **RF-Runner Cond A (GPS-solo)** | **0.9074** | **±0.0777** | **63 / 74** | **+0.4065** | **+40.65%** |
| **RF-Runner Cond B (M1→M2, integrado)** | **0.9034** | **±0.0965** | **63 / 74** | **+0.4025** | **+40.25%** |
| RF-Runner Cond C (GPS+subjetivo real) | 0.9109 | ±0.0877 | 63 / 74 | +0.4100 | +41.00% |
| RF-Runner M2 full (18 características) | 0.9101 | ±0.0891 | 63 / 74 | +0.4092 | +40.92% |

*Nota: La línea base para los cálculos de Δ es el DummyClassifier LOAO (AUC = 0.5009). DE = desviación estándar sobre folds válidos.*

**Resultado 1 — Ganancia vs. clasificador ingenuo (IOV R11).** El sistema integrado M1 → M2 (Condición B) alcanza un AUC-ROC de 0.9034 ± 0.0965 frente al AUC de 0.5009 ± 0.0188 del DummyClassifier, lo que representa una mejora absoluta de **Δ = +0.4025** (mejora relativa del **+40.25%** sobre la base). Este resultado supera ampliamente el umbral mínimo del 5% establecido en el IOV del resultado R11. El IOV queda **cumplido** con un margen de 35.25 puntos porcentuales sobre el criterio de aceptación.

**Resultado 2 — Contribución marginal de M1 (Cond B vs. Cond A).** La comparación entre el sistema integrado (Condición B: AUC = 0.9034) y el modelo GPS-solo (Condición A: AUC = 0.9074) arroja una diferencia de **Δ = −0.0040** (−0.40%). La Condición A exhibe un AUC ligeramente superior al sistema integrado, lo que indica que la predicción de fatiga del modelo M1 no añade señal discriminativa neta sobre las 10 características GPS objetivas en este protocolo de evaluación. Para determinar si esta diferencia es estadísticamente significativa, se aplicó la prueba de Wilcoxon de rangos signados sobre los 63 pares de AUC fold-a-fold disponibles (folds con datos válidos en ambas condiciones).

**Prueba de Wilcoxon (Cond A vs. Cond B).** Los resultados de la prueba estadística se presentan en la Tabla 7.3.

Tabla 7.3. Prueba de Wilcoxon de rangos signados: Condición A (GPS-solo) vs. Condición B (M1→M2), AUC fold-a-fold (n = 63 pares).

| Parámetro | Valor |
|-----------|-------|
| n pares | 63 |
| W (estadístico de Wilcoxon) | 875.0 |
| p-valor (bilateral) | 0.4767 |
| Diferencia media (B − A) | −0.0040 |
| Nivel de significancia α | 0.05 |
| Conclusión | No significativa (p > α) |

Con p = 0.4767 > α = 0.05, no se rechaza la hipótesis nula de que las dos distribuciones de AUC son equivalentes. La diferencia observada de −0.40% entre Cond B y Cond A no es estadísticamente significativa, lo que implica que el modelo M1 no introduce una degradación significativa del rendimiento al reemplazar las variables subjetivas reales, pero tampoco introduce una mejora significativa sobre el modelo GPS-solo.

**Resultado 3 — Coste de eliminar variables subjetivas (Cond B vs. Cond C).** La comparación entre el sistema integrado (Cond B: AUC = 0.9034) y el modelo que emplea variables subjetivas reales (Cond C: AUC = 0.9109) arroja una diferencia de **Δ = −0.0075** (−0.75%). Este valor cuantifica la penalización por reemplazar la recopilación manual diaria de RPE y bienestar por la estimación automática del modelo M1: una pérdida de 0.75 puntos porcentuales de AUC, lo que representa un coste operativo bajo en términos de automatización del flujo de datos de entrada al modelo.

**Resultado 4 — Generalización a dataset primario (RF-Runner vs. LR-PMData).** La comparación entre el mejor modelo sobre el Runner Dataset (Cond A: AUC = 0.9074) y el modelo de regresión logística entrenado sobre PMData (AUC = 0.5517) arroja una diferencia de **Δ = +0.3557** (+35.57%). Esta diferencia refleja la combinación de tres factores: (1) el aumento del conjunto de datos de 16 a 74 atletas con mayor poder estadístico; (2) el cambio de arquitectura de regresión logística a Random Forest con SMOTE; y (3) la incorporación de características de carga de entrenamiento específicas para corredores en lugar de variables de bienestar subjetivo de uso general.

**Indicador objetivamente verificable.** El modelo integrado M1 → M2 demuestra un AUC-ROC de 0.9034 frente al AUC de 0.5009 del DummyClassifier, lo que representa una mejora del 40.25% en la métrica AUC-ROC en comparación con el modelo base. El IOV queda **cumplido**: la mejora observada (40.25%) supera el umbral establecido del 5% con un margen de 35.25 puntos porcentuales. Los resultados completos de la validación comparativa se encuentran en `src/outputs/r11_comparison_results.csv`, el informe de la prueba de Wilcoxon en `src/outputs/wilcoxon_cond_a_vs_b.txt`, y la visualización en el gráfico de barras `src/outputs/plots/r11_auc_comparison.png`, disponibles en el Anexo W.

## 7.3 Discusión

### 7.3.1 Interpretabilidad: dominancia del ACWR y coherencia teórica

El análisis SHAP revela una jerarquía de importancia coherente con la literatura de ciencias del deporte. La dominancia del `acwr` (SHAP medio = 0.1956) como principal predictor de lesión es consistente con los meta-análisis de Gabbett (2016) y Hulin et al. (2016), que identifican el desequilibrio entre carga aguda y crónica como el principal factor de riesgo modificable en corredores. El modelo aprende esta relación sin haber recibido ninguna regla explícita: la señal emerge directamente de los patrones de los datos GPS de 51 atletas durante el entrenamiento.

La segunda posición de `chronic_load_28d` (SHAP medio = 0.1103) es igualmente significativa desde el punto de vista teórico. El nivel de acondicionamiento crónico actúa como denominador del ACWR: un atleta con alta carga crónica puede tolerar cargas agudas elevadas que serían lesivas para un atleta con baja carga crónica ante el mismo numerador. El modelo captura implícitamente esta interacción, como evidencian los gráficos de dependencia, donde valores bajos de `chronic_load_28d` amplifican el valor SHAP de `acwr` para el mismo nivel de razón aguda:crónica.

La brecha entre las dos primeras características y las restantes ocho (importancias de 0.025–0.037 vs. 0.110–0.196) sugiere que el modelo opera con una arquitectura de señal en dos niveles: las variables de carga macro (`acwr`, `chronic_load_28d`) determinan el nivel de riesgo base, mientras que las variables de densidad y calidad del entrenamiento (`nr_sessions_7d`, `high_intensity_km_7d`) y de recuperación (`strength_days_7d`, `nr_rest_days_7d`) ajustan el riesgo en respuesta a características específicas de la semana de entrenamiento.

La posición de `recent_km` (rango 8, SHAP medio = 0.0223) es relevante desde la perspectiva del diseño de la Condición A. Esta variable registra los kilómetros de carrera del día inmediatamente anterior a la observación e integra la señal de carga inmediata que no está completamente capturada por las ventanas de 7 y 28 días. Su presencia entre las 10 variables de la Condición A, junto con su posición moderada en el ranking SHAP, confirma que la decisión de incluirla en el conjunto de características GPS fue correcta: aporta señal incremental sin solaparse con las variables de ventana larga.

### 7.3.2 Contribución marginal de M1: señal real pero no incremental sobre GPS

El resultado más relevante de la validación comparativa es el hallazgo de que la Condición B (sistema integrado M1 → M2, AUC = 0.9034) no supera estadísticamente a la Condición A (GPS-solo, AUC = 0.9074) en el protocolo LOAO (Wilcoxon p = 0.4767). Este resultado, que podría interpretarse superficialmente como un fracaso del modelo M1, es en realidad coherente con el análisis de interpretabilidad SHAP y con la arquitectura del sistema.

La explicación reside en que las 10 características GPS de la Condición A —en particular `acwr` y `chronic_load_28d`— contienen implícitamente buena parte de la señal de fatiga que M1 intenta estimar. El Índice Dinámico de Fatiga (DFI) predicho por M1 es una función de variables de carga GPS, frecuencia cardíaca y patrones de sueño; cuando M2 recibe directamente las variables GPS que son las principales entradas de M1, ya dispone de la información de carga objetiva sin necesidad de pasar por la transformación del modelo de fatiga. La reducción de R² negativa del modelo M1 sobre participantes no vistos (reportada en el Capítulo 6) es consistente con este argumento: M1 no siempre mejora la estimación del esfuerzo percibido sobre participantes heterogéneos, y cuando no lo hace, añadir su salida como característica de M2 introduce ruido en lugar de señal.

Lo que sí queda demostrado por la comparación es que el sistema integrado no degrada significativamente el rendimiento: la diferencia de −0.40% entre Cond B y Cond A es estadísticamente indistinguible de cero (p = 0.4767), y la diferencia de −0.75% entre Cond B y Cond C es igualmente pequeña. Esto implica que el sistema puede operar sin la recopilación manual diaria de RPE y bienestar sin sacrificar capacidad predictiva clínicamente relevante, lo que representa una ventaja operativa real para la implementación del sistema en poblaciones de corredores que no cuentan con infraestructura de reporte subjetivo.

### 7.3.3 Generalización y contextualización frente a PMData

La diferencia de +35.57% en AUC-ROC entre el mejor modelo del Runner Dataset y el modelo de regresión logística sobre PMData no refleja únicamente la calidad de los modelos, sino también la diferencia fundamental en el alcance de los dos datasets. PMData es un dataset de prueba de concepto con 16 participantes y 3.0% de prevalencia de lesión, diseñado para validar la viabilidad del pipeline de ETL y del modelo de riesgo en el Capítulo 6; la regresión logística sobre PMData tiene como función establecer la línea base del sistema integrado, no competir con el modelo principal. El modelo principal de la tesis es el Random Forest LOAO sobre el Runner Dataset, que con 74 atletas y 583 eventos de lesión ofrece el poder estadístico necesario para aprender patrones de riesgo generalizables.

El AUC-ROC de 0.5009 del DummyClassifier LOAO confirma que la tarea de predicción de lesiones es genuinamente difícil: sin información sobre el atleta, la predicción aleatoria produce un AUC cercano a 0.50 por definición. La ganancia de +40.25% que el sistema M1 → M2 logra sobre esta línea base es, por tanto, una medida directa de cuánta información predictiva útil contiene el historial de carga GPS de un atleta. El modelo extrae esa información de forma completamente automática a partir de datos de rastreo GPS rutinariamente recopilados en la actividad de entrenamiento, sin necesidad de evaluaciones clínicas adicionales.

Los 11 folds excluidos del análisis LOAO (atletas sin ningún evento de lesión registrado) son coherentes con la distribución de la población de corredores: no todos los corredores se lesionan durante el período de seguimiento, y en esos casos la métrica AUC-ROC no está definida. La tasa de exclusión del 14.9% (11/74) es aceptable y no sesga los resultados, ya que los folds excluidos corresponden precisamente a los atletas más saludables de la muestra, para quienes el sistema de predicción es igualmente aplicable aunque su desempeño no sea cuantificable con la métrica seleccionada.

### 7.3.4 Limitaciones

El presente análisis presenta tres limitaciones que deben considerarse en la interpretación de los resultados. Primera, la muestra estratificada de 2,000 observaciones utilizada para el cómputo de los valores SHAP introduce variabilidad muestral en el ranking de importancia; aunque el tamaño de muestra es suficiente para identificar las características más relevantes, los rangos entre las posiciones 3–8 (importancias de 0.025–0.037) deben interpretarse con cautela dado su proximidad. Segunda, los valores SHAP se calculan sobre el conjunto de prueba completo (16 atletas), que puede no ser representativo de la distribución de la población general de corredores; la extensión del análisis SHAP al conjunto de entrenamiento o a todos los atletas mediante validación cruzada proporcionaría estimaciones más robustas. Tercera, la comparación estadística (Wilcoxon) opera sobre 63 pares de AUC fold-a-fold y no controla por múltiples comparaciones; en un análisis de confirmación, sería deseable aplicar corrección de Bonferroni o FDR dado el número de comparaciones realizadas.
