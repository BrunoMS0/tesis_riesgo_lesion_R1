
---

## 📊 **Tabla 2 – Objetivo 1 (O1)**

**Objetivo 1:**
Construir un dataset multimodal y procesado para el análisis de la fatiga y las lesiones en corredores, integrando datos de sensores vestibles, registros de entrenamiento y historiales de lesiones.

### 🔹 Resultados

* **R1:** Un modelo de datos formalizado y un conjunto definido de características de ingeniería (features).

  * **Medio de verificación:** Documento técnico de Feature Engineering que define y justifica cada variable creada, con su formulación y relevancia basada en la literatura.
  * **Indicador objetivamente verificable:** El documento es revisado y aprobado al 100% por el asesor de tesis.

* **R2:** Pipeline de software para la Extracción, Transformación y Carga (ETL) de datos de múltiples fuentes.

  * **Medio de verificación:** Código fuente del pipeline de ETL alojado en un repositorio de control de versiones y plan de pruebas documentado que verifique cada etapa del pipeline.
  * **Indicador objetivamente verificable:** El 100% de las pruebas unitarias y de integración definidas para el pipeline se ejecutan exitosamente.

* **R3:** Un dataset final, curado, normalizado y listo para el modelado.

  * **Medio de verificación:** Informe de calidad de datos (Data Quality Report) que resume las propiedades del dataset.
  * **Indicador objetivamente verificable:** El dataset cumple con el 100% de la estructura y las variables definidas en el documento de Feature Engineering (R1) y presenta un 0% de valores nulos en la variable objetivo.

---

## 📊 **Tabla 3 – Objetivo 2 (O2)**

### 🔹 Resultados

* **R4:** Un modelo de Deep Learning implementado para el análisis de la fatiga.

  * **Medio de verificación:** Código fuente de la implementación del modelo de Deep Learning.
  * **Indicador objetivamente verificable:** El informe técnico es revisado y aprobado al 100% por el asesor de tesis y un experto en Machine Learning.

* **R5:** Un modelo de Machine Learning implementado para la predicción del riesgo de lesión.

  * **Medio de verificación:** Código fuente de la implementación del modelo de Machine Learning.
  * **Indicador objetivamente verificable:** El informe técnico es revisado y aprobado al 100% por el asesor de tesis y un experto en Machine Learning.

* **R6:** Un sistema predictivo integrado que orquesta la ejecución de los dos modelos en secuencia.

  * **Medio de verificación:** Script de integración que demuestra el flujo secuencial y plan de pruebas de integración documentado.
  * **Indicador objetivamente verificable:** El script de integración se ejecuta de manera exitosa y sin errores en un conjunto de datos de prueba.

---

---

## 📊 **Tabla 3 – Objetivo 3 (O3)**

### 🔹 Resultados

* **R7:** El modelo de análisis de la fatiga entrenado y validado.

  * **Medio de verificación:** Informe de evaluación que presenta las métricas de rendimiento del modelo de fatiga.
  * **Indicador objetivamente verificable:** El error de predicción del modelo (RMSE) es inferior a 2.0 puntos en la escala RPE (0-10).

* **R8:** El modelo de predicción de riesgo de lesión entrenado y validado.

  * **Medio de verificación:** Informe de evaluación que presenta las métricas de rendimiento del modelo de riesgo de lesión.
  * **Indicador objetivamente verificable:** El rendimiento del clasificador de riesgo de lesión supera un AUC-ROC de 0.70 en el conjunto de validación

* **R9:** Un informe técnico de rendimiento y métricas de validación del sistema integrado.

  * **Medio de verificación:** Documento del informe técnico que consolida los resultados de la validación y las métricas de rendimiento finales de los modelos predictivos implementados.
  * **Indicador objetivamente verificable:** El informe técnico es revisado y aprobado al 100% por el asesor de tesis y un experto en Machine Learning.

---