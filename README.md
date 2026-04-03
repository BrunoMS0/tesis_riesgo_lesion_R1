# Tesis - Riesgo de Lesión (R1)

Instrucciones para preparar el entorno de ejecución, ejecutar el notebook y buenas prácticas de Git.

## Resumen
Este repositorio contiene los notebooks y scripts del proyecto. Los datos crudos están fuera del repo (`pmdata/` está en `.gitignore`). Estas instrucciones describen cómo crear un entorno reproducible en Windows (PowerShell) usando `venv` o `conda`.

---

## Requisitos previos
- Python 3.10 o 3.11 instalado y accesible como `python`.
- Opcional: `conda` (Anaconda/Miniconda) si prefieres usar entornos conda.

## Opción A — Entorno ligero con `venv` (recomendado)

1) Crear y activar el entorno virtual (desde la raíz del repo):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

2) Actualizar pip y herramientas básicas:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

3) Instalar dependencias mínimas (recomendado):

```powershell
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyterlab notebook ipykernel
```

4) Registrar el kernel de Jupyter para usar este entorno desde notebooks:

```powershell
python -m ipykernel install --user --name tesis_riesgo_R1 --display-name "tesis_riesgo_R1 (.venv)"
```

5) (Opcional) Generar `requirements.txt` con las versiones instaladas:

```powershell
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add requirements.txt"
```

6) Ejecutar JupyterLab / Notebook:

```powershell
jupyter lab
# o
jupyter notebook
```

---

## Opción B — Entorno con `conda`

1) Crear y activar un entorno conda:

```powershell
conda create -n tesis_riesgo_R1 python=3.11 -y
conda activate tesis_riesgo_R1
```

2) Instalar dependencias básicas:

```powershell
conda install pandas numpy scipy scikit-learn matplotlib seaborn jupyter -y
pip install ipykernel
python -m ipykernel install --user --name tesis_riesgo_R1 --display-name "tesis_riesgo_R1 (conda)"
```

3) Exportar entorno (opcional):

```powershell
conda env export --name tesis_riesgo_R1 > environment.yml
```

---

## Notas sobre `pmdata/` y Git
- `pmdata/` contiene los datos crudos y debe permanecer fuera del repositorio remoto. Verifica que está en `.gitignore`.

Si `pmdata/` ya fue agregado por error, ejecuta:

```powershell
git rm --cached -r pmdata
git commit -m "Remove pmdata from index and rely on .gitignore"
git push origin main
```

Si al hacer `git push` obtienes un rechazo porque el remoto contiene trabajo (mensaje: "Updates were rejected because the remote contains work that you do not have locally"), usa uno de los flujos siguientes:

- Integrar cambios del remoto (recomendado):

```powershell
git fetch origin
git pull --rebase origin main
# resolver conflictos si aparecen, luego
git push -u origin main
```

- Forzar sobrescritura del remoto (peligroso — perderás historial remoto):

```powershell
git push --force origin main
```

---

## Archivo `requirements.txt` sugerido
Si no quieres usar `pip freeze` ahora, crea un `requirements.txt` de ejemplo con estas líneas:

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
jupyterlab
notebook
ipykernel
```

---

## Estructura del Proyecto

```
tesis_riesgo_lesion_R1/
├── notebooks/
│   ├── R1_Feature_Engineering.ipynb      # R1: Análisis exploratorio y Feature Engineering
│   ├── R3_Feature_Engineering_Document.ipynb  # R3: Documento técnico del dataset
│   └── outputs/                          # Artefactos R1 + R3 (CSVs, tablas)
├── src/
│   ├── etl/
│   │   ├── config.py       # Configuración del pipeline (umbrales, rutas)
│   │   ├── extract.py      # EXTRACT: lectura de PMData (10 fuentes × 16 participantes)
│   │   ├── transform.py    # TRANSFORM: limpieza, feature eng., selección de variables
│   │   ├── load.py         # LOAD: exportación CSV, TFRecord, tf.data.Dataset
│   │   └── pipeline.py     # Orquestador Extract → Transform → Load
│   ├── fatigue/             # R4: Modelo de fatiga (Bi-LSTM + TemporalAttention)
│   │   ├── config.py       # Hiperparámetros y configuración del modelo
│   │   ├── dataset.py      # Carga y windowing de datos
│   │   ├── model.py        # Arquitectura Bi-LSTM + TemporalAttention
│   │   ├── train.py        # Entrenamiento con early stopping
│   │   ├── evaluate.py     # Evaluación (RMSE, MAE en escala DFI)
│   │   ├── predict.py      # Predicción de DFI por participante
│   │   └── pipeline.py     # Orquestador R4 completo
│   ├── injury/              # R5: Modelo de predicción de lesión
│   │   ├── config.py       # Configuración (C_GRID, SMOTE ratio, etc.)
│   │   ├── dataset.py      # Carga, normalización, split por participante
│   │   ├── augment.py      # SMOTE (target_ratio=0.3)
│   │   ├── model.py        # LogisticRegression + Baseline
│   │   ├── train.py        # Entrenamiento + grid search de C
│   │   ├── evaluate.py     # Evaluación (AUC-ROC, F1, threshold en validación)
│   │   ├── validate.py     # LOSO cross-validation
│   │   └── pipeline.py     # Orquestador R5 completo (6 stages)
│   └── integration/         # R6: Pipeline integrado R4 → R5
│       ├── config.py
│       └── pipeline.py     # Orquestador R6 (fatigue → injury handoff)
├── tests/                   # 137 tests (pytest)
│   ├── conftest.py
│   ├── test_extract.py
│   ├── test_transform.py
│   ├── test_load.py
│   ├── test_pipeline.py
│   ├── test_fatigue_*.py
│   ├── test_injury_*.py
│   └── test_integration.py
├── run_pipeline.py          # CLI: python run_pipeline.py -v
├── run_fatigue.py           # CLI: python run_fatigue.py
├── run_injury.py            # CLI: python run_injury.py
└── run_integration.py       # CLI: python run_integration.py
```

---

## Requerimientos Completados

### R1 — Feature Engineering Exploratorio
- **Notebook**: `notebooks/R1_Feature_Engineering.ipynb` (47 celdas)
- Análisis exploratorio, ingeniería de features, selección de variables
- **Salida**: `notebooks/outputs/dataset_modelado_R1.csv` (2398 × 31)

### R2 — Pipeline ETL de Producción
- **Código**: `src/etl/` (5 módulos)
- Pipeline reproducible: Extract → Transform (Clean → Engineer → Select) → Load
- La normalización **no** se realiza en el ETL; cada modelo downstream aplica su propio normalizador (MinMaxScaler en R4, Yeo-Johnson en R5) fit-on-train-only
- **Ejecución**: `python run_pipeline.py -v`
- **Salidas**:
  - `src/outputs/dataset_etl_output.csv` (2398 × 57, todas las features sin normalizar)
  - `src/outputs/train.tfrecord`, `val.tfrecord`, `test.tfrecord`

### R3 — Dataset Final Curado + Documento Técnico
- **Notebook**: `notebooks/R3_Feature_Engineering_Document.ipynb` (26 celdas)
- Documento técnico de Feature Engineering que define y justifica cada variable creada
- **Nota**: Este notebook fue creado cuando el ETL aún incluía estandarización Yeo-Johnson. Las tablas de normalización reflejan ese estado previo. Si se regeneran datos, el notebook debe re-ejecutarse.
- **Salidas**:
  - `notebooks/outputs/R3_feature_engineering_document.csv`
  - `notebooks/outputs/R3_dataset_specification.csv`
  - `notebooks/outputs/R3_validacion_R1_vs_R2.csv`

### R4 — Modelo de Fatiga (Bi-LSTM + TemporalAttention)
- **Código**: `src/fatigue/` (7 módulos)
- Arquitectura: Bi-LSTM con TemporalAttention, ventana de 14 días
- Normalización: MinMaxScaler [0,1] fit-on-train
- Target: DFI = (5 − fatigue) / 4 ∈ [0,1]
- Métrica: RMSE ≤ 0.15 en escala DFI
- **Ejecución**: `python run_fatigue.py`

### R5 — Modelo de Predicción de Lesión
- **Código**: `src/injury/` (8 módulos)
- Modelo: Logistic Regression con grid search C ∈ {0.01, 0.1, 1.0, 10.0}
- Normalización: Yeo-Johnson + z-score fit-on-train
- Oversampling: SMOTE con target_ratio=0.3
- Evaluación: AUC-ROC (primaria), AUC-PR, F1; threshold optimizado sobre validación
- Validación: LOSO cross-validation (folds 0-injury excluidos de métricas, reportados por separado)
- **Ejecución**: `python run_injury.py`

### R6 — Pipeline Integrado (R4 → R5)
- **Código**: `src/integration/` (2 módulos)
- Orquesta la predicción de DFI (R4) → inyección como feature → predicción de lesión (R5)
- **Ejecución**: `python run_integration.py`

---

## Ejecución Rápida

### Notebook R1 (Exploratorio)
1. Activar el entorno y seleccionar el kernel `tesis_riesgo_R1` en Jupyter.
2. Abrir `notebooks/R1_Feature_Engineering.ipynb`.
3. Ejecutar las celdas en orden ("Restart & Run All").

### Pipeline R2 (ETL)
```powershell
.\.venv\Scripts\Activate.ps1
python run_pipeline.py -v
```

### Modelo R4 (Fatiga)
```powershell
python run_fatigue.py
```

### Modelo R5 (Lesión)
```powershell
python run_injury.py
```

### Pipeline Integrado R6 (R4 → R5)
```powershell
python run_integration.py
```

### Notebook R3 (Documento Técnico)
1. Ejecutar primero el pipeline R2 para generar los archivos de salida.
2. Abrir `notebooks/R3_Feature_Engineering_Document.ipynb`.
3. Ejecutar todas las celdas en orden.

### Tests
```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests/ -v
```

---

## Dataset Final

| Aspecto | Valor |
|---------|-------|
| Filas | 2,398 |
| Features (completo CSV) | 54 |
| Features (seleccionadas, TFRecord) | 28 |
| Target | `is_injured` (binaria, 3.0% prevalencia) |
| Participantes | 16 |
| Normalización ETL | Ninguna (cada modelo normaliza internamente) |
| Normalización R4 | MinMaxScaler [0,1] fit-on-train |
| Normalización R5 | Yeo-Johnson + z-score fit-on-train |
| Split (por participante) | Train: 11 pids / Val: 2 pids / Test: 3 pids |
| Formatos | CSV + TFRecord |
