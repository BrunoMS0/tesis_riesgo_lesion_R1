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
│   └── etl/
│       ├── config.py       # Configuración del pipeline (umbrales, rutas)
│       ├── extract.py      # EXTRACT: lectura de PMData (10 fuentes × 16 participantes)
│       ├── transform.py    # TRANSFORM: limpieza, feature eng., estandarización, selección
│       ├── load.py         # LOAD: exportación CSV, TFRecord, tf.data.Dataset
│       └── pipeline.py     # Orquestador Extract → Transform → Load
├── tests/                   # 40 tests (pytest)
│   ├── conftest.py
│   ├── test_extract.py
│   ├── test_transform.py
│   ├── test_load.py
│   └── test_pipeline.py
├── run_pipeline.py          # CLI entry point: python run_pipeline.py -v
└── README.md
```

---

## Requerimientos Completados

### R1 — Feature Engineering Exploratorio
- **Notebook**: `notebooks/R1_Feature_Engineering.ipynb` (47 celdas)
- Análisis exploratorio, ingeniería de features, estandarización Yeo-Johnson, selección de variables
- **Salida**: `notebooks/outputs/dataset_modelado_R1.csv` (2398 × 31)

### R2 — Pipeline ETL de Producción
- **Código**: `src/etl/` (5 módulos)
- Pipeline reproducible: Extract → Transform → Load
- 40 tests unitarios (pytest), todos pasan
- **Ejecución**: `python run_pipeline.py -v`
- **Salidas**:
  - `src/outputs/dataset_modelado_R2.csv` (2398 × 57, todas las features)
  - `src/outputs/train.tfrecord` (1651 examples, 28 features)
  - `src/outputs/val.tfrecord` (295 examples)
  - `src/outputs/test.tfrecord` (452 examples)

### R3 — Dataset Final Curado + Documento Técnico
- **Notebook**: `notebooks/R3_Feature_Engineering_Document.ipynb` (26 celdas)
- Documento técnico de Feature Engineering que define y justifica cada variable creada
- Contenido:
  - Estadísticas descriptivas y distribución de prevalencia de lesiones (3.0%, ratio 1:32)
  - Fórmulas con LaTeX de variables derivadas (ACWR, TRIMP, Sleep Debt, RHR Drift, etc.)
  - Tabla técnica de 29 variables con tipo, fuente, fórmula, normalización y justificación
  - Tratamiento de nulos y outliers
  - Comparación de estandarización Yeo-Johnson vs StandardScaler
  - Análisis de correlación (Spearman) y multicolinealidad
  - PCA: scree plot y varianza acumulada (18 comp. → 85%, 22 → 90%, 28 → 95%)
  - Validación cruzada R1 ↔ R2 (KS test: 28/29 variables equivalentes)
  - Verificación de roundtrip TFRecord (2398/2398 match)
  - 11 referencias bibliográficas
- **Salidas**:
  - `notebooks/outputs/R3_feature_engineering_document.csv`
  - `notebooks/outputs/R3_dataset_specification.csv`
  - `notebooks/outputs/R3_validacion_R1_vs_R2.csv`

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
| Estandarización | Yeo-Johnson (PowerTransformer) |
| Split (por participante) | Train: 11 pids (1,651) / Val: 2 pids (295) / Test: 3 pids (452) |
| Formatos | CSV + TFRecord |
