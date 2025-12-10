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

## Ejecución rápida del notebook principal
1) Activar el entorno y seleccionar el kernel `tesis_riesgo_R1` en Jupyter.
2) Abrir `notebooks/R1_Feature_Engineering.ipynb`.
3) Ejecutar las celdas en orden (usa "Restart & Run All" si quieres una ejecución limpia).

Si encuentras errores sobre columnas faltantes o nombres de campos, revisa las primeras celdas donde se definen `CRITICAL_VARS` y las rutas de datos. Contacta si quieres que arregle alguna celda.

---

## Contacto / Siguientes pasos
- Si quieres, puedo:
  - Generar y commitear un `requirements.txt` exacto desde el entorno `.venv` que tienes localmente (necesito que lo ejecutes y me confirmes o puedo añadir uno de ejemplo).
  - Preparar un script `setup_env.ps1` que automatice la creación del venv, instalación e instalación del kernel.

Dime qué prefieres y lo añado.
