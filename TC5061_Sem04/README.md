# Experimentacion reproducible con MLflow

Esta carpeta migra a un script el flujo del notebook
`baseline/TC5061_Notebook_Baseline_VargasAnuar.ipynb`. El modelo predice la
contratacion de un deposito con UCI Bank Marketing. Se elimina `duration` para
evitar fuga de informacion y se registra el pipeline completo, incluido el
preprocesamiento.

## Requisitos

- Python 3.12
- Conexion a internet en la primera ejecucion si no existe el archivo de datos

El script descarga automaticamente UCI Bank Marketing en `data/` cuando es
necesario.

## Instalacion y ejecucion

Desde `TC5061_Sem04`, en macOS o Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

En Windows PowerShell, la activacion del entorno se realiza con:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

El script funciona sin argumentos. Sus opciones se consultan con
`python train.py --help`. Los hiperparametros principales son `--c`,
`--class-weight` y `--max-iter`; tambien se exponen `--decision-threshold`,
`--test-size` y `--seed`.

Las cinco configuraciones de la comparacion se ejecutaron con:

```bash
python train.py --run-name baseline
python train.py --c 0.01 --run-name regularizacion-fuerte
python train.py --c 0.1 --run-name regularizacion-media
python train.py --c 10 --run-name regularizacion-debil
python train.py --class-weight none --run-name sin-balanceo
```

Todas pertenecen al experimento `bank-marketing-logistic-comparison`. El
resumen ordenado por average precision se guarda en
`resultados_corridas.csv`.

## Reproducibilidad

La semilla predeterminada es `42`. Se aplica a Python, NumPy, la division
estratificada de datos y `LogisticRegression`. Tambien se fija a uno el numero
de hilos de las bibliotecas numericas. El conjunto de prueba permanece igual
en las cinco corridas; ejecutar dos veces los mismos argumentos produce las
mismas metricas. Las versiones exactas del entorno usado estan en
`requirements.txt`.

## Interfaz de MLflow

Desde `TC5061_Sem04`, iniciar la interfaz con:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir `http://127.0.0.1:5000`, elegir el experimento
`bank-marketing-logistic-comparison`, seleccionar las cinco corridas y pulsar
**Compare**. `average_precision` es la metrica principal porque la clase
positiva representa aproximadamente 11.7 % de los ejemplos.
