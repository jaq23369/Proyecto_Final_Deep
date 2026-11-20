# Cupo Seguro

Clasificación de espacios de parqueo como **libre**, **ocupado** o **por verificar** a partir de
cámaras, con la cuantificación del valor de negocio (parqueo comercial en Q/mes y parqueo UVG en
horas por semestre). Proyecto final de CC3092 Deep Learning y Sistemas Inteligentes (UVG),
de Joel Jaquez y Kevin Villagran.

## Contenido

1. [Qué es Cupo Seguro](#qué-es-cupo-seguro)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Instalación](#instalación)
4. [Datos](#datos)
5. [Particiones y protocolo de evaluación](#particiones-y-protocolo-de-evaluación)
6. [Reproducción completa en orden](#reproducción-completa-en-orden)
7. [Reproducir cada iteración](#reproducir-cada-iteración)
8. [Evaluación, tabla de resultados, figuras y EDA](#evaluación-tabla-de-resultados-figuras-y-eda)
9. [Modelo de valor](#modelo-de-valor)
10. [Modelo final: ONNX, latencia, costo y Grad-CAM](#modelo-final-onnx-latencia-costo-y-grad-cam)
11. [Explicabilidad del modelo final](#explicabilidad-del-modelo-final)
12. [App de demostración](#app-de-demostración)
13. [Pruebas](#pruebas)
14. [Consejos de rendimiento](#consejos-de-rendimiento)
15. [Créditos y licencia del dataset](#créditos-y-licencia-del-dataset)

## Qué es Cupo Seguro

En un parqueo lleno, el conductor pierde minutos dando vueltas en busca de un espacio, y el
operador pierde clientes que se van porque "no hay lugar". Instalar un sensor por espacio
resuelve el problema, pero es caro. Cupo Seguro usa las cámaras que el parqueo ya tiene: recorta
cada espacio de la imagen y una red neuronal estima `p = P(ocupado)`. Con dos umbrales
`t_L ≤ t_H`, cada espacio se muestra como **libre** (verde, `p < t_L`), **ocupado** (rojo,
`p ≥ t_H`) o **por verificar** (amarillo, en medio). Los espacios amarillos los revisa un
encargado en la cámara, así que el sistema no tiene que acertar siempre: cuando duda, lo dice.

Los errores no cuestan lo mismo. Mostrar en verde un espacio ocupado manda al conductor a un
lugar que no existe (pierde tiempo y confianza); mostrar en rojo un espacio libre esconde
capacidad y alarga el recorrido. Por eso el proyecto no elige modelos por F1, sino con un
**modelo de valor** (`src/cupo/value_model.py`) que traduce las tasas de error a minutos por
llegada y a quetzales en dos escenarios:

- **Parqueo comercial de pago:** valor neto para el operador en **Q/mes** (clientes que ya no
  abandonan, menos la infraestructura, la inferencia y el personal de verificación).
- **Parqueo UVG:** **horas ahorradas por semestre** por estudiantes y personal, y su equivalente
  en Q menos los costos del sistema.

La pregunta central es **cuánto vale mejorar el modelo**: cuántos minutos se ahorra un conductor
y cuántos Q/mes gana el parqueo con el modelo B frente al modelo A, y si el costo de cómputo extra
de un modelo más grande se paga. Las iteraciones van del status quo (sin sistema) a un clásico
sin deep learning (HOG + regresión logística), una CNN pequeña desde cero (mAlexNet),
MobileNetV3 con fine-tuning y un modelo grande de referencia (EfficientNet-B0). El mejor modelo
es el de **mayor valor comercial en validación** con sus umbrales óptimos.

## Estructura del repositorio

```
proyecto_final_deep/
├─ pyproject.toml            paquete "cupo" (layout src), se instala con pip install -e .
├─ requirements.txt          dependencias con rango compatible (torch se instala aparte)
├─ requirements-lock.txt     entorno exacto de desarrollo (pip freeze, CUDA 12.8)
├─ configs/
│  ├─ b1_hog_lr.yaml  b2_malexnet.yaml  malexnet_aug.yaml  mnv3_head.yaml
│  ├─ mnv3_ft.yaml  mnv3_ft_cost.yaml  efficientnet_b0.yaml  mnv3_ft_seeds.yaml
│  ├─ costs.yaml             costos de cómputo y tipo de cambio
│  └─ value/                 comercial.yaml, uvg.yaml (supuestos del modelo de valor)
├─ src/cupo/
│  ├─ paths.py               rutas del repo y de los datos (DATA_ROOT)
│  ├─ data.py  frames.py  transforms.py    particiones, Dataset, frames completos, preprocesamiento
│  ├─ models.py  losses.py   arquitecturas y pérdida sensible al costo
│  ├─ train.py               entrenamiento (CLI)
│  ├─ evaluate.py  metrics.py  thresholds.py   evaluación, métricas, umbrales óptimos (CLI)
│  ├─ value_model.py         modelo de valor de negocio
│  ├─ calibrate.py           temperature scaling (CLI)
│  ├─ export_onnx.py  benchmark.py         exportación a ONNX y latencia/costo (CLI)
│  ├─ explain.py                           explicabilidad: instancias a explicar, oclusión y fidelidad
│  └─ runs.py  utils.py  classical.py      utilidades de runs/, semillas, HOG
├─ scripts/
│  ├─ download_data.py       descarga y verificación del dataset
│  ├─ eda.py                 análisis exploratorio → results/eda/
│  ├─ train_classical.py     baseline B1 (HOG + regresión logística)
│  ├─ results_table.py       tabla de iteraciones → results/iteraciones*.md/.csv, ranking_sensibilidad_*.csv
│  ├─ uncertainty.py         IC95 bootstrap de métricas y valor en test y OOD → results/incertidumbre.*
│  ├─ make_figures.py        figuras → results/figures/ (con --with-table, también la tabla)
│  ├─ gradcam.py             Grad-CAM → results/figures/
│  ├─ build_gallery.py       galería de demo de la app → app/gallery.json
│  ├─ export_classical.py    baseline B1 para el selector de modelos de la app → models/
│  └─ inference_costs.py     costo de inferencia de B1, B2 y el modelo final → results/costo_inferencia*
├─ app/                      app FastAPI: main.py (API), inference.py (ONNX y clásico), static/ (dashboard)
├─ models/                   best.onnx + best.json del modelo final y los baselines (no se versionan; ver abajo)
├─ notebooks/                explicabilidad.ipynb: Grad-CAM, oclusión y fidelidad del modelo final
├─ results/                  sí se versiona: eda/, figures/, xai/ (explicabilidad), tablas de iteraciones, incertidumbre y costo de inferencia
├─ docs/                     arquitectura.md: diagrama del sistema desplegado y plan de monitoreo
├─ tests/                    pruebas (pytest)
└─ runs/                     una carpeta por corrida (no se versiona)
```

Los datos viven **fuera** del repo (por defecto en `../Data`). `runs/`, `models/*` y los
checkpoints están en `.gitignore`.

**Modelo final:** `models/best.onnx` y `models/best.json` no se versionan en git. Están publicados
como assets del Release [`v1.0`](https://github.com/jaq23369/Proyecto_Final_Deep/releases/tag/v1.0)
del repositorio; para usarlos, descarga los dos archivos y colócalos en `models/` (o genéralos
con `export_onnx`, ver
[Modelo final](#modelo-final-onnx-latencia-costo-y-grad-cam)).

## Instalación

Requiere Python 3.11 o superior (se desarrolló con 3.12 en Windows 11 con una RTX 2060 de 6 GB).
Todos los comandos de este README se ejecutan **desde la raíz del repo** (`proyecto_final_deep/`) y con
el entorno virtual activado.

### 1. Clonar y crear el entorno

Se recomienda esta estructura, para que los datos queden al lado del repo, en `../Data`:

```
Proyecto/            ← carpeta de trabajo (cualquier nombre)
├─ Data/             ← la crea scripts/download_data.py
└─ proyecto_final_deep/      ← el repo
```

**Windows (PowerShell):**

```powershell
git clone <url-del-repo> proyecto_final_deep
cd proyecto_final_deep
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1      # si PowerShell lo bloquea: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
```

**Linux / macOS:**

```bash
git clone <url-del-repo> proyecto_final_deep
cd proyecto_final_deep
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Instalar las dependencias

Hay dos formas:

- **Entorno exacto (recomendado para reproducir resultados, con GPU NVIDIA):**
  `requirements-lock.txt` fija todas las versiones con las que se desarrolló (`pip freeze`,
  torch y torchvision `+cu128`) e incluye el índice de PyTorch para CUDA 12.8.

  ```bash
  pip install -r requirements-lock.txt
  pip install -e . --no-deps
  ```

- **Rango compatible (sin GPU NVIDIA, otra versión de CUDA o si el lock falla):**
  `requirements.txt` declara versiones mínimas con cota en la siguiente versión mayor. torch y
  torchvision se instalan aparte, con el índice que corresponda:

  ```bash
  # GPU NVIDIA con CUDA 12.8 (Windows o Linux):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  # Solo CPU:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

  pip install -r requirements.txt
  pip install -e . --no-deps
  ```

`pip install -e .` instala el paquete `cupo` en modo editable; es lo que permite correr
`python -m cupo.train`, `python -m cupo.evaluate`, etc. (`--no-deps` porque las dependencias ya
vienen de los requirements).

Comprobar que torch ve la GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Sin GPU todo funciona en CPU (más lento); AMP solo se activa con CUDA.

## Datos

Se usa **CNRPark+EXT** (Amato et al., ver [Créditos](#créditos-y-licencia-del-dataset)):

| Subconjunto | Contenido | Imágenes | Uso |
|---|---|---|---|
| CNR-EXT (patches 150×150) | 9 cámaras, sol/nublado/lluvia, nov 2015 – feb 2016 | 144,965 | Entrenamiento, validación y prueba principal |
| CNRPark A/B | 2 cámaras de otro parqueo, julio 2015 | 12,584 | Prueba fuera de dominio (Test-OOD) |
| CNR-EXT FULL 1000×750 | 4,278 frames completos + coordenadas de cada espacio | 4,278 | Demo de frame completo y latencia por frame |

### Descarga

`scripts/download_data.py` descarga los archivos del release de los autores, los extrae y
verifica los conteos. Solo usa la biblioteca estándar, así que se puede correr antes de instalar
las dependencias. Son unos 1.6 GB de descarga (unos 3.3 GB en disco si se conservan los
comprimidos).

```bash
python scripts/download_data.py                     # descarga, extrae y verifica (en ../Data)
python scripts/download_data.py --verify-only       # solo verifica conteos y archivos
python scripts/download_data.py --delete-archives   # borra los .zip/.tar después de extraerlos
python scripts/download_data.py --data-root D:/datos/parqueo
```

- El script salta lo ya descargado o extraído: cada carpeta extraída lleva un archivo centinela
  **`.complete`** que se escribe al terminar la extracción. Si una extracción se interrumpe, no hay
  centinela y se repite. Las descargas interrumpidas se reanudan.
- La verificación compara contra los conteos esperados: train 94,493 · val 18,647 ·
  test 31,825 · OOD 12,584 · 157,549 filas del CSV · 164 espacios únicos · 4,278 frames ·
  9 CSV de cámaras. Cualquier diferencia indica una descarga incompleta.

### `DATA_ROOT` y `--data-root`

El código busca los datos en la variable de entorno **`DATA_ROOT`**; si no está definida, en
`../Data` (relativo al repo). `--data-root` **solo** afecta a `download_data.py`. Si los datos
no están en `../Data`, define además `DATA_ROOT` con la misma ruta para que el resto del código
(entrenamiento, evaluación, app) los encuentre:

```bash
export DATA_ROOT=D:/datos/parqueo            # Linux/macOS o Git Bash
$env:DATA_ROOT = "D:/datos/parqueo"          # PowerShell (solo la sesión actual)
```

Las etiquetas que se usan son las oficiales de `CNR-EXT/LABELS/*.txt`, que difieren de la
columna `occupancy` del CSV en 26 patches (0.018 %).

### Licencia y privacidad

- El dataset está licenciado bajo la **Open Data Commons Open Database License (ODbL) v1.0**
  (según [cnrpark.it](http://cnrpark.it/)). Permite usar, adaptar y redistribuir, incluso con fines comerciales,
  **con atribución** y compartiendo bajo la misma licencia las bases de datos derivadas. Por eso
  los datos no están en el repo: se descargan de la fuente original, y la app y este README
  atribuyen el dataset a sus autores.
- Si se usa el dataset, hay que citar [1] y [2] (ver [Créditos](#créditos-y-licencia-del-dataset)).
- **Privacidad:** los autores redujeron las imágenes completas de 2592×1944 a 1000×750 por
  razones de privacidad, y los patches de 150×150 solo muestran un espacio. El repositorio no
  incluye fotos propias: la galería local opcional de la app (carpeta `galeria_propia` dentro de `app`) está en
  `.gitignore` porque sus fotos pueden tener placas y personas.
  La galería versionada (`app/gallery.json`) usa solo imágenes del dataset (frames de fechas de
  test y patches de CNRPark A/B). Las imágenes que un usuario
  sube a la app se procesan en memoria; lo único que se guarda es el reporte del botón de
  retroalimentación (`app/feedback.jsonl`, sin imágenes).
- El código de los autores (github.com/fabiocarrara/deep-parking) no tiene archivo de licencia:
  no se copió código de ahí; la arquitectura mAlexNet se reimplementó a partir del paper [1].

## Particiones y protocolo de evaluación

Las particiones son **fijas** y nadie las cambia:

| Partición | Origen | Patches | % ocupado |
|---|---|---|---|
| train | `CNR-EXT/LABELS/train.txt` | 94,493 | 50.4 % |
| val | `CNR-EXT/LABELS/val.txt` | 18,647 | 71.9 % |
| test | `CNR-EXT/LABELS/test.txt` | 31,825 | 57.4 % |
| Test-OOD | CNRPark A + B completo | 12,584 | 66.8 % |

- Las particiones oficiales de CNR-EXT están **separadas por fecha**: ningún día de test aparece
  en train, así que no hay fuga entre fotos casi idénticas del mismo espacio. Esto imita el uso
  real (se entrena con días pasados y se usa en días futuros).
- **Test-OOD** = CNRPark A/B: otro parqueo y otras cámaras ("instalar el sistema en un parqueo
  nuevo"). Nunca se usa para entrenar ni para elegir nada.
- **La validación oficial no tiene días de lluvia.** Las métricas se reportan por clima y por
  cámara; la lluvia solo se mide en test, y eso es una limitación declarada.
- **Guarda de test.** La selección de modelos, umbrales, temperatura e hiperparámetros se hace
  **solo con validación**. `train.py` y `train_classical.py` nunca leen test ni OOD.
  `data.load_split('test'|'ood')` lanza `PermissionError` salvo con `allow_test=True`, y
  `python -m cupo.evaluate --split test|ood` exige el flag **`--allow-test`**. Ese flag se usa
  **una sola vez, en el congelamiento del modelo (sábado 14 de noviembre de 2026, 12:00)**,
  para evaluar todas las iteraciones; los umbrales son siempre los elegidos en val (nunca se
  re-optimizan en test).

## Reproducción completa en orden

Resumen de punta a punta; cada paso se detalla en las secciones siguientes. Antes de cada corrida
larga conviene la [prueba de humo](#prueba-de-humo-obligatoria-antes-de-cada-corrida-larga).

1. **Datos:** `python scripts/download_data.py` (y `--verify-only` para confirmar).
2. **Iteraciones B1 a efficientnet_b0** (una corrida por config, semilla 42):

   ```bash
   python scripts/train_classical.py --config configs/b1_hog_lr.yaml
   python -m cupo.train --config configs/b2_malexnet.yaml
   python -m cupo.train --config configs/malexnet_aug.yaml
   python -m cupo.train --config configs/mnv3_head.yaml
   python -m cupo.train --config configs/mnv3_ft.yaml
   python -m cupo.train --config configs/mnv3_ft_cost.yaml
   python -m cupo.train --config configs/efficientnet_b0.yaml
   ```

3. **mnv3_ft_seeds:** `configs/mnv3_ft_seeds.yaml` con `--seed 42`, `43` y `44`, y
   `python -m cupo.calibrate --run runs/<run_id>` en cada uno de los tres runs (ver
   [mnv3_ft_seeds](#mnv3_ft_seeds-semillas-y-calibración)).
4. **Tabla y figuras (solo val):** `python scripts/make_figures.py --with-table`.
5. **Elegir el mejor modelo** por valor comercial en val (ver
   [Encontrar el run del mejor modelo](#encontrar-el-run-del-mejor-modelo)). La selección queda
   fija aquí, antes de mirar test.
6. **Evaluación final única (congelamiento, sábado 14 de noviembre de 2026, 12:00):** test y OOD de
   **todos** los runs completos, no solo del mejor, con `--allow-test`. Luego se regeneran la
   tabla y las figuras, que ya incluyen test y OOD.

   PowerShell:

   ```powershell
   Get-ChildItem runs -Directory | Where-Object { $_.Name -notlike "_*" } | ForEach-Object {
       python -m cupo.evaluate --run $_.FullName --split test --allow-test
       python -m cupo.evaluate --run $_.FullName --split ood --allow-test
   }
   python scripts/make_figures.py --with-table
   ```

   bash:

   ```bash
   for run in runs/*/; do
       case "$(basename "$run")" in _*) continue ;; esac   # salta _smoke/ y otras carpetas auxiliares
       python -m cupo.evaluate --run "$run" --split test --allow-test
       python -m cupo.evaluate --run "$run" --split ood --allow-test
   done
   python scripts/make_figures.py --with-table
   ```

7. **Exportar** el mejor run: `python -m cupo.export_onnx --run runs/<run_id> --out models/best.onnx`.
8. **Benchmark:** `python -m cupo.benchmark --onnx models/best.onnx --run runs/<run_id>`, y volver a
   generar la tabla para que incluya la latencia.
9. **Grad-CAM:** `python scripts/gradcam.py --run runs/<run_id> --split val --n 16`.
10. **App:** `python scripts/build_gallery.py` y `uvicorn app.main:app --port 8000` (para el
    selector de modelos, exportar también B1 y B2; ver [App de demostración](#app-de-demostración)).

## Reproducir cada iteración

Cada corrida se lanza con una config de `configs/` y escribe todo en `runs/<run_id>/`, con
`run_id = {name}_s{seed}_{YYYYmmdd-HHMM}`.

| ID | Modelo y cambio | Config | Comando |
|---|---|---|---|
| B0 | Sin sistema (status quo, valor 0) | — | No se entrena; `results_table.py` agrega la fila |
| B1 | HOG + regresión logística (sin deep learning) | `configs/b1_hog_lr.yaml` | `python scripts/train_classical.py --config configs/b1_hog_lr.yaml` |
| B2 | mAlexNet desde cero, sin aumento, lr constante | `configs/b2_malexnet.yaml` | `python -m cupo.train --config configs/b2_malexnet.yaml` |
| malexnet_aug | B2 + aumento de datos + coseno con warmup | `configs/malexnet_aug.yaml` | `python -m cupo.train --config configs/malexnet_aug.yaml` |
| mnv3_head | MobileNetV3-Small de ImageNet, solo la cabeza | `configs/mnv3_head.yaml` | `python -m cupo.train --config configs/mnv3_head.yaml` |
| mnv3_ft | MobileNetV3-Small con fine-tuning completo + coseno + aumento | `configs/mnv3_ft.yaml` | `python -m cupo.train --config configs/mnv3_ft.yaml` |
| mnv3_ft_cost | mnv3_ft + pérdida sensible al costo (peso 3 a ocupado) | `configs/mnv3_ft_cost.yaml` | `python -m cupo.train --config configs/mnv3_ft_cost.yaml` |
| efficientnet_b0 | EfficientNet-B0 de ImageNet, fine-tuning completo | `configs/efficientnet_b0.yaml` | `python -m cupo.train --config configs/efficientnet_b0.yaml` |
| mnv3_ft_seeds | Mejor modelo (mnv3_ft_cost) × 3 semillas + calibración | `configs/mnv3_ft_seeds.yaml` | ver [mnv3_ft_seeds](#mnv3_ft_seeds-semillas-y-calibración) |

Como referencia, B1 completo tarda alrededor de 1 minuto y B2 unos 2 minutos por época en una
RTX 2060.

### Opciones de `cupo.train`

```
python -m cupo.train --config CONFIG [--seed SEED] [--subset SUBSET] [--epochs EPOCHS]
                     [--run-name RUN_NAME] [--runs-dir RUNS_DIR] [--n-boot N_BOOT]
                     [--num-workers NUM_WORKERS]
```

- `--seed`: sobrescribe `train.seed` de la config (por defecto 42). Se fijan las semillas de
  `random`, `numpy` y `torch`, y el `DataLoader` usa un generador con semilla.
- `--subset`: fracción estratificada de train y val (para pruebas de humo).
- `--epochs`, `--num-workers`, `--n-boot`: sobrescriben la config (`--n-boot` son las réplicas
  bootstrap de los IC95).
- `--run-name`: nombre fijo de la carpeta del run; `--runs-dir`: carpeta de runs alternativa.

`scripts/train_classical.py` acepta `--config`, `--subset`, `--seed`, `--run-name`, `--runs-dir`
y `--n-boot` con el mismo significado.

Cada época mide en val la pérdida, accuracy, F1, AUC y el **valor de negocio** del escenario
`select.scenario` (comercial) con los umbrales óptimos. `best.pt` se guarda en la época de
**máximo valor en val** (desempate: menor pérdida en val) y el early stopping usa el mismo
criterio.

### Qué deja cada run

| Archivo | Contenido |
|---|---|
| `config.yaml` | config resuelta (con la semilla y el subset efectivos) |
| `env.json` | commit de git, versiones de Python/torch, GPU, CPU, fecha |
| `history.csv`, `curves.png` | métricas por época y curvas de entrenamiento |
| `best.pt` (o `model.joblib` en B1) | checkpoint del mejor modelo según valor en val |
| `val_preds.npz` | `y`, `p`, logits y metadatos de cada patch de val |
| `metrics_val.json`, `value_val.json` | métricas, IC95 bootstrap, desgloses por cámara y clima, valor por escenario |
| `thresholds.json` | `t_L` y `t_H` de cada escenario, elegidos en val |
| `summary.json` | resumen para la tabla: parámetros, tamaño, tiempo y costo de entrenamiento, valor en val |

### Prueba de humo (obligatoria antes de cada corrida larga)

Una época con el 2 % de los datos, para confirmar que todo corre antes de gastar GPU:

```bash
python -m cupo.train --config configs/mnv3_ft.yaml --subset 0.02 --epochs 1
python scripts/train_classical.py --config configs/b1_hog_lr.yaml --subset 0.02
```

Con `--subset` < 1 el run se guarda solo en `runs/_smoke/` (si no se pasa `--runs-dir`), así no
entra a la tabla de resultados ni a las figuras. Con mAlexNet tarda menos de un minuto; con
EfficientNet-B0, unos 3 minutos en la RTX 2060.

### mnv3_ft_seeds: semillas y calibración

`configs/mnv3_ft_seeds.yaml` repite la configuración de mnv3_ft_cost (la de mayor valor comercial en val
entre malexnet_aug y efficientnet_b0) con `name: mnv3_ft_seeds`, para que la tabla la muestre como fila mnv3_ft_seeds. Si se
repitiera el experimento y ganara otra candidata, se copian a ese archivo las secciones `model`,
`data`, `optim` y `loss` de la ganadora (sin cambiar `name`).

1. Entrenar con tres semillas:

   ```bash
   python -m cupo.train --config configs/mnv3_ft_seeds.yaml --seed 42
   python -m cupo.train --config configs/mnv3_ft_seeds.yaml --seed 43
   python -m cupo.train --config configs/mnv3_ft_seeds.yaml --seed 44
   ```

2. Calibrar **cada uno** de los tres runs con temperature scaling (ajustado en val):

   ```bash
   python -m cupo.calibrate --run runs/<run_id>      # --n-boot N (por defecto 200)
   ```

   Escribe `runs/<run_id>/calibration.json` con la temperatura, la NLL y el ECE antes y
   después, y los umbrales **re-optimizados en val** con las probabilidades calibradas. Si existe,
   `evaluate`, `export_onnx`, `results_table.py` y `make_figures.py` usan esa temperatura y esos
   umbrales.

`results_table.py` agrupa los runs por `name`: con varias semillas reporta media ± desviación.

## Evaluación, tabla de resultados, figuras y EDA

### Evaluar un run

```bash
python -m cupo.evaluate --run runs/<run_id> --split val
```

`--run` acepta la ruta o solo el `run_id`. Con `--split val` no se sobrescriben los artefactos
del entrenamiento: se escriben `val_eval_preds.npz`, `metrics_val_eval.json` y
`value_val_eval.json` (sirve para comprobar el run cargado, por ejemplo con la temperatura
aplicada). Otras opciones: `--batch-size` (256), `--num-workers` (4), `--n-boot`, `--device`
(`cuda` o `cpu`).

**Solo en el congelamiento** (una vez, para todas las iteraciones):

```bash
python -m cupo.evaluate --run runs/<run_id> --split test --allow-test
python -m cupo.evaluate --run runs/<run_id> --split ood --allow-test
```

Escriben `test_preds.npz`/`ood_preds.npz`, `metrics_{test,ood}.json` y `value_{test,ood}.json`,
con métricas binarias a 0.5, tasas de tres estados con los umbrales de val, IC95 por bootstrap,
desgloses por cámara y por clima, y el valor por escenario.

### Tabla de iteraciones

```bash
python scripts/results_table.py                  # → results/
python scripts/results_table.py --include-smoke  # incluye runs/_smoke/ (solo para probar)
```

Escribe cuatro tipos de archivo en `results/` (opciones: `--runs-dir`, `--out`):

| Archivo | Contenido |
|---|---|
| `iteraciones.md` y `iteraciones.csv` | Una fila por grupo de runs (`name` de la config). Con varias semillas, media ± desviación (columnas `*_std`) y totales de tiempo y costo. Los runs con subconjunto (`subset` < 1) forman un grupo aparte, `"<name> (subset <f>)"`, y no se comparan con los completos. B0 es una fila sintética con valor 0. Columnas: modelo, cambios, parámetros, tamaño, train acc, F1 y tasa ocupado→libre en val, valor comercial con IC95, Δ valor frente a la iteración anterior, horas UVG, valor con un umbral único, personal extra, latencia y costo por predicción (si hay `benchmark.json`), tiempo y costo de entrenamiento, test y OOD (solo después del congelamiento) y `run_ids` (los runs del grupo). |
| `iteraciones_resumen.md` | Una fila por iteración (B0, B1, ..., mnv3_ft_seeds), compacta para el informe. |
| `ranking_sensibilidad_<escenario>.csv` | Para `comercial` y `uvg`: si cambia el modelo ganador al mover cada supuesto (±20 % y ±50 %). |

### Incertidumbre en test y OOD

```bash
python scripts/uncertainty.py --allow-test
    [--runs-dir runs] [--out results] [--n-boot N] [--seed 0]
    [--final mnv3_ft_cost] [--refs b1_hog_lr b2_malexnet] [--iterations results/iteraciones.csv]
```

Solo reporta: no elige ni ajusta nada con test u OOD. Lee de cada run las predicciones del
congelamiento (`test_preds.npz`, `ood_preds.npz`) y su temperatura y umbrales congelados
(`calibration.json` o `thresholds.json`); no necesita el checkpoint. Con las mismas funciones,
réplicas y semilla que `cupo.evaluate` calcula el IC95 por bootstrap del F1 macro y la tasa
ocupado→libre (umbral 0.5) y del valor comercial y UVG (punto de operación). Con varias semillas
reporta media ± desviación y el promedio de los IC95, como la tabla de iteraciones. Además, la
diferencia de valor del modelo final frente a B1 y B2 con un bootstrap pareado (las mismas
imágenes, emparejadas por ruta, para los dos modelos), con su IC95 y la proporción de remuestras
en que gana el final. Escribe `results/incertidumbre.csv` (formato largo) y
`results/incertidumbre.md` (tablas para el informe y conclusiones), y comprueba que las
estimaciones puntuales coinciden con `results/iteraciones.csv`, que solo se lee. Sin
`--allow-test` no corre.

### Encontrar el run del mejor modelo

El mejor modelo es el run con **mayor valor comercial en val** (`val_value_comercial` en
`results/iteraciones.csv`; el run está en la columna `run_ids`). En mnv3_ft_seeds la fila promedia las tres
semillas: el run que se usa es la semilla con mayor valor en val, **ya calibrada**. Si dos
candidatos empatan dentro del IC95, gana el de menor latencia y costo. Para obtener el
`run_id` directamente (entre los runs completos, con la calibración aplicada si existe):

```bash
python -c "import sys; sys.path.insert(0, 'scripts'); from results_table import load_records, best_record; from cupo import paths; r = best_record(load_records(paths.runs_dir())); print(r.run_id if r else 'sin runs')"
```

Ese `run_id` es el que se usa en `export_onnx`, `benchmark` y `gradcam.py`.

### Figuras

```bash
python scripts/make_figures.py                   # → results/figures/
python scripts/make_figures.py --with-table      # figuras + tabla (results/iteraciones.*) en un solo comando
```

Genera las curvas de entrenamiento comparadas, el valor frente al umbral (mejor run, por
escenario), el valor frente a F1 y a la tasa ocupado→libre por iteración, el tornado de
sensibilidad, el valor frente al costo y el tamaño (rendimientos decrecientes), las matrices de
confusión y las métricas por cámara y clima. Todo con val; las figuras de test y OOD solo
aparecen si ya existen sus predicciones. Opciones: `--runs-dir`, `--include-smoke`, `--out` y,
con `--with-table`, `--table-out` (carpeta de la tabla, por defecto `results/`).

### EDA

```bash
python scripts/eda.py                            # → results/eda/
```

Genera las figuras del análisis exploratorio (balance por cámara, clima y hora, fechas por
partición, brillo y contraste, ejemplos), `eda_resumen.md` y `eda_stats.json`. Opciones:
`--sample-per-split N` (patches por partición para las estadísticas de imagen, 3000),
`--seed` (0), `--out`, `--full-hash` (busca duplicados con el hash de todos los archivos de
train y test; más lento). El EDA lee test y OOD solo para describir los datos (conteos,
distribuciones, fechas); no carga modelos ni mide desempeño en ellos.

## Modelo de valor

`src/cupo/value_model.py` convierte las tasas de la decisión de tres estados en minutos y
quetzales. El docstring del módulo tiene la derivación completa, los supuestos y las
limitaciones; aquí va el resumen.

1. **Tasas.** Para ocupados: `q_OL`, `q_OV`, `q_OO` (mostrados libre, por verificar u ocupado);
   para libres: `q_LL`, `q_LV`, `q_LO`. Se reponderan con la ocupación del escenario `π` (no con
   la proporción de clases de val).
2. **Verificación imperfecta.** El amarillo lo revisa un operador que acierta con probabilidad
   `a = precision_verificacion` (0.95 por defecto: los casos en amarillo son los más ambiguos).
   Tras la revisión, los verdes efectivos son `q_LL_eff = q_LL + a·q_LV` (libres) y
   `q_OL_eff = q_OL + (1 − a)·q_OV` (ocupados mostrados en verde por error del operador). El
   costo de revisar se modela por capacidad del personal: el encargado actual absorbe pocos
   amarillos; si la carga lo supera, se paga personal extra.
3. **Tiempo y recorrido.** `e` = probabilidad de que un verde esté ocupado (cada fallo cuesta
   una redirección). Ocultar libres alarga el **recorrido** hasta el verde más cercano
   (`t_acceso_min` + una fracción de `t_recorrido_total_min` que crece al haber menos verdes).
   Con eso se calcula `T_sys`, el tiempo con sistema, que nunca supera `t_base + t_redirección`,
   y el ahorro por llegada `A = t_base − T_sys`.
4. **Dinero.**
   - Comercial: `valor_operador_mes` = ingreso recuperado por menos abandono − infraestructura −
     inferencia − personal extra de verificación. Se reporta aparte el valor para los
     conductores y la comparación con sensores por espacio.
   - UVG: `valor_neto_semestre_Q` = horas ahorradas por semestre × valor del tiempo (mezcla de
     estudiantes y personal) − costos del semestre; se reportan también las horas ahorradas y
     los minutos por llegada.
   - Status quo (B0): ahorro 0 y sin costos → valor 0.
5. **Umbrales.** `thresholds.optimize` recorre la grilla completa `0 ≤ t_L ≤ t_H ≤ 1` (paso
   0.01; incluye umbrales únicos `t_L = t_H` lejos de 0.5 y franjas de verificar que no contienen
   0.5) y elige el par que maximiza el objetivo del escenario, siempre en val.

### Dónde se editan los supuestos

Todos los parámetros viven en `configs/value/comercial.yaml` y `configs/value/uvg.yaml`. Cada uno
tiene la forma:

```yaml
params:
  t_base_min:
    value: 6.0
    source: "Shoup (2006) reporta ~8 min promedio de crucero en zonas congestionadas; ..."
    confidence: literatura      # medido | literatura | estimado
```

Al reemplazar un valor con una medición de campo, actualizar también `source` y `confidence`.
El costo de inferencia por defecto (`costo_por_1000_pred_Q`) y los costos de cómputo están en
`configs/costs.yaml`; un escenario puede sobrescribir `costo_por_1000_pred_Q`. Tras cambiar
supuestos, volver a correr `results_table.py` y `make_figures.py` (los valores se recalculan
desde las predicciones de val guardadas, sin reentrenar).

### Sensibilidad (tornado)

`make_figures.py` genera `results/figures/tornado_comercial.png` y `tornado_uvg.png` (+ `.csv`)
para el mejor run, con cada supuesto multiplicado por 0.5, 0.8, 1.2 y 1.5 (±50 % y ±20 %). Para
explorar a mano:

```python
from cupo.value_model import load_scenario, compute_value, sensitivity

cfg = load_scenario("comercial")
rates = {"q_OL": 0.01, "q_OV": 0.02, "q_OO": 0.97, "q_LL": 0.95, "q_LV": 0.03, "q_LO": 0.02}
print(compute_value(rates, cfg)["objective"])            # Q/mes con estas tasas
tabla = sensitivity(rates, cfg, factors=(0.5, 0.8, 1.2, 1.5))
print(tabla.sort_values("delta").head(10))
```

Con `reoptimize=True, y=..., p=...`, `sensitivity` re-optimiza los umbrales para cada variación.
`value_curve(y, p, cfg)` devuelve la curva de valor frente al umbral.

## Modelo final: ONNX, latencia, costo y Grad-CAM

### Exportar a ONNX

```bash
python -m cupo.export_onnx --run runs/<run_id> --out models/best.onnx
```

Escribe `models/best.onnx` (entrada `[batch, 3, H, W]`, salida `logits`, eje de batch dinámico,
opset 17) y `models/best.json` con el nombre, el run, el tamaño de entrada, la temperatura, los
umbrales de cada escenario y un resumen de métricas de val. Verifica con imágenes de val que ONNX
Runtime y PyTorch coincidan (diferencia máxima < 1e-3); si no, termina con error. Opciones:
`--opset`, `--n-check`. La tarjeta se llama como el `.onnx` (`<stem>.json`: `best.json` para
`best.onnx`). B1 (`model.joblib`) no se exporta a ONNX; para la app se usa
`scripts/export_classical.py` (ver [App de demostración](#app-de-demostración)).

**`best.onnx` y `best.json` van siempre juntos:** se copian, mueven o reemplazan como par. El
`.json` trae la temperatura y los umbrales que la app aplica a los logits del `.onnx`, y
`benchmark --run` lo usa para comprobar que el modelo salió de ese run.

### Latencia, throughput y costo

```bash
python -m cupo.benchmark --onnx models/best.onnx --run runs/<run_id>
```

Mide con patches de val la latencia p50/p95 por espacio (batch 1) y por frame (46 espacios) en
CPU con ONNX Runtime (1 hilo y hilos por defecto) y en GPU con PyTorch si hay CUDA; el throughput
en predicciones/s; `costo_por_1000_pred_Q` en CPU según `cpu_server_usd_per_hour` de
`configs/costs.yaml`; y el volumen: cuántas cámaras a 1 frame cada 30 s soporta 1 núcleo.
Escribe `benchmark.json` en `models/` y, con `--run`, también en el run (lo lee la tabla).
`--run` debe ser el run del que salió el `.onnx`: se compara con el `run_id` del `best.json` que
está al lado; si no coincide o falta `best.json`, no se escribe nada. Opciones: `--n` (300),
`--n-frame`, `--out`, `--no-gpu`.

El baseline clásico B1 se mide con el mismo protocolo (mismas imágenes de val, por espacio y por
frame, 1 hilo y todos los hilos, misma base de costo; sin GPU):

```bash
python -m cupo.benchmark --joblib models/b1_hog_lr.joblib --out <carpeta>
```

Con un `.joblib`, `--out` es obligatorio: así la medición de B1 nunca sobrescribe
`models/benchmark.json`, que es el del modelo servido por defecto.

En B1 la latencia del modelo incluye el cálculo de HOG (su extracción de características, con el
mismo camino que la app) y el preprocesamiento es solo abrir y decodificar la imagen; el JSON lo
indica en `model_latency_includes` y `preprocess_basis`.

### Costo de inferencia por modelo

```bash
python scripts/inference_costs.py
    [--models models/b1_hog_lr.joblib models/b2_malexnet.onnx models/best.onnx] [--ref b1_hog_lr]
    [--n 300] [--n-frame N] [--no-gpu] [--skip-measure] [--out results]
    [--iterations results/iteraciones.csv]
```

Mide B1, B2 y el modelo final uno tras otro en la misma máquina (para que la comparación sea
justa) con `cupo.benchmark` y guarda cada medición en `results/costo_inferencia/<name>.json`; no
escribe en `models/` ni en `runs/`. Luego arma `results/costo_inferencia.csv` y
`results/costo_inferencia.md`: por modelo, parámetros, tamaño, latencia p50 en CPU con 1 hilo
(por espacio y por frame), throughput, costo por 1,000 predicciones, costo de inferencia al mes
con el volumen del escenario comercial (`configs/value/comercial.yaml`) y latencia en GPU si hay
run y CUDA; y, desde `results/iteraciones.csv` (solo lectura), el valor comercial en val, test y
OOD, el costo de entrenamiento y las diferencias frente a B1 (Δ valor y Δ costo de inferencia en
Q/mes). El `.md` cierra con una frase por modelo del tipo "el modelo final genera Q… más al mes
que B1 … con un costo extra de cómputo de Q… al mes". `--skip-measure` reusa los JSON ya medidos y
solo rehace la tabla; `--iterations` cambia la tabla de iteraciones de donde se leen los valores.
El valor absoluto de la latencia cambia entre sesiones y máquinas; lo que se compara es la
diferencia entre modelos medidos en la misma sesión.

### Grad-CAM

```bash
python scripts/gradcam.py --run runs/<run_id> --split val --n 16
```

Grad-CAM propio (hooks sobre la última capa convolucional) sobre aciertos y errores de val, con
prioridad a los errores ocupado→libre. Escribe `results/figures/gradcam_<run_id>_val.png` y
`.csv`. Solo para runs de PyTorch y solo en val: `--split test` y `--split ood` se rechazan
siempre (el flag existe, pero el script termina con error). Opciones: `--out`, `--device`.

## Explicabilidad del modelo final

`notebooks/explicabilidad.ipynb` explica el modelo final (`mnv3_ft_cost`, semilla 42) **sin
reentrenarlo**. Usa Grad-CAM como método principal y oclusión como alternativa y como prueba de
fidelidad, y compara con el baseline interpretable B1 (HOG + regresión logística). La lógica vive en
`src/cupo/explain.py` y Grad-CAM en `scripts/gradcam.py`.

| Parte del notebook | Qué hace | Salida en `results/xai/` |
|---|---|---|
| Explicaciones locales | 6 instancias de test elegidas con reglas fijas: acierto ocupado y libre con confianza alta, error ocupado→libre, caso más cercano a `t_H`, acierto con lluvia y error libre→ocupado | `instancias.csv`, `locales.png`, `locales_compacta.png` |
| Explicación global | Muestra estratificada de test por clase y clima: mapa promedio y parte de la atención en el centro del parche | `global.csv`, `global.png`, `global_compacta.png`, `por_camara.csv` |
| Fidelidad | Tapa la zona que marca Grad-CAM y la compara con tapar la zona menos marcada y zonas al azar; repite con oclusión | `fidelidad.csv` |
| Errores ocupado→libre | Grad-CAM de los errores caros de test | `errores_ocupado_libre.png` |
| Baseline B1 | Contribución de cada bloque HOG al log-odds de "ocupado" | `baseline_b1.png` |
| Costo y sensibilidad | Milisegundos por explicación (Grad-CAM frente a oclusión) y efecto de la capa objetivo y de la ventana de oclusión | `costo.csv`, `sensibilidad.csv` |
| Parámetros y resumen | Semilla, capa objetivo, ventana, fracción tapada, tamaño de la muestra y números clave | `parametros.json`, `resumen.json` |

### Qué necesita

- **Dependencias del notebook**, que no están en `requirements.txt`:

  ```bash
  python -m pip install nbformat nbconvert ipykernel
  ```

- **Los datos** en `DATA_ROOT` (por defecto `../Data`; ver [Datos](#datos)).
- **El run del modelo final** en `runs/mnv3_ft_cost_s42_<fecha>/` con `config.yaml`, `best.pt`,
  `thresholds.json` (y `calibration.json` si lo tiene) y `test_preds.npz`. Si no lo tienes, se
  regenera con:

  ```bash
  python -m cupo.train --config configs/mnv3_ft_cost.yaml
  python -m cupo.evaluate --run runs/<run_id> --split test --allow-test
  ```

  El test se usa aquí **solo para explicar** el modelo ya congelado, nunca para elegir modelos ni
  umbrales.
- **El run de B1** en `runs/b1_hog_lr_s42_<fecha>/` con `model.joblib`
  (`python scripts/train_classical.py --config configs/b1_hog_lr.yaml`).

### Cómo ejecutarlo

Desde la raíz del repositorio, con el entorno virtual activado:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 notebooks/explicabilidad.ipynb
```

O ábrelo en VS Code o Jupyter y ejecuta todas las celdas en orden. El notebook busca el run más
reciente que coincida con `runs/mnv3_ft_cost_s42_*` y usa GPU si hay una disponible.

**Reproducibilidad:** la semilla es 42 (NumPy, PyTorch y la selección de instancias) y cuDNN corre
en modo determinista. En una misma máquina, volver a ejecutarlo deja `results/xai/` idéntico byte a
byte, salvo `costo.csv`, que mide tiempos. En otra GPU las instancias elegidas y los parámetros son
los mismos, pero los mapas pueden variar en la tercera cifra decimal (diferencias relativas de
alrededor de 1 % en las tablas) por cómo cada tarjeta redondea las operaciones de punto flotante; las
conclusiones no cambian.

## App de demostración

App FastAPI (`app/main.py`) con un dashboard HTML (`app/static/`) que clasifica cada espacio en
CPU (`app/inference.py`): con ONNX Runtime para los modelos de deep learning y con HOG +
regresión logística para el baseline clásico. El modelo por defecto es `models/best.onnx` +
`models/best.json`. La forma principal de usarla es la **galería**: solo imágenes del dataset
que el modelo nunca vio al entrenar (ver [Galería de demo](#galería-de-demo)). Además tiene tres
modos para subir imágenes: **Cámara de CNR-EXT** (frame completo de una de las 9 cámaras del
dataset → mapa del parqueo con sus cajas fijas), **Dibujar espacios** (el usuario marca los
espacios sobre otra foto) y **Un espacio**. Muestra la confianza, el estado "por verificar", los
umbrales de cada escenario (comercial / UVG), qué modelo produjo el resultado, el **tiempo que
ahorra cada conductor** en el parqueo UVG con el modelo elegido y un botón "este espacio estaba
ocupado".

El panel **Tiempo del conductor** sale del modelo de valor en validación: minutos ahorrados por
llegada = horas ahorradas por semestre (`val_hours_uvg` de `results/iteraciones.csv`) × 60 /
(llegadas por día × días del semestre de `configs/value/uvg.yaml`). Muestra también el tiempo
base sin sistema (`t_base_min`), el tiempo con el modelo y la diferencia en segundos frente a B1.
El tiempo base viene de la literatura y de estimaciones; se reemplaza con mediciones en campo. El diagrama del sistema
desplegado, las métricas operativas y el plan de monitoreo están en
[docs/arquitectura.md](docs/arquitectura.md).

```bash
# 1. Exportar el modelo final (ver la sección anterior) → models/best.onnx + models/best.json
python -m cupo.export_onnx --run runs/<run_id> --out models/best.onnx
# 2. (Opcional) Exportar los baselines para el selector de modelos
python -m cupo.export_onnx --run runs/<id_b2> --out models/b2_malexnet.onnx
python scripts/export_classical.py --run runs/<id_b1> --out models/b1_hog_lr.joblib
# 3. Preparar la galería de demo → app/gallery.json (solo imágenes nunca vistas en el
#    entrenamiento: frames completos de fechas de test y patches de CNRPark A/B)
python scripts/build_gallery.py
# 4. Levantar la app y abrir http://127.0.0.1:8000
uvicorn app.main:app --port 8000
```

### Selector de modelos (B1, B2 y el modelo final)

La app descubre en la carpeta del modelo por defecto (`models/`, o la de `MODEL_PATH`) todos los
pares `<stem>.onnx` + `<stem>.json` y `<stem>.joblib` + `<stem>.json`; cada uno aparece en el
selector **Modelo** de la barra superior con una etiqueta legible ("B1 · HOG + regresión
logística (clásico)", "B2 · mAlexNet", "Modelo final · MobileNetV3 (mnv3_ft_cost)"; si el nombre
no se conoce, se muestra el `name` de la tarjeta). Cada modelo se carga la primera vez que se
pide, queda en caché y se recarga si cambia su archivo. Al cambiar de modelo, la imagen actual
(de la galería, subida o con cajas dibujadas) se vuelve a analizar sola y el panel "Modelo"
muestra el elegido; la latencia medida de `benchmark.json` solo se muestra para el modelo del que
salió.

- **B2 (mAlexNet):** se exporta a ONNX con la misma herramienta que el modelo final; la tarjeta
  queda como `models/b2_malexnet.json`.
- **B1 (HOG + regresión logística):** no es ONNX. `scripts/export_classical.py` copia
  `model.joblib` del run a `--out` y escribe `<stem>.json` con el mismo esquema que `export_onnx`
  (nombre, run, umbrales por escenario, resumen de val, temperatura 1.0, `kind: "classical"` y
  los parámetros HOG). Verifica con patches de val que la p calculada como en la app coincide
  con la de `val_preds.npz` del run. La app calcula HOG desde la imagen en memoria con
  `cupo.classical.hog_image`, la misma función que usa el entrenamiento.
- API: `GET /api/models` devuelve `{models: [{id, label, name, model, kind, is_default, val_f1,
  val_value_comercial, thresholds, available, error}], default}`; `GET /api/model` con el
  parámetro `model=<id>` da el detalle de un modelo; los tres endpoints de predicción
  (`/api/predict/frame`, `/api/predict/patch` y `/api/predict/custom`) aceptan el campo opcional
  `model` (un id de la lista) y la respuesta trae `model: {id, label, name, run_id, kind}`. Un
  id desconocido devuelve 400 con las opciones válidas. `GET /api/model` incluye además
  `time_benefit` (minutos ahorrados por conductor en el escenario UVG; `null` si el modelo no
  está en `results/iteraciones.csv`).

### Fotos propias y galería local

Las cajas fijas del modo **Cámara de CNR-EXT** solo coinciden con las 9 cámaras del dataset
(1000×750). Con fotos de otro parqueo (por ejemplo, el de la UVG) caen en cualquier lado; por
eso la app avisa si la foto no mide 1000×750 y recomienda la pestaña **Dibujar espacios**. Ahí conviene dibujar cada caja ajustada a un solo espacio (sin topes de concreto ni
árboles). El modelo se entrenó con cámaras elevadas que miran en diagonal, de día: las fotos de
noche o a nivel del suelo están fuera de dominio y se esperan más "por verificar".

- **Guardar cajas** descarga un JSON `{image: <nombre>, boxes: [{slot_id, x, y, w, h}]}` en
  píxeles de la imagen original; **Cargar cajas** lee ese JSON y deja las cajas editables.
- **Galería local (opcional):** si se crea la carpeta `galeria_propia` dentro de `app`, cada foto
  `<foto>.jpg|jpeg|png` con su `<foto>.json` de cajas (el mismo formato) aparece en la sección
  "Fotos propias" de la galería y se analiza con esas cajas; sin JSON se abre directamente en el
  modo de dibujo. La carpeta se cambia con `OWN_GALLERY_DIR`. No existe en el repositorio y está
  en `.gitignore`: las fotos pueden tener personas y placas y el repositorio es público. Sin la
  carpeta, la galería muestra solo imágenes del dataset.

### Galería de demo

`build_gallery.py` elige los ejemplos de forma determinista con metadatos, sin predicciones ni
métricas de test: dos frames de fecha de test por cámara (1-9) y por clima (soleado, nublado,
lluvia), en lo posible uno en cada mitad del día, los más cercanos a una hora objetivo y con etiquetas cerca de
esa hora (sin mirar sus valores), más dos casos difíciles (temprano con lluvia y al atardecer):
56 frames; 8 patches por cámara de CNRPark A/B (4 de la carpeta `busy` y 4 de `free`); y cuatro
mosaicos (dos de A y dos de B) de 12 patches de CNRPark distintos (6 y 6) en una cuadrícula 4×3, que la app compone al vuelo con
celdas de 150×150 y analiza celda por celda. Guarda rutas relativas a `DATA_ROOT` (las imágenes
no se copian al repo). Opción: `--out` (por defecto `app/gallery.json`). En el dashboard los
frames se agrupan por clima y la galería se desplaza dentro de su panel.

### Configuración y casos límite

- **Cambiar el modelo por defecto = reemplazar un archivo:** basta con sobrescribir el par
  `models/best.onnx` + `models/best.json` (con `export_onnx`); la app detecta el cambio y recarga
  el modelo en la siguiente petición, sin reiniciar. También se puede apuntar la variable de
  entorno `MODEL_PATH` a otro `.onnx` (su `.json` con el mismo nombre debe estar al lado; las
  rutas relativas se resuelven desde la raíz del repo):

  ```bash
  MODEL_PATH=models/otro.onnx uvicorn app.main:app --port 8000              # Linux/macOS o Git Bash
  $env:MODEL_PATH = "models/otro.onnx"; uvicorn app.main:app --port 8000    # PowerShell
  ```

- Variables de entorno: `MODEL_PATH` (por defecto `models/best.onnx`), `DATA_ROOT` (datos para
  la galería), `OWN_GALLERY_DIR` (por defecto la carpeta `galeria_propia` dentro de `app`), `FEEDBACK_PATH` (por defecto
  `app/feedback.jsonl`) y `SHOW_GROUND_TRUTH` (desactivada por defecto; ver abajo).
- Endpoints: `GET /` (dashboard), `GET /api/models`, `GET /api/model`, `GET /api/gallery`,
  `GET /api/gallery/{id}/image`, `POST /api/predict/frame`, `POST /api/predict/patch`,
  `POST /api/predict/custom`, `POST /api/feedback`.
- Casos límite: un archivo que no es imagen o está vacío devuelve 400 con un mensaje; uno de
  más de 15 MB o de más de 50 megapíxeles devuelve 413; un frame de menos de 320×240 devuelve
  400; una cámara fuera de 1-9 devuelve 400; un id de modelo desconocido devuelve 400; una imagen
  muy oscura o borrosa devuelve una advertencia visible; sin cajas dibujadas, la app pide dibujar
  al menos una; sin modelo exportado o sin las cajas de la cámara en `DATA_ROOT`, la API responde
  503 con una explicación. El reporte de retroalimentación se valida (estados, escenario, modelo,
  `p_busy` en [0, 1], `slot_id` corto) y su cuerpo no puede pasar de 2 KB.
- La galería del dataset necesita los datos en `DATA_ROOT`; las fotos subidas por el usuario se
  procesan en memoria y no se guardan.
- Enlace directo a un ejemplo (útil como respaldo en la demo):
  `http://127.0.0.1:8000/?ejemplo=<id>&escenario=uvg&modelo=<id_modelo>`, con `<id>` de
  `app/gallery.json` (o `own-<foto>` para una foto propia) y `<id_modelo>` de `GET /api/models`.
- **Etiquetas reales de la galería:** solo después del congelamiento del modelo (sábado 14 de
  noviembre). Por defecto la app no las envía. Para la demo, ya pasado el congelamiento, se
  arranca con `SHOW_GROUND_TRUTH=1` y la galería las muestra como comparación visual (aciertos ✓
  y errores ✗; en los patches y mosaicos de CNRPark salen de la carpeta `busy`/`free`); nunca se
  usan para elegir el modelo ni los umbrales.

  ```bash
  SHOW_GROUND_TRUTH=1 uvicorn app.main:app --port 8000               # solo después del congelamiento
  ```

## Pruebas

```bash
pytest                     # todas las pruebas
pytest -m "not data"       # solo las que no necesitan el dataset
pytest -m data             # solo las que usan el dataset real
```

Las pruebas que necesitan el dataset llevan el marcador `data` (declarado en `pyproject.toml`) y
se saltan solas si no encuentran los datos en `DATA_ROOT`, así que `pytest` corre en verde en una
máquina sin dataset. Con `-m "not data"` esas pruebas se deseleccionan (no se ejecutan ni aparecen
como saltadas).

## Consejos de rendimiento

- **`num_workers` en Windows.** Windows arranca los workers del `DataLoader` con `spawn`, lo que
  tarda unos segundos por worker. Las configs usan `num_workers: 6`; ajústalo a los núcleos de tu
  CPU con `--num-workers N` sin editar la config (por ejemplo, `--num-workers 0` para depurar).
  Con conjuntos de 5,000 imágenes o menos (pruebas de humo) se usa 0 automáticamente.
- **GPU de 6 GB (RTX 2060).** Los tamaños de batch de las configs (256 en mAlexNet, 128 en
  MobileNetV3, 64 en EfficientNet-B0) caben en 6 GB con AMP, que se activa solo si hay CUDA. Si
  aparece un error de memoria, crea una copia de la config con un `data.batch_size` menor.
- **`cudnn_benchmark`.** Está apagado por defecto (`train.cudnn_benchmark: false`): en la RTX 2060
  tarda unos 30 s por cada forma de batch nueva con MobileNet y EfficientNet y casi no acelera.
  Para reproducibilidad bit a bit se puede poner `train.deterministic: true` (más lento).
- **Lectura de disco.** Son unos 145,000 JPEG pequeños; tener `Data/` en un SSD ayuda más que
  subir `num_workers`.
- **Sin GPU.** Todo corre en CPU, pero entrenar MobileNet o EfficientNet completos es lento; para
  probar, usa `--subset` y `--epochs`.

## Créditos y licencia del dataset

Dataset **CNRPark+EXT** de los autores de [1] y [2] ([cnrpark.it](http://cnrpark.it/)), bajo la licencia
**Open Data Commons Open Database License (ODbL) v1.0**. Las imágenes de la galería de la app
provienen de este dataset y se usan con atribución.

- [1] G. Amato, F. Carrara, F. Falchi, C. Gennaro, C. Meghini, C. Vairo, "Deep learning for
  decentralized parking lot occupancy detection", *Expert Systems with Applications*, 72,
  327-334, 2017. DOI: [10.1016/j.eswa.2016.10.055](https://doi.org/10.1016/j.eswa.2016.10.055)
- [2] G. Amato, F. Carrara, F. Falchi, C. Gennaro, C. Vairo, "Car parking occupancy detection
  using smart camera networks and deep learning", *IEEE Symposium on Computers and Communication
  (ISCC) 2016*, 1212-1217. DOI: [10.1109/ISCC.2016.7543901](https://doi.org/10.1109/ISCC.2016.7543901)

La arquitectura mAlexNet de este repo es una reimplementación a partir de [1]. MobileNetV3 y
EfficientNet-B0 usan los pesos de ImageNet de torchvision.
