# Cupo Seguro

Clasificación de espacios de parqueo como **libre**, **ocupado** o **por verificar** a partir de
cámaras, con la cuantificación del valor de negocio (parqueo comercial en Q/mes y parqueo UVG en
horas ahorradas). Proyecto final de CC3092 Deep Learning y Sistemas Inteligentes (UVG).

## Instalación

Requiere Python 3.11 o superior (desarrollado con 3.12). Hay dos formas de instalar las
dependencias:

- **Entorno exacto (recomendado para reproducir resultados):** `requirements-lock.txt` fija todas
  las versiones con las que se desarrolló (Windows 11, Python 3.12, GPU con CUDA 12.8). Incluye
  torch y torchvision `+cu128` y el índice de PyTorch.

  ```bash
  python -m venv .venv
  # Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
  pip install -r requirements-lock.txt
  pip install -e . --no-deps
  ```

- **Rango compatible:** `requirements.txt` declara versiones mínimas y una cota superior en la
  siguiente versión mayor. Úsalo si el lock no aplica a tu máquina (por ejemplo, sin GPU NVIDIA).
  torch y torchvision se instalan aparte, con el índice que corresponda:

  ```bash
  python -m venv .venv
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # o .../whl/cpu
  pip install -r requirements.txt
  pip install -e . --no-deps
  ```

## Datos

Se usa CNRPark+EXT (Amato et al.). Por defecto los datos viven en `../Data` (al lado del repo);
se puede cambiar con la variable de entorno `DATA_ROOT`.

```bash
python scripts/download_data.py                 # descarga, extrae y verifica
python scripts/download_data.py --verify-only   # solo verifica conteos y archivos
python scripts/download_data.py --data-root D:/datos/parqueo
```

`--data-root` solo afecta a este script. Si los datos no están en `../Data`, define además la
variable de entorno `DATA_ROOT` con la misma ruta para que el resto del código (entrenamiento,
evaluación, app) los encuentre:

```bash
export DATA_ROOT=D:/datos/parqueo            # Linux/macOS o Git Bash
$env:DATA_ROOT = "D:/datos/parqueo"          # PowerShell
```

El script salta lo que ya está descargado o extraído (cada carpeta extraída lleva un archivo
`.complete` al terminar) y reanuda descargas interrumpidas.

Particiones (fijas): `CNR-EXT/LABELS/{train,val,test}.txt` (separadas por fecha) y, como prueba
fuera de dominio, CNRPark A+B completo. Las etiquetas que se usan son las oficiales de
`LABELS/*.txt`, que difieren de la columna `occupancy` del CSV en 26 patches (0.018 %).
