"""Utilidades de ``runs/<run_id>/``: identificadores, lectura/escritura y carga de un run.

Formato de cada run (lo consume ``scripts/results_table.py``):
config.yaml, env.json, history.csv, curves.png, best.pt (o model.joblib), val_preds.npz,
metrics_val.json, value_val.json, thresholds.json, summary.json y, opcionalmente,
calibration.json.

Umbrales y temperatura de un run (``operating_point``):
- sin ``calibration.json``: temperatura 1 y umbrales de ``thresholds.json``;
- con ``calibration.json``: su temperatura y los umbrales re-optimizados en val con las
  probabilidades calibradas (los dos se eligen juntos para que sean coherentes).
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cupo import paths

CHECKPOINT_FILE = "best.pt"
CLASSICAL_FILE = "model.joblib"
CALIBRATION_FILE = "calibration.json"
THRESHOLDS_FILE = "thresholds.json"
SMOKE_DIRNAME = "_smoke"  # runs con subset < 1: fuera de runs/*/ para no entrar a la tabla


# ---------------------------------------------------------------------------
# Identificadores y carpetas
# ---------------------------------------------------------------------------

def make_run_id(name: str, seed: int, now: datetime | None = None) -> str:
    """``{name}_s{seed}_{YYYYmmdd-HHMM}``."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M")
    return f"{name}_s{seed}_{stamp}"


def create_run_dir(run_id: str, runs_root: Path | None = None) -> Path:
    """Crea ``runs_root/run_id``; si ya existe, agrega ``-2``, ``-3``, ... al final."""
    root = Path(runs_root) if runs_root is not None else paths.runs_dir()
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / run_id
    k = 2
    while candidate.exists():
        candidate = root / f"{run_id}-{k}"
        k += 1
    candidate.mkdir(parents=True)
    return candidate.resolve()


def choose_runs_root(runs_dir: str | Path | None, subset: float) -> Path:
    """Carpeta donde se crea el run.

    ``runs_dir`` explícito manda; si no se da, ``runs/`` para corridas completas y
    ``runs/_smoke/`` cuando ``subset < 1`` (pruebas de humo), así no contaminan la tabla.
    """
    if runs_dir is not None:
        return Path(runs_dir)
    root = paths.runs_dir()
    return root / SMOKE_DIRNAME if float(subset) < 1 else root


def resolve_run_dir(run: str | Path) -> Path:
    """Acepta una ruta (relativa al cwd o al repo) o solo el ``run_id`` dentro de ``runs/``."""
    run = Path(run)
    for candidate in (run, paths.repo_root() / run, paths.runs_dir() / run):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"No se encontró el run {str(run)!r} (ni en {paths.runs_dir()})")


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Mezcla recursiva: los valores de ``override`` ganan; no modifica las entradas."""
    out = copy.deepcopy(dict(base))
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_yaml(path: str | Path, data: Mapping[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(to_native(data), fh, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# JSON y npz
# ---------------------------------------------------------------------------

def to_native(obj: Any) -> Any:
    """Convierte tipos de numpy a nativos y NaN/inf a ``None`` (JSON estándar)."""
    if isinstance(obj, Mapping):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: str | Path, data: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(to_native(data), fh, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def save_npz(npz_path: str | Path, /, **arrays: Any) -> None:
    """Guarda arreglos comprimidos; las columnas de texto se guardan como ``str``.

    ``npz_path`` es solo posicional para que ``path`` pueda ser el nombre de un arreglo.
    """
    clean = {}
    for key, val in arrays.items():
        arr = np.asarray(val)
        if arr.dtype == object:
            arr = arr.astype(str)
        clean[key] = arr
    np.savez_compressed(npz_path, **clean)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Carga de un run
# ---------------------------------------------------------------------------

@dataclass
class LoadedRun:
    """Un run listo para predecir."""

    run_dir: Path
    config: dict[str, Any]
    kind: str                       # 'torch' o 'classical'
    model: Any                      # nn.Module (en eval) o dict de model.joblib
    input_size: int | None
    temperature: float = 1.0
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    calibrated: bool = False

    @property
    def run_id(self) -> str:
        return self.run_dir.name


def run_kind(run_dir: Path) -> str:
    if (run_dir / CHECKPOINT_FILE).exists():
        return "torch"
    if (run_dir / CLASSICAL_FILE).exists():
        return "classical"
    raise FileNotFoundError(f"{run_dir} no tiene {CHECKPOINT_FILE} ni {CLASSICAL_FILE}")


def operating_point(run_dir: Path) -> tuple[float, dict[str, dict[str, float]], bool]:
    """(temperatura, umbrales por escenario, calibrado?) del run."""
    calib_path = run_dir / CALIBRATION_FILE
    if calib_path.exists():
        calib = load_json(calib_path)
        return float(calib["temperature"]), calib["thresholds"], True
    thr_path = run_dir / THRESHOLDS_FILE
    thresholds = load_json(thr_path) if thr_path.exists() else {}
    return 1.0, thresholds, False


def load_torch_model(run_dir: Path, device: str | Any = "cpu"):
    """Reconstruye el modelo de ``best.pt`` (sin descargar pesos) en modo evaluación."""
    import torch

    from cupo.models import build_model

    ckpt = torch.load(run_dir / CHECKPOINT_FILE, map_location="cpu", weights_only=True)
    mcfg = dict(ckpt["model_cfg"])
    model = build_model(
        mcfg["name"],
        num_classes=int(mcfg.get("num_classes", 2)),
        pretrained=False,
        freeze="none",
        input_size=int(ckpt["input_size"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, int(ckpt["input_size"]), ckpt


def load_run(run: str | Path, device: str | Any = "cpu") -> LoadedRun:
    """Carga config, modelo, temperatura y umbrales de un run (torch o clásico)."""
    run_dir = resolve_run_dir(run)
    config = load_yaml(run_dir / "config.yaml")
    kind = run_kind(run_dir)
    if kind == "torch":
        model, input_size, _ = load_torch_model(run_dir, device)
    else:
        import joblib

        model = joblib.load(run_dir / CLASSICAL_FILE)
        input_size = None
    temperature, thresholds, calibrated = operating_point(run_dir)
    return LoadedRun(
        run_dir=run_dir, config=config, kind=kind, model=model, input_size=input_size,
        temperature=temperature, thresholds=thresholds, calibrated=calibrated,
    )
