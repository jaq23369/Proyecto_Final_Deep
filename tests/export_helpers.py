"""Utilidades de las pruebas de exportación ONNX, benchmark y Grad-CAM.

- ``torch_run_copy``: copia a una carpeta temporal un run de humo de ``runs/_smoke/``
  (mAlexNet o MobileNetV3 calibrado); si no existe, entrena uno diminuto ahí mismo.
- ``fake_classical_run``: run clásico mínimo (``model.joblib``) para probar los rechazos.
- ``assert_no_local_paths``: el texto de un JSON no debe llevar rutas locales.
- ``load_gradcam``: importa ``scripts/gradcam.py`` como módulo.
"""

from __future__ import annotations

import getpass
import importlib.util
import shutil
import sys
from pathlib import Path

from cupo import paths

from run_helpers import REPO, tiny_torch_cfg

SMOKE = paths.runs_dir() / "_smoke"

# prefijo del run de humo -> (config para entrenar uno nuevo, calibrar?)
SOURCES = {
    "malexnet": ("b2_malexnet", "b2_malexnet.yaml", False),
    "mnv3_calibrated": ("k2_mnv3_head", "k2_mnv3_head.yaml", True),
}


def _smoke_candidates(prefix: str, calibrated: bool) -> list[Path]:
    if not SMOKE.is_dir():
        return []
    out = []
    for d in sorted(SMOKE.glob(f"{prefix}_s*"), reverse=True):
        if not (d / "best.pt").is_file() or not (d / "val_preds.npz").is_file():
            continue
        if calibrated and not (d / "calibration.json").is_file():
            continue
        out.append(d)
    return out


def torch_run_copy(kind: str, dest_root: Path) -> Path:
    """Run de PyTorch dentro de ``dest_root`` (copia de un run de humo o uno nuevo diminuto)."""
    prefix, config, calibrated = SOURCES[kind]
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    found = _smoke_candidates(prefix, calibrated)
    if found:
        dst = dest_root / found[0].name
        if not dst.exists():
            shutil.copytree(found[0], dst)
        return dst
    from cupo import train as T

    cfg = tiny_torch_cfg(subset=0.005, config=config)
    run_dir = T.run_training(cfg, run_name=f"{prefix}_s42_tmp", runs_root=dest_root)
    if calibrated:
        from cupo.calibrate import calibrate_run

        calibrate_run(str(run_dir), n_boot=10)
    return run_dir


def fake_classical_run(dest: Path) -> Path:
    """Run clásico mínimo: ``config.yaml`` + ``model.joblib`` + ``thresholds.json``."""
    import joblib

    from cupo.runs import save_json, save_yaml

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    save_yaml(dest / "config.yaml", {"name": "b1_hog_lr", "model": {"name": "hog_lr"}})
    joblib.dump({"kind": "hog_lr"}, dest / "model.joblib")
    save_json(dest / "thresholds.json", {"comercial": {"t_L": 0.4, "t_H": 0.6},
                                         "uvg": {"t_L": 0.4, "t_H": 0.6}})
    return dest


def local_path_markers(*extra: Path) -> list[str]:
    """Fragmentos que delatan una ruta local (usuario, unidad, carpetas del equipo)."""
    user = getpass.getuser()
    markers = ["C:\\\\Users", "C:\\Users", "C:/Users", ":\\\\", ":/Users", "/home/", user,
               str(REPO), str(REPO).replace("\\", "/"), str(paths.data_root())]
    for p in extra:
        markers += [str(p), str(p).replace("\\", "/"), str(p).replace("\\", "\\\\")]
    return [m for m in markers if m]


def assert_no_local_paths(text: str, *extra: Path) -> None:
    for marker in local_path_markers(*extra):
        assert marker not in text, f"ruta local {marker!r} en el JSON"


def load_gradcam():
    """Importa ``scripts/gradcam.py`` (no es parte del paquete)."""
    name = "cupo_scripts_gradcam"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "gradcam.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
