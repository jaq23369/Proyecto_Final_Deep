"""Utilidades compartidas por las pruebas de entrenamiento, evaluación y calibración.

- ``SplitRecorder``: reemplaza ``data.load_split`` para registrar qué particiones se piden.
- ``tiny_torch_cfg`` / ``tiny_classical_cfg``: configs mínimas (subconjunto pequeño, 1 época).
- ``load_train_classical``: importa ``scripts/train_classical.py`` como módulo.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from cupo import paths

REPO = paths.repo_root()
PYTHON = sys.executable
CONFIGS = REPO / "configs"
HAS_DATA = (paths.split_file("train").is_file() and paths.split_file("val").is_file()
            and paths.patches_dir().is_dir())

# Archivos de runs/<id>/ según el formato de los runs (E6)
E6_COMMON = ["config.yaml", "env.json", "history.csv", "curves.png", "val_preds.npz",
             "metrics_val.json", "value_val.json", "thresholds.json", "summary.json"]
E6_HISTORY = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro",
              "val_auc", "val_value", "lr", "epoch_time_s"]
E6_SUMMARY = ["run_id", "name", "model", "params_total", "params_trainable", "size_mb",
              "best_epoch", "train_time_s", "gpu", "train_cost_usd", "train_cost_q",
              "val_f1_macro", "val_rate_busy_as_free", "val_value_comercial", "val_value_uvg"]
E6_NPZ = ["y", "p", "camera", "weather", "date", "path"]


def _reject_constant(token: str):
    raise ValueError(f"JSON no estándar: {token}")


def load_strict_json(path: Path) -> Any:
    """Lee un JSON y falla si contiene NaN, Infinity o -Infinity."""
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_train_classical():
    """Importa ``scripts/train_classical.py`` (no es parte del paquete)."""
    name = "cupo_scripts_train_classical"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "train_classical.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class SplitRecorder:
    """Sustituto de ``load_split`` que registra cada llamada.

    Delega en la función original para train/val (con submuestra opcional) y deja que la
    guarda original rechace test/ood. Si alguien pide test/ood con ``allow_test=True``,
    se registra y se lanza ``AssertionError``.
    """

    def __init__(self, original, frac: float | None = None, seed: int = 0) -> None:
        self.original = original
        self.frac = frac
        self.seed = seed
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, name, allow_test: bool = False):
        self.calls.append((name, bool(allow_test)))
        if name in ("test", "ood") and allow_test:
            raise AssertionError(f"se pidió la partición protegida {name!r}")
        df = self.original(name, allow_test=allow_test)
        if self.frac is not None:
            from cupo.data import subsample

            df = subsample(df, self.frac, self.seed)
        return df

    @property
    def names(self) -> set[str]:
        return {name for name, _ in self.calls}


class CachedLoaderRecorder:
    """Registra las particiones que llegan al cargador interno (detrás de la guarda)."""

    def __init__(self, original) -> None:
        self.original = original
        self.names: list[str] = []

    def __call__(self, name, root):
        self.names.append(name)
        return self.original(name, root)


def install_split_recorders(monkeypatch, frac: float | None = None):
    """Instala los registradores en ``cupo.data`` y en los módulos que importan ``load_split``."""
    import cupo.data
    import cupo.train

    rec = SplitRecorder(cupo.data.load_split, frac=frac)
    monkeypatch.setattr(cupo.data, "load_split", rec)
    monkeypatch.setattr(cupo.train, "load_split", rec)
    inner = CachedLoaderRecorder(cupo.data._load_split_cached)
    monkeypatch.setattr(cupo.data, "_load_split_cached", inner)
    return rec, inner


def tiny_torch_cfg(subset: float = 0.005, config: str = "b2_malexnet.yaml", **extra) -> dict:
    """Config de B2 con un subconjunto pequeño, 1 época y pocas réplicas bootstrap."""
    from cupo.runs import deep_merge
    from cupo.train import resolve_config

    over = {"train": {"epochs": 1}, "data": {"subset": subset, "num_workers": 0},
            "eval": {"n_boot": 10}}
    return resolve_config(CONFIGS / config, deep_merge(over, extra))


def tiny_classical_cfg(subset: float = 0.01) -> dict:
    """Config de B1 con un subconjunto pequeño, HOG en serie y pocas réplicas bootstrap."""
    from cupo.runs import deep_merge, load_yaml

    mod = load_train_classical()
    cfg = deep_merge(mod.DEFAULT_CONFIG, load_yaml(CONFIGS / "b1_hog_lr.yaml"))
    return deep_merge(cfg, {"data": {"subset": subset, "n_jobs": 1}, "eval": {"n_boot": 10}})
