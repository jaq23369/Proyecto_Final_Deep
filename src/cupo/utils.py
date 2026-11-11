"""Utilidades generales: semillas e información del entorno."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cupo.paths import repo_root


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fija las semillas de random, numpy y torch (si está instalado).

    Con ``deterministic=True`` además pide a cuDNN algoritmos deterministas
    (más lento; útil para depurar).

    Nota: ``PYTHONHASHSEED`` se fija en el entorno solo para que lo hereden los
    procesos hijos (por ejemplo, los workers del DataLoader con ``spawn``); no cambia
    el hash del proceso actual, que se decide al arrancar el intérprete.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _run_git(args: list[str]) -> str | None:
    """Ejecuta un comando git en el repo; devuelve None si falla."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info() -> dict[str, Any]:
    """Commit actual y si hay cambios sin confirmar."""
    status = _run_git(["status", "--porcelain"])
    return {
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
    }


def _torch_info() -> dict[str, Any]:
    """Versiones de torch/torchvision y GPU disponible (None si no aplica)."""
    info: dict[str, Any] = {"torch": None, "torchvision": None, "cuda": None, "gpu": None}
    try:
        import torch
    except ImportError:
        return info
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    return info


def env_info() -> dict[str, Any]:
    """Información del entorno para ``runs/<id>/env.json``."""
    return {
        **git_info(),
        "python": sys.version.split()[0],
        **_torch_info(),
        "numpy": np.__version__,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "date": datetime.now().isoformat(timespec="seconds"),
    }


def ensure_dir(path: Path) -> Path:
    """Crea la carpeta si no existe y la devuelve."""
    path.mkdir(parents=True, exist_ok=True)
    return path
