"""Rutas del repositorio y de los datos.

Los datos viven fuera del repo. Se buscan en la variable de entorno ``DATA_ROOT``;
si no está definida, en ``<repo>/../Data``.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "DATA_ROOT"

# Nombres de archivos y carpetas bajo DATA_ROOT
METADATA_CSV = "CNRPark+EXT.csv"
EXT_DIRNAME = "CNR-EXT"
CNRPARK_DIRNAME = "CNRPark"
FULL_DIRNAME = "CNR-EXT_FULL"
FULL_IMAGES_DIRNAME = "FULL_IMAGE_1000x750"


def repo_root() -> Path:
    """Raíz del repo (carpeta que contiene ``src/``)."""
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Raíz de los datos: ``$DATA_ROOT`` o ``<repo>/../Data``."""
    env = os.environ.get(DATA_ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root().parent / "Data").resolve()


def metadata_csv() -> Path:
    """CSV con los metadatos de todos los patches (CNRPark + EXT)."""
    return data_root() / METADATA_CSV


def ext_dir() -> Path:
    return data_root() / EXT_DIRNAME


def labels_dir() -> Path:
    """Carpeta con las particiones oficiales ``{train,val,test}.txt``."""
    return ext_dir() / "LABELS"


def patches_dir() -> Path:
    """Carpeta de patches 150x150 de CNR-EXT (las rutas de LABELS son relativas a ella)."""
    return ext_dir() / "PATCHES"


def split_file(name: str) -> Path:
    return labels_dir() / f"{name}.txt"


def cnrpark_dir() -> Path:
    """Patches de CNRPark A/B (prueba fuera de dominio)."""
    return data_root() / CNRPARK_DIRNAME


def full_dir() -> Path:
    """Frames completos de CNR-EXT y los CSV de cajas ``camera<k>.csv``."""
    return data_root() / FULL_DIRNAME


def full_images_dir() -> Path:
    return full_dir() / FULL_IMAGES_DIRNAME


def camera_boxes_csv(camera: int) -> Path:
    return full_dir() / f"camera{int(camera)}.csv"


def configs_dir() -> Path:
    return repo_root() / "configs"


def runs_dir() -> Path:
    return repo_root() / "runs"


def models_dir() -> Path:
    return repo_root() / "models"


def results_dir() -> Path:
    return repo_root() / "results"
