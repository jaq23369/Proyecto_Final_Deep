"""Particiones oficiales de CNR-EXT, Test-OOD (CNRPark A/B) y Dataset de PyTorch.

Particiones fijas:
- ``train``, ``val``, ``test``: ``CNR-EXT/LABELS/{train,val,test}.txt`` (separadas por fecha).
- ``ood``: CNRPark A+B completo, tomado de ``CNRPark+EXT.csv``.

``test`` y ``ood`` solo se cargan con ``allow_test=True`` (después del congelamiento).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cupo import paths

SplitName = Literal["train", "val", "test", "ood"]
SPLITS: tuple[str, ...] = ("train", "val", "test", "ood")
PROTECTED_SPLITS: frozenset[str] = frozenset({"test", "ood"})
COLUMNS: list[str] = ["path", "label", "camera", "weather", "date", "time", "slot_id", "subset"]
WEATHER_CODES: dict[str, str] = {"S": "SUNNY", "O": "OVERCAST", "R": "RAINY"}

# Ejemplo: SUNNY/2015-11-12/camera8/S_2015-11-12_12.26_C08_280.jpg
_EXT_PATH_RE = (
    r"^(?P<weather>SUNNY|OVERCAST|RAINY)/(?P<date>\d{4}-\d{2}-\d{2})/camera(?P<camera>\d+)/"
    r"[SOR]_\d{4}-\d{2}-\d{2}_(?P<hh>\d{2})\.(?P<mm>\d{2})_C\d+_(?P<slot_id>\d+)\.jpg$"
)


def _read_split_file(name: str) -> pd.DataFrame:
    """Lee ``LABELS/<name>.txt`` → columnas ``rel`` (ruta relativa a PATCHES) y ``label``.

    Privada: no aplica la guarda de test; el acceso público es ``load_split``.
    Las etiquetas de LABELS/*.txt difieren de ``occupancy`` del CSV en 26 patches
    (0.018 %); se usan las oficiales de LABELS.
    """
    return pd.read_csv(
        paths.split_file(name), sep=" ", header=None, names=["rel", "label"],
        dtype={"rel": str, "label": int},
    )


def split_dates(name: str) -> set[str]:
    """Fechas ('YYYY-MM-DD') presentes en una partición oficial de CNR-EXT.

    Lee solo la columna de rutas (no las etiquetas); sirve para comprobar que no
    hay fechas compartidas entre particiones.
    """
    rel = pd.read_csv(paths.split_file(name), sep=" ", header=None, usecols=[0], dtype=str)[0]
    return set(rel.str.split("/", n=2).str[1].unique())


def _parse_ext(rel_paths: pd.DataFrame) -> pd.DataFrame:
    """Construye las columnas estándar a partir de las rutas relativas de EXT."""
    meta = rel_paths["rel"].str.extract(_EXT_PATH_RE)
    bad = meta["date"].isna()
    if bad.any():
        ejemplo = rel_paths.loc[bad, "rel"].iloc[0]
        raise ValueError(f"{int(bad.sum())} rutas con formato inesperado, por ejemplo: {ejemplo}")
    prefix = str(paths.patches_dir()) + os.sep
    return pd.DataFrame({
        "path": prefix + rel_paths["rel"].str.replace("/", os.sep, regex=False),
        "label": rel_paths["label"].astype(int),
        "camera": meta["camera"].astype(int).astype(str),
        "weather": meta["weather"],
        "date": meta["date"],
        "time": meta["hh"] + ":" + meta["mm"],
        "slot_id": meta["slot_id"].astype(int).astype(str),
        "subset": "EXT",
    })


def _load_ood() -> pd.DataFrame:
    """CNRPark A+B con metadatos del CSV (datetime 'YYYYMMDD_HHMM')."""
    csv = pd.read_csv(paths.metadata_csv(), dtype=str)
    csv = csv[csv["camera"].isin(["A", "B"])].reset_index(drop=True)
    dt = csv["datetime"]
    root = str(paths.data_root()) + os.sep
    return pd.DataFrame({
        "path": root + csv["image_url"].str.replace("/", os.sep, regex=False),
        "label": csv["occupancy"].astype(int),
        "camera": csv["camera"],
        "weather": csv["weather"].map(WEATHER_CODES),
        "date": dt.str[0:4] + "-" + dt.str[4:6] + "-" + dt.str[6:8],
        "time": dt.str[9:11] + ":" + dt.str[11:13],
        "slot_id": csv["slot_id"].astype(int).astype(str),
        "subset": "CNRPark",
    })


@lru_cache(maxsize=8)
def _load_split_cached(name: str, root: str) -> pd.DataFrame:
    # ``root`` forma parte de la llave del caché por si cambia DATA_ROOT
    df = _load_ood() if name == "ood" else _parse_ext(_read_split_file(name))
    return df[COLUMNS]


def load_split(name: SplitName, allow_test: bool = False) -> pd.DataFrame:
    """Carga una partición con columnas ``COLUMNS``.

    ``test`` y ``ood`` lanzan ``PermissionError`` si ``allow_test`` es False.
    """
    if name not in SPLITS:
        raise ValueError(f"Partición desconocida: {name!r}. Opciones: {SPLITS}")
    if name in PROTECTED_SPLITS and not allow_test:
        raise PermissionError(
            f"La partición {name!r} está bloqueada hasta el congelamiento del modelo; "
            "usa allow_test=True solo para la evaluación final."
        )
    return _load_split_cached(name, str(paths.data_root())).copy()


def subsample(df: pd.DataFrame, frac: float, seed: int = 42) -> pd.DataFrame:
    """Submuestra estratificada por ``label`` (al menos 1 ejemplo por clase).

    Mantiene el orden original de las filas e índice reiniciado.
    """
    if not 0 < frac <= 1:
        raise ValueError(f"frac debe estar en (0, 1], se recibió {frac}")
    df = df.reset_index(drop=True)
    if frac >= 1:
        return df.copy()
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for _, group in df.groupby("label", sort=True):
        n = max(1, int(round(frac * len(group))))
        keep.append(rng.choice(group.index.to_numpy(), size=n, replace=False))
    idx = np.sort(np.concatenate(keep))
    return df.loc[idx].reset_index(drop=True)


class ParkingDataset(Dataset):
    """Dataset de patches: devuelve ``(transform(imagen RGB), label)``.

    Guarda solo listas de rutas y etiquetas para que sea picklable
    (Windows usa ``spawn`` con ``num_workers > 0``).
    """

    def __init__(self, df: pd.DataFrame, transform: Callable | None = None) -> None:
        self.paths: list[str] = df["path"].astype(str).tolist()
        self.labels: list[int] = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def make_loader(
    df: pd.DataFrame,
    transform: Callable | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """DataLoader reproducible (generador con semilla) sobre ``ParkingDataset``."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ParkingDataset(df, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )
