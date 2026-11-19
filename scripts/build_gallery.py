"""Genera ``app/gallery.json``: la galería de demo de la app, solo con imágenes nunca vistas.

Contenido:
- Frames completos de CNR-EXT de **fechas de test** (ninguna fecha de test aparece en train
  ni en val): uno por cámara (1-9) y por clima (soleado, nublado, lluvia), 27 en total, más
  dos casos difíciles (temprano con lluvia y al atardecer).
- Patches de CNRPark A/B (otro parqueo y otras cámaras: fuera de dominio): 8 por cámara,
  4 de la carpeta ``busy`` y 4 de ``free``.
- Dos mosaicos (uno de A y uno de B): cuadrícula 4x3 de 12 patches de CNRPark (6 ``busy`` y
  6 ``free``, distintos de los anteriores). La app compone la imagen al vuelo y usa cada celda
  como un espacio.

La selección es determinista y usa solo metadatos (clima, cámara, hora y carpeta de la
clase en CNRPark); no usa predicciones de ningún modelo ni métricas de test, así que no
influye en la selección de modelos ni de umbrales. Para los frames solo se comprueba que
existan etiquetas en el CSV cerca de esa hora (cobertura), sin mirar sus valores.

El JSON guarda rutas relativas a DATA_ROOT y metadatos; las imágenes no se copian al repo.

Uso:
    python scripts/build_gallery.py [--out app/gallery.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cupo import data, frames, paths  # noqa: E402

DEFAULT_OUT = paths.repo_root() / "app" / "gallery.json"
SEED = 0
WEATHER_ES = {"SUNNY": "soleado", "OVERCAST": "nublado", "RAINY": "lluvia"}
WEATHERS: tuple[str, ...] = ("SUNNY", "OVERCAST", "RAINY")
CAMERAS: tuple[int, ...] = tuple(range(1, 10))
PATCHES_PER_FOLDER = 4          # 4 busy + 4 free por cámara de CNRPark
MOSAIC_PER_FOLDER = 6           # 6 busy + 6 free por mosaico
MOSAIC_GRID = {"cols": 4, "rows": 3}
MOSAIC_CELL = 150
FOLDERS: tuple[str, ...] = ("busy", "free")
# Horas objetivo: rotan con la cámara y el clima para que la galería cubra el día
TARGET_TIMES: tuple[str, ...] = ("09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30",
                                 "11:00", "13:00")


@dataclass(frozen=True)
class FrameSpec:
    """Frame buscado: clima, cámara y hora objetivo ('HH:MM'); se toma el más cercano."""

    weather: str
    camera: int
    target_time: str
    note: str = ""
    hard: bool = False


# Horas de la versión anterior de la galería (se conservan esos ejemplos)
CURATED: dict[tuple[str, int], tuple[str, str]] = {
    ("SUNNY", 1): ("12:00", ""),
    ("SUNNY", 5): ("15:00", "sombras largas de la tarde"),
    ("OVERCAST", 2): ("11:00", ""),
    ("OVERCAST", 7): ("14:00", ""),
    ("RAINY", 3): ("10:00", ""),
    ("RAINY", 9): ("13:00", ""),
}
HARD_SPECS: tuple[FrameSpec, ...] = (
    FrameSpec("RAINY", 6, "07:15", "caso difícil: temprano, con lluvia y poca luz", hard=True),
    FrameSpec("SUNNY", 8, "17:20", "caso difícil: al atardecer, con muy poca luz", hard=True),
)


def frame_specs() -> list[FrameSpec]:
    """Uno por clima y cámara (27) y después los casos difíciles."""
    specs = []
    for w_idx, weather in enumerate(WEATHERS):
        for camera in CAMERAS:
            default = TARGET_TIMES[(camera - 1 + 3 * w_idx) % len(TARGET_TIMES)]
            target, note = CURATED.get((weather, camera), (default, ""))
            specs.append(FrameSpec(weather, camera, target, note))
    return specs + list(HARD_SPECS)


def _minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def _relative(path: str | Path) -> str:
    """Ruta relativa a DATA_ROOT con separadores '/' (portable entre sistemas)."""
    return Path(path).resolve().relative_to(paths.data_root()).as_posix()


def _has_labels(row: pd.Series) -> bool:
    """True si el CSV tiene etiquetas de ese frame (solo cobertura; no se usan los valores)."""
    # Fecha de test: lectura permitida solo para armar la demo, no para elegir modelos
    gt = frames.ground_truth(int(row["camera"]), row["date"], row["time"], allow_test=True)
    return len(gt) > 0


def pick_frame(test_frames: pd.DataFrame, spec: FrameSpec) -> pd.Series:
    """Frame de fecha de test con ese clima y cámara más cercano a la hora objetivo."""
    cand = test_frames[(test_frames["weather"] == spec.weather) & (test_frames["camera"] == spec.camera)]
    if cand.empty:
        raise ValueError(f"No hay frames de test para {spec}")
    cand = cand.assign(delta=(cand["time"].map(_minutes) - _minutes(spec.target_time)).abs())
    for _, row in cand.sort_values(["delta", "date", "time"]).iterrows():
        if _has_labels(row):
            return row
    raise ValueError(f"Ningún frame de {spec} tiene etiquetas en el CSV")


def frame_item(row: pd.Series, spec: FrameSpec) -> dict[str, Any]:
    weather = WEATHER_ES[row["weather"]]
    item = {
        "id": f"ext-c{row['camera']}-{row['date']}-{row['time'].replace(':', '')}",
        "kind": "frame",
        "subset": "CNR-EXT",
        "path": _relative(row["path"]),
        "camera": int(row["camera"]),
        "weather": row["weather"],
        "date": row["date"],
        "time": row["time"],
        "title": f"Cámara {row['camera']}, {weather}, {row['time']}",
        "note": spec.note,
        "split_origin": "fecha de test (nunca vista en entrenamiento)",
    }
    if spec.hard:
        item["hard"] = True
    return item


def build_frames() -> list[dict[str, Any]]:
    all_frames = frames.list_frames(allow_test=True)
    test_frames = all_frames[all_frames["is_test_date"]]
    items: dict[str, dict[str, Any]] = {}
    for spec in frame_specs():
        item = frame_item(pick_frame(test_frames, spec), spec)
        # Si un caso difícil coincide con un frame ya elegido, no se duplica
        items.setdefault(item["id"], item)
    return list(items.values())


def _folder(path: str) -> str:
    """Carpeta de la clase en CNRPark (``busy`` o ``free``)."""
    return Path(path).parts[-2]


def split_patches(cam_df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(patches sueltos, celdas del mosaico) de una cámara, balanceados por carpeta y sin repetir."""
    singles: dict[str, pd.DataFrame] = {}
    cells: list[pd.DataFrame] = []
    folders = cam_df["path"].map(_folder)
    for folder in FOLDERS:
        group = cam_df[folders == folder]
        idx = rng.choice(len(group), size=PATCHES_PER_FOLDER + MOSAIC_PER_FOLDER, replace=False)
        singles[folder] = group.iloc[np.sort(idx[:PATCHES_PER_FOLDER])]
        cells.append(group.iloc[np.sort(idx[PATCHES_PER_FOLDER:])])
    # Patches sueltos alternando busy/free; celdas del mosaico en un orden aleatorio fijo
    single = pd.concat([singles[f].iloc[[i]] for i in range(PATCHES_PER_FOLDER) for f in FOLDERS])
    mosaic = pd.concat(cells)
    return single, mosaic.iloc[rng.permutation(len(mosaic))]


def patch_item(row: pd.Series) -> dict[str, Any]:
    weather = WEATHER_ES.get(row["weather"], str(row["weather"]).lower())
    return {
        "id": f"ood-{row['camera'].lower()}-{Path(row['path']).stem}",
        "kind": "patch",
        "subset": "CNRPark",
        "path": _relative(row["path"]),
        "camera": row["camera"],
        "weather": row["weather"],
        "date": row["date"],
        "time": row["time"],
        "title": f"CNRPark {row['camera']}, espacio {row['slot_id']}, {weather}, {row['time']}",
        "note": "otro parqueo y otra cámara (fuera de dominio)",
        "split_origin": "CNRPark A/B (nunca usado para entrenar)",
    }


def mosaic_item(camera: str, cells: pd.DataFrame) -> dict[str, Any]:
    """Mosaico de CNRPark: la app compone la imagen y usa cada celda como un espacio."""
    return {
        "id": f"mosaic-{camera.lower()}",
        "kind": "mosaic",
        "subset": "CNRPark",
        "camera": camera,
        "title": f"CNRPark {camera}: mosaico de {len(cells)} espacios",
        "note": "otro parqueo: patches de 150×150 en una cuadrícula 4×3 (fuera de dominio)",
        "split_origin": "CNRPark A/B (nunca usado para entrenar)",
        "grid": dict(MOSAIC_GRID),
        "cell_size": MOSAIC_CELL,
        "cells": [{"slot_id": str(i), "path": _relative(row["path"]), "source_slot": str(row["slot_id"]),
                   "date": row["date"], "time": row["time"]}
                  for i, (_, row) in enumerate(cells.iterrows(), start=1)],
    }


def build_patches() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(patches sueltos, mosaicos) de CNRPark A y B."""
    # CNRPark se lee solo para armar la demo; no se evalúa ni se usa para elegir nada
    ood = data.load_split("ood", allow_test=True).sort_values("path").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    singles: list[dict[str, Any]] = []
    mosaics: list[dict[str, Any]] = []
    for cam in ("A", "B"):
        single, cells = split_patches(ood[ood["camera"] == cam], rng)
        singles += [patch_item(row) for _, row in single.iterrows()]
        mosaics.append(mosaic_item(cam, cells))
    return singles, mosaics


def build_gallery() -> dict[str, Any]:
    singles, mosaics = build_patches()
    return {
        "description": ("Galería de demo: solo imágenes nunca vistas en el entrenamiento "
                        "(frames de fechas de test de CNR-EXT y patches y mosaicos de CNRPark A/B)."),
        "generated_by": "scripts/build_gallery.py",
        "paths_relative_to": "DATA_ROOT",
        "seed": SEED,
        "items": build_frames() + mosaics + singles,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Genera la galería de demo de la app.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="ruta del JSON (por defecto app/gallery.json)")
    args = parser.parse_args(argv)
    gallery = build_gallery()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gallery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in gallery["items"]:
        where = item.get("path") or f"{len(item.get('cells', []))} patches"
        print(f"{item['id']:<40} {where}")
    print(f"{len(gallery['items'])} elementos escritos en {out}")


if __name__ == "__main__":
    main()
