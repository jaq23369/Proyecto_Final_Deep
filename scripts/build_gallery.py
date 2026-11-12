"""Genera ``app/gallery.json``: la galería de demo de la app, solo con imágenes nunca vistas.

Contenido:
- Frames completos de CNR-EXT de **fechas de test** (ninguna fecha de test aparece en train
  ni en val): ~2 por clima, varias cámaras y casos difíciles (temprano y al atardecer).
- Patches de CNRPark A/B (otro parqueo y otras cámaras: fuera de dominio).

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
PATCHES_PER_CAMERA = 3


@dataclass(frozen=True)
class FrameSpec:
    """Frame buscado: clima, cámara y hora objetivo ('HH:MM'); se toma el más cercano."""

    weather: str
    camera: int
    target_time: str
    note: str = ""


# Lista curada: 2 por clima, 8 cámaras distintas y dos casos difíciles
FRAME_SPECS: tuple[FrameSpec, ...] = (
    FrameSpec("SUNNY", 1, "12:00"),
    FrameSpec("SUNNY", 5, "15:00", "sombras largas de la tarde"),
    FrameSpec("OVERCAST", 2, "11:00"),
    FrameSpec("OVERCAST", 7, "14:00"),
    FrameSpec("RAINY", 3, "10:00"),
    FrameSpec("RAINY", 9, "13:00"),
    FrameSpec("RAINY", 6, "07:15", "caso difícil: temprano, con lluvia y poca luz"),
    FrameSpec("SUNNY", 8, "17:20", "caso difícil: al atardecer, con muy poca luz"),
)


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
    return {
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


def build_frames() -> list[dict[str, Any]]:
    all_frames = frames.list_frames(allow_test=True)
    test_frames = all_frames[all_frames["is_test_date"]]
    return [frame_item(pick_frame(test_frames, spec), spec) for spec in FRAME_SPECS]


def _pick_patches(cam_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Un patch por clase (carpeta busy/free) y uno más al azar, sin repetir."""
    picks = [cam_df[cam_df["label"] == lbl].iloc[[rng.integers(0, (cam_df["label"] == lbl).sum())]]
             for lbl in (1, 0)]
    rest = cam_df.drop(index=pd.concat(picks).index)
    picks.append(rest.iloc[[rng.integers(0, len(rest))]])
    return pd.concat(picks)


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


def build_patches() -> list[dict[str, Any]]:
    # CNRPark se lee solo para armar la demo; no se evalúa ni se usa para elegir nada
    ood = data.load_split("ood", allow_test=True).sort_values("path").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    items: list[dict[str, Any]] = []
    for cam in ("A", "B"):
        sel = _pick_patches(ood[ood["camera"] == cam], rng)
        items += [patch_item(row) for _, row in sel.iterrows()]
    return items


def build_gallery() -> dict[str, Any]:
    return {
        "description": ("Galería de demo: solo imágenes nunca vistas en el entrenamiento "
                        "(frames de fechas de test de CNR-EXT y patches de CNRPark A/B)."),
        "generated_by": "scripts/build_gallery.py",
        "paths_relative_to": "DATA_ROOT",
        "seed": SEED,
        "items": build_frames() + build_patches(),
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
        print(f"{item['id']:<40} {item['path']}")
    print(f"{len(gallery['items'])} elementos escritos en {out}")


if __name__ == "__main__":
    main()
