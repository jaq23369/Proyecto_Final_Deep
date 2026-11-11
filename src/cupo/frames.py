"""Frames completos de CNR-EXT (1000x750): listado, cajas de espacios, recortes y etiquetas.

- Frames: ``CNR-EXT_FULL/FULL_IMAGE_1000x750/<WEATHER>/<YYYY-MM-DD>/camera<k>/<YYYY-MM-DD>_<HHMM>.jpg``.
- Cajas: ``CNR-EXT_FULL/camera<k>.csv`` con ``SlotId,X,Y,W,H`` en coordenadas 2592x1944.
- Etiquetas: ``CNRPark+EXT.csv`` (EXT usa camera '01'..'09' y datetime 'YYYY-MM-DD_HH.MM').

Este módulo no depende de torch (lo usa la app con ONNX Runtime).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache

import pandas as pd
from PIL import Image

from cupo import paths

FULL_SIZE: tuple[int, int] = (1000, 750)      # (ancho, alto) de los frames completos
ORIGINAL_SIZE: tuple[int, int] = (2592, 1944)  # (ancho, alto) de las coordenadas de los CSV
FRAME_COLUMNS: list[str] = ["path", "weather", "date", "time", "camera", "is_test_date"]

# Ejemplo: SUNNY/2015-11-12/camera1/2015-11-12_0709.jpg
_FRAME_RE = (
    r"(?P<weather>SUNNY|OVERCAST|RAINY)[\\/](?P<date>\d{4}-\d{2}-\d{2})[\\/]camera(?P<camera>\d+)[\\/]"
    r"\d{4}-\d{2}-\d{2}_(?P<hh>\d{2})(?P<mm>\d{2})\.jpg$"
)


@lru_cache(maxsize=4)
def _test_dates(root: str) -> frozenset[str]:
    """Fechas de ``test.txt``; solo se usan para marcar o excluir frames, nunca para evaluar."""
    rel = pd.read_csv(paths.split_file("test"), sep=" ", header=None, usecols=[0], dtype=str)[0]
    return frozenset(rel.str.split("/", n=2).str[1].unique())


def list_frames(allow_test: bool = False) -> pd.DataFrame:
    """Lista los frames completos con su clima, fecha, hora ('HH:MM') y cámara (int).

    ``is_test_date`` indica si la fecha aparece en ``test.txt``. Con ``allow_test=False``
    esos frames se excluyen.
    """
    files = sorted(str(p) for p in paths.full_images_dir().glob("*/*/camera*/*.jpg"))
    meta = pd.Series(files, dtype=str).str.extract(_FRAME_RE)
    df = pd.DataFrame({
        "path": files,
        "weather": meta["weather"],
        "date": meta["date"],
        "time": meta["hh"] + ":" + meta["mm"],
        "camera": meta["camera"].astype("Int64"),
    }).dropna(subset=["date"])
    df["camera"] = df["camera"].astype(int)
    df["is_test_date"] = df["date"].isin(_test_dates(str(paths.data_root())))
    if not allow_test:
        df = df[~df["is_test_date"]]
    return df[FRAME_COLUMNS].reset_index(drop=True)


def slot_boxes(camera: int) -> pd.DataFrame:
    """Cajas de los espacios de una cámara escaladas a 1000x750 → ``[slot_id, x, y, w, h]``."""
    raw = pd.read_csv(paths.camera_boxes_csv(camera))
    sx = FULL_SIZE[0] / ORIGINAL_SIZE[0]
    sy = FULL_SIZE[1] / ORIGINAL_SIZE[1]
    x0 = (raw["X"] * sx).round().clip(0, FULL_SIZE[0])
    y0 = (raw["Y"] * sy).round().clip(0, FULL_SIZE[1])
    x1 = ((raw["X"] + raw["W"]) * sx).round().clip(0, FULL_SIZE[0])
    y1 = ((raw["Y"] + raw["H"]) * sy).round().clip(0, FULL_SIZE[1])
    return pd.DataFrame({
        "slot_id": raw["SlotId"].astype(int).astype(str),
        "x": x0.astype(int),
        "y": y0.astype(int),
        "w": (x1 - x0).astype(int),
        "h": (y1 - y0).astype(int),
    })


def _iter_boxes(boxes: pd.DataFrame | Iterable) -> Iterable[tuple[float, float, float, float]]:
    """Acepta un DataFrame con x,y,w,h, dicts con esas llaves o tuplas (x, y, w, h)."""
    if isinstance(boxes, pd.DataFrame):
        yield from boxes[["x", "y", "w", "h"]].itertuples(index=False, name=None)
        return
    for box in boxes:
        if isinstance(box, Mapping):
            yield box["x"], box["y"], box["w"], box["h"]
        else:
            x, y, w, h = box
            yield x, y, w, h


def crop_slots(img: Image.Image, boxes: pd.DataFrame | Iterable) -> list[Image.Image]:
    """Recorta cada caja (en píxeles de ``img``) y devuelve los recortes en RGB."""
    rgb = img.convert("RGB")
    crops = []
    for x, y, w, h in _iter_boxes(boxes):
        crops.append(rgb.crop((int(x), int(y), int(x) + int(w), int(y) + int(h))))
    return crops


@lru_cache(maxsize=4)
def _ext_labels(root: str) -> pd.DataFrame:
    """Etiquetas de patches de EXT: camera (int), date, minuto del día, slot_id (str), label."""
    csv = pd.read_csv(
        paths.metadata_csv(), dtype=str,
        usecols=["camera", "datetime", "hour", "minute", "slot_id", "occupancy"],
    )
    csv = csv[~csv["camera"].isin(["A", "B"])]
    return pd.DataFrame({
        "camera": csv["camera"].astype(int),
        "date": csv["datetime"].str[:10],
        "minutes": csv["hour"].astype(int) * 60 + csv["minute"].astype(int),
        "slot_id": csv["slot_id"].astype(int).astype(str),
        "label": csv["occupancy"].astype(int),
    }).reset_index(drop=True)


def _to_minutes(time: str) -> int:
    """'HH:MM' (o 'HHMM' / 'HH.MM') → minutos desde medianoche."""
    digits = time.replace(":", "").replace(".", "")
    return int(digits[:2]) * 60 + int(digits[2:4])


def ground_truth(
    camera: int, date: str, time: str, tol_min: int = 2, allow_test: bool = False
) -> dict[str, int]:
    """Etiquetas de los espacios de un frame: ``{slot_id: 0|1}``.

    Busca en el CSV los patches de esa cámara y fecha con hora dentro de ±``tol_min``
    minutos; si un espacio tiene varios, usa el más cercano en el tiempo.
    Si la fecha es de test, lanza ``PermissionError`` salvo que ``allow_test`` sea True.
    """
    root = str(paths.data_root())
    if not allow_test and date in _test_dates(root):
        raise PermissionError(
            f"La fecha {date} pertenece a test; sus etiquetas están bloqueadas hasta el "
            "congelamiento del modelo (usa allow_test=True solo después)."
        )
    labels = _ext_labels(root)
    target = _to_minutes(time)
    sel = labels[(labels["camera"] == int(camera)) & (labels["date"] == date)]
    delta = (sel["minutes"] - target).abs()
    sel = sel.assign(delta=delta)[delta <= tol_min]
    if sel.empty:
        return {}
    best = sel.sort_values(["delta", "minutes"]).drop_duplicates("slot_id", keep="first")
    return {str(s): int(v) for s, v in zip(best["slot_id"], best["label"])}
