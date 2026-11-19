"""API y dashboard de Cupo Seguro.

Arranque (desde la raíz del repo):
    uvicorn app.main:app --port 8000

Variables de entorno:
- ``MODEL_PATH``: modelo por defecto (``models/best.onnx``; ``best.json`` va al lado). Los demás
  pares ``<stem>.onnx|.joblib`` + ``<stem>.json`` de esa carpeta se pueden elegir con ``model``.
- ``DATA_ROOT``: datos de CNRPark+EXT para la galería (por defecto ``../Data``).
- ``OWN_GALLERY_DIR``: fotos propias con sus cajas (por defecto ``app/galeria_propia``).
- ``FEEDBACK_PATH``: archivo de retroalimentación (por defecto ``app/feedback.jsonl``).
- ``SHOW_GROUND_TRUTH``: con ``1`` la galería muestra las etiquetas reales como comparación
  visual. Desactivada por defecto: solo se activa después del congelamiento del modelo.

Endpoints: ``GET /``, ``GET /api/models``, ``GET /api/model``, ``GET /api/gallery``,
``GET /api/gallery/{id}/image``, ``POST /api/predict/frame``, ``POST /api/predict/patch``,
``POST /api/predict/custom`` y ``POST /api/feedback``.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from app.inference import (
    SCENARIOS,
    STATE_BUSY,
    STATE_FREE,
    STATE_VERIFY,
    ModelStore,
    ModelUnavailableError,
    Predictor,
    UnknownModelError,
    decide,
    default_model_id,
    image_quality,
    list_models,
    load_benchmark,
    model_path,
    resolve_model,
)
from cupo import frames, paths
from cupo.value_model import load_scenario

logger = logging.getLogger("cupo.app")

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
GALLERY_PATH = APP_DIR / "gallery.json"
FEEDBACK_ENV = "FEEDBACK_PATH"
SHOW_GT_ENV = "SHOW_GROUND_TRUTH"
OWN_GALLERY_ENV = "OWN_GALLERY_DIR"
DEFAULT_OWN_GALLERY = APP_DIR / "galeria_propia"

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_FORM_OVERHEAD = 64 * 1024          # margen para los demás campos del formulario
MAX_JSON_BYTES = 256 * 1024            # cuerpo JSON de predicción (cajas incluidas)
MAX_FEEDBACK_BYTES = 2 * 1024
MAX_PIXELS = 50_000_000
MIN_FRAME_SIZE = (320, 240)            # por debajo no se distinguen los espacios de un frame
MAX_BOXES = 300
MIN_BOX_SIDE = 4
EXT_CAMERAS = range(1, 10)
PATCH_MAX_SIDE = 600                   # más grande que esto ya no parece la foto de un solo espacio
ASPECT_TOLERANCE = 0.03
MOSAIC_CELL = 150                      # lado de cada celda de los mosaicos de CNRPark
ALL_STATES = (STATE_FREE, STATE_BUSY, STATE_VERIFY)
REPORTED_STATES = (STATE_BUSY, STATE_FREE)
# \Z y no $: $ también acepta un salto de línea final
SLOT_ID_RE = re.compile(r"^[\w-]{1,32}\Z")
UPLOAD_SOURCE = "upload"
# Fotos propias: <nombre>.jpg|jpeg|png + <nombre>.json; el id es "own-<nombre>"
OWN_PREFIX = "own-"
OWN_IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
OWN_NAME_RE = re.compile(r"^\w[\w .-]{0,79}\Z")
DRAW_TAB = "'Dibujar espacios'"

GT_NOTE = ("Etiquetas reales del dataset (SHOW_GROUND_TRUTH=1), solo como comparación visual; "
           "no se usan para elegir modelos ni umbrales.")
OTHER_CAMERA_WARNING = (
    "Este modo solo sirve para fotos de las 9 cámaras de CNR-EXT (1000×750): las cajas de los "
    "espacios son fijas por cámara y en otra foto caen en cualquier lado. Si es de otro parqueo, "
    f"usa {DRAW_TAB} y dibuja cada espacio."
)

app = FastAPI(title="Cupo Seguro", version="1.1",
              description="Clasificación de espacios de parqueo: libre, ocupado o por verificar.")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

store = ModelStore()
_feedback_lock = threading.Lock()


@app.exception_handler(ModelUnavailableError)
async def _model_unavailable(_: Request, exc: ModelUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(UnknownModelError)
async def _unknown_model(_: Request, exc: UnknownModelError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
    # Último recurso: mensaje claro sin detalles internos (el detalle queda en el log)
    logger.exception("Error no controlado: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={
        "detail": "Error interno al procesar la petición; intenta con otra imagen o revisa el log del servidor."})


def show_ground_truth() -> bool:
    """Etiquetas reales en la galería: solo con ``SHOW_GROUND_TRUTH=1`` (tras el congelamiento)."""
    return os.environ.get(SHOW_GT_ENV, "").strip() == "1"


def _short(name: str, limit: int = 60) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Lectura de la petición
# ---------------------------------------------------------------------------

@dataclass
class Payload:
    """Campos de la petición (JSON o formulario) y la imagen subida, si hay."""

    fields: dict[str, Any] = field(default_factory=dict)
    file_bytes: bytes | None = None
    filename: str = ""


def _check_length(request: Request, limit: int) -> None:
    """Rechaza por ``Content-Length`` antes de leer el cuerpo."""
    raw = request.headers.get("content-length")
    if raw and raw.isdigit() and int(raw) > limit:
        raise HTTPException(413, f"La petición supera el máximo de {limit // 1024} KB.")


async def _read_json(request: Request, limit: int) -> dict[str, Any]:
    _check_length(request, limit)
    body = await request.body()
    if len(body) > limit:
        raise HTTPException(413, f"La petición supera el máximo de {limit // 1024} KB.")
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(400, "El cuerpo de la petición no es un JSON válido.")
    if not isinstance(data, dict):
        raise HTTPException(400, "El JSON debe ser un objeto con campos.")
    return data


async def read_payload(request: Request) -> Payload:
    """Acepta ``application/json`` o ``multipart/form-data`` (campo ``file`` o ``image``)."""
    ctype = request.headers.get("content-type", "").lower()
    if ctype.startswith("application/json"):
        return Payload(fields=await _read_json(request, MAX_JSON_BYTES))
    if ctype.startswith("multipart/form-data") or ctype.startswith("application/x-www-form-urlencoded"):
        _check_length(request, MAX_UPLOAD_BYTES + MAX_FORM_OVERHEAD)
        try:
            form = await request.form()
        except Exception:  # cuerpo multipart mal formado
            raise HTTPException(400, "El formulario no es válido; envía la imagen como multipart/form-data.")
        payload = Payload(fields={k: v for k, v in form.items() if not isinstance(v, UploadFile)})
        upload = form.get("file") or form.get("image")
        if isinstance(upload, UploadFile):
            payload.file_bytes = await upload.read(MAX_UPLOAD_BYTES + 1)
            payload.filename = _short(upload.filename or "archivo")
        return payload
    raise HTTPException(400, "Envía JSON ({gallery_id: ...}) o un formulario multipart con la imagen.")


def open_upload(data: bytes, filename: str) -> Image.Image:
    """Abre la imagen subida en RGB (respeta la orientación EXIF de las fotos de celular)."""
    if not data:
        raise HTTPException(400, f"El archivo '{filename}' está vacío.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"El archivo supera el máximo de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise HTTPException(413, f"La imagen '{filename}' tiene demasiados píxeles.")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise HTTPException(400, f"El archivo '{filename}' no es una imagen válida (usa JPG o PNG).")
    # Image.open solo lee la cabecera: el tamaño se revisa antes de decodificar
    if img.width * img.height > MAX_PIXELS:
        raise HTTPException(413, f"La imagen '{filename}' ({img.width}x{img.height}) supera el máximo de "
                                 f"{MAX_PIXELS // 1_000_000} megapíxeles.")
    try:
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB")
    except Image.DecompressionBombError:
        raise HTTPException(413, f"La imagen '{filename}' tiene demasiados píxeles.")
    except (OSError, SyntaxError, ValueError):
        raise HTTPException(400, f"El archivo '{filename}' no es una imagen válida o está dañado.")


def parse_scenario(fields: dict[str, Any]) -> str:
    scenario = str(fields.get("scenario") or SCENARIOS[0]).strip().lower()
    if scenario not in SCENARIOS:
        raise HTTPException(400, f"Escenario inválido: '{_short(scenario, 30)}'. Opciones: {', '.join(SCENARIOS)}.")
    return scenario


def parse_model(value: Any) -> str | None:
    """Id de modelo pedido (``None`` = el de por defecto); uno desconocido da 400."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise HTTPException(400, "model debe ser el id de un modelo de GET /api/models.")
    return resolve_model(value)[0]


def parse_camera(raw: Any) -> int:
    if raw is None or str(raw).strip() == "":
        raise HTTPException(400, "Indica la cámara (1-9) para ubicar los espacios. Para una foto de "
                                 f"otro parqueo usa el modo {DRAW_TAB}.")
    try:
        camera = int(str(raw).strip())
    except ValueError:
        camera = -1
    if camera not in EXT_CAMERAS:
        raise HTTPException(400, f"Cámara inválida: '{_short(str(raw), 20)}'. Usa un número del 1 al 9 "
                                 "(cámaras de CNR-EXT).")
    return camera


# ---------------------------------------------------------------------------
# Galería del dataset (frames, patches y mosaicos)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def gallery_items() -> dict[str, dict[str, Any]]:
    try:
        items = json.loads(GALLERY_PATH.read_text(encoding="utf-8")).get("items", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        logger.warning("No se pudo leer %s; la galería queda vacía", GALLERY_PATH.name)
        return {}
    return {item["id"]: item for item in items if isinstance(item, dict) and "id" in item}


def gallery_item(gallery_id: Any) -> dict[str, Any]:
    """Ejemplo de la galería del dataset o de las fotos propias; 404 si no existe."""
    key = str(gallery_id)
    item = gallery_items().get(key)
    if item is None and key.startswith(OWN_PREFIX):
        item = own_items().get(key)
    if item is None:
        raise HTTPException(404, f"No existe el ejemplo '{_short(key, 40)}' en la galería.")
    return item


def item_paths(item: dict[str, Any]) -> list[str]:
    """Rutas (relativas a DATA_ROOT) que necesita un ejemplo del dataset."""
    if item.get("kind") == "mosaic":
        return [cell["path"] for cell in item.get("cells", [])]
    return [item["path"]] if "path" in item else []


def item_available(item: dict[str, Any]) -> bool:
    if item.get("kind") == "own":
        return Path(item["_file"]).is_file()
    root = paths.data_root()
    rels = item_paths(item)
    return bool(rels) and all((root / rel).is_file() for rel in rels)


def _missing_data(item: dict[str, Any]) -> HTTPException:
    return HTTPException(404, f"La imagen '{item['id']}' no está en DATA_ROOT; descarga los datos con "
                              "python scripts/download_data.py.")


def gallery_file(item: dict[str, Any]) -> Path:
    path = paths.data_root() / item["path"]
    if not path.is_file():
        raise _missing_data(item)
    return path


def open_gallery_image(item: dict[str, Any]) -> Image.Image:
    try:
        with Image.open(gallery_file(item)) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(503, f"No se pudo leer la imagen '{item['id']}' de la galería en DATA_ROOT.")


def mosaic_layout(item: dict[str, Any]) -> tuple[int, int, int]:
    """(columnas, filas, lado de celda) del mosaico."""
    grid = item.get("grid") or {}
    return int(grid.get("cols", 4)), int(grid.get("rows", 3)), int(item.get("cell_size", MOSAIC_CELL))


def mosaic_boxes(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Una caja por celda, en el orden de ``cells`` (fila por fila)."""
    cols, _, cell = mosaic_layout(item)
    return [{"slot_id": str(c["slot_id"]), "x": (i % cols) * cell, "y": (i // cols) * cell, "w": cell, "h": cell}
            for i, c in enumerate(item["cells"])]


@lru_cache(maxsize=8)
def _mosaic_cached(gallery_id: str, root: str) -> Image.Image:
    # ``root`` forma parte de la llave del caché por si cambia DATA_ROOT
    item = gallery_items()[gallery_id]
    cols, rows, cell = mosaic_layout(item)
    canvas = Image.new("RGB", (cols * cell, rows * cell), (40, 40, 40))
    for box, c in zip(mosaic_boxes(item), item["cells"]):
        with Image.open(Path(root) / c["path"]) as patch:
            tile = patch.convert("RGB")
        if tile.size != (cell, cell):
            tile = tile.resize((cell, cell), Image.Resampling.BILINEAR)
        canvas.paste(tile, (box["x"], box["y"]))
    return canvas


def mosaic_image(item: dict[str, Any]) -> Image.Image:
    """Compone el mosaico al vuelo con los patches de CNRPark (cuadrícula de celdas de 150)."""
    if not item_available(item):
        raise _missing_data(item)
    try:
        return _mosaic_cached(item["id"], str(paths.data_root())).copy()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(503, f"No se pudo leer algún patch del mosaico '{item['id']}' en DATA_ROOT.")


def _folder_label(rel_path: str) -> int | None:
    """Etiqueta de un patch de CNRPark según su carpeta (busy = 1, free = 0)."""
    parts = Path(rel_path).parts
    if "busy" in parts:
        return 1
    if "free" in parts:
        return 0
    return None


def frame_ground_truth(item: dict[str, Any]) -> dict[str, int] | None:
    """Etiquetas reales de un frame de la galería (fecha de test), solo con ``SHOW_GROUND_TRUTH=1``.

    Es una demostración posterior al congelamiento del modelo: las etiquetas solo se muestran
    como comparación visual y no se usan para seleccionar modelos, umbrales ni nada más.
    """
    if not show_ground_truth():
        return None
    return _frame_ground_truth(item["id"])


def _frame_ground_truth(gallery_id: str) -> dict[str, int] | None:
    return _frame_ground_truth_cached(gallery_id, str(paths.data_root()))


@lru_cache(maxsize=64)
def _frame_ground_truth_cached(gallery_id: str, root: str) -> dict[str, int] | None:
    item = gallery_item(gallery_id)
    try:
        # allow_test=True: solo para la comparación visual (ver frame_ground_truth)
        return frames.ground_truth(item["camera"], item["date"], item["time"], allow_test=True)
    except Exception:  # las etiquetas son opcionales: sin CSV la demo sigue sin comparación
        logger.warning("Sin etiquetas para %s", gallery_id)
        return None


def patch_ground_truth(item: dict[str, Any]) -> int | None:
    """Etiqueta de un patch de CNRPark de la galería según su carpeta (busy/free).

    Solo con ``SHOW_GROUND_TRUTH=1`` (después del congelamiento) y como comparación visual.
    """
    if not show_ground_truth():
        return None
    return _folder_label(item["path"])


def mosaic_ground_truth(item: dict[str, Any]) -> dict[str, int] | None:
    """Etiquetas de las celdas de un mosaico (carpeta busy/free), solo con ``SHOW_GROUND_TRUTH=1``."""
    if not show_ground_truth():
        return None
    out = {str(c["slot_id"]): _folder_label(c["path"]) for c in item["cells"]}
    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------------------
# Galería de fotos propias (carpeta local, no se versiona)
# ---------------------------------------------------------------------------

def own_gallery_dir() -> Path:
    """``$OWN_GALLERY_DIR`` (las relativas, desde el repo) o ``app/galeria_propia``."""
    raw = os.environ.get(OWN_GALLERY_ENV)
    if not raw:
        return DEFAULT_OWN_GALLERY
    path = Path(raw).expanduser()
    return path if path.is_absolute() else paths.repo_root() / path


def read_boxes_file(path: Path) -> list[Any] | None:
    """Cajas de ``<foto>.json`` (``{image, boxes: [...]}`` o una lista); None si no hay o no es válido."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("No se pudo leer %s", path.name)
        return None
    boxes = data.get("boxes") if isinstance(data, dict) else data
    return boxes if isinstance(boxes, list) and boxes else None


def _own_item(image: Path) -> dict[str, Any]:
    boxes_file = image.with_suffix(".json")
    boxes = read_boxes_file(boxes_file)
    return {
        "id": OWN_PREFIX + image.stem,
        "kind": "own",
        "subset": "UVG",
        "title": image.stem.replace("_", " "),
        "filename": image.name,
        "has_boxes": boxes is not None,
        "n_boxes": len(boxes) if boxes else 0,
        "note": "" if boxes else "sin cajas guardadas: se abre en modo dibujar",
        "_file": str(image),
        "_boxes_file": str(boxes_file),
    }


def own_items() -> dict[str, dict[str, Any]]:
    """Fotos propias ``<foto>.jpg|jpeg|png`` de la carpeta local (se vuelve a leer en cada llamada).

    Los ids salen del listado de la carpeta y los nombres se validan: no se aceptan rutas, así que
    un id no puede apuntar fuera de la carpeta.
    """
    folder = own_gallery_dir()
    if not folder.is_dir():
        return {}
    base = folder.resolve()
    items: dict[str, dict[str, Any]] = {}
    for image in sorted(folder.iterdir()):
        if (image.suffix.lower() not in OWN_IMAGE_TYPES or not OWN_NAME_RE.match(image.stem)
                or ".." in image.name or not image.is_file()):
            continue
        if image.resolve().parent != base:  # enlaces que apuntan fuera de la carpeta
            continue
        item = _own_item(image)
        items.setdefault(item["id"], item)
    return items


def open_own_image(item: dict[str, Any]) -> Image.Image:
    path = Path(item["_file"])
    try:
        data = path.read_bytes()
    except OSError:
        raise HTTPException(404, f"La foto '{item['filename']}' ya no está en la galería local.")
    return open_upload(data, item["filename"])


def own_boxes(item: dict[str, Any], img: Image.Image) -> tuple[list[dict[str, Any]], list[str]]:
    raw = read_boxes_file(Path(item["_boxes_file"]))
    if raw is None:
        stem = Path(item["filename"]).stem
        raise HTTPException(400, f"La foto '{item['filename']}' no tiene cajas guardadas ({stem}.json): "
                                 f"dibújalas en {DRAW_TAB} y guárdalas con 'Guardar cajas'.")
    return parse_boxes(raw, img.width, img.height)


def open_item_image(item: dict[str, Any]) -> Image.Image:
    kind = item.get("kind")
    if kind == "mosaic":
        return mosaic_image(item)
    if kind == "own":
        return open_own_image(item)
    return open_gallery_image(item)


def source_info(item: dict[str, Any] | None, filename: str = "") -> dict[str, Any]:
    if item is None:
        return {"type": UPLOAD_SOURCE, "filename": filename}
    keys = ("id", "kind", "subset", "title", "camera", "weather", "date", "time", "note")
    return {"type": "gallery", **{k: item.get(k) for k in keys}}


# ---------------------------------------------------------------------------
# Predicción y respuesta
# ---------------------------------------------------------------------------

def summarize(states: list[str]) -> dict[str, int]:
    return {"libres": states.count(STATE_FREE), "ocupados": states.count(STATE_BUSY),
            "por_verificar": states.count(STATE_VERIFY), "total": len(states)}


def compare_ground_truth(states: list[str], truth: list[int | None]) -> dict[str, int]:
    """Aciertos y errores contra las etiquetas reales (los amarillos no cuentan como error)."""
    out = {"aciertos": 0, "errores": 0, "ocupado_como_libre": 0, "libre_como_ocupado": 0,
           "por_verificar": 0, "sin_etiqueta": 0}
    for state, gt in zip(states, truth):
        if gt is None:
            out["sin_etiqueta"] += 1
        elif state == STATE_VERIFY:
            out["por_verificar"] += 1
        elif (state == STATE_BUSY) == (gt == 1):
            out["aciertos"] += 1
        else:
            out["errores"] += 1
            out["ocupado_como_libre" if gt == 1 else "libre_como_ocupado"] += 1
    return out


def run_prediction(pred: Predictor, img: Image.Image, boxes: list[dict[str, Any]], scenario: str,
                   truth: dict[str, int] | None = None) -> dict[str, Any]:
    """Recorta, predice y arma la lista de espacios con sus estados en cada escenario.

    ``truth`` solo llega con ``SHOW_GROUND_TRUTH=1``; si es None, la respuesta no trae etiquetas.
    """
    p, model_ms = pred.predict(frames.crop_slots(img, boxes))
    slots: list[dict[str, Any]] = []
    for box, pv in zip(boxes, p):
        states = {s: decide(float(pv), pred.thresholds[s]["t_L"], pred.thresholds[s]["t_H"]) for s in SCENARIOS}
        slot = {"slot_id": str(box["slot_id"]),
                "box": {k: int(box[k]) for k in ("x", "y", "w", "h")},
                "p_busy": round(float(pv), 4), "state": states[scenario], "states": states}
        if truth is not None:
            slot["ground_truth"] = truth.get(str(box["slot_id"]))
        slots.append(slot)
    by_scenario = {s: summarize([sl["states"][s] for sl in slots]) for s in SCENARIOS}
    result: dict[str, Any] = {
        "scenario": scenario,
        "thresholds": pred.thresholds,
        "image": {"width": img.width, "height": img.height},
        "slots": slots,
        "summary": by_scenario[scenario],
        "summary_by_scenario": by_scenario,
        "model_ms": model_ms,
    }
    if truth is not None:
        gts = [sl["ground_truth"] for sl in slots]
        result["ground_truth"] = {
            "available": any(g is not None for g in gts), "note": GT_NOTE,
            "by_scenario": {s: compare_ground_truth([sl["states"][s] for sl in slots], gts) for s in SCENARIOS},
        }
    return result


def model_brief(pred: Predictor) -> dict[str, Any]:
    """Qué modelo produjo la predicción (va en cada respuesta)."""
    return {"id": pred.model_id, "label": pred.label, "name": pred.card.get("name"),
            "run_id": pred.card.get("run_id"), "kind": pred.kind}


def finish(result: dict[str, Any], pred: Predictor, mode: str, source: dict[str, Any],
           quality: dict[str, Any], warnings: list[str], t0: float) -> dict[str, Any]:
    model_ms = result.pop("model_ms")
    result.update({
        "mode": mode,
        "source": source,
        "model": model_brief(pred),
        "latency_ms": {"total": round((time.perf_counter() - t0) * 1000.0, 1),
                       "model": round(model_ms, 1), "n_slots": len(result["slots"])},
        "quality": {k: v for k, v in quality.items() if k != "warnings"},
        "warnings": quality["warnings"] + warnings,
    })
    return result


@lru_cache(maxsize=16)
def _camera_boxes_cached(camera: int, root: str) -> tuple[tuple[str, int, int, int, int], ...]:
    # ``root`` forma parte de la llave del caché por si cambia DATA_ROOT
    boxes = frames.slot_boxes(camera)
    return tuple((str(r.slot_id), int(r.x), int(r.y), int(r.w), int(r.h)) for r in boxes.itertuples(index=False))


def camera_boxes(camera: int) -> tuple[tuple[str, int, int, int, int], ...]:
    """Cajas de una cámara en 1000x750 (se leen del CSV una sola vez)."""
    try:
        return _camera_boxes_cached(camera, str(paths.data_root()))
    except Exception:  # CSV ausente o con otro formato
        raise HTTPException(503, f"Faltan las cajas de la cámara {camera} en DATA_ROOT; usa el modo "
                                 f"{DRAW_TAB}.")


def scaled_camera_boxes(camera: int, img: Image.Image) -> tuple[list[dict[str, Any]], list[str]]:
    """Cajas de la cámara (en 1000x750) escaladas al tamaño de la imagen recibida."""
    full_w, full_h = frames.FULL_SIZE
    sx, sy = img.width / full_w, img.height / full_h
    warnings: list[str] = []
    if abs(sx / sy - 1.0) > ASPECT_TOLERANCE:
        warnings.append(f"La imagen ({img.width}x{img.height}) no tiene la proporción 4:3 de las cámaras "
                        "de CNR-EXT; las cajas pueden no coincidir con los espacios.")
    out = [{"slot_id": sid, "x": round(x * sx), "y": round(y * sy), "w": round(w * sx), "h": round(h * sy)}
           for sid, x, y, w, h in camera_boxes(camera)]
    return out, warnings


def _check_frame_size(img: Image.Image) -> None:
    if img.width < MIN_FRAME_SIZE[0] or img.height < MIN_FRAME_SIZE[1]:
        raise HTTPException(400, f"La imagen ({img.width}x{img.height}) es demasiado pequeña para un frame "
                                 f"completo (mínimo {MIN_FRAME_SIZE[0]}x{MIN_FRAME_SIZE[1]}). Para la foto de "
                                 "un solo espacio usa el modo 'un espacio'.")


@dataclass
class FrameInput:
    """Imagen, cajas y etiquetas (opcionales) de una predicción de frame."""

    img: Image.Image
    boxes: list[dict[str, Any]]
    warnings: list[str]
    truth: dict[str, int] | None
    source: dict[str, Any]


def gallery_frame_input(item: dict[str, Any]) -> FrameInput:
    """Frame de CNR-EXT (cajas de su cámara), mosaico de CNRPark (celdas) o foto propia (su JSON)."""
    kind = item.get("kind")
    if kind == "patch":
        raise HTTPException(400, f"'{item['id']}' es la foto de un solo espacio; usa /api/predict/patch.")
    img = open_item_image(item)
    if kind == "mosaic":
        return FrameInput(img, mosaic_boxes(item), [], mosaic_ground_truth(item), source_info(item))
    if kind == "own":
        boxes, warnings = own_boxes(item, img)
        return FrameInput(img, boxes, warnings, None, source_info(item))
    camera = int(item["camera"])
    _check_frame_size(img)
    boxes, warnings = scaled_camera_boxes(camera, img)
    return FrameInput(img, boxes, warnings, frame_ground_truth(item), source_info(item) | {"camera": camera})


def upload_frame_input(payload: Payload) -> FrameInput:
    """Foto subida de una cámara de CNR-EXT: se usan las cajas fijas de esa cámara."""
    camera = parse_camera(payload.fields.get("camera"))
    img = open_upload(payload.file_bytes or b"", payload.filename)
    _check_frame_size(img)
    boxes, warnings = scaled_camera_boxes(camera, img)
    if img.size != frames.FULL_SIZE:
        warnings.append(f"La imagen mide {img.width}x{img.height}. " + OTHER_CAMERA_WARNING)
    return FrameInput(img, boxes, warnings, None, source_info(None, payload.filename) | {"camera": camera})


def predict_frame_sync(payload: Payload) -> dict[str, Any]:
    t0 = time.perf_counter()
    scenario = parse_scenario(payload.fields)
    model_id = parse_model(payload.fields.get("model"))
    gallery_id = payload.fields.get("gallery_id")
    if payload.file_bytes is None and gallery_id:
        frame = gallery_frame_input(gallery_item(gallery_id))
    elif payload.file_bytes is not None:
        frame = upload_frame_input(payload)
    else:
        raise HTTPException(400, "Elige un ejemplo de la galería (gallery_id) o sube una imagen con su cámara.")
    pred = store.get(model_id)
    result = run_prediction(pred, frame.img, frame.boxes, scenario, frame.truth)
    return finish(result, pred, "frame", frame.source, image_quality(frame.img), frame.warnings, t0)


def predict_patch_sync(payload: Payload) -> dict[str, Any]:
    t0 = time.perf_counter()
    scenario = parse_scenario(payload.fields)
    model_id = parse_model(payload.fields.get("model"))
    gallery_id = payload.fields.get("gallery_id")
    item = None
    if payload.file_bytes is None and gallery_id:
        item = gallery_item(gallery_id)
        if item["kind"] != "patch":
            raise HTTPException(400, f"'{item['id']}' no es la foto de un solo espacio; usa /api/predict/frame.")
    elif payload.file_bytes is None:
        raise HTTPException(400, "Sube la foto de un espacio o elige un ejemplo de la galería (gallery_id).")
    img = open_gallery_image(item) if item else open_upload(payload.file_bytes or b"", payload.filename)
    pred = store.get(model_id)
    warnings: list[str] = []
    if max(img.size) > PATCH_MAX_SIDE:
        warnings.append(f"La imagen ({img.width}x{img.height}) parece mostrar más de un espacio; este modo "
                        f"espera la foto de un solo espacio. Para varios, usa {DRAW_TAB}.")
    box = {"slot_id": item["id"] if item else "espacio", "x": 0, "y": 0, "w": img.width, "h": img.height}
    truth = None
    if item is not None and show_ground_truth():
        gt = patch_ground_truth(item)
        truth = {str(box["slot_id"]): gt} if gt is not None else {}
    result = run_prediction(pred, img, [box], scenario, truth)
    return finish(result, pred, "patch", source_info(item, payload.filename), image_quality(img), warnings, t0)


def _box_slot_id(b: Any, index: int) -> str:
    """Identificador de la caja: el del cliente si es corto y seguro; si no, su número."""
    raw = b.get("slot_id") if isinstance(b, dict) else None
    sid = str(raw) if raw is not None else ""
    return sid if SLOT_ID_RE.match(sid) else str(index)


def parse_boxes(raw: Any, width: int, height: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Valida las cajas dibujadas (``[{x, y, w, h, slot_id?}]`` en píxeles de la imagen)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else []
        except json.JSONDecodeError:
            raise HTTPException(400, "Las cajas deben ser una lista JSON de objetos {x, y, w, h}.")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise HTTPException(400, "Las cajas deben ser una lista JSON de objetos {x, y, w, h}.")
    if not raw:
        raise HTTPException(400, "No hay espacios dibujados: dibuja al menos una caja sobre la imagen.")
    if len(raw) > MAX_BOXES:
        raise HTTPException(400, f"Demasiadas cajas ({len(raw)}); el máximo es {MAX_BOXES}.")
    boxes: list[dict[str, Any]] = []
    dropped = 0
    for i, b in enumerate(raw, start=1):
        try:
            x, y, w, h = (float(b[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, f"La caja {i} no es válida: usa números x, y, w, h.")
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise HTTPException(400, f"La caja {i} no es válida: usa números finitos.")
        if w < 0:
            x, w = x + w, -w
        if h < 0:
            y, h = y + h, -h
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(width), x + w), min(float(height), y + h)
        if x1 - x0 < MIN_BOX_SIDE or y1 - y0 < MIN_BOX_SIDE:
            dropped += 1
            continue
        boxes.append({"slot_id": _box_slot_id(b, i), "x": round(x0), "y": round(y0),
                      "w": round(x1 - x0), "h": round(y1 - y0)})
    if not boxes:
        raise HTTPException(400, "Todas las cajas quedaron fuera de la imagen o son demasiado pequeñas.")
    warnings = [f"Se ignoraron {dropped} caja(s) fuera de la imagen o demasiado pequeñas."] if dropped else []
    return boxes, warnings


def predict_custom_sync(payload: Payload) -> dict[str, Any]:
    t0 = time.perf_counter()
    scenario = parse_scenario(payload.fields)
    model_id = parse_model(payload.fields.get("model"))
    gallery_id = payload.fields.get("gallery_id")
    item = None
    if payload.file_bytes is None and gallery_id:
        item = gallery_item(gallery_id)
    elif payload.file_bytes is None:
        raise HTTPException(400, "Sube una imagen (o elige una de la galería) y dibuja los espacios.")
    img = open_item_image(item) if item else open_upload(payload.file_bytes or b"", payload.filename)
    boxes, warnings = parse_boxes(payload.fields.get("boxes"), img.width, img.height)
    pred = store.get(model_id)
    result = run_prediction(pred, img, boxes, scenario)
    return finish(result, pred, "custom", source_info(item, payload.filename), image_quality(img), warnings, t0)


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/models")
def models() -> dict[str, Any]:
    """Modelos seleccionables (pares ``<stem>.onnx|.joblib`` + ``<stem>.json`` de la carpeta del modelo)."""
    return {"models": list_models(), "default": default_model_id()}


ITERATIONS_CSV = paths.results_dir() / "iteraciones.csv"
B1_NAME = "b1_hog_lr"


@lru_cache(maxsize=4)
def _hours_uvg_cached(csv_path: str, mtime: float) -> dict[str, float]:
    """``name`` → horas ahorradas por semestre en val (escenario UVG) de la tabla de iteraciones."""
    import csv

    out: dict[str, float] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["name"]] = float(row["val_hours_uvg"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def hours_uvg_by_name() -> dict[str, float]:
    try:
        return _hours_uvg_cached(str(ITERATIONS_CSV), ITERATIONS_CSV.stat().st_mtime)
    except OSError:
        return {}


def time_benefit(name: Any) -> dict[str, Any] | None:
    """Minutos que ahorra cada conductor en el parqueo UVG con este modelo (val, modelo de valor).

    minutos por llegada = horas ahorradas por semestre × 60 / (llegadas por día × días del semestre).
    """
    hours = hours_uvg_by_name()
    if not isinstance(name, str) or name not in hours:
        return None
    params = load_scenario("uvg")["params"]
    arrivals = float(params["llegadas_dia"]) * float(params["dias_semestre"])
    t_base = float(params["t_base_min"])
    saved = hours[name] * 60.0 / arrivals
    # t_base − ahorro = T_sys del modelo de valor porque en UVG no hay abandono (abandono_base = 0)
    out: dict[str, Any] = {
        "escenario": "uvg",
        "t_base_min": t_base,
        "minutos_ahorrados": round(saved, 2),
        "minutos_con_sistema": round(max(t_base - saved, 0.0), 2),
        "horas_semestre": round(hours[name]),
        "llegadas_dia": int(params["llegadas_dia"]),
        "dias_semestre": int(params["dias_semestre"]),
        "fuente": "results/iteraciones.csv (validación) y configs/value/uvg.yaml",
    }
    if name != B1_NAME and B1_NAME in hours:
        out["segundos_vs_b1"] = round((hours[name] - hours[B1_NAME]) * 3600.0 / arrivals, 1)
    return out


@app.get("/api/model")
def model_info(model: str | None = None) -> dict[str, Any]:
    pred = store.get(parse_model(model))
    card = pred.card
    model_file = pred.model_file
    bench = load_benchmark(model_file)
    if bench is not None:
        # El benchmark de la carpeta es de un solo modelo: los demás no lo muestran
        bench["matches_model"] = bench.get("run_id") == card.get("run_id")
    try:
        size_mb: float | None = round(model_file.stat().st_size / (1024 * 1024), 3)
    except OSError:
        size_mb = None
    return {
        "id": pred.model_id,
        "label": pred.label,
        "kind": pred.kind,
        "is_default": pred.model_id == default_model_id(),
        "name": card.get("name"),
        "run_id": card.get("run_id"),
        "model": card.get("model"),
        "onnx_file": model_file.name,
        "model_file": model_file.name,
        "size_mb": size_mb,
        "input_size": pred.input_size,
        "hog": card.get("hog"),
        "temperature": pred.temperature,
        "calibrated": bool(card.get("calibrated", False)),
        "select_scenario": card.get("select_scenario"),
        "thresholds": pred.thresholds,
        "val_metrics": card.get("val_metrics_summary"),
        "benchmark": bench,
        "time_benefit": time_benefit(card.get("name")),
        "exported_at": card.get("exported_at"),
        "rule": "LIBRE si p < t_L; OCUPADO si p >= t_H; POR VERIFICAR en el resto (p = P(ocupado)).",
    }


GALLERY_KEYS = ("id", "kind", "subset", "title", "camera", "weather", "date", "time", "note",
                "split_origin", "hard")


@app.get("/api/gallery")
def gallery() -> dict[str, Any]:
    items = []
    for item in gallery_items().values():
        extra = {"n_cells": len(item.get("cells", []))} if item.get("kind") == "mosaic" else {}
        items.append({k: item.get(k) for k in GALLERY_KEYS} | extra
                     | {"image_url": f"/api/gallery/{item['id']}/image", "available": item_available(item)})
    own = [{k: v for k, v in it.items() if not k.startswith("_")}
           | {"image_url": f"/api/gallery/{it['id']}/image", "available": True}
           for it in own_items().values()]
    return {"items": items + own, "ground_truth_enabled": show_ground_truth(),
            "note": ("Solo imágenes nunca vistas en el entrenamiento: frames de fechas de test de "
                     "CNR-EXT y patches de CNRPark A/B (otro parqueo)."),
            "own_note": ("Fotos propias de la carpeta local (no se versiona): se analizan con sus "
                         "cajas guardadas en <foto>.json.")}


@app.get("/api/gallery/{gallery_id}/image", include_in_schema=False)
def gallery_image(gallery_id: str) -> Response:
    item = gallery_item(gallery_id)
    kind = item.get("kind")
    if kind == "mosaic":
        buf = io.BytesIO()
        mosaic_image(item).save(buf, "JPEG", quality=95)
        return Response(buf.getvalue(), media_type="image/jpeg")
    if kind == "own":
        path = Path(item["_file"])
        if not path.is_file():
            raise HTTPException(404, f"La foto '{item['filename']}' ya no está en la galería local.")
        return FileResponse(path, media_type=OWN_IMAGE_TYPES[path.suffix.lower()])
    return FileResponse(gallery_file(item), media_type="image/jpeg")


@app.post("/api/predict/frame")
async def predict_frame(request: Request) -> dict[str, Any]:
    return await run_in_threadpool(predict_frame_sync, await read_payload(request))


@app.post("/api/predict/patch")
async def predict_patch(request: Request) -> dict[str, Any]:
    return await run_in_threadpool(predict_patch_sync, await read_payload(request))


@app.post("/api/predict/custom")
async def predict_custom(request: Request) -> dict[str, Any]:
    return await run_in_threadpool(predict_custom_sync, await read_payload(request))


def feedback_path() -> Path:
    raw = os.environ.get(FEEDBACK_ENV)
    return Path(raw).expanduser() if raw else APP_DIR / "feedback.jsonl"


def _optional_choice(fields: dict[str, Any], key: str, options: tuple[str, ...]) -> str | None:
    value = fields.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value not in options:
        raise HTTPException(400, f"{key} debe ser uno de: {', '.join(options)} (o no enviarse).")
    return value


def _optional_probability(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(400, "p_busy debe ser un número entre 0 y 1 (o no enviarse).")
    p = float(value)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise HTTPException(400, "p_busy debe ser un número entre 0 y 1 (o no enviarse).")
    return p


def _feedback_source(value: Any) -> str | None:
    """Origen del reporte: un id de la galería (del dataset o foto propia) o 'upload'."""
    if value is None:
        return None
    if value == UPLOAD_SOURCE or (isinstance(value, str) and (
            value in gallery_items() or (value.startswith(OWN_PREFIX) and value in own_items()))):
        return value
    raise HTTPException(400, "source debe ser un id de la galería o 'upload' (o no enviarse).")


def _feedback_model(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return parse_model(value)
    except UnknownModelError:
        raise HTTPException(400, "model debe ser el id de un modelo de GET /api/models (o no enviarse).")


def validate_feedback(fields: dict[str, Any]) -> dict[str, Any]:
    slot_id = fields.get("slot_id")
    if not isinstance(slot_id, (str, int)) or isinstance(slot_id, bool) or not SLOT_ID_RE.match(str(slot_id)):
        raise HTTPException(400, "slot_id es obligatorio: letras, números, '_' o '-' (máximo 32).")
    reported = fields.get("reported_state")
    if reported not in REPORTED_STATES:
        raise HTTPException(400, f"reported_state debe ser uno de: {', '.join(REPORTED_STATES)}.")
    return {
        "slot_id": str(slot_id),
        "reported_state": reported,
        "predicted_state": _optional_choice(fields, "predicted_state", ALL_STATES),
        "p_busy": _optional_probability(fields.get("p_busy")),
        "scenario": _optional_choice(fields, "scenario", SCENARIOS),
        "source": _feedback_source(fields.get("source")),
        "model_id": _feedback_model(fields.get("model")),
    }


@app.post("/api/feedback")
async def feedback(request: Request) -> dict[str, Any]:
    if not request.headers.get("content-type", "").lower().startswith("application/json"):
        raise HTTPException(400, "Envía el reporte como JSON.")
    try:
        fields = await _read_json(request, MAX_FEEDBACK_BYTES)
    except HTTPException as exc:
        # Cualquier cuerpo inválido o demasiado grande se responde con 400
        raise HTTPException(400, exc.detail)
    record = validate_feedback(fields)
    model_name = None
    try:
        pred = store.get(record["model_id"])
        model_name = pred.card.get("name")
        record["model_id"] = pred.model_id
    except ModelUnavailableError:
        pass
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record, "model": model_name}
    path = feedback_path()
    with _feedback_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True, "message": f"Gracias: se registró que el espacio {record['slot_id']} estaba "
                                   f"{record['reported_state']}."}


@app.get("/api/health", include_in_schema=False)
def health() -> dict[str, Any]:
    """Estado rápido para el despliegue: si hay modelo y si se ven los datos de la galería."""
    try:
        store.get()
        model_ok, detail = True, None
    except ModelUnavailableError as exc:
        model_ok, detail = False, str(exc)
    return {"model": model_ok, "model_file": model_path().name, "detail": detail,
            "data_root_exists": paths.data_root().exists()}
