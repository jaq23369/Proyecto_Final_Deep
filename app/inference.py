"""Inferencia de la app en CPU, sin torch: modelos ONNX (ONNX Runtime) y el baseline clásico.

- Registro de modelos: en la carpeta del modelo por defecto (``models/`` o la de
  ``$MODEL_PATH``) cada par ``<stem>.onnx`` + ``<stem>.json`` o ``<stem>.joblib`` + ``<stem>.json``
  es un modelo seleccionable con id ``<stem>``. El modelo por defecto es ``best.onnx`` (o
  ``$MODEL_PATH``).
- ONNX: cada recorte se preprocesa con ``cupo.transforms.preprocess_numpy`` (igual que en
  evaluación) y ``p = softmax(logits / T)[:, 1]`` = P(ocupado), con la temperatura de la tarjeta.
- Clásico (B1): HOG con ``cupo.classical.hog_image`` (el mismo camino que en entrenamiento) y
  regresión logística; ``p = softmax([0, z] / T)[:, 1]`` con T = 1.
- Estados con los umbrales del escenario: LIBRE si p < t_L, OCUPADO si p >= t_H y
  POR VERIFICAR en el resto.

Cada modelo se carga la primera vez que se pide y queda en caché; se recarga solo si cambia su
archivo o su tarjeta: cambiar de modelo = reemplazar el par (por ejemplo ``best.onnx`` y
``best.json``).
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cupo import paths
from cupo.transforms import preprocess_numpy

MODEL_PATH_ENV = "MODEL_PATH"
DEFAULT_MODEL_PATH = Path("models") / "best.onnx"
SCENARIOS: tuple[str, ...] = ("comercial", "uvg")
STATE_FREE, STATE_BUSY, STATE_VERIFY = "libre", "ocupado", "verificar"
KIND_ONNX, KIND_CLASSICAL = "onnx", "classical"
MODEL_SUFFIXES: dict[str, str] = {".onnx": KIND_ONNX, ".joblib": KIND_CLASSICAL}
EXPORT_HINT = "python -m cupo.export_onnx --run runs/<id> --out models/best.onnx"
EXPORT_HINT_CLASSICAL = "python scripts/export_classical.py --run runs/<id_b1> --out models/b1_hog_lr.joblib"
# Ids de modelo: el stem del archivo (letras, números, '_', '-' o '.')
MODEL_ID_RE = re.compile(r"^[\w][\w.-]{0,63}\Z")
# Etiquetas legibles por ``name`` de la tarjeta; si no se conoce, se muestra el name
KNOWN_LABELS: dict[str, str] = {
    "b1_hog_lr": "B1 · HOG + regresión logística (clásico)",
    "b2_malexnet": "B2 · mAlexNet",
    "mnv3_ft_cost": "Modelo final · MobileNetV3 (mnv3_ft_cost)",
}

# Calidad de imagen (medido en frames y patches de train/val): los frames normales tienen
# brillo medio > 80 salvo de noche y varianza del Laplaciano > 40; un desenfoque gaussiano
# de radio 2 la baja a ~8. Los patches pequeños son más suaves, así que su umbral es menor.
DARK_THRESHOLD = 40.0
BLUR_THRESHOLD = 15.0
BLUR_THRESHOLD_SMALL = 4.0
SMALL_IMAGE_SIDE = 400
QUALITY_MAX_SIDE = 1000
BATCH_SIZE = 64  # bloques de inferencia: acota la memoria con muchas cajas dibujadas


class ModelUnavailableError(RuntimeError):
    """No hay modelo exportado o no se pudo cargar (la API responde 503)."""


class UnknownModelError(LookupError):
    """El id de modelo pedido no está en el registro (la API responde 400)."""


def model_path() -> Path:
    """Ruta del modelo por defecto: ``$MODEL_PATH`` o ``models/best.onnx``; las relativas, desde el repo."""
    raw = Path(os.environ.get(MODEL_PATH_ENV) or DEFAULT_MODEL_PATH).expanduser()
    return raw if raw.is_absolute() else (paths.repo_root() / raw)


def display_path(path: Path) -> str:
    """Ruta relativa al repo si se puede (no expone rutas absolutas en los mensajes)."""
    try:
        return path.resolve().relative_to(paths.repo_root()).as_posix()
    except ValueError:
        return path.name


def model_kind(path: Path) -> str:
    """'onnx' o 'classical' según la extensión del archivo del modelo."""
    return MODEL_SUFFIXES.get(path.suffix.lower(), KIND_ONNX)


def export_hint(path: Path) -> str:
    return EXPORT_HINT_CLASSICAL if model_kind(path) == KIND_CLASSICAL else EXPORT_HINT


# ---------------------------------------------------------------------------
# Registro de modelos
# ---------------------------------------------------------------------------

def default_model_id() -> str:
    return model_path().stem


def discover_models() -> dict[str, Path]:
    """``{id: archivo}`` de los modelos junto al modelo por defecto (que siempre está).

    Un modelo cuenta si su archivo (``.onnx`` o ``.joblib``) tiene una tarjeta ``.json`` con
    el mismo nombre. Si un stem tiene los dos formatos, gana el ``.onnx``.
    """
    default = model_path()
    found: dict[str, Path] = {}
    folder = default.parent
    if folder.is_dir():
        for suffix in MODEL_SUFFIXES:  # .onnx primero
            for path in sorted(folder.glob(f"*{suffix}")):
                if (path.is_file() and MODEL_ID_RE.match(path.stem) and path.stem not in found
                        and path.with_suffix(".json").is_file()):
                    found[path.stem] = path
    found[default.stem] = default  # el por defecto se lista aunque falte (la API dará 503)
    return found


def resolve_model(model_id: str | None) -> tuple[str, Path]:
    """(id, archivo) del modelo pedido; ``None`` o vacío = el modelo por defecto."""
    if model_id is None or str(model_id).strip() == "":
        return default_model_id(), model_path()
    key = str(model_id).strip()
    registry = discover_models()
    if key not in registry:
        shown = key if len(key) <= 40 else key[:39] + "…"
        raise UnknownModelError(f"Modelo desconocido: '{shown}'. Opciones: {', '.join(sorted(registry))}.")
    return key, registry[key]


def model_label(card: dict[str, Any] | None, model_id: str) -> str:
    """Etiqueta legible derivada del ``name`` de la tarjeta (o el id si no hay tarjeta)."""
    name = str((card or {}).get("name") or model_id)
    return KNOWN_LABELS.get(name, name)


def softmax_busy(logits: np.ndarray, temperature: float) -> np.ndarray:
    """P(ocupado) = softmax(logits / T)[:, 1], estable numéricamente."""
    z = logits.astype(np.float64) / float(temperature)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e[:, 1] / e.sum(axis=1)).astype(np.float64)


def decide(p: float, t_low: float, t_high: float) -> str:
    """Estado de un espacio según los dos umbrales del escenario."""
    if p < t_low:
        return STATE_FREE
    if p >= t_high:
        return STATE_BUSY
    return STATE_VERIFY


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

def _classical_features(bundle: dict[str, Any], crops: Sequence[Image.Image]) -> np.ndarray:
    """HOG de cada recorte con los parámetros guardados en el ``.joblib`` del entrenamiento."""
    from cupo.classical import classical_features

    return classical_features(bundle["hog"], crops)


def _classical_run(bundle: dict[str, Any], feats: np.ndarray) -> np.ndarray:
    from cupo.classical import classical_logits

    return classical_logits(bundle, feats)


@dataclass
class Predictor:
    """Modelo cargado + su tarjeta (``<stem>.json``).

    ``session`` es la sesión de ONNX Runtime o, en el modelo clásico, el dict del ``.joblib``
    (``pipeline`` + parámetros ``hog``). ``onnx_path`` es el archivo del modelo (``.onnx`` o
    ``.joblib``; el nombre se conserva por compatibilidad).
    """

    session: Any
    card: dict[str, Any]
    onnx_path: Path
    input_name: str
    input_size: int
    temperature: float
    thresholds: dict[str, dict[str, float]]
    load_ms: float = 0.0
    kind: str = KIND_ONNX
    model_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def model_file(self) -> Path:
        return self.onnx_path

    @property
    def label(self) -> str:
        return model_label(self.card, self.model_id or self.onnx_path.stem)

    def _prepare(self, chunk: Sequence[Image.Image]) -> np.ndarray:
        if self.kind == KIND_CLASSICAL:
            return _classical_features(self.session, chunk)
        return np.concatenate([preprocess_numpy(c, self.input_size) for c in chunk], axis=0)

    def _logits(self, batch: np.ndarray) -> np.ndarray:
        if self.kind == KIND_CLASSICAL:
            return _classical_run(self.session, batch)
        return np.asarray(self.session.run(None, {self.input_name: batch})[0])

    def predict(self, crops: Sequence[Image.Image]) -> tuple[np.ndarray, float]:
        """P(ocupado) de cada recorte y milisegundos del modelo (sin preprocesamiento ni HOG)."""
        if not crops:
            return np.zeros(0, dtype=np.float64), 0.0
        probs: list[np.ndarray] = []
        model_ms = 0.0
        for start in range(0, len(crops), BATCH_SIZE):
            batch = self._prepare(crops[start:start + BATCH_SIZE])
            t0 = time.perf_counter()
            logits = self._logits(batch)
            model_ms += (time.perf_counter() - t0) * 1000.0
            probs.append(softmax_busy(np.asarray(logits), self.temperature))
        return np.concatenate(probs), model_ms

    def states(self, p: np.ndarray, scenario: str) -> list[str]:
        th = self.thresholds[scenario]
        return [decide(float(v), th["t_L"], th["t_H"]) for v in p]


def _read_card(model_file: Path) -> dict[str, Any]:
    card_path = model_file.with_suffix(".json")
    hint = export_hint(model_file)
    if not card_path.exists():
        raise ModelUnavailableError(
            f"Falta {card_path.name} junto a {model_file.name} (umbrales y temperatura); "
            f"vuelve a exportar con {hint}"
        )
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailableError(
            f"No se pudo leer {card_path.name} ({type(exc).__name__}); vuelve a exportar con {hint}"
        ) from exc
    if not isinstance(card, dict):
        raise ModelUnavailableError(
            f"{card_path.name} no es un objeto JSON ({type(card).__name__}); vuelve a exportar con {hint}"
        )
    return card


def _thresholds(card: dict[str, Any], card_name: str = "best.json") -> dict[str, dict[str, float]]:
    raw = card.get("thresholds") or {}
    out: dict[str, dict[str, float]] = {}
    for name in SCENARIOS:
        th = raw.get(name) if isinstance(raw, dict) else None
        if not isinstance(th, dict) or "t_L" not in th or "t_H" not in th:
            raise ModelUnavailableError(f"{card_name} no trae los umbrales del escenario '{name}'")
        try:
            t_low, t_high = float(th["t_L"]), float(th["t_H"])
        except (TypeError, ValueError):
            t_low = t_high = float("nan")
        # Cualquier par 0 <= t_L <= t_H <= 1 es válido (no se supone t_L <= 0.5 <= t_H)
        if not (0.0 <= t_low <= t_high <= 1.0):
            raise ModelUnavailableError(
                f"Umbrales inválidos en {card_name} para '{name}': se espera 0 <= t_L <= t_H <= 1"
            )
        out[name] = {"t_L": t_low, "t_H": t_high}
    return out


def _temperature(card: dict[str, Any], card_name: str = "best.json") -> float:
    raw = card.get("temperature", 1.0)
    try:
        temperature = float(1.0 if raw is None else raw)
    except (TypeError, ValueError):
        temperature = float("nan")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ModelUnavailableError(f"Temperatura inválida en {card_name}: debe ser un número mayor que 0")
    return temperature


def _missing_model(model_file: Path) -> ModelUnavailableError:
    return ModelUnavailableError(
        f"No hay modelo exportado ({display_path(model_file)}); ejecuta {export_hint(model_file)} "
        "(o define MODEL_PATH con la ruta del .onnx)."
    )


def _load_onnx_session(model_file: Path) -> tuple[Any, str]:
    import onnxruntime as ort  # import local: la app arranca aunque falte el modelo

    try:
        session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    except Exception as exc:  # onnxruntime lanza varios tipos según el error
        # El mensaje de onnxruntime trae rutas absolutas: se resume para el usuario
        raise ModelUnavailableError(
            f"No se pudo cargar {model_file.name} ({type(exc).__name__}): el archivo no es un modelo "
            f"ONNX válido; vuelve a exportarlo con {EXPORT_HINT}"
        ) from exc
    return session, session.get_inputs()[0].name


def _load_classical_bundle(model_file: Path) -> dict[str, Any]:
    import joblib  # solo se importa si se pide el modelo clásico

    try:
        bundle = joblib.load(model_file)
    except Exception as exc:  # joblib/pickle lanzan varios tipos según el error
        raise ModelUnavailableError(
            f"No se pudo cargar {model_file.name} ({type(exc).__name__}); vuelve a exportarlo con "
            f"{EXPORT_HINT_CLASSICAL}"
        ) from exc
    if not isinstance(bundle, dict) or "pipeline" not in bundle or not isinstance(bundle.get("hog"), dict):
        raise ModelUnavailableError(
            f"{model_file.name} no tiene el formato del baseline (pipeline + hog); vuelve a exportarlo con "
            f"{EXPORT_HINT_CLASSICAL}"
        )
    return bundle


def load_predictor(model_file: Path, model_id: str | None = None) -> Predictor:
    """Carga el modelo (ONNX o clásico) en CPU; lanza ``ModelUnavailableError`` con un mensaje claro."""
    if not model_file.exists():
        raise _missing_model(model_file)
    card = _read_card(model_file)
    card_name = model_file.with_suffix(".json").name
    thresholds = _thresholds(card, card_name)
    temperature = _temperature(card, card_name)
    kind = model_kind(model_file)
    t0 = time.perf_counter()
    if kind == KIND_CLASSICAL:
        session: Any = _load_classical_bundle(model_file)
        input_name = "hog"
        input_size = int(session["hog"].get("image_size", 64))
    else:
        session, input_name = _load_onnx_session(model_file)
        input_size = int(card.get("input_size", 224))
    load_ms = (time.perf_counter() - t0) * 1000.0
    return Predictor(
        session=session,
        card=card,
        onnx_path=model_file,
        input_name=input_name,
        input_size=input_size,
        temperature=temperature,
        thresholds=thresholds,
        load_ms=load_ms,
        kind=kind,
        model_id=model_id or model_file.stem,
    )


def _signature(model_file: Path) -> tuple:
    stamp: list[tuple[int, int] | None] = []
    for p in (model_file, model_file.with_suffix(".json")):
        try:
            st = p.stat()
            stamp.append((st.st_mtime_ns, st.st_size))
        except OSError:  # el archivo puede desaparecer entre llamadas
            stamp.append(None)
    return (str(model_file), *stamp)


class ModelStore:
    """Caché de modelos por id: cada uno se carga al pedirlo y se recarga si cambia su archivo."""

    def __init__(self, loader: Callable[[Path, str], Predictor] | None = None) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[tuple, Predictor]] = {}
        self._loader = loader or load_predictor

    _signature = staticmethod(_signature)

    def get(self, model_id: str | None = None) -> Predictor:
        """Modelo pedido (por defecto, el de ``MODEL_PATH``); ``UnknownModelError`` si no existe."""
        key_id, model_file = resolve_model(model_id)
        key = _signature(model_file)
        with self._lock:
            cached = self._cache.get(key_id)
            if cached is not None and cached[0] == key:
                return cached[1]
            self._cache.pop(key_id, None)
            predictor = self._loader(model_file, key_id)
            self._cache[key_id] = (key, predictor)
            return predictor


def _val_summary(card: dict[str, Any]) -> tuple[float | None, float | None]:
    """(F1 macro en val con t = 0.5, valor comercial en val) de la tarjeta, si están."""
    summary = card.get("val_metrics_summary")
    if not isinstance(summary, dict):
        return None, None
    binary = summary.get("binary_t05")
    value = summary.get("value")
    f1 = binary.get("f1_macro") if isinstance(binary, dict) else None
    com = value.get("comercial") if isinstance(value, dict) else None
    obj = com.get("objective_value") if isinstance(com, dict) else None
    return (float(f1) if isinstance(f1, (int, float)) else None,
            float(obj) if isinstance(obj, (int, float)) else None)


def model_entry(model_id: str, model_file: Path, is_default: bool) -> dict[str, Any]:
    """Resumen de un modelo del registro para ``GET /api/models`` (solo lee la tarjeta)."""
    entry: dict[str, Any] = {"id": model_id, "label": model_id, "name": None, "model": None,
                             "kind": model_kind(model_file), "is_default": is_default,
                             "val_f1": None, "val_value_comercial": None, "thresholds": None,
                             "available": False, "error": None}
    try:
        if not model_file.exists():
            raise _missing_model(model_file)
        card = _read_card(model_file)
        card_name = model_file.with_suffix(".json").name
        entry["thresholds"] = _thresholds(card, card_name)
        _temperature(card, card_name)
    except ModelUnavailableError as exc:
        entry["error"] = str(exc)
        return entry
    entry["val_f1"], entry["val_value_comercial"] = _val_summary(card)
    entry.update({"label": model_label(card, model_id), "name": card.get("name"),
                  "model": card.get("model"), "available": True})
    return entry


def list_models() -> list[dict[str, Any]]:
    """Modelos del registro ordenados por etiqueta (B1, B2, modelo final, ...)."""
    default_id = default_model_id()
    entries = [model_entry(mid, path, mid == default_id) for mid, path in discover_models().items()]
    return sorted(entries, key=lambda e: (e["label"].lower(), e["id"]))


def load_benchmark(onnx_path: Path) -> dict[str, Any] | None:
    """Resumen de ``benchmark.json`` (junto al modelo) si existe y corresponde a ese modelo."""
    bench_path = onnx_path.parent / "benchmark.json"
    if not bench_path.exists():
        return None
    try:
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cpu = bench.get("cpu_all_threads") or {}
    cpu1 = bench.get("cpu_1_thread") or {}
    return {
        "model_name": (bench.get("model") or {}).get("name"),
        "run_id": (bench.get("model") or {}).get("run_id"),
        "cpu_per_slot_p50_ms": (cpu.get("per_slot") or {}).get("p50_ms"),
        "cpu_per_frame_p50_ms": (cpu.get("per_frame") or {}).get("p50_ms"),
        "cpu_per_frame_p95_ms": (cpu.get("per_frame") or {}).get("p95_ms"),
        "cpu_1_thread_per_frame_p50_ms": (cpu1.get("per_frame") or {}).get("p50_ms"),
        "slots_per_frame": (cpu.get("per_frame") or {}).get("batch"),
        "costo_por_1000_pred_Q": bench.get("costo_por_1000_pred_Q"),
        "cameras_per_core": (bench.get("volume") or {}).get("cameras_per_core"),
        "cpu": (bench.get("env") or {}).get("cpu"),
        "measured_at": bench.get("measured_at"),
    }


# ---------------------------------------------------------------------------
# Calidad de imagen
# ---------------------------------------------------------------------------

def _gray(img: Image.Image) -> np.ndarray:
    g = img.convert("L")
    if max(g.size) > QUALITY_MAX_SIDE:
        s = QUALITY_MAX_SIDE / max(g.size)
        g = g.resize((max(1, round(g.width * s)), max(1, round(g.height * s))), Image.Resampling.BILINEAR)
    return np.asarray(g, dtype=np.float32)


def laplacian_variance(gray: np.ndarray) -> float:
    """Varianza del Laplaciano (kernel de 4 vecinos); baja = imagen borrosa."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           - 4.0 * gray[1:-1, 1:-1])
    return float(lap.var())


def image_quality(img: Image.Image) -> dict[str, Any]:
    """Brillo medio (0-255), varianza del Laplaciano y advertencias para el usuario."""
    gray = _gray(img)
    brightness = float(gray.mean())
    sharpness = laplacian_variance(gray)
    blur_thr = BLUR_THRESHOLD_SMALL if max(img.size) < SMALL_IMAGE_SIDE else BLUR_THRESHOLD
    warnings: list[str] = []
    if brightness < DARK_THRESHOLD:
        warnings.append(
            f"La imagen está muy oscura (brillo medio {brightness:.0f}/255); las predicciones "
            "pueden no ser fiables. Conviene verificar en sitio."
        )
    elif sharpness < blur_thr:
        # Solo se avisa del desenfoque si no está oscura: la oscuridad también baja la varianza
        warnings.append(
            f"La imagen parece borrosa (varianza del Laplaciano {sharpness:.1f}); las "
            "predicciones pueden no ser fiables."
        )
    return {"brightness": round(brightness, 1), "laplacian_var": round(sharpness, 1),
            "dark": brightness < DARK_THRESHOLD,
            "blurry": brightness >= DARK_THRESHOLD and sharpness < blur_thr,
            "warnings": warnings}
