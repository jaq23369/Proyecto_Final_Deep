"""Inferencia de la app con ONNX Runtime (CPU), sin torch.

- Carga ``best.onnx`` y el ``best.json`` que lo acompaña (mismo nombre, extensión ``.json``).
- Preprocesa cada recorte con ``cupo.transforms.preprocess_numpy`` (igual que en evaluación).
- ``p = softmax(logits / T)[:, 1]`` = P(ocupado), con la temperatura de ``best.json``.
- Estados con los umbrales del escenario: LIBRE si p < t_L, OCUPADO si p >= t_H y
  POR VERIFICAR en el resto.

El modelo se recarga solo si cambia el archivo: cambiar de modelo = reemplazar ``best.onnx``
y ``best.json``.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Sequence
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
EXPORT_HINT = "python -m cupo.export_onnx --run runs/<id> --out models/best.onnx"

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


def model_path() -> Path:
    """Ruta del modelo: ``$MODEL_PATH`` o ``models/best.onnx``; las relativas, desde el repo."""
    raw = Path(os.environ.get(MODEL_PATH_ENV) or DEFAULT_MODEL_PATH).expanduser()
    return raw if raw.is_absolute() else (paths.repo_root() / raw)


def display_path(path: Path) -> str:
    """Ruta relativa al repo si se puede (no expone rutas absolutas en los mensajes)."""
    try:
        return path.resolve().relative_to(paths.repo_root()).as_posix()
    except ValueError:
        return path.name


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


@dataclass
class Predictor:
    """Sesión de ONNX Runtime + tarjeta del modelo (``best.json``)."""

    session: Any
    card: dict[str, Any]
    onnx_path: Path
    input_name: str
    input_size: int
    temperature: float
    thresholds: dict[str, dict[str, float]]
    load_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def predict(self, crops: Sequence[Image.Image]) -> tuple[np.ndarray, float]:
        """P(ocupado) de cada recorte y milisegundos del modelo (sin preprocesamiento)."""
        if not crops:
            return np.zeros(0, dtype=np.float64), 0.0
        probs: list[np.ndarray] = []
        model_ms = 0.0
        for start in range(0, len(crops), BATCH_SIZE):
            chunk = crops[start:start + BATCH_SIZE]
            batch = np.concatenate([preprocess_numpy(c, self.input_size) for c in chunk], axis=0)
            t0 = time.perf_counter()
            logits = self.session.run(None, {self.input_name: batch})[0]
            model_ms += (time.perf_counter() - t0) * 1000.0
            probs.append(softmax_busy(np.asarray(logits), self.temperature))
        return np.concatenate(probs), model_ms

    def states(self, p: np.ndarray, scenario: str) -> list[str]:
        th = self.thresholds[scenario]
        return [decide(float(v), th["t_L"], th["t_H"]) for v in p]


def _read_card(onnx_path: Path) -> dict[str, Any]:
    card_path = onnx_path.with_suffix(".json")
    if not card_path.exists():
        raise ModelUnavailableError(
            f"Falta {card_path.name} junto a {onnx_path.name} (umbrales y temperatura); "
            f"vuelve a exportar con {EXPORT_HINT}"
        )
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailableError(
            f"No se pudo leer {card_path.name} ({type(exc).__name__}); vuelve a exportar con {EXPORT_HINT}"
        ) from exc
    if not isinstance(card, dict):
        raise ModelUnavailableError(
            f"{card_path.name} no es un objeto JSON ({type(card).__name__}); vuelve a exportar con {EXPORT_HINT}"
        )
    return card


def _thresholds(card: dict[str, Any]) -> dict[str, dict[str, float]]:
    raw = card.get("thresholds") or {}
    out: dict[str, dict[str, float]] = {}
    for name in SCENARIOS:
        th = raw.get(name)
        if not isinstance(th, dict) or "t_L" not in th or "t_H" not in th:
            raise ModelUnavailableError(f"best.json no trae los umbrales del escenario '{name}'")
        try:
            t_low, t_high = float(th["t_L"]), float(th["t_H"])
        except (TypeError, ValueError):
            t_low = t_high = float("nan")
        # Cualquier par 0 <= t_L <= t_H <= 1 es válido (no se supone t_L <= 0.5 <= t_H)
        if not (0.0 <= t_low <= t_high <= 1.0):
            raise ModelUnavailableError(
                f"Umbrales inválidos en best.json para '{name}': se espera 0 <= t_L <= t_H <= 1"
            )
        out[name] = {"t_L": t_low, "t_H": t_high}
    return out


def _temperature(card: dict[str, Any]) -> float:
    raw = card.get("temperature", 1.0)
    try:
        temperature = float(1.0 if raw is None else raw)
    except (TypeError, ValueError):
        temperature = float("nan")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ModelUnavailableError("Temperatura inválida en best.json: debe ser un número mayor que 0")
    return temperature


def load_predictor(onnx_path: Path) -> Predictor:
    """Carga el modelo ONNX en CPU; lanza ``ModelUnavailableError`` con un mensaje claro."""
    if not onnx_path.exists():
        raise ModelUnavailableError(
            f"No hay modelo exportado ({display_path(onnx_path)}); ejecuta {EXPORT_HINT} "
            "(o define MODEL_PATH con la ruta del .onnx)."
        )
    card = _read_card(onnx_path)
    import onnxruntime as ort  # import local: la app arranca aunque falte el modelo

    t0 = time.perf_counter()
    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # onnxruntime lanza varios tipos según el error
        # El mensaje de onnxruntime trae rutas absolutas: se resume para el usuario
        raise ModelUnavailableError(
            f"No se pudo cargar {onnx_path.name} ({type(exc).__name__}): el archivo no es un modelo "
            f"ONNX válido; vuelve a exportarlo con {EXPORT_HINT}"
        ) from exc
    load_ms = (time.perf_counter() - t0) * 1000.0
    return Predictor(
        session=session,
        card=card,
        onnx_path=onnx_path,
        input_name=session.get_inputs()[0].name,
        input_size=int(card.get("input_size", 224)),
        temperature=_temperature(card),
        thresholds=_thresholds(card),
        load_ms=load_ms,
    )


class ModelStore:
    """Mantiene el modelo cargado y lo recarga si cambia el ``.onnx`` o el ``.json``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._predictor: Predictor | None = None

    @staticmethod
    def _signature(onnx_path: Path) -> tuple:
        stamp: list[tuple[int, int] | None] = []
        for p in (onnx_path, onnx_path.with_suffix(".json")):
            try:
                st = p.stat()
                stamp.append((st.st_mtime_ns, st.st_size))
            except OSError:  # el archivo puede desaparecer entre llamadas
                stamp.append(None)
        return (str(onnx_path), *stamp)

    def get(self) -> Predictor:
        onnx_path = model_path()
        key = self._signature(onnx_path)
        with self._lock:
            if self._predictor is None or key != self._key:
                self._predictor = None
                self._key = None
                self._predictor = load_predictor(onnx_path)
                self._key = key
            return self._predictor


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
