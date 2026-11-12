"""Utilidades de las pruebas de la app (``app/``).

- ``make_model_dir``: carpeta con ``best.onnx`` + ``best.json`` + ``benchmark.json``. Por defecto
  arma un modelo ONNX diminuto y determinista (promedio por canal + capa lineal), sin datos ni
  GPU. Con la variable ``CUPO_TEST_ONNX`` se copia ese modelo exportado (y su ``.json``).
- Imágenes sintéticas: PNG en memoria, PNG con muchos píxeles y pocos bytes.
- ``fake_data_root``: DATA_ROOT mínimo con los CSV de cajas de algunas cámaras.
- ``has_key`` y ``assert_private``: revisan la estructura y la privacidad de las respuestas.
"""

from __future__ import annotations

import getpass
import io
import json
import os
import shutil
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TEST_ONNX_ENV = "CUPO_TEST_ONNX"
TINY_INPUT_SIZE = 64
TINY_TEMPERATURE = 1.5
# Pesos de la clase "ocupado": el logit depende del color medio del recorte
TINY_W = np.array([[0.0, 0.0, 0.0], [3.0, -2.0, 1.0]], dtype=np.float32)
TINY_B = np.array([0.0, 0.2], dtype=np.float32)
TINY_THRESHOLDS = {"comercial": {"t_L": 0.3, "t_H": 0.7}, "uvg": {"t_L": 0.4, "t_H": 0.6}}


def make_tiny_onnx(path: Path) -> None:
    """Modelo ``input [N,3,H,W] -> logits [N,2]``: ReduceMean(H,W) seguido de Gemm."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 2])
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["pooled"], axes=[2, 3], keepdims=0),
        helper.make_node("Gemm", ["pooled", "W", "B"], ["logits"], transB=1),
    ]
    inits = [numpy_helper.from_array(TINY_W, "W"), numpy_helper.from_array(TINY_B, "B")]
    graph = helper.make_graph(nodes, "tiny_test", [inp], [out], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())


def tiny_card() -> dict[str, Any]:
    return {
        "name": "tiny_test",
        "run_id": "tiny_test_s0",
        "model": "tiny",
        "input_size": TINY_INPUT_SIZE,
        "input_name": "input",
        "temperature": TINY_TEMPERATURE,
        "calibrated": True,
        "thresholds": json.loads(json.dumps(TINY_THRESHOLDS)),
        "select_scenario": "comercial",
        "val_metrics_summary": {"source": "sintético"},
        "exported_at": "2026-01-01T00:00:00",
    }


def tiny_benchmark(run_id: str) -> dict[str, Any]:
    return {
        "model": {"name": "tiny_test", "run_id": run_id},
        "cpu_all_threads": {"per_slot": {"p50_ms": 0.5},
                            "per_frame": {"p50_ms": 10.0, "p95_ms": 12.0, "batch": 40}},
        "cpu_1_thread": {"per_frame": {"p50_ms": 20.0}},
        "costo_por_1000_pred_Q": 0.01,
    }


def make_model_dir(dest: Path) -> Path:
    """Crea ``dest/best.onnx`` (+ ``.json`` y ``benchmark.json``) y devuelve la ruta del ``.onnx``."""
    dest.mkdir(parents=True, exist_ok=True)
    onnx_path = dest / "best.onnx"
    src = os.environ.get(TEST_ONNX_ENV)
    if src:
        src_path = Path(src)
        shutil.copy(src_path, onnx_path)
        shutil.copy(src_path.with_suffix(".json"), dest / "best.json")
        bench = src_path.parent / "benchmark.json"
        if bench.is_file():
            shutil.copy(bench, dest / "benchmark.json")
        return onnx_path
    make_tiny_onnx(onnx_path)
    card = tiny_card()
    write_card(onnx_path, card)
    (dest / "benchmark.json").write_text(json.dumps(tiny_benchmark(card["run_id"])), encoding="utf-8")
    return onnx_path


def read_card(onnx_path: Path) -> dict[str, Any]:
    return json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))


def write_card(onnx_path: Path, card: Any) -> None:
    onnx_path.with_suffix(".json").write_text(json.dumps(card), encoding="utf-8")


# ----------------------------------------------------------------- imágenes

def noise_image(w: int, h: int, seed: int = 0) -> Image.Image:
    """Imagen RGB con ruido de semilla fija (nítida y con brillo medio normal)."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), mode="RGB")


def tinted_image(w: int, h: int, rgb: tuple[int, int, int], seed: int = 0) -> Image.Image:
    """Ruido alrededor de un color: cambia el color medio y, con él, la p del modelo diminuto."""
    rng = np.random.default_rng(seed)
    arr = np.clip(np.asarray(rgb, dtype=np.int16) + rng.integers(-40, 41, (h, w, 3)), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def huge_png_bytes(width: int, height: int) -> bytes:
    """PNG válido en blanco y negro (1 bit) con ``width x height`` píxeles y pocos bytes."""
    row = b"\x00" + b"\x00" * ((width + 7) // 8)
    comp = zlib.compressobj(9)
    parts = [comp.compress(row) for _ in range(height)]
    parts.append(comp.flush())
    ihdr = struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", b"".join(parts)) + _png_chunk(b"IEND", b""))


# ------------------------------------------------------------------- datos

FAKE_BOXES = (  # SlotId, X, Y, W, H en coordenadas de 2592x1944
    (101, 100, 100, 400, 300),
    (102, 700, 100, 400, 300),
    (103, 1300, 900, 500, 400),
)


def fake_data_root(root: Path, cameras: tuple[int, ...] = (1,)) -> Path:
    """DATA_ROOT mínimo: ``CNR-EXT_FULL/camera<k>.csv`` con tres cajas."""
    full = root / "CNR-EXT_FULL"
    full.mkdir(parents=True, exist_ok=True)
    for cam in cameras:
        lines = ["SlotId,X,Y,W,H"] + [",".join(map(str, b)) for b in FAKE_BOXES]
        (full / f"camera{cam}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


# ------------------------------------------------------------- respuestas

def has_key(obj: Any, key: str) -> bool:
    """True si ``key`` aparece como llave en cualquier nivel del JSON."""
    if isinstance(obj, dict):
        return key in obj or any(has_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(has_key(v, key) for v in obj)
    return False


def private_markers() -> list[str]:
    _u = "Us" + "ers"  # se arma por partes para no dejar rutas literales en el repo
    bs = chr(92)  # barra invertida
    markers = ["C:" + bs + _u, "C:/" + _u, "/" + _u + "/", bs + _u + bs]
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    if user and len(user) >= 3:
        markers.append(user)
    return markers


def leaks(text: str, *extra: str) -> bool:
    """True si el texto (crudo o con las barras de JSON sin escapar) trae algún marcador."""
    variants = (text, text.replace("\\\\", "\\"))
    markers = private_markers() + [str(e) for e in extra if e]
    return any(m in v for m in markers for v in variants)


def assert_private(resp, *extra: str) -> None:
    """El texto de la respuesta no trae rutas locales ni el nombre de usuario."""
    leaked = leaks(resp.text, *extra)  # booleano: el mensaje de fallo no repite el marcador
    assert leaked is False, f"la respuesta {resp.status_code} expone una ruta local o el usuario"
