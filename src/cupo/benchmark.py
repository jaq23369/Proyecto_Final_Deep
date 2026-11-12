"""Benchmark de inferencia: latencia, throughput, costo por predicción y volumen.

Uso:
    python -m cupo.benchmark --onnx models/best.onnx [--run runs/<id>] [--n 300] [--out DIR]

Mide, con patches de val preprocesados con ``transforms.preprocess_numpy`` (nunca test):

- latencia p50/p95 por espacio (batch 1) y por frame (batch = 46 espacios);
- en CPU con ONNX Runtime, con 1 hilo (equivale a 1 núcleo) y con los hilos por defecto;
- en GPU con PyTorch si se pasa ``--run`` y hay CUDA (con AMP si el run lo usaba);
- throughput en predicciones/s;
- ``costo_por_1000_pred_Q`` en CPU: tiempo de 1 núcleo por cada 1,000 predicciones en modo
  frame, incluido el preprocesamiento, valorado a ``cpu_server_usd_per_hour``
  (``configs/costs.yaml``) suponiendo que esa tarifa paga 1 núcleo dedicado. Es una
  estimación con 1 núcleo medido en el CPU de ``env.cpu``; también se guarda la variante
  solo con el modelo (``cost.model_only``);
- volumen: con 1 frame cada ``refresh_s`` (30 s) por cámara, cuántas cámaras soporta
  1 núcleo y si 1 núcleo alcanza para los escenarios de ``configs/value/``.

La latencia del modelo no incluye abrir ni preprocesar la imagen; ese tiempo se mide
aparte (``preprocess_ms_per_patch``: abrir patches desde disco; en producción sería
decodificar un frame + 46 recortes) y sí entra en el costo y en el volumen, con la misma
base: ``(frame_mean_ms + 46 · preprocess_ms) / 46`` por predicción.

Escribe ``benchmark.json`` en ``--out`` (por defecto ``models/``) y, si se pasa ``--run``,
también en el run. ``--run`` debe ser el run del que salió el ``.onnx`` (``run_id`` del
``best.json`` que lo acompaña); si no coincide o no hay ``best.json``, no se escribe nada.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cupo import paths
from cupo.runs import load_json, save_json

SLOTS_PER_FRAME = 46       # espacios por frame (batch del modo frame)
REFRESH_S = 30.0           # 1 frame cada 30 s por cámara
TARGET_UTILIZATION = 0.7   # margen operativo: no planear a más del 70 % del núcleo
WARMUP = 10
SAMPLE_SEED = 0
SCENARIOS: tuple[str, ...] = ("comercial", "uvg")
PREPROCESS_BASIS = ("abrir patches desde disco + preprocess_numpy; en producción sería "
                    "decodificar un frame + 46 recortes")


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def load_val_patches(input_size: int, n_images: int = SLOTS_PER_FRAME,
                     seed: int = SAMPLE_SEED) -> tuple[np.ndarray, list[float]]:
    """Patches de val preprocesados ``[n, 3, H, W]`` y el tiempo (ms) de abrir + preprocesar cada uno."""
    from PIL import Image

    from cupo.data import load_split
    from cupo.transforms import preprocess_numpy

    df = load_split("val")
    rows = df.sample(n=min(n_images, len(df)), random_state=seed)
    arrays, times_ms = [], []
    for path in rows["path"]:
        t0 = time.perf_counter()
        with Image.open(path) as im:
            arrays.append(preprocess_numpy(im, input_size))
        times_ms.append((time.perf_counter() - t0) * 1000)
    return np.concatenate(arrays, axis=0), times_ms


def make_frame_batch(patches: np.ndarray, batch: int = SLOTS_PER_FRAME) -> np.ndarray:
    """Batch de ``batch`` patches (repite si hay menos)."""
    idx = np.arange(batch) % len(patches)
    return np.ascontiguousarray(patches[idx])


# ---------------------------------------------------------------------------
# Medición
# ---------------------------------------------------------------------------

def latency_stats(times_ms: Sequence[float], batch: int) -> dict[str, float]:
    """p50, p95 y media en ms + throughput (predicciones/s) a partir de la media."""
    arr = np.asarray(times_ms, dtype=np.float64)
    mean = float(arr.mean())
    return {
        "batch": int(batch),
        "n": int(arr.size),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": mean,
        "throughput_pred_s": batch / (mean / 1000) if mean > 0 else None,
    }


def time_calls(fn: Callable[[int], Any], n: int, warmup: int = WARMUP,
               sync: Callable[[], None] | None = None) -> list[float]:
    """Tiempo (ms) de ``n`` llamadas a ``fn(i)`` después de ``warmup`` llamadas."""
    for i in range(warmup):
        fn(i)
    if sync is not None:
        sync()
    out = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        if sync is not None:
            sync()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def bench_onnx_cpu(onnx_path: Path, patches: np.ndarray, frame: np.ndarray, n: int,
                   n_frame: int, threads: int | None) -> dict[str, Any]:
    """Latencia de ONNX Runtime en CPU por espacio (batch 1) y por frame."""
    from cupo.export_onnx import onnx_session

    sess = onnx_session(onnx_path, threads=threads)
    name_in = sess.get_inputs()[0].name
    singles = [patches[i:i + 1] for i in range(len(patches))]

    def run_single(i: int) -> None:
        sess.run(None, {name_in: singles[i % len(singles)]})

    def run_frame(_: int) -> None:
        sess.run(None, {name_in: frame})

    return {
        "engine": "onnxruntime",
        "threads": threads if threads is not None else "default",
        "per_slot": latency_stats(time_calls(run_single, n), 1),
        "per_frame": latency_stats(time_calls(run_frame, n_frame), len(frame)),
    }


def bench_torch_gpu(run: str, patches: np.ndarray, frame: np.ndarray, n: int,
                    n_frame: int) -> dict[str, Any] | None:
    """Latencia de PyTorch en GPU (incluye copiar la entrada y traer los logits a CPU)."""
    import torch

    from cupo.runs import load_run

    if not torch.cuda.is_available():
        return None
    loaded = load_run(run, device="cuda")
    if loaded.kind != "torch":
        return None
    amp = bool(loaded.config.get("train", {}).get("amp", True))
    model = loaded.model

    def infer(x: np.ndarray) -> None:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            model(torch.from_numpy(x).to("cuda", non_blocking=False)).float().cpu()

    singles = [patches[i:i + 1] for i in range(len(patches))]
    sync = torch.cuda.synchronize
    return {
        "engine": "torch",
        "device": torch.cuda.get_device_name(0),
        "amp_fp16": amp,
        "per_slot": latency_stats(time_calls(lambda i: infer(singles[i % len(singles)]), n, sync=sync), 1),
        "per_frame": latency_stats(time_calls(lambda _: infer(frame), n_frame, sync=sync), len(frame)),
    }


# ---------------------------------------------------------------------------
# Costo y volumen
# ---------------------------------------------------------------------------

def load_costs(path: Path | None = None) -> dict[str, float]:
    with (path or paths.configs_dir() / "costs.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def cost_per_1000(throughput_pred_s: float, usd_per_hour: float, gtq_per_usd: float) -> dict[str, float]:
    """Costo de 1,000 predicciones con 1 núcleo a ``throughput_pred_s``."""
    seconds = 1000.0 / throughput_pred_s
    usd = seconds / 3600.0 * usd_per_hour
    return {"core_seconds_per_1000": seconds, "usd": usd, "q": usd * gtq_per_usd}


def volume(frame_mean_ms: float, preprocess_ms: float, slots: int = SLOTS_PER_FRAME,
           refresh_s: float = REFRESH_S) -> dict[str, float]:
    """Cámaras por núcleo: cada cámara pide 1 frame (``slots`` espacios) cada ``refresh_s``."""
    frame_s = (frame_mean_ms + slots * preprocess_ms) / 1000.0
    cams = refresh_s / frame_s
    return {
        "slots_per_frame": slots,
        "refresh_s": refresh_s,
        "seconds_per_frame_1core": frame_s,
        "cameras_per_core": int(np.floor(cams)),
        "cameras_per_core_at_70pct": int(np.floor(cams * TARGET_UTILIZATION)),
        "slots_per_core": int(np.floor(cams * slots)),
    }


def scenario_volume(frame_s: float, slots: int = SLOTS_PER_FRAME) -> dict[str, dict[str, Any]]:
    """¿Alcanza 1 núcleo para el parqueo de cada escenario? (espacios, cámaras y refresco)."""
    from cupo.value_model import load_scenario

    out: dict[str, dict[str, Any]] = {}
    for name in SCENARIOS:
        prm = load_scenario(name)["params"]
        n_cams = int(prm.get("n_camaras", 1))
        refresh = float(prm.get("refresh_s", REFRESH_S))
        espacios = int(prm.get("espacios", n_cams * slots))
        # Cada cámara procesa sus espacios; se cobra el frame completo de 46 aunque tenga menos
        busy_s = n_cams * frame_s
        out[name] = {
            "espacios": espacios,
            "n_camaras": n_cams,
            "refresh_s": refresh,
            "pred_per_s_needed": espacios / refresh,
            "core_utilization": busy_s / refresh,
            "target_utilization": TARGET_UTILIZATION,
            "fits_1_core": busy_s / refresh <= TARGET_UTILIZATION,
        }
    return out


# ---------------------------------------------------------------------------
# Entorno
# ---------------------------------------------------------------------------

def cpu_name() -> str:
    """Nombre comercial del CPU (registro de Windows, /proc/cpuinfo o ``platform``)."""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def environment() -> dict[str, Any]:
    import onnxruntime as ort

    return {"cpu": cpu_name(), "cpu_count": os.cpu_count(), "onnxruntime": ort.__version__,
            "python": platform.python_version(), "platform": platform.platform()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latencia, throughput y costo de inferencia.")
    parser.add_argument("--onnx", default=str(paths.models_dir() / "best.onnx"), help="modelo ONNX")
    parser.add_argument("--run", default=None, help="run de PyTorch para medir también en GPU")
    parser.add_argument("--n", type=int, default=300, help="repeticiones por espacio (batch 1)")
    parser.add_argument("--n-frame", type=int, default=None,
                        help="repeticiones por frame (por defecto min(n, 100))")
    parser.add_argument("--out", default=str(paths.models_dir()), help="carpeta de benchmark.json")
    parser.add_argument("--no-gpu", action="store_true", help="omite la medición en GPU")
    return parser.parse_args(argv)


def onnx_input_size(onnx_path: Path) -> int:
    """Tamaño de entrada (H) del modelo, leído del grafo ONNX."""
    import onnx

    dims = onnx.load(str(onnx_path)).graph.input[0].type.tensor_type.shape.dim
    return int(dims[2].dim_value)


def model_info(onnx_path: Path) -> dict[str, Any]:
    """Nombre y run del modelo según el ``best.json`` que acompaña al ``.onnx`` (si existe)."""
    card_path = onnx_path.with_suffix(".json")
    card = load_json(card_path) if card_path.exists() else {}
    # Solo el nombre del archivo: el JSON puede publicarse y no debe llevar rutas locales
    return {"onnx": onnx_path.name, "size_mb": onnx_path.stat().st_size / 1024**2,
            "name": card.get("name"), "run_id": card.get("run_id"), "model": card.get("model")}


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    onnx_path = Path(args.onnx).resolve()
    n_frame = args.n_frame or min(args.n, 100)
    input_size = onnx_input_size(onnx_path)
    patches, pre_ms = load_val_patches(input_size)
    frame = make_frame_batch(patches)

    cpu_1 = bench_onnx_cpu(onnx_path, patches, frame, args.n, n_frame, threads=1)
    cpu_all = bench_onnx_cpu(onnx_path, patches, frame, args.n, n_frame, threads=None)
    gpu = None if (args.run is None or args.no_gpu) else bench_torch_gpu(args.run, patches, frame, args.n, n_frame)

    costs = load_costs()
    usd_h, gtq = float(costs["cpu_server_usd_per_hour"]), float(costs["gtq_per_usd"])
    # El primer patch incluye la carga en frío del disco; se descarta
    preprocess_ms = float(np.median(pre_ms[1:] if len(pre_ms) > 1 else pre_ms))
    frame_ms = cpu_1["per_frame"]["mean_ms"]
    slots = len(frame)
    # Misma base que el volumen: modelo por frame + preprocesar cada espacio
    pipeline_pred_s = slots / ((frame_ms + slots * preprocess_ms) / 1000.0)
    cost = cost_per_1000(pipeline_pred_s, usd_h, gtq)
    cost_model = cost_per_1000(cpu_1["per_frame"]["throughput_pred_s"], usd_h, gtq)
    vol = volume(frame_ms, preprocess_ms, slots)
    return {
        "model": model_info(onnx_path),
        "input_size": input_size,
        "data": {"split": "val", "n_images": int(len(patches)), "seed": SAMPLE_SEED},
        "warmup": WARMUP,
        "n_slot": int(args.n),
        "n_frame": int(n_frame),
        "preprocess_ms_per_patch": preprocess_ms,
        "preprocess_basis": PREPROCESS_BASIS,
        "cpu_1_thread": cpu_1,
        "cpu_all_threads": cpu_all,
        "gpu_torch": gpu,
        "costo_por_1000_pred_Q": cost["q"],
        "cost": {**cost,
                 "basis": "1 núcleo (ONNX Runtime, 1 hilo), modo frame, modelo + preprocesamiento",
                 "pipeline_pred_s": pipeline_pred_s,
                 "model_only": {**cost_model, "basis": "1 núcleo, modo frame, solo el modelo"},
                 "estimate_note": f"estimación con 1 núcleo medido en {cpu_name()}",
                 "cpu_server_usd_per_hour": usd_h, "gtq_per_usd": gtq},
        "volume": vol,
        "volume_by_scenario": scenario_volume(vol["seconds_per_frame_1core"]),
        "env": environment(),
        "measured_at": datetime.now().isoformat(timespec="seconds"),
    }


def _fmt(block: dict[str, Any]) -> str:
    return (f"p50 {block['p50_ms']:.2f} ms  p95 {block['p95_ms']:.2f} ms  "
            f"({block['throughput_pred_s']:,.0f} pred/s)")


def print_report(res: dict[str, Any]) -> None:
    print(f"Modelo: {res['model']['name'] or res['model']['onnx']} ({res['model']['size_mb']:.2f} MB), "
          f"CPU: {res['env']['cpu']}")
    print(f"Preprocesamiento por patch (abrir + preprocess_numpy): {res['preprocess_ms_per_patch']:.2f} ms")
    for key, label in (("cpu_1_thread", "CPU 1 hilo"), ("cpu_all_threads", "CPU todos los hilos"),
                       ("gpu_torch", "GPU torch")):
        block = res[key]
        if block is None:
            continue
        print(f"[{label}] por espacio: {_fmt(block['per_slot'])}")
        print(f"[{label}] por frame ({block['per_frame']['batch']}): {_fmt(block['per_frame'])}")
    vol = res["volume"]
    print(f"Costo por 1,000 predicciones (1 núcleo, modelo + preprocesamiento): "
          f"Q{res['costo_por_1000_pred_Q']:.6f} (solo modelo: Q{res['cost']['model_only']['q']:.6f})")
    print(f"Volumen: 1 núcleo procesa un frame de {vol['slots_per_frame']} espacios en "
          f"{vol['seconds_per_frame_1core'] * 1000:.1f} ms, {vol['cameras_per_core']} cámaras a 1 frame "
          f"cada {vol['refresh_s']:.0f} s ({vol['cameras_per_core_at_70pct']} al 70 %)")
    for name, sc in res["volume_by_scenario"].items():
        print(f"  {name}: {sc['n_camaras']} cámaras, {sc['espacios']} espacios: uso de 1 núcleo "
              f"{sc['core_utilization'] * 100:.2f} % ({'alcanza' if sc['fits_1_core'] else 'NO alcanza'})")


def check_run_matches(onnx_path: Path, run: str) -> Path:
    """Carpeta de ``run`` si es el run del que salió el ``.onnx``; si no, ``ValueError``."""
    from cupo.runs import resolve_run_dir

    run_dir = resolve_run_dir(run)
    card_path = onnx_path.with_suffix(".json")
    if not card_path.exists():
        raise ValueError(f"No existe {card_path.name} junto a {onnx_path.name}: no se puede comprobar "
                         f"que el modelo sea del run {run_dir.name}. Exporta con python -m cupo.export_onnx.")
    card_run = load_json(card_path).get("run_id")
    if card_run != run_dir.name:
        raise ValueError(f"El modelo {onnx_path.name} es del run {card_run!r}, no de {run_dir.name!r}: "
                         "la latencia se guardaría en otro run.")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        print(f"Error: no existe el modelo {args.onnx}; exporta uno con python -m cupo.export_onnx",
              file=sys.stderr)
        return 1
    run_dir = None
    if args.run is not None:
        try:
            run_dir = check_run_matches(onnx_path, args.run)
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    res = run_benchmark(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / "benchmark.json"]
    if run_dir is not None:
        targets.append(run_dir / "benchmark.json")
    for target in targets:
        save_json(target, res)
    print_report(res)
    print("Escrito en: " + ", ".join(str(t) for t in targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
