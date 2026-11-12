"""Exportación de un run de PyTorch a ONNX para la app.

Uso:
    python -m cupo.export_onnx --run runs/<id> --out models/best.onnx

- Entrada ``input`` ``[batch, 3, H, W]`` float32 (preprocesada con
  ``transforms.preprocess_numpy``); salida ``logits`` ``[batch, 2]`` sin calibrar.
  La app calcula P(ocupado) = softmax(logits / T)[:, 1] con la ``temperature`` de
  ``best.json`` y aplica los umbrales ``t_L``/``t_H`` del escenario.
- Eje de batch dinámico y opset 17.
- Escribe ``best.json`` junto al ``.onnx`` con nombre, run, tamaño de entrada,
  temperatura, umbrales y resumen de métricas de validación (todo de ``runs.load_run``,
  que respeta ``calibration.json``).
- Verifica con imágenes de val que ONNX Runtime y PyTorch den los mismos logits
  (diferencia absoluta máxima < 1e-3); si no, termina con error.

Notas de exportación:
- Se usa el exportador clásico de ``torch.onnx`` (``dynamo=False``): no requiere
  ``onnxscript`` y produce un grafo que ONNX Runtime ejecuta sin problemas.
- La exportación se hace en CPU y fp32, sin autocast. La LRN de mAlexNet está envuelta
  en ``FP32LocalResponseNorm`` (que desactiva autocast); en fp32 sus conversiones de
  tipo son identidades y la LRN se traza como Pad + AveragePool + Pow + Div. Se verificó
  numéricamente contra PyTorch (la diferencia queda en el orden de 1e-7). Se probó
  reemplazarla por el nodo ``LRN`` nativo de ONNX y en ONNX Runtime (CPU) resultó más
  lenta (~1.5x), así que se deja la versión trazada.
- Los runs clásicos (B1, ``model.joblib``) no se exportan: no son modelos de PyTorch.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cupo import paths
from cupo.runs import LoadedRun, load_json, load_run, resolve_run_dir, run_kind, save_json

OPSET = 17
INPUT_NAME = "input"
OUTPUT_NAME = "logits"
SCENARIOS: tuple[str, ...] = ("comercial", "uvg")
CHECK_TOL = 1e-3
CHECK_N = 8
CHECK_SEED = 0
CLASS_NAMES: list[str] = ["libre", "ocupado"]


# ---------------------------------------------------------------------------
# Carga y validaciones
# ---------------------------------------------------------------------------

def load_exportable_run(run: str | Path) -> LoadedRun:
    """Carga el run en CPU; falla con un mensaje claro si es un run clásico."""
    run_dir = resolve_run_dir(run)
    if run_kind(run_dir) == "classical":
        raise ValueError(
            f"El run {run_dir.name} es clásico (model.joblib, HOG + regresión logística): "
            "B1 no se exporta a ONNX. Exporta un run de PyTorch (best.pt)."
        )
    loaded = load_run(run_dir, device="cpu")
    missing = [s for s in SCENARIOS if s not in loaded.thresholds]
    if missing:
        raise ValueError(f"El run {loaded.run_id} no tiene umbrales para {missing} "
                         "(thresholds.json o calibration.json).")
    return loaded


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------

def export_model(model, input_size: int, out_path: Path, opset: int = OPSET) -> None:
    """Exporta ``model`` (en CPU, fp32, modo evaluación) con batch dinámico."""
    import torch

    model = model.to("cpu").float().eval()
    dummy = torch.zeros(2, 3, input_size, input_size, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # Solo se silencian los avisos conocidos del exportador clásico
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.onnx")
        warnings.filterwarnings("ignore", message=r".*legacy TorchScript-based ONNX export.*")
        warnings.filterwarnings("ignore", message=r"Constant folding - Only steps=1.*")
        with torch.no_grad():
            torch.onnx.export(
                model, (dummy,), str(out_path),
                dynamo=False,
                opset_version=opset,
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
                do_constant_folding=True,
            )
    import onnx

    onnx.checker.check_model(onnx.load(str(out_path)))


# ---------------------------------------------------------------------------
# Verificación ONNX contra PyTorch
# ---------------------------------------------------------------------------

def sample_val_batch(input_size: int, n: int = CHECK_N, seed: int = CHECK_SEED) -> tuple[np.ndarray, list[str]]:
    """``n`` patches de val (muestra fija con semilla) preprocesados con ``preprocess_numpy``."""
    from PIL import Image

    from cupo.data import load_split
    from cupo.transforms import preprocess_numpy

    df = load_split("val")
    rows = df.sample(n=min(n, len(df)), random_state=seed)
    batch = []
    for path in rows["path"]:
        with Image.open(path) as im:
            batch.append(preprocess_numpy(im, input_size))
    return np.concatenate(batch, axis=0), rows["path"].tolist()


def onnx_session(onnx_path: Path, threads: int | None = None):
    """Sesión de ONNX Runtime en CPU (``threads=None`` usa los valores por defecto)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads is not None:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def compare_with_torch(model, onnx_path: Path, batch: np.ndarray, temperature: float) -> dict[str, Any]:
    """Logits de ONNX y PyTorch (fp32, CPU) sobre ``batch``; con batch completo y de 1 en 1."""
    import torch

    from cupo.evaluate import probs_from_logits

    with torch.no_grad():
        ref = model.to("cpu").float().eval()(torch.from_numpy(batch)).numpy()
    sess = onnx_session(onnx_path)
    got = sess.run([OUTPUT_NAME], {INPUT_NAME: batch})[0]
    # El eje de batch es dinámico: también se prueba con batch 1
    got_single = np.concatenate([sess.run([OUTPUT_NAME], {INPUT_NAME: batch[i:i + 1]})[0]
                                 for i in range(len(batch))])
    diff = float(max(np.abs(got - ref).max(), np.abs(got_single - ref).max()))
    p_diff = float(np.abs(probs_from_logits(got, temperature) - probs_from_logits(ref, temperature)).max())
    return {"n": int(len(batch)), "max_abs_diff_logits": diff, "max_abs_diff_p": p_diff,
            "tolerance": CHECK_TOL, "ok": diff < CHECK_TOL}


# ---------------------------------------------------------------------------
# best.json
# ---------------------------------------------------------------------------

def _binary_summary(run_dir: Path) -> dict[str, Any]:
    """Métricas binarias de val (umbral 0.5) de ``metrics_val.json`` o ``summary.json``."""
    keys = ("accuracy", "f1_macro", "auc", "rate_busy_as_free")
    metrics_path = run_dir / "metrics_val.json"
    if metrics_path.exists():
        binary = load_json(metrics_path).get("binary", {})
        return {k: binary.get(k) for k in keys}
    summary_path = run_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    return {k: summary.get(f"val_{k}") for k in keys}


def _scenario_summary(run: LoadedRun) -> dict[str, dict[str, Any]]:
    """Valor en val por escenario, coherente con la temperatura y los umbrales exportados."""
    from cupo.value_model import load_scenario

    out: dict[str, dict[str, Any]] = {}
    if run.calibrated:
        source = load_json(run.run_dir / "calibration.json").get("value_val", {})
        for name in SCENARIOS:
            v = source.get(name, {})
            out[name] = {"objective_name": load_scenario(name)["objective"],
                         "objective_value": v.get("objective_value"), "ci95": v.get("ci95"),
                         "verify_rate": v.get("verify_rate")}
        return out
    value_path = run.run_dir / "value_val.json"
    source = load_json(value_path) if value_path.exists() else {}
    for name in SCENARIOS:
        v = source.get(name, {})
        out[name] = {"objective_name": v.get("objective_name"),
                     "objective_value": v.get("objective_value"),
                     "ci95": v.get("ci95", {}).get("objective"),
                     "verify_rate": v.get("rates", {}).get("verify_rate")}
    return out


def val_metrics_summary(run: LoadedRun) -> dict[str, Any]:
    return {
        "source": "calibration.json" if run.calibrated else "value_val.json",
        "binary_t05": _binary_summary(run.run_dir),
        "value": _scenario_summary(run),
    }


def model_card(run: LoadedRun, onnx_path: Path, check: dict[str, Any], opset: int) -> dict[str, Any]:
    """Contenido de ``best.json`` (lo lee la app junto con el ``.onnx``)."""
    from cupo.transforms import IMAGENET_MEAN, IMAGENET_STD

    model_cfg = run.config.get("model", {})
    return {
        "name": run.config.get("name", run.run_id),
        "run_id": run.run_id,
        "model": model_cfg.get("name"),
        "onnx_file": onnx_path.name,
        "input_size": int(run.input_size),
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "output": "logits sin calibrar; P(ocupado) = softmax(logits / temperature)[:, 1]",
        "classes": CLASS_NAMES,
        "preprocessing": {
            "color": "RGB",
            "size": [int(run.input_size), int(run.input_size)],
            "resize": "bilinear",
            "scale": "1/255",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "layout": "NCHW",
            "function": "cupo.transforms.preprocess_numpy",
        },
        "opset": opset,
        "temperature": float(run.temperature),
        "calibrated": bool(run.calibrated),
        "thresholds": {s: {"t_L": float(run.thresholds[s]["t_L"]), "t_H": float(run.thresholds[s]["t_H"])}
                       for s in SCENARIOS},
        "select_scenario": run.config.get("select", {}).get("scenario", "comercial"),
        "val_metrics_summary": val_metrics_summary(run),
        "onnx_check": check,
        "note": ("p calculada con ONNX en fp32 puede diferir ~0.01 de la de val_preds.npz, que se "
                 "calculó en GPU con AMP (fp16); los umbrales se eligieron con esa p."),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta un run de PyTorch a ONNX + best.json.")
    parser.add_argument("--run", required=True, help="runs/<run_id> o solo el run_id")
    parser.add_argument("--out", default=str(paths.models_dir() / "best.onnx"),
                        help="ruta del .onnx (por defecto models/best.onnx); best.json va al lado")
    parser.add_argument("--opset", type=int, default=OPSET, help=f"opset de ONNX (>= {OPSET})")
    parser.add_argument("--n-check", type=int, default=CHECK_N,
                        help="imágenes de val para comparar ONNX contra PyTorch")
    args = parser.parse_args(argv)
    if args.opset < OPSET:
        parser.error(f"--opset debe ser >= {OPSET}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run = load_exportable_run(args.run)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    out_path = Path(args.out).resolve()
    export_model(run.model, run.input_size, out_path, args.opset)
    batch, _ = sample_val_batch(run.input_size, args.n_check)
    check = compare_with_torch(run.model, out_path, batch, run.temperature)
    card = model_card(run, out_path, check, args.opset)
    card_path = out_path.with_suffix(".json")
    save_json(card_path, card)

    size_mb = out_path.stat().st_size / 1024**2
    print(f"ONNX: {out_path} ({size_mb:.2f} MB, opset {args.opset}, entrada {run.input_size}x{run.input_size})")
    print(f"Ficha: {card_path} (T={run.temperature:.4f}, calibrado={run.calibrated})")
    for name, thr in card["thresholds"].items():
        print(f"  {name}: t_L={thr['t_L']:.2f}  t_H={thr['t_H']:.2f}")
    print(f"Verificación con {check['n']} imágenes de val: max |ONNX - torch| logits = "
          f"{check['max_abs_diff_logits']:.2e}, p = {check['max_abs_diff_p']:.2e} (tolerancia {CHECK_TOL:g})")
    if not check["ok"]:
        print("Error: ONNX y PyTorch no coinciden dentro de la tolerancia.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
