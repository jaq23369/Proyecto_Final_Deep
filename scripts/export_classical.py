"""Exporta el baseline clásico B1 (HOG + regresión logística) para la app.

Uso:
    python scripts/export_classical.py --run runs/<id_b1> --out models/b1_hog_lr.joblib

- Copia ``model.joblib`` del run a ``--out`` (no es un modelo de PyTorch: no se exporta a ONNX).
- Escribe la tarjeta ``<stem de --out>.json`` al lado, con el mismo esquema que
  ``cupo.export_onnx`` (nombre, run, umbrales por escenario, temperatura 1.0, resumen de
  métricas de val) más ``kind: "classical"`` y los parámetros HOG.
- Verifica con patches de val que la p calculada como en la app (HOG desde la imagen PIL en
  memoria con ``cupo.classical.hog_image``) coincide con la de ``val_preds.npz`` del run; si
  no, termina con error.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cupo import paths  # noqa: E402
from cupo.export_onnx import SCENARIOS, val_metrics_summary  # noqa: E402
from cupo.runs import CLASSICAL_FILE, LoadedRun, load_run, resolve_run_dir, run_kind, save_json  # noqa: E402

KIND = "classical"
CHECK_N = 16
CHECK_SEED = 0
CHECK_TOL = 1e-4
CLASS_NAMES: list[str] = ["libre", "ocupado"]


def load_classical_run(run: str | Path) -> LoadedRun:
    """Carga un run clásico (``model.joblib``); falla con un mensaje claro si no lo es."""
    run_dir = resolve_run_dir(run)
    if run_kind(run_dir) != KIND:
        raise ValueError(f"El run {run_dir.name} no es clásico (tiene best.pt): expórtalo con "
                         "python -m cupo.export_onnx.")
    loaded = load_run(run_dir)
    missing = [s for s in SCENARIOS if s not in loaded.thresholds]
    if missing:
        raise ValueError(f"El run {loaded.run_id} no tiene umbrales para {missing} (thresholds.json).")
    return loaded


def p_like_app(bundle: dict[str, Any], image_paths: Sequence[str]) -> np.ndarray:
    """P(ocupado) con el mismo camino que la app: imagen PIL en RGB → HOG → regresión logística."""
    from PIL import Image

    from cupo.classical import classical_logits, hog_image

    feats = []
    for path in image_paths:
        with Image.open(path) as im:
            # La app recorta siempre sobre una imagen en RGB
            feats.append(hog_image(im.convert("RGB"), **bundle["hog"]))
    logits = classical_logits(bundle, np.stack(feats)).astype(np.float64)
    return 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))


def check_against_val_preds(run: LoadedRun, n: int = CHECK_N, seed: int = CHECK_SEED) -> dict[str, Any]:
    """Compara la p de la app con ``val_preds.npz`` en ``n`` patches de val (muestra fija)."""
    from cupo.data import load_split

    preds_path = run.run_dir / "val_preds.npz"
    if not preds_path.exists():
        return {"n": 0, "ok": False, "detail": "el run no tiene val_preds.npz"}
    preds = np.load(preds_path, allow_pickle=False)
    val = load_split("val").reset_index(drop=True)
    if len(val) != len(preds["p"]):
        return {"n": 0, "ok": False, "detail": "val_preds.npz no tiene las mismas filas que val"}
    idx = np.sort(np.random.default_rng(seed).choice(len(val), size=min(n, len(val)), replace=False))
    # Las rutas del npz son absolutas de la máquina del run: se emparejan por metadatos
    for key in ("camera", "date", "time", "slot_id"):
        if not np.array_equal(val[key].to_numpy(str)[idx], preds[key][idx].astype(str)):
            return {"n": 0, "ok": False, "detail": f"val_preds.npz no coincide con val en '{key}'"}
    got = p_like_app(run.model, val["path"].to_numpy()[idx].tolist())
    diff = float(np.abs(got - preds["p"][idx]).max())
    return {"n": int(len(idx)), "max_abs_diff_p": diff, "tolerance": CHECK_TOL, "ok": diff < CHECK_TOL}


def model_card(run: LoadedRun, out_path: Path, check: dict[str, Any]) -> dict[str, Any]:
    """Tarjeta del modelo (mismo esquema que ``export_onnx.model_card`` + ``kind`` y ``hog``)."""
    hog = dict(run.model["hog"])
    return {
        "name": run.config.get("name", run.run_id),
        "run_id": run.run_id,
        "model": run.config.get("model", {}).get("name"),
        "kind": KIND,
        "model_file": out_path.name,
        "input_size": int(hog["image_size"]),
        "output": "z = decision_function(HOG); P(ocupado) = softmax([0, z] / temperature)[:, 1]",
        "classes": CLASS_NAMES,
        "preprocessing": {
            "color": "L (gris, desde RGB)",
            "size": [int(hog["image_size"]), int(hog["image_size"])],
            "resize": "bilinear",
            "scale": "1/255",
            "features": "HOG (block_norm L2-Hys) + StandardScaler",
            "function": "cupo.classical.hog_image",
        },
        "hog": hog,
        "C": run.model.get("C"),
        "temperature": 1.0,
        "calibrated": False,
        "thresholds": {s: {"t_L": float(run.thresholds[s]["t_L"]), "t_H": float(run.thresholds[s]["t_H"])}
                       for s in SCENARIOS},
        "select_scenario": run.config.get("select", {}).get("scenario", "comercial"),
        "val_metrics_summary": val_metrics_summary(run),
        "check": check,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta el baseline B1 (model.joblib) + su tarjeta .json.")
    parser.add_argument("--run", required=True, help="runs/<run_id> o solo el run_id (run clásico)")
    parser.add_argument("--out", default=str(paths.models_dir() / "b1_hog_lr.joblib"),
                        help="ruta del .joblib (por defecto models/b1_hog_lr.joblib); el .json va al lado")
    parser.add_argument("--n-check", type=int, default=CHECK_N,
                        help="patches de val para comparar con val_preds.npz")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run = load_classical_run(args.run)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    out_path = Path(args.out).resolve()
    if out_path.suffix != ".joblib":
        print("Error: --out debe terminar en .joblib", file=sys.stderr)
        return 1
    # Se verifica antes de escribir: si falla, la app no debe listar un modelo inconsistente
    check = check_against_val_preds(run, args.n_check)
    if not check.get("ok"):
        print(f"Error: la verificación con val_preds.npz falló: {check}", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run.run_dir / CLASSICAL_FILE, out_path)
    card_path = out_path.with_suffix(".json")
    save_json(card_path, model_card(run, out_path, check))

    print(f"Modelo: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB, run {run.run_id})")
    print(f"Tarjeta: {card_path.name}")
    for name in SCENARIOS:
        thr = run.thresholds[name]
        print(f"  {name}: t_L={float(thr['t_L']):.2f}  t_H={float(thr['t_H']):.2f}")
    print(f"Verificación con {check['n']} patches de val: max |p_app - p_val_preds| = "
          f"{check['max_abs_diff_p']:.2e} (tolerancia {CHECK_TOL:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
