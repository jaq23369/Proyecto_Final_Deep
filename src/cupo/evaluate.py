"""Evaluación de un run en val, test u OOD.

Uso:
    python -m cupo.evaluate --run runs/<id> --split val
    python -m cupo.evaluate --run runs/<id> --split test --allow-test   # solo tras el congelamiento

Escribe en el run (``<tag>`` = ``test`` u ``ood``; con ``--split val`` es ``val_eval``):
- ``<tag>_preds.npz``: y, p, logits, camera, weather, date, time, slot_id, path;
- ``metrics_<tag>.json``: métricas binarias a 0.5, IC95 por bootstrap, tasas de tres estados
  con los umbrales del run, desgloses por cámara y por clima y el valor por escenario;
- ``value_<tag>.json``: ``{escenario: {t_L, t_H, objective_value, components, rates, ci95}}``.

Con ``--split val`` no se sobrescriben los artefactos de val del entrenamiento
(``val_preds.npz``, ``metrics_val.json``, ``value_val.json``): se escriben
``val_eval_preds.npz``, ``metrics_val_eval.json`` y ``value_val_eval.json``. La fuente de
verdad de val para la tabla de resultados es ``summary.json`` (más ``calibration.json`` si
existe); la re-evaluación en val sirve para comprobar el run cargado (por ejemplo con la
temperatura aplicada o en val completo si se entrenó con un subconjunto).

Los umbrales son siempre los elegidos en validación (``thresholds.json`` o
``calibration.json``); aquí nunca se re-optimizan. Este módulo también expone las
funciones de reporte que usan ``train.py`` y ``scripts/train_classical.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cupo import metrics as M
from cupo.runs import LoadedRun, load_run, save_json, save_npz
from cupo.thresholds import optimize
from cupo.value_model import compute_value, load_scenario

SCENARIOS: tuple[str, ...] = ("comercial", "uvg")
META_COLUMNS: tuple[str, ...] = ("camera", "weather", "date", "time", "slot_id", "path")
DEFAULT_N_BOOT = 1000


# ---------------------------------------------------------------------------
# Escenarios y probabilidades
# ---------------------------------------------------------------------------

def scenario_names(primary: str = "comercial") -> list[str]:
    """Escenarios con el de selección primero."""
    if primary not in SCENARIOS:
        raise ValueError(f"Escenario desconocido: {primary!r}. Opciones: {SCENARIOS}")
    return [primary, *[s for s in SCENARIOS if s != primary]]


def load_scenarios(primary: str = "comercial") -> dict[str, dict[str, Any]]:
    return {name: load_scenario(name) for name in scenario_names(primary)}


def probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """P(ocupado) = softmax(logits / T)[:, 1], calculado de forma estable."""
    logits = np.asarray(logits, dtype=np.float64)
    z = (logits[:, 1] - logits[:, 0]) / float(temperature)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

def logits_from_loader(model, loader, device, amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Logits ``[N, 2]`` (float32) y etiquetas de un DataLoader."""
    import torch

    use_amp = amp and torch.device(device).type == "cuda"
    model.eval()
    out_logits, out_y = [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            out_logits.append(logits.float().cpu().numpy())
            out_y.append(np.asarray(y))
    return np.concatenate(out_logits), np.concatenate(out_y).astype(int)


def predict_torch(model, df: pd.DataFrame, input_size: int, device, batch_size: int = 256,
                  num_workers: int = 4, amp: bool = True) -> np.ndarray:
    """Logits de un modelo de torch sobre las imágenes de ``df``."""
    from cupo.data import make_loader
    from cupo.transforms import build_transforms

    tfm = build_transforms(input_size, train=False, augment=False)
    loader = make_loader(df, tfm, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits, _ = logits_from_loader(model, loader, device, amp)
    return logits


def classical_logits(bundle: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    """Logits ``[0, z]`` de la regresión logística: softmax da sigmoid(z) = P(ocupado)."""
    z = bundle["pipeline"].decision_function(features)
    return np.stack([np.zeros_like(z), z], axis=1).astype(np.float32)


def predict_classical(bundle: Mapping[str, Any], df: pd.DataFrame, n_jobs: int = -1) -> np.ndarray:
    """Logits del baseline HOG + regresión logística sobre las imágenes de ``df``."""
    from cupo.classical import hog_features

    feats = hog_features(df["path"].tolist(), bundle["hog"], n_jobs=n_jobs)
    return classical_logits(bundle, feats)


def predict_run(run: LoadedRun, df: pd.DataFrame, device, batch_size: int, num_workers: int) -> np.ndarray:
    """Logits sin calibrar de un run (torch o clásico)."""
    if run.kind == "torch":
        amp = bool(run.config.get("train", {}).get("amp", True))
        return predict_torch(run.model, df, run.input_size, device, batch_size, num_workers, amp)
    return predict_classical(run.model, df)


# ---------------------------------------------------------------------------
# Métricas, valor e intervalos
# ---------------------------------------------------------------------------

def _accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == (y == 1)))


def _f1_macro(y: np.ndarray, p: np.ndarray) -> float:
    pred = p >= 0.5
    tp = np.sum(pred & (y == 1))
    tn = np.sum(~pred & (y == 0))
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    f1_b = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_f = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    return float((f1_b + f1_f) / 2)


def _rate_busy_as_free(y: np.ndarray, p: np.ndarray) -> float:
    busy = y == 1
    return float(np.mean(p[busy] < 0.5)) if busy.any() else float("nan")


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < y.size else float("nan")


CI_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": _accuracy,
    "f1_macro": _f1_macro,
    "rate_busy_as_free": _rate_busy_as_free,
    "auc": _auc,
}


def metric_cis(y: np.ndarray, p: np.ndarray, n_boot: int, seed: int = 0) -> dict[str, list[float]]:
    """IC95 por bootstrap de las métricas principales (umbral 0.5)."""
    return {name: list(M.bootstrap_ci(fn, y, p, n_boot=n_boot, seed=seed))
            for name, fn in CI_METRICS.items()}


def choose_thresholds(y: np.ndarray, p: np.ndarray, scenarios: Mapping[str, Mapping]) -> dict[str, dict[str, float]]:
    """Umbrales que maximizan el valor de cada escenario (solo se llama con val)."""
    out = {}
    for name, cfg in scenarios.items():
        best = optimize(y, p, cfg)
        out[name] = {"t_L": float(best["t_L"]), "t_H": float(best["t_H"])}
    return out


def value_report(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
    scenarios: Mapping[str, Mapping],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Valor de cada escenario con umbrales fijos + IC95 del objetivo por bootstrap."""
    report: dict[str, dict[str, Any]] = {}
    for name, cfg in scenarios.items():
        if name not in thresholds:
            continue
        t_L, t_H = float(thresholds[name]["t_L"]), float(thresholds[name]["t_H"])
        rates = M.three_state_rates(y, p, t_L, t_H)
        comp = compute_value(rates, cfg)

        def objective(yb: np.ndarray, pb: np.ndarray, _cfg=cfg, _tl=t_L, _th=t_H) -> float:
            return float(compute_value(M.three_state_rates(yb, pb, _tl, _th), _cfg)["objective"])

        report[name] = {
            "t_L": t_L,
            "t_H": t_H,
            "objective_name": cfg["objective"],
            "objective_value": float(comp["objective"]),
            "components": comp,
            "rates": rates,
            "ci95": {"objective": list(M.bootstrap_ci(objective, y, p, n_boot=n_boot, seed=seed))},
        }
    return report


def _three_state_by_group(df: pd.DataFrame, by: str, t_L: float, t_H: float) -> dict[str, dict]:
    return {str(g): M.three_state_rates(sub["y"].to_numpy(), sub["p"].to_numpy(), t_L, t_H)
            for g, sub in df.groupby(by, sort=True)}


def metrics_report(
    y: np.ndarray,
    p: np.ndarray,
    meta: pd.DataFrame,
    thresholds: Mapping[str, Mapping[str, float]],
    values: Mapping[str, Mapping[str, Any]],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """Contenido de ``metrics_{split}.json``."""
    df = pd.DataFrame({"y": y, "p": p, "camera": meta["camera"].astype(str).to_numpy(),
                       "weather": meta["weather"].astype(str).to_numpy()})
    primary = next(iter(values), None)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "binary": M.binary_metrics(y, p, 0.5),
        "ci95": metric_cis(y, p, n_boot, seed),
        "three_state": {name: v["rates"] for name, v in values.items()},
        "thresholds": {name: {"t_L": v["t_L"], "t_H": v["t_H"]} for name, v in values.items()},
        "value": {name: v["objective_value"] for name, v in values.items()},
        "value_ci95": {name: v["ci95"]["objective"] for name, v in values.items()},
        "breakdown": {
            "camera": M.breakdown(df, by="camera"),
            "weather": M.breakdown(df, by="weather"),
        },
    }
    if primary is not None:
        t_L, t_H = thresholds[primary]["t_L"], thresholds[primary]["t_H"]
        out["breakdown_three_state"] = {
            "scenario": primary,
            "camera": _three_state_by_group(df, "camera", t_L, t_H),
            "weather": _three_state_by_group(df, "weather", t_L, t_H),
        }
    return out


def write_split_outputs(
    run_dir: Path,
    split: str,
    df: pd.DataFrame,
    logits: np.ndarray,
    p: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
    scenarios: Mapping[str, Mapping],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    extra: Mapping[str, Any] | None = None,
    tag: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Escribe ``{tag}_preds.npz``, ``metrics_{tag}.json`` y ``value_{tag}.json``.

    ``tag`` es el sufijo de los archivos; por defecto, el nombre de la partición.
    """
    tag = tag or split
    y = df["label"].to_numpy().astype(int)
    save_npz(run_dir / f"{tag}_preds.npz", y=y, p=p, logits=logits,
             **{c: df[c].astype(str).to_numpy() for c in META_COLUMNS})
    values = value_report(y, p, thresholds, scenarios, n_boot, seed)
    metrics = {"run_id": run_dir.name, "split": split, "tag": tag, **(extra or {}),
               **metrics_report(y, p, df, thresholds, values, n_boot, seed)}
    save_json(run_dir / f"metrics_{tag}.json", metrics)
    save_json(run_dir / f"value_{tag}.json", values)
    return metrics, values


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def output_tag(split: str) -> str:
    """Sufijo de los archivos de salida: ``val_eval`` para val, el nombre de la partición si no."""
    return "val_eval" if split == "val" else split


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa un run en val, test u OOD.")
    parser.add_argument("--run", required=True, help="runs/<run_id> o solo el run_id")
    parser.add_argument("--split", required=True, choices=["val", "test", "ood"])
    parser.add_argument("--allow-test", action="store_true",
                        help="obligatorio para test/ood (solo tras el congelamiento)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=None,
                        help=f"réplicas bootstrap (por defecto eval.n_boot de la config o {DEFAULT_N_BOOT})")
    parser.add_argument("--device", default=None, help="cuda o cpu (por defecto cuda si hay)")
    args = parser.parse_args(argv)
    if args.split in ("test", "ood") and not args.allow_test:
        parser.error(
            f"--split {args.split} está bloqueado hasta el congelamiento del modelo. "
            "Agrega --allow-test solo para la evaluación final única."
        )
    return args


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _print_summary(split: str, metrics: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    b = metrics["binary"]
    print(f"[{split}] n={metrics['n']}  acc={b['accuracy']:.4f}  f1_macro={b['f1_macro']:.4f}  "
          f"auc={b['auc']:.4f}  ocupado->libre={b['rate_busy_as_free']:.4f}")
    for name, v in values.items():
        lo, hi = v["ci95"]["objective"]
        print(f"  {name}: t_L={v['t_L']:.2f} t_H={v['t_H']:.2f}  {v['objective_name']}="
              f"{v['objective_value']:,.1f} (IC95 {lo:,.1f} a {hi:,.1f})  "
              f"verificar={v['rates']['verify_rate']:.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    from cupo.data import load_split

    args = parse_args(argv)
    device = args.device or _default_device()
    run = load_run(args.run, device=device)
    if not run.thresholds:
        print(f"El run {run.run_id} no tiene umbrales (thresholds.json).", file=sys.stderr)
        return 1
    primary = run.config.get("select", {}).get("scenario", "comercial")
    scenarios = {k: v for k, v in load_scenarios(primary).items() if k in run.thresholds}
    n_boot = args.n_boot or int(run.config.get("eval", {}).get("n_boot", DEFAULT_N_BOOT))

    df = load_split(args.split, allow_test=args.allow_test)
    t0 = time.perf_counter()
    logits = predict_run(run, df, device, args.batch_size, args.num_workers)
    pred_s = time.perf_counter() - t0
    p = probs_from_logits(logits, run.temperature)
    extra = {"temperature": run.temperature, "calibrated": run.calibrated,
             "model_kind": run.kind, "predict_time_s": pred_s}
    tag = output_tag(args.split)
    metrics, values = write_split_outputs(run.run_dir, args.split, df, logits, p,
                                          run.thresholds, scenarios, n_boot, extra=extra, tag=tag)
    print(f"Predicción de {len(df):,} imágenes en {pred_s:.1f} s (T={run.temperature:.3f})")
    _print_summary(args.split, metrics, values)
    print(f"Resultados en {run.run_dir} (metrics_{tag}.json, value_{tag}.json, {tag}_preds.npz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
