"""Runs sintéticos (formato de ``runs/<id>/``) para probar la tabla de iteraciones y las figuras.

Nada se entrena: las predicciones de val salen de logits sintéticos con semilla fija.
- ``make_run``: escribe summary.json, config.yaml, history.csv, thresholds.json,
  value_val.json y val_preds.npz (y, p, logits, camera, weather, date, path); opcionalmente
  calibration.json, metrics_test.json, metrics_ood.json, benchmark.json y *_preds.npz.
- ``load_scripts``: importa ``scripts/results_table.py`` y ``scripts/make_figures.py``.
- ``manual_rates`` / ``sigmoid``: cálculo independiente de las tasas de tres estados.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cupo import paths
from cupo.thresholds import optimize
from cupo.value_model import compute_value, load_scenario

REPO = paths.repo_root()
SCRIPTS = REPO / "scripts"
SCENARIOS = ("comercial", "uvg")
HISTORY_COLS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro",
                "val_auc", "val_value", "lr", "epoch_time_s"]
CAMERAS = np.array(["1", "2", "3"])
WEATHERS = np.array(["SUNNY", "OVERCAST"])

_SCEN_CACHE: dict[str, dict] = {}


def scenarios() -> dict[str, dict]:
    if not _SCEN_CACHE:
        _SCEN_CACHE.update({name: load_scenario(name) for name in SCENARIOS})
    return _SCEN_CACHE


def load_scripts():
    """(results_table, make_figures) importados como módulos con esos nombres."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import make_figures  # noqa: PLC0415
    import results_table  # noqa: PLC0415

    return results_table, make_figures


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def manual_rates(y: np.ndarray, p: np.ndarray, t_L: float, t_H: float) -> dict[str, float]:
    """Tasas de tres estados contadas a mano (p < t_L libre, p ≥ t_H ocupado, resto verificar)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    free_shown = p < t_L
    busy_shown = (p >= t_H) & ~free_shown
    verify = ~(free_shown | busy_shown)
    nb, nf = (y == 1).sum(), (y == 0).sum()
    return {
        "q_OL": ((y == 1) & free_shown).sum() / nb,
        "q_OV": ((y == 1) & verify).sum() / nb,
        "q_OO": ((y == 1) & busy_shown).sum() / nb,
        "q_LL": ((y == 0) & free_shown).sum() / nf,
        "q_LV": ((y == 0) & verify).sum() / nf,
        "q_LO": ((y == 0) & busy_shown).sum() / nf,
        "verify_rate": verify.mean(),
    }


def synthetic_preds(n: int, sep: float, noise: float, seed: int, busy_frac: float = 0.8):
    """y, logits [n, 2] y metadatos (cámara, clima, fecha) con semilla fija."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, busy_frac, n)
    z = np.where(y == 1, sep, -sep) + rng.normal(0.0, noise, n)
    logits = np.stack([np.zeros(n), z], axis=1).astype(np.float32)
    camera = CAMERAS[rng.integers(0, len(CAMERAS), n)]
    weather = WEATHERS[rng.integers(0, len(WEATHERS), n)]
    date = np.array(["2015-11-20"] * n)
    path = np.array([f"SUNNY/2015-11-20/camera{c}/x_{i}.jpg" for i, c in enumerate(camera)])
    return y, logits, camera, weather, date, path


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_preds(path: Path, y, logits, camera, weather, date, files, temperature: float = 1.0):
    z = (logits[:, 1].astype(float) - logits[:, 0].astype(float)) / temperature
    np.savez_compressed(path, y=y, p=sigmoid(z), logits=logits, camera=camera, weather=weather,
                        date=date, path=files)


def make_run(
    root: Path,
    name: str,
    seed: int = 42,
    *,
    sep: float = 2.0,
    noise: float = 1.2,
    n: int = 400,
    data_seed: int | None = None,
    subset: float = 1.0,
    stored_values: dict[str, float] | None = None,
    calibration: dict[str, Any] | None = None,
    metrics_test: dict[str, Any] | None = None,
    metrics_ood: dict[str, Any] | None = None,
    benchmark: dict[str, Any] | None = None,
    extra_split_preds: tuple[str, ...] = (),
    summary_extra: dict[str, Any] | None = None,
    params_total: int = 50_000,
    train_time_s: float = 100.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Crea ``root/<run_id>/`` con el formato de los runs y devuelve sus datos clave.

    Los umbrales salen de ``thresholds.optimize`` sobre las p de val (T = 1) y el valor
    guardado es ``compute_value`` en ese punto, salvo que ``stored_values`` lo sobrescriba.
    """
    run_id = run_id or f"{name}_s{seed}_20261111-1200"
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True)
    y, logits, camera, weather, date, files = synthetic_preds(
        n, sep, noise, seed if data_seed is None else data_seed)
    p = sigmoid(logits[:, 1] - logits[:, 0])
    _write_preds(run_dir / "val_preds.npz", y, logits, camera, weather, date, files)

    scen = scenarios()
    thresholds, values, value_val = {}, {}, {}
    for scen_name, cfg in scen.items():
        opt = optimize(y, p, cfg)
        thresholds[scen_name] = {"t_L": float(opt["t_L"]), "t_H": float(opt["t_H"])}
        obj = float(compute_value(manual_rates(y, p, opt["t_L"], opt["t_H"]), cfg)["objective"])
        values[scen_name] = obj
        value_val[scen_name] = {"t_L": opt["t_L"], "t_H": opt["t_H"], "objective_value": obj,
                                "ci95": {"objective": [obj - 10.0, obj + 10.0]}}
    if stored_values:
        values.update(stored_values)
    _write_json(run_dir / "thresholds.json", thresholds)
    _write_json(run_dir / "value_val.json", value_val)

    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    f1_busy = 2 * tp / (2 * tp + fp + fn)
    f1_free = 2 * tn / (2 * tn + fp + fn)
    summary = {
        "run_id": run_id, "name": name, "model": "sintetico", "notes": f"run sintético {name}",
        "params_total": params_total, "params_trainable": params_total, "size_mb": params_total * 4 / 2**20,
        "best_epoch": 2, "train_time_s": train_time_s, "gpu": "ninguna",
        "train_cost_usd": train_time_s / 3600 * 0.35, "train_cost_q": train_time_s / 3600 * 0.35 * 7.7,
        "seed": seed, "subset": subset, "select_scenario": "comercial",
        "train_acc": 0.9, "val_f1_macro": (f1_busy + f1_free) / 2,
        "val_rate_busy_as_free": fn / int((y == 1).sum()),
        "val_value_comercial": values["comercial"],
        "val_value_comercial_ci95": [values["comercial"] - 10.0, values["comercial"] + 10.0],
        "val_value_uvg": values["uvg"],
        "val_value_uvg_ci95": [values["uvg"] - 10.0, values["uvg"] + 10.0],
    }
    summary.update(summary_extra or {})
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "config.yaml").write_text(
        f"name: {name}\nseed: {seed}\nnotes: run sintético\ndata:\n  subset: {subset}\n",
        encoding="utf-8")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HISTORY_COLS)
        for epoch in (1, 2, 3):
            writer.writerow([epoch, 0.6 / epoch, 0.8 + 0.03 * epoch, 0.5 / epoch, 0.85, 0.8,
                             0.9, values["comercial"] * (0.8 + 0.05 * epoch), 1e-3, 5.0])

    temperature = 1.0
    if calibration is not None:
        _write_json(run_dir / "calibration.json", calibration)
        temperature = float(calibration.get("temperature", 1.0))
    if metrics_test is not None:
        _write_json(run_dir / "metrics_test.json", metrics_test)
    if metrics_ood is not None:
        _write_json(run_dir / "metrics_ood.json", metrics_ood)
    if benchmark is not None:
        _write_json(run_dir / "benchmark.json", benchmark)
    for k, split in enumerate(extra_split_preds):
        ys, lg, cam, wea, dt, fl = synthetic_preds(200, sep, noise, 1000 + k)
        _write_preds(run_dir / f"{split}_preds.npz", ys, lg, cam, wea, dt, fl, temperature)
    return {"run_dir": run_dir, "run_id": run_id, "y": y, "logits": logits, "p": p,
            "thresholds": thresholds, "values": values, "summary": summary}


def metrics_split(f1: float, rate: float, value_comercial: float) -> dict[str, Any]:
    """Contenido mínimo de ``metrics_{test,ood}.json`` (formato de ``evaluate``)."""
    return {"binary": {"f1_macro": f1, "rate_busy_as_free": rate},
            "value": {"comercial": value_comercial, "uvg": value_comercial * 3}}


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """{ruta relativa: (tamaño, mtime_ns)} de todos los archivos bajo ``root``."""
    if not root.exists():
        return {}
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}
