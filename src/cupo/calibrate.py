"""Calibración por temperature scaling ajustada en val.

Uso:
    python -m cupo.calibrate --run runs/<id>

Lee los logits de ``val_preds.npz`` (siempre sin calibrar), ajusta una temperatura ``T``
minimizando la NLL en val con LBFGS y re-optimiza en val los umbrales de cada escenario
con las probabilidades calibradas (la temperatura cambia la escala de ``p``, así que los
umbrales sin calibrar dejarían de ser los óptimos). Escribe ``calibration.json``:
``{temperature, nll_before, nll_after, ece_before, ece_after, thresholds, value_val}``.
``evaluate`` y la exportación usan la temperatura y estos umbrales juntos.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
import torch

from cupo.evaluate import choose_thresholds, load_scenarios, probs_from_logits, value_report
from cupo.runs import CALIBRATION_FILE, load_npz, load_yaml, resolve_run_dir, save_json


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 100) -> float:
    """Temperatura que minimiza la NLL; se optimiza ``log T`` para que T > 0."""
    z = torch.from_numpy(np.asarray(logits, dtype=np.float64))
    target = torch.from_numpy(np.asarray(y, dtype=np.int64))
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), target)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error sobre la confianza de la clase predicha."""
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = (p >= 0.5) == (y == 1)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total)


def calibrate_run(run: str, n_boot: int = 200) -> dict:
    """Ajusta T en val, re-optimiza umbrales en val y escribe ``calibration.json``."""
    run_dir = resolve_run_dir(run)
    preds = load_npz(run_dir / "val_preds.npz")
    if "logits" not in preds:
        raise KeyError(f"{run_dir / 'val_preds.npz'} no tiene logits")
    y, logits = preds["y"].astype(int), preds["logits"]
    cfg = load_yaml(run_dir / "config.yaml")
    scenarios = load_scenarios(cfg.get("select", {}).get("scenario", "comercial"))

    temperature = fit_temperature(logits, y)
    p0, p1 = probs_from_logits(logits, 1.0), probs_from_logits(logits, temperature)
    thresholds = choose_thresholds(y, p1, scenarios)
    values = value_report(y, p1, thresholds, scenarios, n_boot=n_boot)
    result = {
        "temperature": temperature,
        "fitted_on": "val",
        "n": int(y.size),
        "nll_before": nll(p0, y),
        "nll_after": nll(p1, y),
        "ece_before": ece(p0, y),
        "ece_after": ece(p1, y),
        "thresholds": thresholds,
        "value_val": {k: {"objective_value": v["objective_value"], "ci95": v["ci95"]["objective"],
                          "verify_rate": v["rates"]["verify_rate"]} for k, v in values.items()},
    }
    save_json(run_dir / CALIBRATION_FILE, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Temperature scaling en val para un run.")
    parser.add_argument("--run", required=True, help="runs/<run_id> o solo el run_id")
    parser.add_argument("--n-boot", type=int, default=200, help="réplicas bootstrap del IC95 del valor")
    args = parser.parse_args(argv)
    res = calibrate_run(args.run, n_boot=args.n_boot)
    print(f"T = {res['temperature']:.4f}  NLL {res['nll_before']:.4f} -> {res['nll_after']:.4f}  "
          f"ECE {res['ece_before']:.4f} -> {res['ece_after']:.4f}")
    for name, thr in res["thresholds"].items():
        val = res["value_val"][name]["objective_value"]
        print(f"  {name}: t_L={thr['t_L']:.2f} t_H={thr['t_H']:.2f} valor val={val:,.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
