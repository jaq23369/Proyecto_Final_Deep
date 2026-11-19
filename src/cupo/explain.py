"""Explicabilidad del modelo final: selección de instancias, oclusión y pruebas de fidelidad.

Este módulo no entrena nada. Trabaja con las predicciones ya guardadas de un run
(``<split>_preds.npz``) y con el modelo ya entrenado.

- ``load_split_table``: tabla de un split con la probabilidad recalculada con la temperatura del run.
- ``select_instances``: elige las instancias locales que se explican (reglas fijas y semilla).

Grad-CAM vive en ``scripts/gradcam.py``; aquí se usa desde el notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cupo.evaluate import probs_from_logits
from cupo.runs import LoadedRun, load_npz

SEED = 42

HIT_BUSY = "acierto ocupado (confianza alta)"
HIT_FREE = "acierto libre (confianza alta)"
ERR_BUSY_AS_FREE = "error ocupado→libre"
ERR_FREE_AS_BUSY = "error libre→ocupado"
NEAR_THRESHOLD = "cerca del umbral t_H"
RAIN_HIT = "acierto con lluvia"
OOD_CASE = "otro parqueo (OOD)"

# Una probabilidad cercana a 1 (o a 0) cuenta como confianza alta.
HIGH_BUSY = 0.999
HIGH_FREE = 0.002


@dataclass(frozen=True)
class Operating:
    """Punto de operación de un run: temperatura y umbrales del escenario de selección."""

    scenario: str
    t_L: float
    t_H: float
    temperature: float


def operating_point(run: LoadedRun) -> Operating:
    scenario = run.config.get("select", {}).get("scenario", "comercial")
    if scenario not in run.thresholds:
        scenario = next(iter(run.thresholds))
    thr = run.thresholds[scenario]
    return Operating(scenario, float(thr["t_L"]), float(thr["t_H"]), float(run.temperature))


def load_split_table(run: LoadedRun, split: str) -> pd.DataFrame:
    """Predicciones de un split (``val``, ``test`` u ``ood``) con ``p`` recalculada con la temperatura."""
    data = load_npz(run.run_dir / f"{split}_preds.npz")
    if "logits" in data:
        p = probs_from_logits(data["logits"], run.temperature)
    elif run.calibrated:
        raise ValueError(f"{split}_preds.npz no trae logits y el run está calibrado.")
    else:
        p = data["p"]
    table = pd.DataFrame({
        "path": data["path"], "y": data["y"].astype(int), "p": p,
        "camera": data["camera"], "weather": data["weather"],
        "date": data["date"], "time": data["time"], "slot_id": data["slot_id"],
    })
    return table


def decision(p: np.ndarray, t_L: float, t_H: float) -> np.ndarray:
    """'libre' si p < t_L, 'ocupado' si p >= t_H y 'por verificar' en medio."""
    p = np.asarray(p)
    return np.where(p >= t_H, "ocupado", np.where(p < t_L, "libre", "por verificar"))


def _pick(pool: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Una fila al azar (con la semilla dada) de un conjunto de candidatos."""
    if pool.empty:
        raise ValueError("No hay candidatos para esta regla.")
    return pool.loc[rng.choice(pool.index.to_numpy())]


def select_instances(table: pd.DataFrame, t_L: float, t_H: float, seed: int = SEED) -> pd.DataFrame:
    """Instancias locales con reglas fijas. Devuelve una fila por instancia, con su motivo.

    Reglas:
    1. acierto ocupado con p >= 0.999 (sol);
    2. acierto libre con p < 0.002 (sol);
    3. error ocupado→libre, el de menor p (el más seguro de su error);
    4. el caso más cercano al umbral t_H (en log-odds), de cualquier clase;
    5. acierto ocupado con lluvia, en una cámara distinta a la del error;
    6. error libre→ocupado, el de mayor p (el más seguro de su error).
    """
    rng = np.random.default_rng(seed)
    dec = decision(table.p.to_numpy(), t_L, t_H)
    df = table.assign(decision=dec)
    sunny = df.weather == "SUNNY"

    rows: list[pd.Series] = []

    def add(label: str, row: pd.Series) -> pd.Series:
        row = row.copy()
        row["motivo"] = label
        rows.append(row)
        return row

    add(HIT_BUSY, _pick(df[(df.y == 1) & (df.p >= HIGH_BUSY) & sunny], rng))
    add(HIT_FREE, _pick(df[(df.y == 0) & (df.p < HIGH_FREE) & sunny], rng))
    err = add(ERR_BUSY_AS_FREE, df[(df.y == 1) & (df.decision == "libre")].sort_values("p").iloc[0])

    logit = np.log(np.clip(df.p, 1e-12, 1 - 1e-12) / (1 - np.clip(df.p, 1e-12, 1 - 1e-12)))
    gap = (logit - np.log(t_H / (1 - t_H))).abs()
    add(NEAR_THRESHOLD, df.loc[gap.idxmin()])

    rain = df[(df.weather == "RAINY") & (df.y == 1) & (df.p >= HIGH_BUSY) & (df.camera != err.camera)]
    add(RAIN_HIT, _pick(rain, rng))
    add(ERR_FREE_AS_BUSY, df[(df.y == 0) & (df.decision == "ocupado")].sort_values("p").iloc[-1])

    out = pd.DataFrame(rows).reset_index().rename(columns={"index": "row"})
    return out[["motivo", "row", "path", "y", "p", "decision", "camera", "weather", "date", "time", "slot_id"]]
