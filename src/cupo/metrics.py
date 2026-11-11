"""Métricas técnicas de clasificación de espacios de parqueo.

Convenciones:
- ``y`` es la etiqueta real: 1 = ocupado (positivo), 0 = libre.
- ``p`` es P(ocupado) que entrega el modelo, en [0, 1].
- Con un umbral único ``thr`` se predice ocupado si ``p >= thr``.
- Con dos umbrales ``t_L <= t_H``: ``p < t_L`` → LIBRE, ``p >= t_H`` → OCUPADO,
  en otro caso (``t_L <= p < t_H``) → VERIFICAR. Con ``t_L == t_H`` VERIFICAR
  queda vacío y la decisión coincide con ``binary_metrics(thr=t_L)``.

Todas las funciones devuelven tipos nativos de Python (float, int, list), así
que sus resultados se pueden escribir directamente en JSON.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_recall_fscore_support, roc_auc_score

# Recorte para evitar log(0) en la pérdida logarítmica.
_EPS = 1e-7


def _as_arrays(y, p) -> tuple[np.ndarray, np.ndarray]:
    """Convierte etiquetas y probabilidades a arreglos 1D validados."""
    y_arr = np.asarray(y).astype(int).ravel()
    p_arr = np.asarray(p, dtype=float).ravel()
    if y_arr.shape != p_arr.shape:
        raise ValueError(f"y y p deben tener el mismo tamaño: {y_arr.shape} vs {p_arr.shape}")
    return y_arr, p_arr


def _safe_div(num: float, den: float) -> float:
    """División que devuelve NaN cuando el denominador es 0."""
    return float(num / den) if den > 0 else float("nan")


def binary_metrics(y, p, thr: float = 0.5) -> dict:
    """Métricas binarias con un umbral único (ocupado si ``p >= thr``).

    Devuelve accuracy, precisión/recall/F1 de cada clase (``busy`` y ``free``),
    F1 macro, AUC, log loss, la tasa ocupado→libre (el error más caro), la tasa
    libre→ocupado y la matriz de confusión ``[[TN, FP], [FN, TP]]`` con
    positivo = ocupado. AUC es NaN si solo hay una clase presente.
    """
    y_arr, p_arr = _as_arrays(y, p)
    pred = (p_arr >= thr).astype(int)

    tp = int(np.sum((y_arr == 1) & (pred == 1)))
    tn = int(np.sum((y_arr == 0) & (pred == 0)))
    fp = int(np.sum((y_arr == 0) & (pred == 1)))
    fn = int(np.sum((y_arr == 1) & (pred == 0)))
    n_busy, n_free = tp + fn, tn + fp

    # Orden de etiquetas [libre, ocupado] para separar las métricas por clase.
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_arr, pred, labels=[0, 1], zero_division=0
    )
    both_classes = n_busy > 0 and n_free > 0
    auc = float(roc_auc_score(y_arr, p_arr)) if both_classes else float("nan")
    p_clip = np.clip(p_arr, _EPS, 1 - _EPS)
    ll = float(log_loss(y_arr, p_clip, labels=[0, 1])) if y_arr.size else float("nan")

    return {
        "n": int(y_arr.size),
        "n_busy": n_busy,
        "n_free": n_free,
        "threshold": float(thr),
        "accuracy": _safe_div(tp + tn, y_arr.size),
        "precision_busy": float(prec[1]),
        "recall_busy": float(rec[1]),
        "f1_busy": float(f1[1]),
        "precision_free": float(prec[0]),
        "recall_free": float(rec[0]),
        "f1_free": float(f1[0]),
        "f1_macro": float((f1[0] + f1[1]) / 2),
        "auc": auc,
        "log_loss": ll,
        "rate_busy_as_free": _safe_div(fn, n_busy),
        "rate_free_as_busy": _safe_div(fp, n_free),
        "confusion": [[tn, fp], [fn, tp]],
    }


def three_state_rates(y, p, t_L: float, t_H: float) -> dict:
    """Tasas condicionales de la decisión de tres estados.

    ``q_O*`` se condiciona en los espacios ocupados y ``q_L*`` en los libres:
    L = mostrado libre, V = por verificar, O = mostrado ocupado.
    ``verify_rate`` es la fracción global (sin reponderar) en VERIFICAR.
    Regla: ``p < t_L`` → LIBRE, ``p >= t_H`` → OCUPADO, resto → VERIFICAR.
    """
    if t_L > t_H:
        raise ValueError(f"Se requiere t_L <= t_H (t_L={t_L}, t_H={t_H})")
    y_arr, p_arr = _as_arrays(y, p)
    show_free = p_arr < t_L
    show_busy = (p_arr >= t_H) & ~show_free
    show_verify = ~(show_free | show_busy)

    busy, free = y_arr == 1, y_arr == 0
    n_busy, n_free = int(busy.sum()), int(free.sum())
    return {
        "t_L": float(t_L),
        "t_H": float(t_H),
        "n_busy": n_busy,
        "n_free": n_free,
        "q_OL": _safe_div(np.sum(busy & show_free), n_busy),
        "q_OV": _safe_div(np.sum(busy & show_verify), n_busy),
        "q_OO": _safe_div(np.sum(busy & show_busy), n_busy),
        "q_LL": _safe_div(np.sum(free & show_free), n_free),
        "q_LV": _safe_div(np.sum(free & show_verify), n_free),
        "q_LO": _safe_div(np.sum(free & show_busy), n_free),
        "verify_rate": _safe_div(show_verify.sum(), y_arr.size),
    }


def bootstrap_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y,
    p,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Intervalo de confianza por bootstrap de percentiles para ``fn(y, p)``.

    Se remuestrean pares (y, p) con reemplazo ``n_boot`` veces. Las réplicas
    que dan NaN (p. ej. AUC con una sola clase) se ignoran.
    """
    y_arr, p_arr = _as_arrays(y, p)
    rng = np.random.default_rng(seed)
    n = y_arr.size
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = fn(y_arr[idx], p_arr[idx])
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def breakdown(
    df_preds: pd.DataFrame,
    by: str = "camera",
    thr: float = 0.5,
    y_col: str = "y",
    p_col: str = "p",
) -> dict:
    """Métricas binarias por grupo (``by`` = 'camera' o 'weather').

    ``df_preds`` debe tener las columnas ``y_col``, ``p_col`` y ``by``.
    Devuelve ``{grupo: binary_metrics(...)}`` con las claves como texto.
    """
    missing = {y_col, p_col, by} - set(df_preds.columns)
    if missing:
        raise KeyError(f"Faltan columnas en df_preds: {sorted(missing)}")
    out: dict[str, dict] = {}
    for group, sub in df_preds.groupby(by, sort=True):
        out[str(group)] = binary_metrics(sub[y_col].to_numpy(), sub[p_col].to_numpy(), thr)
    return out
