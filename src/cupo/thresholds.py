"""Búsqueda de los umbrales (t_L, t_H) que maximizan el valor de negocio.

La búsqueda es vectorizada: se ordenan las probabilidades de cada clase una
sola vez y, con ``np.searchsorted``, se obtienen las tasas de los tres estados
para toda la grilla. Luego ``compute_value`` evalúa la grilla completa por
difusión de arreglos (``t_L`` en el eje 0 y ``t_H`` en el eje 1).

La grilla cubre todos los pares ``0 ≤ t_L ≤ t_H ≤ 1`` con paso ``grid_step``
(5,151 pares con paso 0.01). Así se evalúan también los umbrales únicos
(``t_L = t_H = t``) lejos de 0.5 y las franjas de verificar que no contienen
0.5. Las celdas con ``t_L > t_H`` se descartan (valor −inf).

Regla de decisión (igual que ``metrics.three_state_rates``): ``p < t_L`` →
LIBRE, ``p >= t_H`` → OCUPADO, en otro caso VERIFICAR. Con ``t_L == t_H``
VERIFICAR queda vacío y coincide con ``metrics.binary_metrics``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cupo.metrics import three_state_rates
from cupo.value_model import compute_value

# Tolerancia relativa para considerar empates en el valor de la grilla.
_TIE_RTOL = 1e-9


def _grid(lo: float, hi: float, step: float) -> np.ndarray:
    """Grilla cerrada [lo, hi] con paso ``step`` (incluye ambos extremos exactos)."""
    n = int(round((hi - lo) / step)) + 1
    return np.round(np.linspace(lo, hi, n), 10)


def threshold_grid(grid_step: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Grillas de t_L y t_H, ambas en [0, 1]; los pares válidos cumplen t_L ≤ t_H."""
    grid = _grid(0.0, 1.0, grid_step)
    return grid, grid.copy()


class _ClassScores:
    """Probabilidades ordenadas de cada clase para contar rápido por umbral."""

    def __init__(self, y: Any, p: Any) -> None:
        y_arr = np.asarray(y).astype(int).ravel()
        p_arr = np.asarray(p, dtype=float).ravel()
        if y_arr.shape != p_arr.shape:
            raise ValueError("y y p deben tener el mismo tamaño")
        self.busy = np.sort(p_arr[y_arr == 1])
        self.free = np.sort(p_arr[y_arr == 0])
        if self.busy.size == 0 or self.free.size == 0:
            raise ValueError("Se necesitan ejemplos de ambas clases para optimizar umbrales")

    @staticmethod
    def below(sorted_p: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Cuántos p < t."""
        return np.searchsorted(sorted_p, t, side="left")

    @staticmethod
    def at_or_above(sorted_p: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Cuántos p >= t."""
        return sorted_p.size - np.searchsorted(sorted_p, t, side="left")

    def rates(self, t_L: np.ndarray, t_H: np.ndarray) -> dict[str, np.ndarray]:
        """Tasas q_* que se difunden entre las formas de ``t_L`` y ``t_H``."""
        n_b, n_f = self.busy.size, self.free.size
        busy_free = self.below(self.busy, t_L)
        free_free = self.below(self.free, t_L)
        busy_busy = self.at_or_above(self.busy, t_H)
        free_busy = self.at_or_above(self.free, t_H)
        # Conteos enteros: con t_L == t_H la franja da exactamente 0, sin residuos de redondeo.
        busy_verify = np.maximum(n_b - busy_free - busy_busy, 0)
        free_verify = np.maximum(n_f - free_free - free_busy, 0)
        return {
            "q_OL": busy_free / n_b, "q_OV": busy_verify / n_b, "q_OO": busy_busy / n_b,
            "q_LL": free_free / n_f, "q_LV": free_verify / n_f, "q_LO": free_busy / n_f,
            "verify_rate": (busy_verify + free_verify) / (n_b + n_f),
        }


def _objective_grid(
    scores: _ClassScores, scenario_cfg: Mapping[str, Any], t_L: np.ndarray, t_H: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Valor del objetivo en toda la grilla, forma (len(t_L), len(t_H)).

    Las celdas con ``t_L > t_H`` no son decisiones válidas y valen −inf.
    """
    rates = scores.rates(t_L[:, None], t_H[None, :])
    values = compute_value(rates, scenario_cfg)["objective"]
    shape = (t_L.size, t_H.size)
    valid = t_L[:, None] <= t_H[None, :]
    full = np.broadcast_to(np.asarray(values, dtype=float), shape)
    return np.where(valid, full, -np.inf), rates


def _pick_best(values: np.ndarray, t_L: np.ndarray, t_H: np.ndarray) -> tuple[int, int]:
    """Índice del máximo con desempate robusto.

    Entre puntos con el mismo valor (tolerancia relativa 1e-9) gana la franja de
    verificar más angosta y, si también empata, el punto cuyo centro
    ``(t_L + t_H)/2`` está más cerca de 0.5. Cuando un rango de umbrales da el
    mismo valor en validación (por ejemplo, un modelo saturado sin
    probabilidades entre 0.03 y 0.97), elegir el centro deja el mayor margen
    frente a desplazamientos de ``p`` en datos nuevos (otro clima, otra cámara),
    en vez de quedar pegado al borde de una de las clases.
    """
    best = np.nanmax(values)
    tol = _TIE_RTOL * max(1.0, abs(best))
    cand_i, cand_j = np.nonzero(values >= best - tol)
    width = np.round(t_H[cand_j] - t_L[cand_i], 10)
    center_dist = np.round(np.abs((t_L[cand_i] + t_H[cand_j]) / 2 - 0.5), 10)
    k = int(np.lexsort((center_dist, width))[0])
    return int(cand_i[k]), int(cand_j[k])


def optimize(y: Any, p: Any, scenario_cfg: Mapping[str, Any], grid_step: float = 0.01) -> dict[str, Any]:
    """Umbrales (t_L, t_H) que maximizan ``scenario_cfg['objective']``.

    Grilla de todos los pares 0 ≤ t_L ≤ t_H ≤ 1 con paso ``grid_step`` (antes se
    limitaba a t_L ≤ 0.5 ≤ t_H y no evaluaba umbrales únicos lejos de 0.5). En
    empate gana la franja de verificar más angosta y luego el centro más cercano
    a 0.5 (ver ``_pick_best``). Devuelve
    ``{t_L, t_H, value, components, rates}``; ``components`` es la salida de
    ``compute_value`` en el punto óptimo.

    Usar solo con predicciones de validación; los umbrales nunca se
    re-optimizan en test ni en OOD.
    """
    scores = _ClassScores(y, p)
    t_L, t_H = threshold_grid(grid_step)
    values, _ = _objective_grid(scores, scenario_cfg, t_L, t_H)
    i, j = _pick_best(values, t_L, t_H)
    best_L, best_H = float(t_L[i]), float(t_H[j])
    rates = three_state_rates(y, p, best_L, best_H)
    components = compute_value(rates, scenario_cfg)
    return {
        "t_L": best_L,
        "t_H": best_H,
        "value": float(components["objective"]),
        "components": components,
        "rates": rates,
    }


def curve_by_t_H(y: Any, p: Any, scenario_cfg: Mapping[str, Any], grid_step: float = 0.01) -> pd.DataFrame:
    """Valor frente a t_H ∈ [0, 1] con el t_L óptimo (t_L ≤ t_H) para cada t_H."""
    scores = _ClassScores(y, p)
    t_L, t_H = threshold_grid(grid_step)
    values, rates = _objective_grid(scores, scenario_cfg, t_L, t_H)
    best_i = np.nanargmax(values, axis=0)
    cols = np.arange(t_H.size)
    full = {k: np.broadcast_to(v, values.shape) for k, v in rates.items()}
    return pd.DataFrame({
        "t_H": t_H,
        "t_L_opt": t_L[best_i],
        "value": values[best_i, cols],
        "q_OL": full["q_OL"][best_i, cols],
        "q_LO": full["q_LO"][best_i, cols],
        "verify_rate": full["verify_rate"][best_i, cols],
    })


def curve_single_threshold(
    y: Any, p: Any, scenario_cfg: Mapping[str, Any], grid_step: float = 0.01
) -> pd.DataFrame:
    """Decisión de dos estados (t_L = t_H = t): valor y métricas técnicas frente a t.

    Con ``t_L = t_H = t`` VERIFICAR queda vacío: ocupado si ``p >= t``, igual
    que ``metrics.binary_metrics``.
    """
    scores = _ClassScores(y, p)
    t = _grid(0.0, 1.0, grid_step)
    rates = scores.rates(t, t)
    values = np.asarray(compute_value(rates, scenario_cfg)["objective"], dtype=float)

    n_b, n_f = scores.busy.size, scores.free.size
    fn = scores.below(scores.busy, t)
    tn = scores.below(scores.free, t)
    tp, fp = n_b - fn, n_f - tn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_busy = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_free = np.where(2 * tn + fp + fn > 0, 2 * tn / (2 * tn + fp + fn), 0.0)
    return pd.DataFrame({
        "t": t,
        "value": np.broadcast_to(values, t.shape),
        "accuracy": (tp + tn) / (n_b + n_f),
        "f1_macro": (f1_busy + f1_free) / 2,
        "rate_busy_as_free": fn / n_b,
        "verify_rate": rates["verify_rate"],
    })
