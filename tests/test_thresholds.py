"""Pruebas de cupo.thresholds: búsqueda de (t_L, t_H) por valor de negocio."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from conftest import synthetic_scores
from cupo import metrics, thresholds
from cupo import value_model as vm


def _unit_grid(step):
    return np.round(np.linspace(0.0, 1.0, int(round(1.0 / step)) + 1), 10)


def _reference_search(y, p, cfg, step):
    """Búsqueda en bucle simple sobre todos los pares 0 <= t_L <= t_H <= 1, con three_state_rates."""
    grid = _unit_grid(step)
    table = {}
    for tl in grid:
        for th in grid:
            if tl > th:
                continue
            rates = metrics.three_state_rates(y, p, float(tl), float(th))
            table[(float(tl), float(th))] = vm.compute_value(rates, cfg)["objective"]
    return table


def _reference_pick(table, rtol=1e-9):
    """Máximo; en empate la franja más angosta, luego el centro más cercano a 0.5 y luego t_L menor."""
    best = max(table.values())
    tol = rtol * max(1.0, abs(best))
    ties = [k for k, v in table.items() if v >= best - tol]

    def key(k):
        return round(k[1] - k[0], 10), round(abs((k[0] + k[1]) / 2 - 0.5), 10), k[0]

    return min(ties, key=key), best


def _best_single_threshold(y, p, cfg, step=0.01):
    return max(vm.compute_value(metrics.three_state_rates(y, p, float(t), float(t)), cfg)["objective"]
               for t in _unit_grid(step))


def test_threshold_grid():
    t_L, t_H = thresholds.threshold_grid(0.01)
    for g in (t_L, t_H):
        assert g.size == 101 and g[0] == 0.0 and g[-1] == 1.0
        np.testing.assert_allclose(np.diff(g), 0.01)
    assert np.isin(0.5, t_L) and np.isin(0.5, t_H)
    valid = t_L[:, None] <= t_H[None, :]
    assert valid.sum() == 101 * 102 // 2  # 5151 pares válidos


def test_invalid_cells_are_minus_inf(com, good_scores):
    y, p = good_scores
    scores = thresholds._ClassScores(y[:2000], p[:2000])
    t_L, t_H = thresholds.threshold_grid(0.05)
    values, _ = thresholds._objective_grid(scores, com, t_L, t_H)
    invalid = t_L[:, None] > t_H[None, :]
    assert np.all(values[invalid] == -np.inf)
    assert np.all(np.isfinite(values[~invalid]))


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
@pytest.mark.parametrize("which", ["good", "weak"])
def test_optimize_valid_and_beats_default(scenario, which, good_scores, weak_scores, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    y, p = good_scores if which == "good" else weak_scores
    best = thresholds.optimize(y, p, cfg)
    assert 0.0 <= best["t_L"] <= best["t_H"] <= 1.0
    assert best["t_L"] in thresholds.threshold_grid(0.01)[0]
    v05 = vm.compute_value(metrics.three_state_rates(y, p, 0.5, 0.5), cfg)["objective"]
    assert best["value"] >= v05 - 1e-9
    # El valor reportado coincide con evaluar las tasas exactas del punto.
    r = metrics.three_state_rates(y, p, best["t_L"], best["t_H"])
    assert best["value"] == pytest.approx(vm.compute_value(r, cfg)["objective"])
    assert best["components"]["objective"] == pytest.approx(best["value"])
    assert set(best) >= {"t_L", "t_H", "value", "components"}
    json.dumps(best)


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
@pytest.mark.parametrize("which", ["good", "weak"])
def test_optimize_fast_20k(scenario, which, good_scores, weak_scores, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    y, p = good_scores if which == "good" else weak_scores
    assert y.size == 20_000
    t0 = time.perf_counter()
    thresholds.optimize(y, p, cfg, grid_step=0.01)  # grilla completa: 5151 pares
    assert time.perf_counter() - t0 < 2.0


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
@pytest.mark.parametrize("rounded", [False, True])
def test_optimize_matches_loop_reference(scenario, rounded, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    y, p = synthetic_scores(3000, sep=1.0, noise=1.5, seed=11)
    if rounded:
        # Empates exactos con los umbrales de la grilla (múltiplos de 0.05).
        p = np.round(p, 2)
    step = 0.05
    table = _reference_search(y, p, cfg, step)
    assert len(table) == 21 * 22 // 2
    (ref_L, ref_H), best_ref = _reference_pick(table)
    best = thresholds.optimize(y, p, cfg, grid_step=step)
    assert 0.0 <= best["t_L"] <= best["t_H"] <= 1.0
    assert best["value"] == pytest.approx(best_ref, rel=1e-9, abs=1e-6)
    assert table[(best["t_L"], best["t_H"])] == pytest.approx(best_ref, rel=1e-9, abs=1e-6)
    # Mismo desempate: franja más angosta y luego centro más cercano a 0.5.
    assert (best["t_L"], best["t_H"]) == (ref_L, ref_H)


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
@pytest.mark.parametrize("which", ["good", "weak"])
def test_optimize_beats_best_single_threshold(scenario, which, good_scores, weak_scores, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    y, p = good_scores if which == "good" else weak_scores
    y, p = y[:5000], p[:5000]
    best = thresholds.optimize(y, p, cfg)
    single = _best_single_threshold(y, p, cfg)
    assert best["value"] >= single - 1e-9 * max(1.0, abs(single))
    curve = vm.value_curve(y, p, cfg)["single_threshold"]
    assert curve["value"].max() == pytest.approx(single, rel=1e-9)


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_optimum_with_t_H_below_half(scenario, request):
    # Modelo mal calibrado hacia abajo: ocupados con p en [0.25, 0.40), libres en [0, 0.15).
    # Separa perfecto con un umbral único en [0.15, 0.25]; cualquier t_H >= 0.5 deja
    # todos los ocupados en verificar o mostrados libres.
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    rng = np.random.default_rng(21)
    y = rng.binomial(1, 0.8, 4000)
    p = np.where(y == 1, rng.uniform(0.25, 0.40, y.size), rng.uniform(0.0, 0.15, y.size))
    best = thresholds.optimize(y, p, cfg)
    assert best["t_H"] < 0.5
    assert best["t_L"] <= best["t_H"]
    # Desempate: franja de ancho 0 y, entre los umbrales que separan perfecto
    # (0.15 a 0.25), el más cercano a 0.5.
    assert (best["t_L"], best["t_H"]) == (0.25, 0.25)
    r = best["rates"]
    assert r["q_OL"] == 0.0 and r["q_LL"] == 1.0 and r["verify_rate"] == 0.0
    perfect = vm.compute_value({"q_OL": 0.0, "q_OV": 0.0, "q_OO": 1.0,
                                "q_LL": 1.0, "q_LV": 0.0, "q_LO": 0.0}, cfg)["objective"]
    assert best["value"] == pytest.approx(perfect)
    # Con la grilla anterior (t_L <= 0.5 <= t_H) no se llegaba a este valor.
    v_half = vm.compute_value(metrics.three_state_rates(y, p, 0.5, 0.5), cfg)["objective"]
    assert best["value"] > v_half


def test_vectorized_rates_match_three_state_rates():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.6, 1000)
    p = np.round(rng.random(1000), 2)
    scores = thresholds._ClassScores(y, p)
    for tl, th in [(0.3, 0.7), (0.5, 0.5), (0.0, 1.0), (0.41, 0.59), (0.05, 0.95)]:
        g = scores.rates(np.array(tl), np.array(th))
        d = metrics.three_state_rates(y, p, tl, th)
        for k in (*vm.RATE_KEYS, "verify_rate"):
            assert float(g[k]) == pytest.approx(d[k], abs=1e-12), (tl, th, k)


def test_optimize_requires_both_classes(com):
    with pytest.raises(ValueError):
        thresholds.optimize([1, 1, 1], [0.2, 0.5, 0.9], com)


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_value_curve_consistent_with_optimize(scenario, good_scores, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    y, p = good_scores
    best = thresholds.optimize(y, p, cfg)
    curves = vm.value_curve(y, p, cfg)
    by_t_H = curves["by_t_H"]
    assert by_t_H["value"].max() == pytest.approx(best["value"], rel=1e-9)
    assert (by_t_H["t_L_opt"] <= by_t_H["t_H"]).all()

    single = curves["single_threshold"]
    row = single.loc[np.isclose(single["t"], 0.5)].iloc[0]
    bm = metrics.binary_metrics(y, p, 0.5)
    assert row["accuracy"] == pytest.approx(bm["accuracy"])
    assert row["f1_macro"] == pytest.approx(bm["f1_macro"])
    assert row["rate_busy_as_free"] == pytest.approx(bm["rate_busy_as_free"])
    # Con t_L == t_H la franja está vacía: verify_rate exactamente 0.
    assert (single["verify_rate"] == 0.0).all(), single["verify_rate"].abs().max()
    v05 = vm.compute_value(metrics.three_state_rates(y, p, 0.5, 0.5), cfg)["objective"]
    assert row["value"] == pytest.approx(v05)


def test_grid_verify_rates_in_unit_interval(good_scores, weak_scores):
    for y, p in (good_scores, weak_scores):
        scores = thresholds._ClassScores(y, p)
        t_L, t_H = thresholds.threshold_grid(0.01)
        r = scores.rates(t_L[:, None], t_H[None, :])
        for k in ("q_OV", "q_LV", "verify_rate"):
            assert np.all(r[k] >= 0.0) and np.all(r[k] <= 1.0), k
        # Diagonal t_L == t_H: franja vacía exacta.
        diag = np.arange(t_L.size)
        assert np.all(r["q_OV"][diag, diag] == 0.0) and np.all(r["q_LV"][diag, diag] == 0.0)
        t = np.round(np.linspace(0.0, 1.0, 101), 10)
        same = scores.rates(t, t)
        assert np.all(same["q_OV"] == 0.0) and np.all(same["q_LV"] == 0.0), (
            same["q_OV"].max(), same["q_LV"].max())
        assert np.all(same["verify_rate"] == 0.0)


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_saturated_model_picks_center(scenario, request):
    # Libres con p en [0, 0.03] y ocupados en [0.97, 1]: todo umbral único en [0.03, 0.97]
    # separa perfecto; el desempate elige el centro (0.5, 0.5).
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    rng = np.random.default_rng(33)
    y = rng.binomial(1, 0.8, 5000)
    p = np.where(y == 1, rng.uniform(0.97, 1.0, y.size), rng.uniform(0.0, 0.03, y.size))
    best = thresholds.optimize(y, p, cfg)
    assert (best["t_L"], best["t_H"]) == (0.5, 0.5)
    r = best["rates"]
    assert r["q_OL"] == 0.0 and r["q_LL"] == 1.0 and r["verify_rate"] == 0.0


def test_pick_best_tie_break_order():
    t = np.round(np.linspace(0.0, 1.0, 11), 10)
    values = np.full((t.size, t.size), -np.inf)
    valid = t[:, None] <= t[None, :]
    values[valid] = 1.0
    # Empate total: ancho 0 y centro 0.5.
    i, j = thresholds._pick_best(values, t, t)
    assert (t[i], t[j]) == (0.5, 0.5)
    # Solo empatan franjas de ancho 0.2: gana la centrada en 0.5 (0.4, 0.6).
    values = np.where(valid & np.isclose(t[None, :] - t[:, None], 0.2), 1.0, 0.0)
    values[~valid] = -np.inf
    i, j = thresholds._pick_best(values, t, t)
    assert (t[i], t[j]) == (0.4, 0.6)
    # Un máximo estricto gana aunque su franja sea ancha y descentrada.
    values[0, 10] = 2.0
    i, j = thresholds._pick_best(values, t, t)
    assert (t[i], t[j]) == (0.0, 1.0)
