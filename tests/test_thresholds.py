"""Pruebas de cupo.thresholds: búsqueda de (t_L, t_H) por valor de negocio."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from conftest import synthetic_scores
from cupo import metrics, thresholds
from cupo import value_model as vm


def _reference_search(y, p, cfg, step):
    """Búsqueda en bucle simple sobre la misma grilla, con three_state_rates."""
    t_L_grid = np.round(np.linspace(0.0, 0.5, int(round(0.5 / step)) + 1), 10)
    t_H_grid = np.round(np.linspace(0.5, 1.0, int(round(0.5 / step)) + 1), 10)
    table = {}
    for tl in t_L_grid:
        for th in t_H_grid:
            rates = metrics.three_state_rates(y, p, float(tl), float(th))
            table[(float(tl), float(th))] = vm.compute_value(rates, cfg)["objective"]
    return table


def test_threshold_grid():
    t_L, t_H = thresholds.threshold_grid(0.01)
    assert t_L[0] == 0.0 and t_L[-1] == 0.5 and t_L.size == 51
    assert t_H[0] == 0.5 and t_H[-1] == 1.0 and t_H.size == 51
    assert np.all(t_L[:, None] <= t_H[None, :])


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
@pytest.mark.parametrize("which", ["good", "weak"])
def test_optimize_valid_and_beats_default(scenario, which, good_scores, weak_scores, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    y, p = good_scores if which == "good" else weak_scores
    best = thresholds.optimize(y, p, cfg)
    assert 0.0 <= best["t_L"] <= best["t_H"] <= 1.0
    assert 0.0 <= best["t_L"] <= 0.5 <= best["t_H"]
    v05 = vm.compute_value(metrics.three_state_rates(y, p, 0.5, 0.5), cfg)["objective"]
    assert best["value"] >= v05 - 1e-9
    # El valor reportado coincide con evaluar las tasas exactas del punto.
    r = metrics.three_state_rates(y, p, best["t_L"], best["t_H"])
    assert best["value"] == pytest.approx(vm.compute_value(r, cfg)["objective"])
    assert best["components"]["objective"] == pytest.approx(best["value"])
    assert set(best) >= {"t_L", "t_H", "value", "components"}
    json.dumps(best)


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_optimize_fast_20k(scenario, good_scores, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    y, p = good_scores
    assert y.size == 20_000
    t0 = time.perf_counter()
    thresholds.optimize(y, p, cfg, grid_step=0.01)
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
    best_ref = max(table.values())
    best = thresholds.optimize(y, p, cfg, grid_step=step)
    assert best["value"] == pytest.approx(best_ref, rel=1e-9, abs=1e-6)
    assert table[(best["t_L"], best["t_H"])] == pytest.approx(best_ref, rel=1e-9, abs=1e-6)


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
    # Con t_L == t_H la franja está vacía (salvo ruido de punto flotante de 1 − a − b).
    np.testing.assert_allclose(single["verify_rate"], 0.0, atol=1e-12)
    v05 = vm.compute_value(metrics.three_state_rates(y, p, 0.5, 0.5), cfg)["objective"]
    assert row["value"] == pytest.approx(v05)
