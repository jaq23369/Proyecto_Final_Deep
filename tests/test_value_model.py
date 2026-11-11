"""Pruebas del modelo de valor (cupo.value_model) con casos calculados a mano."""

from __future__ import annotations

import copy
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import make_rates
from cupo import metrics, thresholds
from cupo import value_model as vm

REPO = Path(__file__).resolve().parents[1]
CONFIDENCE_LEVELS = {"medido", "literatura", "estimado"}


def binomial_expected_inv(q: float, F: int) -> float:
    """E[1/(K + 1)] con K ~ Bin(F, q), por suma directa de la distribución."""
    return sum(math.comb(F, k) * q**k * (1 - q) ** (F - k) / (k + 1) for k in range(F + 1))


def binomial_travel_fraction(q: float, F: int) -> float:
    """E[1/(K + 1) | K >= 1] por suma directa (solo para q no despreciable)."""
    p_pos = 1 - (1 - q) ** F
    return sum(math.comb(F, k) * q**k * (1 - q) ** (F - k) / (k + 1) for k in range(1, F + 1)) / p_pos


# ---------------------------------------------------------------------------
# Configuración (YAML)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["comercial", "uvg"])
def test_yaml_params_have_value_source_confidence(name):
    raw = yaml.safe_load((REPO / "configs" / "value" / f"{name}.yaml").read_text(encoding="utf-8"))
    assert raw["name"] == name
    assert raw["params"]
    for key, item in raw["params"].items():
        assert isinstance(item, dict), key
        assert {"value", "source", "confidence"} <= set(item), key
        assert isinstance(item["value"], (int, float)) and not isinstance(item["value"], bool), key
        assert isinstance(item["source"], str) and item["source"].strip(), key
        assert item["confidence"] in CONFIDENCE_LEVELS, key


def test_load_scenarios(com, uvg):
    assert com["objective"] == "valor_operador_mes" and com["periodo"] == "mes"
    assert uvg["objective"] == "valor_neto_semestre_Q" and uvg["periodo"] == "semestre"
    for cfg in (com, uvg):
        assert set(cfg["params"]) == set(cfg["sources"])
        for src in cfg["sources"].values():
            assert src["source"] and src["confidence"] in CONFIDENCE_LEVELS
        # Valor por defecto tomado de configs/costs.yaml.
        assert cfg["params"]["costo_por_1000_pred_Q"] == pytest.approx(0.02)


def test_default_values_match_spec(com, uvg):
    c, u = com["params"], uvg["params"]
    expected_common = {
        "espacios": 164, "horas_operacion_dia": 14, "refresh_s": 30, "t_acceso_min": 0.85,
        "t_recorrido_total_min": 4.0,
        "t_redireccion_min": 1.5, "segundos_por_revision": 10, "revisiones_por_hora_amber": 30,
        "fraccion_tiempo_operador": 0.5, "operadores_incluidos": 1, "salario_operador_Q_h": 25,
        "n_camaras": 9, "costo_camara_Q": 1500, "costo_servidor_mes_Q": 400, "meses_amortizacion": 36,
        "costo_sensor_instalado_Q": 770, "mantenimiento_sensores_mes_Q": 300,
    }
    for k, v in expected_common.items():
        assert c[k] == pytest.approx(v), k
        assert u[k] == pytest.approx(v), k
    assert "t_directo_min" not in c and "t_directo_min" not in u
    assert (c["ocupacion"], c["llegadas_dia"], c["dias_mes"], c["t_base_min"]) == (0.85, 1100, 30, 6.0)
    assert (c["abandono_base"], c["ingreso_por_visita_Q"], c["valor_tiempo_Q_h"]) == (0.08, 25, 25)
    assert (u["ocupacion"], u["llegadas_dia"], u["dias_semestre"], u["meses_semestre"]) == (0.90, 900, 90, 4.5)
    assert (u["t_base_min"], u["abandono_base"]) == (7.0, 0.0)
    assert (u["valor_tiempo_estudiante_Q_h"], u["valor_tiempo_personal_Q_h"], u["frac_personal"]) == (15, 45, 0.2)


def test_costs_yaml():
    costs = vm.load_costs()
    assert costs["gtq_per_usd"] == pytest.approx(7.7)
    assert costs["gpu_usd_per_hour"] == pytest.approx(0.35)
    assert costs["cpu_server_usd_per_hour"] == pytest.approx(0.05)
    assert costs["costo_por_1000_pred_Q"] == pytest.approx(0.02)
    assert vm.default_cost_per_1000() == pytest.approx(0.02)


def test_unknown_scenario():
    with pytest.raises(FileNotFoundError):
        vm.load_scenario("no_existe")


# ---------------------------------------------------------------------------
# Casos límite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_perfect_classifier(scenario, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    prm = cfg["params"]
    r = vm.compute_value(make_rates(0.0, 0.0, 1.0, 0.0), cfg)
    assert r["e"] == 0.0 and r["P0"] == 0.0
    # Se ven todos los libres: el más cercano está a L/(F + 1) del inicio.
    F = r["F"]
    t_perfect = prm["t_acceso_min"] + prm["t_recorrido_total_min"] / (F + 1)
    assert r["frac_recorrido"] == pytest.approx(1 / (F + 1))
    assert r["factor_recorrido"] == pytest.approx(1.0)
    assert r["t_directo_eff"] == pytest.approx(t_perfect)
    assert r["T_sys"] == pytest.approx(t_perfect, abs=1e-12)
    expected = {"comercial": 0.85 + 4 / 26, "uvg": 0.85 + 4 / 17}[scenario]
    assert r["T_sys"] == pytest.approx(expected, abs=1e-12)
    assert r["T_sys"] == pytest.approx({"comercial": 1.003846, "uvg": 1.085294}[scenario], abs=1e-6)
    assert r["A"] == pytest.approx(prm["t_base_min"] - t_perfect, abs=1e-12)
    assert r["t_sys_capped"] is False
    assert r["n_amber"] == 0.0 and r["costo_verif_" + cfg["periodo"]] == 0.0


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_never_shows_free(scenario, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    prm = cfg["params"]
    r = vm.compute_value(make_rates(0.0, 0.0, 0.0, 0.0), cfg)
    assert r["P0"] == 1.0
    assert r["e"] == 1.0  # denominador 0 → e = 1
    assert r["T_sys"] == pytest.approx(prm["t_base_min"])
    assert r["A"] == pytest.approx(0.0)
    sfx = cfg["periodo"]
    assert r[f"horas_ahorradas_{sfx}"] == pytest.approx(0.0)
    # Sin ahorro, el objetivo es solo menos los costos del sistema.
    assert r["objective"] == pytest.approx(-r[f"costos_{sfx}"])


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_status_quo_is_zero(scenario, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    r = vm.status_quo(cfg)
    sfx = cfg["periodo"]
    assert r["objective"] == 0.0
    assert r["T_sys"] == cfg["params"]["t_base_min"] and r["A"] == 0.0
    assert r[f"costos_{sfx}"] == 0.0 and r["P0"] == 1.0
    assert r[f"valor_operador_{sfx}"] == 0.0 and r[f"valor_neto_{sfx}_Q"] == 0.0
    json.dumps(r)


# ---------------------------------------------------------------------------
# Casos a mano
# ---------------------------------------------------------------------------

def test_commercial_hand_case(com):
    # π = 0.85, N = 164 → F = round(0.15·164 = 24.6) = 25
    r = vm.compute_value(make_rates(0.02, 0.0, 0.97, 0.0), com)
    e =0.85 * 0.02 / (0.85 * 0.02 + 0.15 * 0.97)  # 0.017 / 0.1625
    P0 = 0.03 ** 25
    frac = binomial_travel_fraction(0.97, 25)  # ≈ 1/(26·0.97) = 0.039651
    t_direct = min(0.85 + 4.0 * frac, 6.0)  # 1.008604
    T = (1 - P0) * (t_direct + e / (1 - e) * 1.5) + P0 * 6.0
    dab = 0.08 - 0.08 * T / 6.0
    income = 1100 * 30 * dab * 25
    infra = 9 * 1500 / 36 + 400  # 775
    pred = 164 * (3600 / 30) * 14 * 30  # 8 265 600
    inf = pred / 1000 * 0.02  # 165.312
    assert r["F"] == 25
    assert r["e"] == pytest.approx(e, abs=1e-12)
    assert r["e"] == pytest.approx(0.1046154, abs=1e-6)
    assert r["T_sys"] == pytest.approx(T)
    assert r["frac_recorrido"] == pytest.approx(frac, rel=1e-12)
    assert r["frac_recorrido"] == pytest.approx(0.0396511, abs=1e-7)
    assert r["t_directo_eff"] == pytest.approx(t_direct)
    assert r["T_sys"] == pytest.approx(1.183862, abs=1e-6)
    assert r["A"] == pytest.approx(4.816138, abs=1e-6)
    assert r["A"] == pytest.approx(6.0 - T)
    assert r["costo_infra_mes"] == pytest.approx(775.0)
    assert r["pred_mes"] == pytest.approx(8_265_600)
    assert r["costo_inf_mes"] == pytest.approx(165.312)
    assert r["costo_verif_mes"] == 0.0
    assert r["ingreso_recuperado_mes"] == pytest.approx(income)
    assert r["valor_operador_mes"] == pytest.approx(income - infra - inf)
    assert r["valor_operador_mes"] == pytest.approx(52037.21, abs=0.01)
    assert r["objective"] == r["valor_operador_mes"]
    assert r["t_sys_capped"] is False
    json.dumps(r)


def test_uvg_hand_case_with_verify_band(uvg):
    r = vm.compute_value(make_rates(0.02, 0.03, 0.95, 0.02), uvg)
    vt = 0.8 * 15 + 0.2 * 45  # 21 Q/h
    F = 16  # round(0.1·164 = 16.4)
    q_eff = 0.97
    e = 0.9 * 0.02 / (0.9 * 0.02 + 0.1 * q_eff)
    P0 = (1 - q_eff) ** F
    frac = binomial_travel_fraction(q_eff, F)  # ≈ 1/(17·0.97) = 0.060643
    t_direct = min(0.85 + 4.0 * frac, 7.0)
    T = (1 - P0) * (t_direct + e / (1 - e) * 1.5) + P0 * 7.0
    hours = 900 * 90 * (7.0 - T) / 60
    n_amber = 164 * (0.9 * 0.03 + 0.1 * 0.02)  # 4.756
    extra = max(0.0, n_amber * 30 * 10 / 3600 - 0.5)  # 0 (0.396 < 0.5)
    costs = (9 * 1500 / 36 + 400) * 4.5 + 164 * 120 * 14 * 90 / 1000 * 0.02 + extra * 25 * 14 * 90
    assert r["F"] == F
    assert r["valor_tiempo_Q_h"] == pytest.approx(21.0)
    assert r["n_amber"] == pytest.approx(4.756)
    assert r["operadores_extra"] == 0.0
    assert r["meses_periodo"] == 4.5
    assert r["costo_infra_semestre"] == pytest.approx(775.0 * 4.5)
    assert r["horas_ahorradas_semestre"] == pytest.approx(hours)
    assert r["valor_neto_semestre_Q"] == pytest.approx(hours * vt - costs)
    assert r["valor_neto_semestre_Q"] == pytest.approx(155600.93, abs=0.01)
    assert r["T_sys"] == pytest.approx(1.370922, abs=1e-6)
    assert r["objective"] == r["valor_neto_semestre_Q"]
    assert r["minutos_ahorrados_por_llegada"] == pytest.approx(7.0 - T)


def test_q_LL_eff_includes_verified_free(com):
    r = vm.compute_value(make_rates(0.02, 0.05, 0.90, 0.07), com)
    assert r["q_LL_eff"] == pytest.approx(0.97)
    # Mismo e y P0 que un clasificador con q_LL = 0.97 sin franja (q_OL no cambia).
    ref = vm.compute_value(make_rates(0.02, 0.0, 0.97, 0.0), com)
    assert r["e"] == pytest.approx(ref["e"])
    assert r["P0"] == pytest.approx(ref["P0"])
    assert r["T_sys"] == pytest.approx(ref["T_sys"])
    # Y distinto de ignorar la franja (q_LL solo).
    e_without = 0.85 * 0.02 / (0.85 * 0.02 + 0.15 * 0.90)
    assert r["e"] != pytest.approx(e_without)


def test_t_sys_cap_active(com):
    r = vm.compute_value(make_rates(0.5, 0.0, 0.3, 0.0), com)
    e_c = min(0.85 * 0.5 / (0.85 * 0.5 + 0.15 * 0.3), 0.95)
    uncapped = 1.0 + e_c / (1 - e_c) * 1.5  # P0 = 0.7^25 ≈ 1.3e-4
    assert uncapped > 7.5
    assert r["t_sys_capped"] is True
    assert r["T_sys"] == pytest.approx(7.5)  # t_base + t_redireccion
    assert r["A"] == pytest.approx(-1.5)


def test_e_cap(com):
    r = vm.compute_value(make_rates(0.9, 0.0, 0.01, 0.0), com)
    assert r["e"] > 0.95
    assert r["e_c"] == pytest.approx(0.95)


def test_verification_cost_hand_case(com):
    r = vm.compute_value(make_rates(0.0, 0.20, 0.50, 0.30), com)
    n_amber = 164 * (0.85 * 0.20 + 0.15 * 0.30)  # 35.26
    load = n_amber * 30 * 10 / 3600  # 2.938333
    extra = load - 1 * 0.5  # 2.438333
    assert r["n_amber"] == pytest.approx(35.26)
    assert r["carga_h"] == pytest.approx(load)
    assert r["disponible_h"] == pytest.approx(0.5)
    assert r["operadores_extra"] == pytest.approx(extra)
    assert r["costo_verif_mes"] == pytest.approx(extra * 25 * 14 * 30)
    assert r["costo_verif_mes"] == pytest.approx(25602.50, abs=0.005)
    assert r["costos_mes"] == pytest.approx(r["costo_verif_mes"] + 775.0 + 165.312)


def test_verification_cost_zero_below_capacity(com):
    r = vm.compute_value(make_rates(0.02, 0.01, 0.95, 0.02), com)
    # n_amber = 164·(0.85·0.01 + 0.15·0.02) = 1.886 → carga 0.157 h < 0.5 h
    assert r["n_amber"] == pytest.approx(1.886)
    assert r["carga_h"] < r["disponible_h"]
    assert r["operadores_extra"] == 0.0
    assert r["costo_verif_mes"] == 0.0


def test_verification_load_function(com):
    prm = com["params"]
    out = vm.verification_load(np.array([0.0, 6.0, 12.0]), prm)
    # 6 espacios · 30 · 10 / 3600 = 0.5 h → justo la capacidad
    np.testing.assert_allclose(out["carga_h"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(out["operadores_extra"], [0.0, 0.0, 0.5])


def test_months_fixed_in_commercial(com):
    base = vm.compute_value(make_rates(0.02, 0.0, 0.97, 0.0), com)
    for days in (15, 31):
        cfg = vm.with_param(com, "dias_mes", days)
        r = vm.compute_value(make_rates(0.02, 0.0, 0.97, 0.0), cfg)
        assert r["meses_periodo"] == 1.0
        assert r["costo_infra_mes"] == pytest.approx(775.0)
        assert r["costo_sensores_mes"] == pytest.approx(base["costo_sensores_mes"])
        assert r["llegadas_mes"] == pytest.approx(1100 * days)
        assert r["pred_mes"] == pytest.approx(164 * 120 * 14 * days)


def test_sensor_comparison(com):
    r = vm.compute_value(make_rates(0.02, 0.0, 0.97, 0.0), com)
    sensors = 164 * 770 / 36 + 300
    assert r["costo_sensores_mes"] == pytest.approx(sensors)
    assert r["ahorro_vs_sensores_mes"] == pytest.approx(sensors - r["costos_mes"])


def test_cost_per_1000_override(com):
    cfg = copy.deepcopy(com)
    cfg["params"]["costo_por_1000_pred_Q"] = 0.05
    r = vm.compute_value(make_rates(0.02, 0.0, 0.97, 0.0), cfg)
    assert r["costo_inf_mes"] == pytest.approx(8265.6 * 0.05)


@pytest.mark.parametrize("missing", vm.RATE_KEYS)
def test_missing_rate_raises(com, missing):
    rates = make_rates(0.02, 0.01, 0.95, 0.02)
    del rates[missing]
    with pytest.raises(ValueError):
        vm.compute_value(rates, com)


def test_compute_value_json_serializable(com, uvg, good_scores):
    y, p = good_scores
    for cfg in (com, uvg):
        r = vm.compute_value(metrics.three_state_rates(y, p, 0.3, 0.7), cfg)
        text = json.dumps(r)
        back = json.loads(text)
        assert back["objective"] == pytest.approx(r["objective"])
        assert isinstance(back["t_sys_capped"], bool)


def test_vectorized_matches_scalar(com):
    q_OL = np.array([0.0, 0.02, 0.1, 0.5])
    q_LL = np.array([1.0, 0.97, 0.8, 0.3])
    arr = vm.compute_value(make_rates(q_OL, 0.0 * q_OL, q_LL, 0.0 * q_LL), com)
    for i in range(q_OL.size):
        s = vm.compute_value(make_rates(float(q_OL[i]), 0.0, float(q_LL[i]), 0.0), com)
        assert arr["objective"][i] == pytest.approx(s["objective"])
        assert arr["T_sys"][i] == pytest.approx(s["T_sys"])


# ---------------------------------------------------------------------------
# Monotonía
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_more_q_OL_lowers_value(scenario, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    values = [vm.compute_value(make_rates(q, 0.0, 0.95, 0.0), cfg)["objective"]
              for q in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)]
    assert all(a > b for a, b in zip(values, values[1:])), values


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_more_t_base_raises_savings(scenario, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    rates = make_rates(0.02, 0.0, 0.95, 0.0)
    sfx = cfg["periodo"]
    prev_A, prev_h = -math.inf, -math.inf
    for t_base in (4.0, 6.0, 8.0, 10.0):
        r = vm.compute_value(rates, vm.with_param(cfg, "t_base_min", t_base))
        assert r["A"] > prev_A and r[f"horas_ahorradas_{sfx}"] > prev_h
        prev_A, prev_h = r["A"], r[f"horas_ahorradas_{sfx}"]


# ---------------------------------------------------------------------------
# Sensibilidad
# ---------------------------------------------------------------------------

def test_sensitivity_clips_occupancy(com, weak_scores):
    y, p = weak_scores
    best = thresholds.optimize(y, p, com)
    df = vm.sensitivity(best["rates"], com, params=["ocupacion", "abandono_base"])
    expected_cols = {"param", "factor", "base_param_value", "param_value", "clipped",
                     "objective", "base_objective", "delta", "t_L", "t_H"}
    assert expected_cols <= set(df.columns)
    assert len(df) == 2 * 4
    occ = df[df.param == "ocupacion"].set_index("factor")
    assert bool(occ.loc[1.2, "clipped"]) and occ.loc[1.2, "param_value"] == 0.97
    assert bool(occ.loc[1.5, "clipped"]) and occ.loc[1.5, "param_value"] == 0.97
    assert not occ.loc[0.8, "clipped"] and occ.loc[0.8, "param_value"] == pytest.approx(0.68)
    assert not df.loc[df.param == "abandono_base", "clipped"].any()
    assert df["base_objective"].unique().size == 1
    assert df["base_objective"].iloc[0] == pytest.approx(best["value"])
    np.testing.assert_allclose(df["delta"], df["objective"] - df["base_objective"])


def test_sensitivity_default_params_and_reoptimize(com):
    from conftest import synthetic_scores

    y, p = synthetic_scores(2000, sep=1.5, noise=1.5, seed=4)
    df = vm.sensitivity(None, com, params=["salario_operador_Q_h"], reoptimize=True, y=y, p=p)
    assert len(df) == 4
    assert (df["t_L"] <= df["t_H"]).all()
    rates = metrics.three_state_rates(y, p, 0.3, 0.7)
    full = vm.sensitivity(rates, com)
    assert set(full["param"]) == {k for k, v in com["params"].items() if v != 0}
    with pytest.raises(ValueError):
        vm.sensitivity(None, com)
    with pytest.raises(ValueError):
        vm.sensitivity(None, com, reoptimize=True)


def test_with_param_does_not_mutate(com):
    before = copy.deepcopy(com)
    cfg = vm.with_param(com, "ocupacion", 2.0)
    assert cfg["params"]["ocupacion"] == 0.97
    assert com == before
    with pytest.raises(KeyError):
        vm.with_param(com, "no_existe", 1.0)


# ---------------------------------------------------------------------------
# Capacidad oculta: recorrido hasta el libre verde más cercano
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("F", [1, 2, 5, 16, 25, 60])
@pytest.mark.parametrize("q", [0.001, 0.01, 0.1, 0.24, 0.5, 0.75, 0.97, 1.0])
def test_expected_inv_and_travel_fraction_match_binomial_sum(q, F):
    assert float(vm.expected_inv_k_plus_1(q, F)) == pytest.approx(binomial_expected_inv(q, F), rel=1e-10)
    # Forma cerrada (1 − (1 − q)^(F+1)) / ((F + 1)·q)
    closed = (1 - (1 - q) ** (F + 1)) / ((F + 1) * q)
    assert float(vm.expected_inv_k_plus_1(q, F)) == pytest.approx(closed, rel=1e-10)
    frac = float(vm.travel_fraction(q, F))
    assert frac == pytest.approx(binomial_travel_fraction(q, F), rel=1e-8)
    # Entre 1/(F + 1) (todos visibles) y 1/2 (un solo verde).
    assert 1 / (F + 1) - 1e-12 <= frac <= 0.5 + 1e-12


def test_travel_fraction_vectorized():
    q = np.array([0.01, 0.3, 0.97, 1.0])
    out = vm.travel_fraction(q, 25)
    assert out.shape == q.shape
    for qi, oi in zip(q, out):
        assert oi == pytest.approx(binomial_travel_fraction(float(qi), 25), rel=1e-8)


def exact_travel_fraction(q: float, F: int) -> float:
    """E[1/(K + 1) | K >= 1] en aritmética racional exacta (referencia sin cancelación)."""
    qf = Fraction(q)
    num = sum(math.comb(F, k) * qf**k * (1 - qf) ** (F - k) / (k + 1) for k in range(1, F + 1))
    return float(num / (1 - (1 - qf) ** F))


@pytest.mark.parametrize("F", [1, 16, 25])
def test_limit_q_to_zero(F):
    # E[1/(K + 1)] → 1 y la fracción condicionada a K >= 1 → 1/2 (un solo verde).
    assert float(vm.expected_inv_k_plus_1(0.0, F)) == 1.0
    assert float(vm.travel_fraction(0.0, F)) == 0.5
    for q in (1e-16, 1e-15, 1e-14):  # 1 − P0 < 1e-12 → límite 1/2
        assert float(vm.expected_inv_k_plus_1(q, F)) == pytest.approx(1.0, abs=1e-12)
        assert float(vm.travel_fraction(q, F)) == 0.5
    # Rango alcanzable con tasas reales (q = k / n_libres, n_libres ~ 1e5): precisión alta.
    for q in (1e-6, 1e-5, 1e-4, 1e-3):
        f = float(vm.travel_fraction(q, F))
        assert f == pytest.approx(exact_travel_fraction(q, F), abs=1e-9)
        assert f == pytest.approx(0.5 - (F - 1) * q / 12, abs=1e-6)  # desarrollo de primer orden


@pytest.mark.xfail(reason="travel_fraction pierde precisión por cancelación en E − P0 "
                          "cuando 1 − P0 apenas supera 1e-12 (q ~ 1e-13): devuelve > 1/2",
                   strict=False)
@pytest.mark.parametrize("F", [16, 25])
def test_travel_fraction_precision_near_cutoff(F):
    for q in (1e-13, 1e-12, 1e-11):
        f = float(vm.travel_fraction(q, F))
        assert f <= 0.5
        assert f == pytest.approx(exact_travel_fraction(q, F), abs=1e-6)


def test_travel_fraction_q_one():
    assert float(vm.travel_fraction(1.0, 25)) == pytest.approx(1 / 26)
    assert float(vm.expected_inv_k_plus_1(1.0, 25)) == pytest.approx(1 / 26)


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_t_sys_strictly_decreasing_in_q_LL(scenario, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    t_base = cfg["params"]["t_base_min"]
    q_LL = np.linspace(0.0, 1.0, 201)
    r = vm.compute_value(make_rates(0.0 * q_LL, 0.0 * q_LL, q_LL, 0.0 * q_LL), cfg)
    T = np.asarray(r["T_sys"])
    assert T[0] == pytest.approx(t_base)  # sin verdes se busca como hoy
    assert np.all(T <= t_base + 1e-12)
    assert np.all(np.diff(T) < 0), np.diff(T).max()
    assert not np.any(r["t_sys_capped"])
    # Escalar en un punto intermedio coincide con el arreglo.
    s = vm.compute_value(make_rates(0.0, 0.0, float(q_LL[37]), 0.0), cfg)
    assert s["T_sys"] == pytest.approx(T[37])


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_degenerate_loses_to_good(scenario, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    degenerate = make_rates(0.0, 0.0, 0.24, 0.0)  # q_LO = 0.76: casi todo marcado ocupado
    good = make_rates(0.02, 0.0, 0.97, 0.0)
    assert degenerate["q_LO"] == pytest.approx(0.76)
    r_deg = vm.compute_value(degenerate, cfg)
    r_good = vm.compute_value(good, cfg)
    assert r_deg["objective"] < r_good["objective"]
    assert r_deg["T_sys"] > r_good["T_sys"]
    assert r_deg["factor_recorrido"] > r_good["factor_recorrido"] >= 1.0


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_t_directo_eff_capped_at_t_base(scenario, request):
    cfg = vm.with_param(request.getfixturevalue("com" if scenario == "comercial" else "uvg"),
                        "t_recorrido_total_min", 100.0)
    r = vm.compute_value(make_rates(0.0, 0.0, 0.05, 0.0), cfg)
    assert r["t_directo_eff"] == pytest.approx(cfg["params"]["t_base_min"])
    assert r["T_sys"] <= cfg["params"]["t_base_min"] + 1e-12


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_more_travel_time_lowers_value(scenario, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    rates = make_rates(0.02, 0.0, 0.9, 0.0)
    vals = [vm.compute_value(rates, vm.with_param(cfg, "t_recorrido_total_min", L))["objective"]
            for L in (1.0, 2.0, 4.0, 8.0)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# JSON estricto
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_status_quo_travel_nan_and_json_safe(scenario, request):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    r = vm.status_quo(cfg)
    assert math.isnan(r["frac_recorrido"]) and math.isnan(r["factor_recorrido"])
    with pytest.raises(ValueError):
        json.dumps(r, allow_nan=False)
    safe = vm.json_safe(r)
    assert safe["frac_recorrido"] is None and safe["factor_recorrido"] is None
    back = json.loads(json.dumps(safe, allow_nan=False))
    assert back["objective"] == 0.0


@pytest.mark.parametrize("scenario", ["comercial", "uvg"])
def test_compute_value_json_safe_strict(scenario, request, good_scores):
    cfg = request.getfixturevalue("com" if scenario == "comercial" else "uvg")
    y, p = good_scores
    for rates in (metrics.three_state_rates(y, p, 0.3, 0.7), make_rates(0.0, 0.0, 1.0, 0.0),
                  make_rates(0.0, 0.0, 0.0, 0.0), make_rates(0.5, 0.0, 0.3, 0.0)):
        r = vm.compute_value(rates, cfg)
        back = json.loads(json.dumps(vm.json_safe(r), allow_nan=False))
        assert back["objective"] == pytest.approx(r["objective"])
    # Arreglos (búsqueda vectorizada) → listas.
    q = np.array([0.0, 0.5, 1.0])
    arr = vm.compute_value(make_rates(0.0 * q, 0.0 * q, q, 0.0 * q), cfg)
    safe = vm.json_safe(arr)
    assert isinstance(safe["T_sys"], list) and len(safe["T_sys"]) == 3
    json.dumps(safe, allow_nan=False)


def test_json_safe_converts_nonfinite():
    out = vm.json_safe({"a": float("nan"), "b": float("inf"), "c": np.array([1.0, np.nan]), "d": "x", "e": 2})
    assert out == {"a": None, "b": None, "c": [1.0, None], "d": "x", "e": 2}
