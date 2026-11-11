"""Modelo de valor de negocio de Cupo Seguro.

Traduce el desempeño del clasificador (tasas de la decisión de tres estados)
a minutos ahorrados por conductor y a quetzales para cada escenario.

Entradas
--------
- ``rates``: salida de ``metrics.three_state_rates``. Son obligatorias las seis
  tasas ``q_OL, q_OV, q_OO`` (condicionadas en ocupado) y ``q_LL, q_LV, q_LO``
  (condicionadas en libre); L = mostrado libre, V = por verificar, O = mostrado
  ocupado.
- ``scenario_cfg``: escenario cargado con ``load_scenario`` (configs/value/*.yaml).
  Cada parámetro del YAML tiene la forma ``{value, source, confidence}``.

Las tasas se reponderan con la ocupación del escenario ``π``; no se usa la
proporción de clases del conjunto de validación.

Franja "por verificar"
----------------------
Los espacios en amarillo los revisa un operador en la cámara. Tras la revisión,
los libres verificados pasan a verde y los ocupados verificados pasan a rojo.
Por eso, para el conductor, la tasa efectiva de libres mostrados en verde es
``q_LL_eff = q_LL + q_LV``, mientras que ``q_OL`` no cambia (la verificación
nunca convierte un ocupado en verde). El precio de ese beneficio es el costo
de revisión de la franja.

Limitaciones conocidas
----------------------
1. ``q_LO`` (libre mostrado ocupado, capacidad oculta) no se penaliza más allá
   de ``P0``. Con ocupación alta (π ≈ 0.85-0.90, F ≈ 16-25), ``P0`` ≈ 0 aunque
   se oculten muchos libres, porque al conductor le basta un verde correcto; el
   error caro es ocupado→libre (``q_OL``, que entra por ``e``).
2. Consecuencia: subir ``t_H`` solo pasa espacios de rojo a amarillo, así que
   por construcción el ``t_H`` óptimo tiende a 0.5 y la franja de verificar
   queda en ``[t_L, 0.5)``; su beneficio viene de bajar ``t_L``.

Paso 1: probabilidades
----------------------
    N, π     = espacios, ocupacion
    F        = max(1, round((1 − π)·N))            espacios realmente libres (redondeo .5 hacia arriba)
    q_LL_eff = q_LL + q_LV                          libres en verde después de verificar
    e        = π·q_OL / (π·q_OL + (1 − π)·q_LL_eff) P(un "libre" mostrado esté ocupado); si el denominador es 0, e = 1
    e_c      = min(e, 0.95)                        tope numérico
    P0       = (1 − q_LL_eff)^F                    ningún libre real se muestra en verde → se busca como hoy

Paso 2: tiempo y ahorro (minutos por llegada)
---------------------------------------------
    T_guiado = (1 − P0)·[t_directo + (e_c/(1 − e_c))·t_redireccion] + P0·t_base
    T_sys    = min(T_guiado, t_base + t_redireccion)
    A        = t_base − T_sys                      ahorro por llegada (puede ser negativo)

El tope de ``T_sys`` refleja que, tras un intento fallido, el conductor deja de
confiar en la app y busca como hoy: lo peor que pasa es perder una redirección
y luego buscar con el método actual. ``t_sys_capped`` indica si el tope se activó.

Paso 3: abandono y volúmenes (por periodo: mes o semestre)
----------------------------------------------------------
    p_ab(t)   = clip(abandono_base · t / t_base, 0, 1)
    Δab       = p_ab(t_base) − p_ab(T_sys)
    llegadas  = llegadas_dia · dias_periodo
    pred      = N · (3600/refresh_s) · horas_operacion_dia · dias_periodo
    n_amber   = N · (π·q_OV + (1 − π)·q_LV)    espacios en amarillo en promedio

Paso 4: costos y valor
----------------------
    carga_h          = n_amber · revisiones_por_hora_amber · segundos_por_revision / 3600   horas-operador necesarias por hora
    disponible_h     = operadores_incluidos · fraccion_tiempo_operador                      capacidad sin costo incremental
    operadores_extra = max(0, carga_h − disponible_h)                                       continuo (tiempo parcial)
    costo_verif  = operadores_extra · salario_operador_Q_h · horas_operacion_dia · dias_periodo
    costo_infra  = [(n_camaras·costo_camara_Q)/meses_amortizacion + costo_servidor_mes_Q] · meses_periodo
    costo_inf    = pred/1000 · costo_por_1000_pred_Q
    costos       = costo_infra + costo_inf + costo_verif
    ingreso_recuperado = llegadas · Δab · ingreso_por_visita_Q
    valor_operador     = ingreso_recuperado − costos
    horas_ahorradas    = llegadas · A / 60
    valor_usuarios_Q   = horas_ahorradas · valor_tiempo_Q_h
    valor_neto_Q       = valor_usuarios_Q − costos
    costo_sensores     = [(N·costo_sensor_instalado_Q)/meses_amortizacion + mantenimiento_sensores_mes_Q] · meses_periodo
    ahorro_vs_sensores = costo_sensores − costos          (comparativa, no entra al objetivo)

El costo de verificación se modela por capacidad del personal. El parqueo ya
tiene un encargado (``operadores_incluidos``) que dedica una fracción de su
tiempo a revisar cámaras sin costo incremental. Con un modelo bueno pocos
espacios quedan en amarillo y ese encargado los atiende sin costo extra; con un
modelo débil la carga supera su capacidad y hay que contratar personal
(``operadores_extra``, a tiempo parcial si es fraccionario). En el escenario
comercial esto impide que un modelo débil "compre" precisión barata mandando
casi todo a verificar. En UVG, donde el minuto ahorrado vale más, un modelo
débil sí puede justificar contratar personal; por eso la tabla de resultados
reporta también el valor con 2 estados y ``operadores_extra``.

``costo_por_1000_pred_Q`` tiene su valor por defecto en ``configs/costs.yaml``;
un escenario puede sobrescribirlo de forma opcional.

Escenarios
----------
- Comercial (periodo = mes, ``meses_periodo = 1`` fijo, aunque varíe ``dias_mes``):
  objetivo ``valor_operador_mes``.
- UVG (periodo = semestre, ``dias_semestre`` en lugar de ``dias_mes``, sin abandono):
  ``valor_tiempo_Q_h = (1 − frac_personal)·valor_tiempo_estudiante + frac_personal·valor_tiempo_personal``;
  objetivo ``valor_neto_semestre_Q = horas_ahorradas_semestre·valor_tiempo − costos_semestre``.
  ``meses_periodo = meses_semestre`` (meses calendario en que corre la infraestructura).
- Status quo (B0, sin sistema): ``T_sys = t_base``, ``A = 0``, sin costos → valor 0.

Los nombres de los componentes llevan el sufijo del periodo, por ejemplo
``costo_infra_mes`` o ``horas_ahorradas_semestre``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Raíz del repositorio: src/cupo/../../
REPO_ROOT = Path(__file__).resolve().parents[2]
VALUE_DIR = REPO_ROOT / "configs" / "value"
COSTS_PATH = REPO_ROOT / "configs" / "costs.yaml"

E_CAP = 0.95  # tope numérico de e para que e/(1−e) no explote
MONTH_DAYS = 30.0  # días por mes cuando el YAML no define los meses del periodo
COST_PER_1000_KEY = "costo_por_1000_pred_Q"

RATE_KEYS = ("q_OL", "q_OV", "q_OO", "q_LL", "q_LV", "q_LO")

# Parámetros que son probabilidades: el análisis de sensibilidad los recorta a estos rangos.
_PROB_BOUNDS: dict[str, tuple[float, float]] = {
    "ocupacion": (0.0, 0.97),
    "abandono_base": (0.0, 1.0),
    "frac_personal": (0.0, 1.0),
    "fraccion_tiempo_operador": (0.0, 1.0),
}

_REQUIRED_PARAMS = (
    "espacios", "ocupacion", "llegadas_dia", "horas_operacion_dia", "refresh_s",
    "t_base_min", "t_directo_min", "t_redireccion_min",
    "n_camaras", "costo_camara_Q", "costo_servidor_mes_Q", "meses_amortizacion",
    "costo_sensor_instalado_Q", "mantenimiento_sensores_mes_Q",
    "segundos_por_revision", "revisiones_por_hora_amber", "fraccion_tiempo_operador",
    "operadores_incluidos", "salario_operador_Q_h",
)


# ---------------------------------------------------------------------------
# Carga de configuración
# ---------------------------------------------------------------------------

def flatten_params(raw: Mapping[str, Any]) -> dict[str, float]:
    """Aplana ``{nombre: {value, source, confidence}}`` a ``{nombre: value}``.

    Acepta también parámetros que ya son números (se devuelven tal cual).
    """
    flat: dict[str, float] = {}
    for key, item in raw.items():
        flat[key] = item["value"] if isinstance(item, Mapping) else item
    return flat


def load_costs(path: str | Path = COSTS_PATH) -> dict[str, float]:
    """Carga ``configs/costs.yaml`` (tipo de cambio, cómputo y costo por predicción)."""
    with Path(path).open(encoding="utf-8") as fh:
        return flatten_params(yaml.safe_load(fh))


def default_cost_per_1000() -> float:
    """Costo por 1000 predicciones por defecto (``configs/costs.yaml``)."""
    return float(load_costs()[COST_PER_1000_KEY])


def _resolve_scenario_path(name: str | Path) -> Path:
    """Acepta el nombre del escenario ('comercial', 'uvg') o una ruta a un YAML."""
    candidate = Path(name)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    path = VALUE_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No existe el escenario '{name}' en {VALUE_DIR}")
    return path


def _param_sources(raw_params: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Fuente y nivel de confianza de cada parámetro del YAML."""
    return {
        k: {"source": v.get("source", ""), "confidence": v.get("confidence", "")}
        for k, v in raw_params.items()
        if isinstance(v, Mapping)
    }


def load_scenario(name: str | Path) -> dict[str, Any]:
    """Carga un escenario de ``configs/value/<name>.yaml``.

    Devuelve un dict con ``name``, ``description``, ``objective``, ``periodo``,
    ``params`` (valores aplanados), ``sources`` (fuente y confianza de cada
    parámetro) y ``path``. Si el escenario no define ``costo_por_1000_pred_Q``
    se toma el valor por defecto de ``configs/costs.yaml``.
    """
    path = _resolve_scenario_path(name)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for key in ("name", "objective", "periodo", "params"):
        if key not in raw:
            raise KeyError(f"El escenario {path} no define '{key}'")
    params = flatten_params(raw["params"])
    sources = _param_sources(raw["params"])
    if COST_PER_1000_KEY not in params:
        params[COST_PER_1000_KEY] = default_cost_per_1000()
        sources[COST_PER_1000_KEY] = {
            "source": "Valor por defecto de configs/costs.yaml; se reemplaza con el resultado de benchmark",
            "confidence": "estimado",
        }
    cfg = {
        "name": raw["name"],
        "description": raw.get("description", ""),
        "objective": raw["objective"],
        "periodo": raw["periodo"],
        "params": params,
        "sources": sources,
        "path": str(path),
    }
    _validate_params(cfg)
    return cfg


def scenario_params(scenario_cfg: Mapping[str, Any]) -> dict[str, float]:
    """Parámetros numéricos del escenario, aplanados."""
    return flatten_params(scenario_cfg["params"])


def _validate_params(scenario_cfg: Mapping[str, Any]) -> None:
    """Verifica que el escenario tenga todos los parámetros necesarios."""
    prm = scenario_params(scenario_cfg)
    sfx = scenario_cfg["periodo"]
    missing = [k for k in (*_REQUIRED_PARAMS, f"dias_{sfx}") if k not in prm]
    if "valor_tiempo_Q_h" not in prm and not _has_mixed_time_value(prm):
        missing.append("valor_tiempo_Q_h")
    if missing:
        raise KeyError(f"Faltan parámetros en el escenario '{scenario_cfg['name']}': {missing}")


# ---------------------------------------------------------------------------
# Piezas del cálculo (aceptan escalares o arreglos de numpy que se difunden)
# ---------------------------------------------------------------------------

def _has_mixed_time_value(prm: Mapping[str, float]) -> bool:
    keys = ("valor_tiempo_estudiante_Q_h", "valor_tiempo_personal_Q_h", "frac_personal")
    return all(k in prm for k in keys)


def time_value_q_h(prm: Mapping[str, float]) -> float:
    """Valor del tiempo en Q/h; en UVG es la mezcla estudiante/personal."""
    if _has_mixed_time_value(prm):
        frac = prm["frac_personal"]
        return (1 - frac) * prm["valor_tiempo_estudiante_Q_h"] + frac * prm["valor_tiempo_personal_Q_h"]
    return prm["valor_tiempo_Q_h"]


def verification_load(n_amber: Any, prm: Mapping[str, float]) -> dict[str, Any]:
    """Carga de verificación frente a la capacidad del personal (por hora de operación).

    carga_h          = n_amber · revisiones_por_hora_amber · segundos_por_revision / 3600
    disponible_h     = operadores_incluidos · fraccion_tiempo_operador
    operadores_extra = max(0, carga_h − disponible_h)
    """
    load = n_amber * prm["revisiones_por_hora_amber"] * prm["segundos_por_revision"] / 3600.0
    available = prm["operadores_incluidos"] * prm["fraccion_tiempo_operador"]
    return {
        "carga_h": load,
        "disponible_h": available,
        "operadores_extra": np.maximum(0.0, load - available),
    }


def _cost_per_1000(prm: Mapping[str, float]) -> float:
    """Costo por 1000 predicciones: el del escenario o el de costs.yaml."""
    if COST_PER_1000_KEY in prm:
        return float(prm[COST_PER_1000_KEY])
    return default_cost_per_1000()


def _period(scenario_cfg: Mapping[str, Any], prm: Mapping[str, float]) -> tuple[str, float, float]:
    """Sufijo del periodo, días de operación y meses calendario del periodo.

    Con periodo mensual los meses son 1 fijo, así variar ``dias_mes`` no escala
    los costos mensuales de infraestructura.
    """
    sfx = scenario_cfg["periodo"]
    days = float(prm[f"dias_{sfx}"])
    if sfx == "mes":
        return sfx, days, 1.0
    months = float(prm.get(f"meses_{sfx}", days / MONTH_DAYS))
    return sfx, days, months


def free_spaces(prm: Mapping[str, float]) -> int:
    """F = max(1, round((1 − π)·N)), con redondeo .5 hacia arriba."""
    return max(1, int(np.floor((1 - prm["ocupacion"]) * prm["espacios"] + 0.5)))


def _check_rates(rates: Mapping[str, Any]) -> None:
    """Exige las seis tasas; no se asume 0 en silencio."""
    missing = [k for k in RATE_KEYS if k not in rates]
    if missing:
        raise ValueError(f"Faltan tasas en rates: {missing}; se requieren {list(RATE_KEYS)}")


def _probabilities(rates: Mapping[str, Any], prm: Mapping[str, float]) -> dict[str, Any]:
    """Paso 1: F, q_LL_eff, e, e_c y P0 (los libres verificados pasan a verde)."""
    pi = prm["ocupacion"]
    q_OL = np.asarray(rates["q_OL"], dtype=float)
    q_LL_eff = np.asarray(rates["q_LL"], dtype=float) + np.asarray(rates["q_LV"], dtype=float)
    num = pi * q_OL
    den = num + (1 - pi) * q_LL_eff
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
    F = free_spaces(prm)
    return {
        "F": F,
        "q_LL_eff": q_LL_eff,
        "e": e,
        "e_c": np.minimum(e, E_CAP),
        "P0": np.clip(1 - q_LL_eff, 0.0, 1.0) ** F,
    }


def _time_with_system(prob: Mapping[str, Any], prm: Mapping[str, float]) -> tuple[Any, Any]:
    """Paso 2: T_sys con tope en t_base + t_redireccion, y si el tope se activó."""
    e_c, P0 = prob["e_c"], prob["P0"]
    guided = prm["t_directo_min"] + (e_c / (1 - e_c)) * prm["t_redireccion_min"]
    t_guided = (1 - P0) * guided + P0 * prm["t_base_min"]
    cap = prm["t_base_min"] + prm["t_redireccion_min"]
    return np.minimum(t_guided, cap), t_guided > cap


def _abandonment(t: Any, prm: Mapping[str, float]) -> Any:
    """p_ab(t) = clip(abandono_base · t / t_base, 0, 1)."""
    return np.clip(prm.get("abandono_base", 0.0) * t / prm["t_base_min"], 0.0, 1.0)


def _amber_spaces(rates: Mapping[str, Any], prm: Mapping[str, float]) -> Any:
    """n_amber = N·(π·q_OV + (1 − π)·q_LV)."""
    pi = prm["ocupacion"]
    q_OV = np.asarray(rates["q_OV"], dtype=float)
    q_LV = np.asarray(rates["q_LV"], dtype=float)
    return prm["espacios"] * (pi * q_OV + (1 - pi) * q_LV)


def _system_costs(prm: Mapping[str, float], days: float, months: float, extra_ops: Any) -> dict[str, Any]:
    """Paso 4 (costos): verificación (personal extra), infraestructura e inferencia del periodo."""
    hours = prm["horas_operacion_dia"]
    pred = prm["espacios"] * (3600.0 / prm["refresh_s"]) * hours * days
    infra_month = (prm["n_camaras"] * prm["costo_camara_Q"]) / prm["meses_amortizacion"] \
        + prm["costo_servidor_mes_Q"]
    return {
        "pred": pred,
        "costo_verif": extra_ops * prm["salario_operador_Q_h"] * hours * days,
        "costo_infra": infra_month * months,
        "costo_inf": pred / 1000.0 * _cost_per_1000(prm),
    }


def _sensor_costs(prm: Mapping[str, float], months: float) -> float:
    """Costo de la alternativa con un sensor por espacio en el periodo (comparativa)."""
    monthly = (prm["espacios"] * prm["costo_sensor_instalado_Q"]) / prm["meses_amortizacion"] \
        + prm["mantenimiento_sensores_mes_Q"]
    return monthly * months


def _assemble(
    scenario_cfg: Mapping[str, Any],
    prm: Mapping[str, float],
    T_sys: Any,
    n_amber: Any,
    with_system: bool,
) -> dict[str, Any]:
    """Pasos 2-4 a partir de T_sys: ahorro, abandono, volúmenes, costos y valor."""
    sfx, days, months = _period(scenario_cfg, prm)
    t_base = prm["t_base_min"]
    A = t_base - T_sys
    p_ab_base = _abandonment(t_base, prm)
    p_ab_sys = _abandonment(T_sys, prm)
    delta_ab = p_ab_base - p_ab_sys
    arrivals = prm["llegadas_dia"] * days

    staff = verification_load(n_amber, prm)
    costs = _system_costs(prm, days, months, staff["operadores_extra"])
    if not with_system:
        costs = {k: 0.0 for k in costs}
    total_costs = costs["costo_infra"] + costs["costo_inf"] + costs["costo_verif"]
    income = arrivals * delta_ab * prm.get("ingreso_por_visita_Q", 0.0)
    hours_saved = arrivals * A / 60.0
    vt = time_value_q_h(prm)
    users_value = hours_saved * vt
    sensors = _sensor_costs(prm, months)

    out = {
        "T_sys": T_sys,
        "A": A,
        "minutos_ahorrados_por_llegada": A,
        "p_ab_base": p_ab_base,
        "p_ab_sys": p_ab_sys,
        "delta_abandono": delta_ab,
        "n_amber": n_amber,
        **staff,
        "costo_por_1000_pred_Q": _cost_per_1000(prm),
        "valor_tiempo_Q_h": vt,
        "meses_periodo": months,
        f"llegadas_{sfx}": arrivals,
        f"pred_{sfx}": costs["pred"],
        f"costo_verif_{sfx}": costs["costo_verif"],
        f"costo_infra_{sfx}": costs["costo_infra"],
        f"costo_inf_{sfx}": costs["costo_inf"],
        f"costos_{sfx}": total_costs,
        f"ingreso_recuperado_{sfx}": income,
        f"valor_operador_{sfx}": income - total_costs,
        f"horas_ahorradas_{sfx}": hours_saved,
        f"valor_usuarios_{sfx}_Q": users_value,
        f"valor_neto_{sfx}_Q": users_value - total_costs,
        f"costo_sensores_{sfx}": sensors,
        f"ahorro_vs_sensores_{sfx}": (sensors - total_costs) if with_system else 0.0,
    }
    objective = scenario_cfg["objective"]
    if objective not in out:
        raise KeyError(f"El objetivo '{objective}' no es un componente válido: {sorted(out)}")
    out["objective"] = out[objective]
    return out


def _to_native(values: dict[str, Any]) -> dict[str, Any]:
    """Convierte escalares de numpy a bool/float; deja los arreglos como arreglos."""
    native: dict[str, Any] = {}
    for key, val in values.items():
        if isinstance(val, (str, bool, int)):
            native[key] = val
        elif np.ndim(val) == 0:
            native[key] = bool(val) if np.asarray(val).dtype == bool else float(val)
        else:
            native[key] = np.asarray(val)
    return native


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def compute_value(rates: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Calcula todos los componentes de valor y ``objective`` para un escenario.

    ``rates`` debe traer las seis tasas ``q_OL, q_OV, q_OO, q_LL, q_LV, q_LO``
    (``ValueError`` si falta alguna). Pueden ser escalares (salida de
    ``three_state_rates``) o arreglos de numpy que se difunden entre sí; en ese
    caso cada componente es un arreglo (lo usa la búsqueda vectorizada de umbrales).
    """
    _check_rates(rates)
    prm = scenario_params(scenario_cfg)
    prob = _probabilities(rates, prm)
    T_sys, capped = _time_with_system(prob, prm)
    n_amber = _amber_spaces(rates, prm)
    out = {
        "scenario": scenario_cfg["name"],
        "objective_name": scenario_cfg["objective"],
        **prob,
        "t_sys_capped": capped,
        **_assemble(scenario_cfg, prm, T_sys, n_amber, with_system=True),
    }
    return _to_native(out)


def status_quo(scenario_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Valor sin sistema (B0): T_sys = t_base, A = 0 y sin costos → objetivo 0.

    No hay recomendación, así que ``e`` no aplica y se reporta 0; ``P0`` = 1.
    """
    prm = scenario_params(scenario_cfg)
    out = {
        "scenario": scenario_cfg["name"],
        "objective_name": scenario_cfg["objective"],
        "F": free_spaces(prm),
        "q_LL_eff": 0.0,
        "e": 0.0,
        "e_c": 0.0,
        "P0": 1.0,
        "t_sys_capped": False,
        **_assemble(scenario_cfg, prm, float(prm["t_base_min"]), 0.0, with_system=False),
    }
    return _to_native(out)


def with_param(scenario_cfg: Mapping[str, Any], name: str, value: float) -> dict[str, Any]:
    """Copia del escenario con un parámetro reemplazado (recortado si es probabilidad)."""
    cfg = copy.deepcopy(dict(scenario_cfg))
    cfg["params"] = scenario_params(cfg)
    if name not in cfg["params"]:
        raise KeyError(f"Parámetro desconocido: {name}")
    lo, hi = _PROB_BOUNDS.get(name, (-np.inf, np.inf))
    cfg["params"][name] = float(np.clip(value, lo, hi))
    return cfg


def _default_sensitivity_params(scenario_cfg: Mapping[str, Any]) -> list[str]:
    """Todos los parámetros numéricos distintos de cero (escalar 0 no cambia nada)."""
    return [k for k, v in scenario_params(scenario_cfg).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0]


def _evaluate_point(
    rates: Mapping[str, Any] | None,
    scenario_cfg: Mapping[str, Any],
    reoptimize: bool,
    y: Any,
    p: Any,
) -> tuple[float, float | None, float | None]:
    """Objetivo con tasas fijas o re-optimizando los umbrales."""
    if reoptimize:
        from cupo.thresholds import optimize  # importación local para evitar ciclo

        best = optimize(y, p, scenario_cfg)
        return best["value"], best["t_L"], best["t_H"]
    res = compute_value(rates, scenario_cfg)
    return res["objective"], rates.get("t_L"), rates.get("t_H")


def sensitivity(
    rates: Mapping[str, Any] | None,
    scenario_cfg: Mapping[str, Any],
    params: Iterable[str] | None = None,
    factors: Iterable[float] = (0.5, 0.8, 1.2, 1.5),
    reoptimize: bool = False,
    y: Any = None,
    p: Any = None,
) -> pd.DataFrame:
    """Tabla para el gráfico tornado: cada parámetro multiplicado por cada factor.

    Con ``reoptimize=False`` se mantienen las tasas ``rates`` (umbrales fijos);
    con ``reoptimize=True`` se re-optimizan ``t_L`` y ``t_H`` con ``y`` y ``p``
    para cada variación. Las probabilidades se recortan a su rango válido
    (``ocupacion`` ≤ 0.97) y esas filas se marcan con ``clipped=True``.
    Columnas: param, factor, base_param_value, param_value, clipped, objective,
    base_objective, delta, t_L, t_H.
    """
    if reoptimize and (y is None or p is None):
        raise ValueError("reoptimize=True requiere y y p")
    if not reoptimize and rates is None:
        raise ValueError("reoptimize=False requiere rates")
    names = list(params) if params is not None else _default_sensitivity_params(scenario_cfg)
    base_prm = scenario_params(scenario_cfg)
    base_obj, _, _ = _evaluate_point(rates, scenario_cfg, reoptimize, y, p)

    rows = []
    for name in names:
        for factor in factors:
            target = base_prm[name] * factor
            cfg = with_param(scenario_cfg, name, target)
            obj, t_L, t_H = _evaluate_point(rates, cfg, reoptimize, y, p)
            rows.append({
                "param": name,
                "factor": float(factor),
                "base_param_value": float(base_prm[name]),
                "param_value": float(cfg["params"][name]),
                "clipped": bool(not np.isclose(cfg["params"][name], target)),
                "objective": float(obj),
                "base_objective": float(base_obj),
                "delta": float(obj - base_obj),
                "t_L": t_L,
                "t_H": t_H,
            })
    return pd.DataFrame(rows)


def value_curve(y: Any, p: Any, scenario_cfg: Mapping[str, Any], grid_step: float = 0.01) -> dict[str, pd.DataFrame]:
    """Curvas de valor de un modelo.

    - ``by_t_H``: valor en función de ``t_H`` con el ``t_L`` óptimo para cada
      ``t_H``, junto con q_OL, q_LO y la tasa de verificación en ese punto.
    - ``single_threshold``: decisión de dos estados (``t_L = t_H = t``); valor
      frente a ``t`` junto con accuracy, F1 macro y la tasa ocupado→libre, para
      graficar valor frente a una métrica técnica.
    """
    from cupo.thresholds import curve_by_t_H, curve_single_threshold  # evita ciclo

    return {
        "by_t_H": curve_by_t_H(y, p, scenario_cfg, grid_step),
        "single_threshold": curve_single_threshold(y, p, scenario_cfg, grid_step),
    }
